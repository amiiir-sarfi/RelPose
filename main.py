import os
import clip
import tqdm
import yaml
import torch
import wandb
import torchvision
from torchvision import transforms as T

from model    import RelPose
from loader   import PairWiseDataset
from datetime import datetime
from utils    import _convert_image_to_rgb, BICUBIC

from config import Config

# Config and Wandb
cfg = Config().parse(None)
if not cfg.no_wandb:
    proj_name = "Rel-Pose" if cfg.wandb_project_name=='' else cfg.wandb_project_name
    wandb.init(
        project=proj_name, 
        entity=cfg.wandb_entity, 
        name=cfg.wandb_experiment_name, 
        config=cfg, 
        mode = cfg.wandb_offline
    )
    with open(cfg.save_root / "config.yml", 'w') as outfile:
        yaml.dump(wandb.config.as_dict(), outfile, default_flow_style=False)


# Device
device = "cuda:0"
torch.cuda.set_device(device)


if not cfg.embedding == "clip":
    tfms = T.Compose([
        T.RandomCrop(224, padding=cfg.crop_padding, fill=255),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
else:
    # _, tfms = clip.load("ViT-B/32")
    tfms  = T.Compose([
        T.Resize(224, interpolation=BICUBIC),
        # T.CenterCrop(224),
        T.RandomCrop(224, padding=cfg.crop_padding, fill=255),
        _convert_image_to_rgb,
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    
    
ds = PairWiseDataset(
    folder_path=cfg.data_dir,
    transform=tfms
)

dl = torch.utils.data.DataLoader(
    ds,
    batch_size=cfg.batch_size,
    num_workers=cfg.num_workers,
)

use_clip = cfg.embedding == "clip"
md = RelPose(
    use_clip,
    num_layers=4,                
    num_pe_bases=8,
    hidden_size=256,
    sample_mode=cfg.sampling_mode,
    recursion_level=cfg.recursion_level,
    num_queries=36864,  # To match recursion level = 3
    num_images=2,
    metadata_size=0,
    freeze_encoder=cfg.freeze_encoder,
    )
md.cuda()

params = list(md.feature_extractor_params) + list(md.embed_feature.parameters()) + list(md.embed_query.parameters())

optimizer = torch.optim.AdamW(
    params,
    lr=cfg.lr
)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=wandb.config["Learning_Rate"], steps_per_epoch=1, epochs=wandb.config["Epochs"])


for e in range(cfg.epochs):
    if e > cfg.debug_epochs and cfg.debug:
        break
    t_loop = tqdm.tqdm(dl, disable=False)
    for idx, data in enumerate(t_loop):
        it = idx + e*len(dl)
        if it > cfg.debug_batches and cfg.debug:
            break

        imgs1, imgs2 = data["image_one"].cuda(), data["image_two"].cuda()
        relative_rotation = data["R"].cuda()
        metadata = None # REMOVE LATER TODO
        
        queries, logits = md(
            images1=data["image_one"].cuda(),
            images2=data["image_two"].cuda(),
            gt_rotation=data["R"].cuda(),
        )

        log_prob = torch.log_softmax(logits, dim=-1)
        loss = -torch.mean(log_prob[:, 0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        t_loop.set_description("Loss = %f" % loss.item())


        # visualization
        if it % cfg.vis_interval == 0:
            with torch.no_grad():
                visuals = md.make_visualization(
                    images1=imgs1.cuda(),
                    images2=imgs2.cuda(),
                    rotations=queries,
                    probabilities=logits.softmax(dim=-1)
                )

                for v, image in enumerate(visuals):
                    torchvision.utils.save_image(torch.from_numpy(image).permute(2, 0, 1) / 255.0, cfg.progress_path / f"iteration_{it}_{v}.png" )

                if not cfg.no_wandb:
                    wandb.log({
                        "Loss": loss.item(),
                        "LR": optimizer.param_groups[0]["lr"],
                        "Image": wandb.Image(str(cfg.progress_path / f"iteration_{it}_{v}.png"))
                    })

        else:
            if not cfg.no_wandb:
                wandb.log({
                    "Loss": loss.item(),
                    "LR": optimizer.param_groups[0]["lr"],
                })
            
        if it % cfg.save_interval == 0:
            md.save_model(cfg.ckpt_path / f"iteration_{it}.pth")