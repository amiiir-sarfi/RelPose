import os
import clip
import tqdm
import yaml
import torch
import wandb
import torchvision
from torchvision import transforms as T
import argparse
from pathlib import Path

from model    import RelPose
from loader   import OOD
from datetime import datetime
from utils    import _convert_image_to_rgb, BICUBIC

from config import Config

# Config and Wandb
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/10-05-2022_23-38-21/clip/weights/iteration_145000.pth")
    # parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--random_order", type=bool, default=True)
    parser.add_argument("--mode", type=str, default="loop", choices=["loop", "mst"])
    parser.add_argument('--save_root', default="./outputs/10-05-2022_23-38-21", type=str,help='root directory for saving checkpoints')
    parser.add_argument("--embedding", default='clip', type=str,
                            choices=["clip", "resnet50"],
                            help='What type of embeddings per image'
                            )
    return parser

cfg = get_parser().parse_args()
cfg.vis_path = Path(cfg.save_root) / cfg.embedding / 'visual'
# Device
device = "cuda:0"
torch.cuda.set_device(device)

if not cfg.embedding == "clip":
    tfms = T.Compose([
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
else:
    # _, tfms = clip.load("ViT-B/32")
    tfms  = T.Compose([
        T.Resize(224, interpolation=BICUBIC),
        T.CenterCrop(224),
        _convert_image_to_rgb,
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


use_clip = cfg.embedding == "clip"
md = RelPose(
    use_clip,
    num_layers=4,                
    num_pe_bases=8,
    hidden_size=256,
    num_queries=36864,  # To match recursion level = 3
    num_images=2,
    metadata_size=0,
    )
md.cuda()
md.load_model(cfg.checkpoint)
md.eval()
for base_deg in [15,45,90,150,180,195+15, 240, 285, 315]:
    ds = OOD(
        transform=tfms,
        base_rotation=base_deg
    )

    dl = torch.utils.data.DataLoader(
        ds,
        shuffle=False,
        batch_size=len(ds),
        num_workers=0,
    )


    # evaluate
    data = next(iter(dl))
    azims1 = data["azim1"]
    azims2 = data["azim2"]
    with torch.no_grad():
        imgs1, imgs2 = data["image_one"].cuda(), data["image_two"].cuda()
        relative_rotation = data["R"].cuda()
        queries, logits = md(
            images1=imgs1,
            images2=imgs2,
            gt_rotation=relative_rotation,
        )
        visuals = md.make_visualization(
            images1=imgs1.cuda(),
            images2=imgs2.cuda(),
            rotations=queries,
            probabilities=logits.softmax(dim=-1),
            num_vis=400
        )
        assert len(visuals) == len(azims1)
        for v, image in enumerate(visuals):
            torchvision.utils.save_image(torch.from_numpy(image).permute(2, 0, 1) / 255.0, cfg.vis_path / f"{azims1[v]}_{azims2[v]}.png" )
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=wandb.config["Learning_Rate"], steps_per_epoch=1, epochs=wandb.config["Epochs"])

