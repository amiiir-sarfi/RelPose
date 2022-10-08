import os
import clip
import torch
import torchvision

from loader import OOD
from model import Model
from utils import Video

MODEL_V = "CLIP"
PATH = "output/%s/weights/epoch_99500.pth" % MODEL_V
OUTPUT_PATH = "output/%s/visual" % MODEL_V
CLIP = True

device = torch.device("cuda:" + "0")
torch.cuda.set_device(device)

if CLIP:
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    normalize = torchvision.transforms.Normalize(mean=mean, std=std)

    tfms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomCrop(224),
        normalize,
    ])
else:
    _, tfms = clip.load("ViT-B/32")

md = Model(CLIP)
md.load_model(PATH)
md.cuda()


ds = OOD("./custom", transform=tfms)

# for IDX in range(5, 10):
    
IDX = 0
video = Video(os.path.join(OUTPUT_PATH), name="Car_%d_ood.mp4" % IDX)

with torch.no_grad():

    data = ds[IDX]

    for idx, img_two in enumerate(data["image_two"]):

        queries, logits = md(
            data["image_one"][0].unsqueeze(0).cuda(),
            img_two.unsqueeze(0).cuda(),
            data["R"][idx].unsqueeze(0).cuda(),
        )

        visuals = md.make_visualization(
            images1=data["image_one"][0].unsqueeze(0).cuda(),
            images2=img_two.unsqueeze(0),
            rotations=queries,
            probabilities=logits.softmax(dim=-1),
        )

        for v, image in enumerate(visuals):
            # torchvision.utils.save_image(torch.from_numpy(image).permute(2, 0, 1) / 255.0, os.path.join(OUTPUT_PATH, "angle_%d.png" % (idx)))
            video.ready_image( torchvision.utils.make_grid(torch.from_numpy(image) / 255.0) )

video.writer.close()