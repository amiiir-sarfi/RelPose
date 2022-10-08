# import glm
import PIL
from PIL import Image
import torch
import numpy as np
from pathlib import Path

from os                 import walk, path
from torch              import from_numpy
from random             import randint, choice, choices
from torch.utils.data   import Dataset
from utils              import look_at_view_transform
from torchvision import transforms as T

class OOD(Dataset):

    def __init__(self, folder_path='./original', base_rotation=325, num_workers=1, transform=None):

        assert (base_rotation % 15 == 0 and base_rotation<360, 
            "rotation should be a multiply of 15 and not bigger than 345")
        
        self.folder_path    = folder_path
        self.azimuths       = torch.arange(0, 359, 15).tolist

        self.transform = transform

        self.distance  = 1.0
        self.elevation = 30.0
        self.classes   = 15
        
        paths = sorted(list(Path(folder_path).glob('*.png')))
        self.images = [(Image.open(p).convert("RGB"), int(p.stem)) for p in paths] # image, label
        # self.images = [(Image.fromarray(np.zeros((224,224,3), dtype=np.uint8)), int(p.stem)) for p in paths]
        self.base_rotation = base_rotation

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        base_img_idx = self.base_rotation // 15
        img1, azim1 = self.images[base_img_idx]
        img2, azim2 = self.images[idx]
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
            
        rotations, _ =  look_at_view_transform(
            dist=self.distance,
            elev=self.elevation,
            azim=[azim1, azim2],
            up=((0, -1, 0),)
        )

        R1, R2 = rotations
        
        return {
            "image_one": img1,
            "image_two": img2,
            "R1": R1,
            "R2": R2,
            "azim1": azim1,
            "azim2": azim2,
            "R": R2 @ R1.T,
            "R_inv": R1 @ R2.T
        }

# class ElementWise(Dataset):

#     def __init__(self, folder_path, num_workers=1, transform=None):

#         self.folder_path    = folder_path
#         self.object_ids     = [ full_path.split("_")[0] for full_path in next(walk(path.join(folder_path, "0")), (None, None, []))[2] ]
#         self.azimuths       = sorted([ int(x) for x in next(walk(folder_path))[1] ])

#         self.transform = transform

#         self.distance  = 1.0
#         self.elevation = 30.0
#         self.classes   = 15

#     def __len__(self):
#         return len(self.object_ids)

#     def __getitem__(self, index):
        
#         id_object       = self.object_ids[index]

#         imgs = []
#         for azim in range(360): 
#             if self.transform:
#                 imgs.append(
#                     self.transform(PIL.Image.open(path.join(self.folder_path, str(azim), "%s_%d.png" % (id_object, azim))))
#                 )

#         rotations, _ =  look_at_view_transform(
#             dist=self.distance,
#             elev=self.elevation,
#             azim=torch.arange(0, 360, 1),
#             up=((0, -1, 0),)
#         )

#         return {
#             "image_one": imgs[0].unsqueeze(0).repeat(359, 1, 1, 1),
#             "image_two": torch.stack(imgs[1:]),
#             "R1": rotations[0],
#             "R2": rotations[1:],
#             "R": torch.bmm(
#                 rotations[1:],
#                 rotations[0].unsqueeze(0).repeat(359, 1, 1).transpose(1, 2)
#             )
#         }

class PairWiseDataset(Dataset):

    def __init__(self, folder_path="../Datasets/cars/", transform=None):

        self.folder_path    = folder_path
        # self.dataset_length = batch_size * num_workers
        self.object_ids     = [ full_path.split("_")[0] for full_path in next(walk(path.join(folder_path, "0")), (None, None, []))[2] ]
        self.azimuths       = sorted([ int(x) for x in next(walk(folder_path))[1] ])

        if transform is None:
            transform = T.Compose(
                [
                    # T.Resize((image_size, image_size)),
                    T.ToTensor(),
                    # T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
        self.transform = transform

        self.distance  = 1.0
        self.elevation = 30.0
        
    def __len__(self):
        return len(self.object_ids)

    def __getitem__(self, idx):
        az_one, az_two  = choices(self.azimuths, k=2)
        id_object       = self.object_ids[idx]

        az_one_img = PIL.Image.open( path.join(self.folder_path, str(az_one), "%s_%d.png" % (id_object, az_one) ) )
        az_two_img = PIL.Image.open( path.join(self.folder_path, str(az_two), "%s_%d.png" % (id_object, az_two) ) )

        if self.transform:
            az_one_img = self.transform(az_one_img)
            az_two_img = self.transform(az_two_img)

        rotations, _ =  look_at_view_transform(
            dist=self.distance,
            elev=self.elevation,
            azim=[az_one, az_two],
            up=((0, -1, 0),)
        )

        R1 = rotations[0]
        R2 = rotations[1]

        return {
            "image_one": az_one_img,
            "image_two": az_two_img,
            "R1": R1,
            "R2": R2,
            "R": R2 @ R1.T,
            "R_inv": R1 @ R2.T
        }
        
if __name__ == "__main__":
    OOD()
    # pwd = PairWiseDataset()
    # sample = pwd[2]
    # i1, i2, R1, R2, R, R_inv = sample.values()
    # print(i1.shape, i2.shape, R)