# Python System Packages
import time
import sys
import os

# Scientific Packages
import numpy as np
from PIL import Image

# PyTorch Packages
import torch
import torchvision
from torchvision import models
from torchvision import transforms
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

# GradCAM Packages
from gradcam.utils import visualize_cam, Normalize
from gradcam.gradcam import GradCAM, GradCAMpp



# -----------------------------------------------------------------------------

data_path = 'data/images/'

# normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# torch_img = torch.from_numpy(np.asarray(pil_img)).permute(2, 0, 1).unsqueeze(0).float().div(255).cuda()
# torch_img = F.upsample(torch_img, size=(224, 224), mode='bilinear', align_corners=False)
# normed_torch_img = normalizer(torch_img)

coco_dataset = torchvision.datasets.ImageFolder(
    root=data_path,
    transform=torchvision.transforms.ToTensor()
)
coco_loader = torch.utils.data.DataLoader(
    coco_dataset,
    batch_size=64,
    num_workers=0,
    shuffle=True
)

# print(type(coco_loader))

# for img in coco_loader:
#     print(type(img[2]))

for img in coco_loader:    
    normalizer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    
    torch_img = img[0].unsqueeze(0).float().div(255).cuda()
    torch_img = F.upsample(torch_img, size=(224, 224), mode='bilinear', align_corners=False)
    normed_torch_img = normalizer(torch_img)

# img_dir = 'data/images/train/'
# img_name = 'COCO_train2014_000000000009.jpg'
# img_path = os.path.join(img_dir, img_name)
# pil_img = PIL.Image.open(img_path)
# # pil_img.show()
# print(type(pil_img)

