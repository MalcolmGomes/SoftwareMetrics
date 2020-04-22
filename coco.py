import torch
import time
import sys
import requests
import os
import torchvision
from torchvision import models
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.utils import make_grid, save_image
# GradCAM Packages
from gradcam.utils import visualize_cam, Normalize
from gradcam.gradcam import GradCAM, GradCAMpp

def generate_saliency_map(img, img_name, directory, model_dict):

    # Normalize PIL Image, Must be in PIL format
    normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    torch_img = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).unsqueeze(0).float().div(255).cuda()
    torch_img = F.upsample(torch_img, size=(512, 512), mode='bilinear', align_corners=False)
    normed_torch_img = normalizer(torch_img)
    # gradcam = GradCAM(model_dict, True)    
    gradcam_pp = GradCAMpp(model_dict, True)    
    mask_pp, _ = gradcam_pp(normed_torch_img)
    mask_pp_np = mask_pp.detach().cpu().numpy()
    heatmap_pp, result_pp = visualize_cam(mask_pp_np, torch_img)

    # Only going to use result_pp (Saliency Map from GradCAM++)
    os.makedirs(directory, exist_ok=True)
    output_name = img_name
    output_path = os.path.join(directory, output_name)
    save_image(result_pp, output_path)

    return output_path

# Seem to be running out of GPU memory with the below code.
images_dict = [
    {
        'directory_name' : 'data/images/train/',
        'output_directory': 'output/densenet/images/train/',        
    },
    {
        'directory_name' : 'data/images/test/',
        'output_directory': 'output/densenet/images/test/',        
    },
    {
        'directory_name' : 'data/images/val/',
        'output_directory': 'output/densenet/images/val/',        
    }
]


# alexnet = models.alexnet(pretrained=True)
# alexnet.eval(), alexnet.cuda();
# alexnet_model_dict = dict(type='alexnet', arch=alexnet, layer_name='features_11', input_size=(512, 512))

# vgg = models.vgg16(pretrained=True)
# vgg.eval(), vgg.cuda();
# vgg_model_dict = dict(type='vgg', arch=vgg, layer_name='features_29', input_size=(512, 512))

# resnet = models.resnet101(pretrained=True)
# resnet.eval(), resnet.cuda();
# resnet_model_dict = dict(type='resnet', arch=resnet, layer_name='layer4', input_size=(512, 512))

densenet = models.densenet161(pretrained=True)
densenet.eval(), densenet.cuda();
densenet_model_dict = dict(type='densenet', arch=densenet, layer_name='features_norm5', input_size=(512, 512))

# squeezenet = models.squeezenet1_1(pretrained=True)
# squeezenet.eval(), squeezenet.cuda();
# squeezenet_model_dict = dict(type='squeezenet', arch=squeezenet, layer_name='features_12_expand3x3_activation', input_size=(512, 512))


for images_dir in images_dict:
    directory_name = images_dir['directory_name']
    output_directory = images_dir['output_directory']
    directory = os.fsencode(directory_name)
    model_dict = densenet_model_dict
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        image_path = directory_name + filename
        img = Image.open(image_path)
        generate_saliency_map(img, filename, output_directory, model_dict)
        torch.cuda.empty_cache() 


# data_path = 'data/images/'

# coco_dataset = torchvision.datasets.ImageFolder(
#     root=data_path,
#     transform=torchvision.transforms.ToTensor()
# )
# coco_loader = torch.utils.data.DataLoader(
#     coco_dataset,
#     batch_size=1,
#     num_workers=0,
# )

# for coco_img in enumerate(coco_loader):
#     generate_saliency_map(coco_img)
