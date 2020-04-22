# Python System Packages
import time
import sys
import os

# Scientific Packages
import numpy as np
from PIL import Image

# PyTorch Packages
import torch
from torchvision import models
from torchvision import transforms
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

# GradCAM Packages
from gradcam.utils import visualize_cam, Normalize
from gradcam.gradcam import GradCAM, GradCAMpp


# ------------------------------------------------------------------

alexnet = models.alexnet(pretrained=True)
alexnet.eval(), alexnet.cuda();

vgg = models.vgg16(pretrained=True)
vgg.eval(), vgg.cuda();

resnet = models.resnet101(pretrained=True)
resnet.eval(), resnet.cuda();

densenet = models.densenet161(pretrained=True)
densenet.eval(), densenet.cuda();

squeezenet = models.squeezenet1_1(pretrained=True)
squeezenet.eval(), squeezenet.cuda();


cam_dict = dict()

alexnet_model_dict = dict(type='alexnet', arch=alexnet, layer_name='features_11', input_size=(224, 224))
alexnet_gradcam = GradCAM(alexnet_model_dict, True)
alexnet_gradcampp = GradCAMpp(alexnet_model_dict, True)
cam_dict['alexnet'] = [alexnet_gradcam, alexnet_gradcampp]

vgg_model_dict = dict(type='vgg', arch=vgg, layer_name='features_29', input_size=(224, 224))
vgg_gradcam = GradCAM(vgg_model_dict, True)
vgg_gradcampp = GradCAMpp(vgg_model_dict, True)
cam_dict['vgg'] = [vgg_gradcam, vgg_gradcampp]

resnet_model_dict = dict(type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
resnet_gradcam = GradCAM(resnet_model_dict, True)
resnet_gradcampp = GradCAMpp(resnet_model_dict, True)
cam_dict['resnet'] = [resnet_gradcam, resnet_gradcampp]

densenet_model_dict = dict(type='densenet', arch=densenet, layer_name='features_norm5', input_size=(224, 224))
densenet_gradcam = GradCAM(densenet_model_dict, True)
densenet_gradcampp = GradCAMpp(densenet_model_dict, True)
cam_dict['densenet'] = [densenet_gradcam, densenet_gradcampp]

squeezenet_model_dict = dict(type='squeezenet', arch=squeezenet, layer_name='features_12_expand3x3_activation', input_size=(224, 224))
squeezenet_gradcam = GradCAM(squeezenet_model_dict, True)
squeezenet_gradcampp = GradCAMpp(squeezenet_model_dict, True)
cam_dict['squeezenet'] = [squeezenet_gradcam, squeezenet_gradcampp]






# ------------------------------------------------------------------

# images = []
# for gradcam, gradcam_pp in cam_dict.values():
#     mask, _ = gradcam(normed_torch_img)
#     mask_np = mask.detach().cpu().numpy()
#     heatmap, result = visualize_cam(mask.detach().cpu().numpy(), torch_img)

#     mask_pp, _ = gradcam_pp(normed_torch_img)
#     mask_pp_np = mask_pp.detach().cpu().numpy()
#     heatmap_pp, result_pp = visualize_cam(mask_pp_np, torch_img)
    
#     images.append(torch.stack([torch_img.squeeze().cpu(), heatmap, heatmap_pp, result, result_pp], 0))
    
# images = make_grid(torch.cat(images, 0), nrow=5)