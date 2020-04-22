# import PIL as Image
from PIL import Image
import numpy as np

image_path = './data/images/train/COCO_train2014_000000000510.jpg'
alexnet_gradcam_path = './output/alexnet/images/train/COCO_train2014_000000000510.jpg'
resnet_gradcam_path = './output/resnet101/images/train/COCO_train2014_000000000510.jpg'
densenet_gradcam_path = './output/densenet/images/train/COCO_train2014_000000000510.jpg'

image = Image.open(image_path)
alexnet_gradcam = Image.open(alexnet_gradcam_path)
resnet_gradcam = Image.open(resnet_gradcam_path)
densenet_gradcam = Image.open(densenet_gradcam_path)

# image.show(title=image_path)
# alexnet_gradcam.show(title=alexnet_gradcam_path)
# resnet_gradcam.show(title=resnet_gradcam_path)
# densenet_gradcam.show(title=densenet_gradcam_path)

np_alex = np.array(alexnet_gradcam)
np_res = np.array(resnet_gradcam)

print(np_res.shape)

print(np_res[0].shape)

print(np.amax(np_res))

redChannel = Image.fromarray(np_res[0])

# a = np.array(np_res)
# a[:,:,0] *=0
# a[:,:,1] *=0
# a = Image.fromarray(a)
# a.show()

# a = np.array(np_res)
# a[:,:,1] *=0
# a[:,:,2] *=0
# a = Image.fromarray(a)
# a.show()

# a = np.array(np_res)
# a[:,:,0] *=0
# a[:,:,2] *=0
# a = Image.fromarray(a)
# a.show()

red, green, blue = resnet_gradcam.split()
# red.show()
# green.show()
# blue.show()