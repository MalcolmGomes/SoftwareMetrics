import numpy as np
from PIL import Image

def get_salient_patch(img, lo, hi):
    # Make HSV version
    HSVim = img.convert('HSV')
    # Make numpy versions
    RGBna = np.array(img)
    HSVna = np.array(HSVim)
    # Extract Hue
    H = HSVna[:,:,0]
    # Find all red pixels, i.e. where 340 < Hue < 100
    # Rescale to 0-255, rather than 0-360 because we are using uint8
    lo = int((lo * 255) / 360)
    hi = int((hi * 255) / 360)
    red = np.where((H>lo) | (H<hi))
    # Make all red pixels black in original image
    RGBna[:] = [0,0,0]
    RGBna[red] = [255,255,255]
    count = red[0].size
    print("Pixels matched: {}".format(count))
    result = Image.fromarray(RGBna)
    return result

def get_difference_image(gradcam, fixation_map):
    fixation_map = fixation_map.resize(size)
    gradcam_array = np.array(gradcam)
    fixation_map_array = np.array(fixation_map)

    difference_array = fixation_map_array * gradcam_array 
    difference = Image.fromarray(difference_array)
    return difference

def get_score(difference, size):
    black = 0
    white = 0
    for pixel in difference.getdata():
        if pixel == (0, 0, 0): black +=1
        if pixel == (255, 255, 255): white +=1

    total = black + white
    pixel_count = size * size
    # score = total / pixel_count 
    score = pixel_count / total
    alt = total / pixel_count
    print('White Pixels:', white, 'Black Pixels:', black, 'Total Pixel Count:', total, 'Score:', score)
    return score, alt

#---------------------------------------------- Start Main ---------------------------------------------------
image_name = 'COCO_train2014_000000001014'
size = 512, 512
lo,hi = 340, 50

image_path = './data/images/train/' + image_name + '.jpg'
alexnet_gradcam_path = './output/alexnet/images/train/' + image_name + '.jpg'
resnet_gradcam_path = './output/resnet101/images/train/' + image_name + '.jpg'
densenet_gradcam_path = './output/densenet/images/train/' + image_name + '.jpg'

image = Image.open(image_path).convert('RGB').resize(size)
gradcam_img = Image.open(resnet_gradcam_path)
gradcam = get_salient_patch(gradcam_img, lo,hi)


fixation_map_path = 'data/maps/train/' + image_name + '.png'
fixation_map = Image.open(fixation_map_path).convert('RGB')

difference = get_difference_image(gradcam, fixation_map)
score, alt = get_score(difference, 512)

# print(score)
image.show()
gradcam.show()
gradcam.save('gradcam.jpg')
fixation_map.show()
fixation_map.save('fixation.jpg')
difference.show()
difference.save('difference.jpg')