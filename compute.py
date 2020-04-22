import numpy as np
import os
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
    # print("Pixels matched: {}".format(count))
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
    # print('White Pixels:', white, 'Black Pixels:', black, 'Total Pixel Count:', total, 'Score:', score)
    return score, alt

#---------------------------------------------- Start Main ---------------------------------------------------
# image_name = 'COCO_train2014_000000000510'
# size = 512, 512
# lo,hi = 340, 50

# image_path = './data/images/train/' + image_name + '.jpg'
# alexnet_gradcam_path = './output/alexnet/images/train/' + image_name + '.jpg'
# resnet_gradcam_path = './output/resnet101/images/train/' + image_name + '.jpg'
# densenet_gradcam_path = './output/densenet/images/train/' + image_name + '.jpg'

# image = Image.open(image_path).convert('RGB').resize(size)
# gradcam_img = Image.open(resnet_gradcam_path)
# gradcam = get_salient_patch(gradcam_img, lo,hi)


# fixation_map_path = 'data/maps/train/' + image_name + '.png'
# fixation_map = Image.open(fixation_map_path).convert('RGB')

# difference = get_difference_image(gradcam, fixation_map)
# score, alt = get_score(difference, 512)



f = open("scores.csv", "a")
dirs = ['./output/alexnet/images/train/', './output/resnet101/images/train/', './output/densenet/images/train/']
size = 512, 512
lo,hi = 340, 50
f.write("# Writting to the csv file.")

checker = 0
counter = 0
for directory_name in dirs:
    directory = os.fsencode(directory_name)
    f.write("\n\nFrom" + directory_name)
    for image_file in os.listdir(directory):
        filename = os.fsdecode(image_file)
        image_path = directory_name + filename
        fixation_map_path = 'data/maps/train/' + filename[0:28] + 'png'
        gradcam_img = Image.open(image_path)
        gradcam = get_salient_patch(gradcam_img, lo,hi)
        fixation_map = Image.open(fixation_map_path).convert('RGB')
        difference = get_difference_image(gradcam, fixation_map)
        score, alt = get_score(difference, 512)
        rowtext = "\n" + filename + ',' + str(score) + ',' + str(alt)
        f.write(rowtext)
        checker+=1
        counter+=1
        if(checker == 500): 
            print(counter, 'files have been processed.')
            checker = 0
    print('Finished processing', directory_name)
    
f.close()
print('Done!')





# print(score)
# image.show()
# gradcam.show()
# fixation_map.show()
# difference.show()