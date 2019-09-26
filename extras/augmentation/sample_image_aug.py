import imgaug as ia
from imgaug import augmenters as iaa
import cv2
import numpy as np
import os
import matplotlib

matplotlib.use('TkAgg')

ia.seed(1)

transformations = [

    iaa.Crop(percent=(0, 0.1)),  # random crops
    # iaa.Fliplr(0.5), # horizontal flips
    # iaa.Flipud(0.4), # vertically flip 20% of all images

    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),

    # Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.75, 1.5)),

    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),

    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),

    # Sharpen each image, overlay the result with the original
    # image using an alpha between 0 (no sharpening) and 1
    # (full sharpening effect).
    # iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

    # Add a value of -10 to 10 to each pixel.
    iaa.Add((-10, 10), per_channel=0.5),
]


def my_fun(image, target_folder, idx1):
    global transformations
    seq = iaa.Sequential(transformations[:], random_order=True)  # apply augmenters in random order
    images = np.array([image for _ in range(30)], dtype=np.uint8)
    images_aug = seq.augment_images(images)
    idx2 = 1
    for img in images_aug:
        cv2.imwrite(target_folder + '/' + str(idx1).zfill(5) + '_' + str(idx2).zfill(5) + '.jpg', img)
        idx2 += 1


def checkdir(path_dir):
    if not os.path.exists(path_dir):
        os.mkdir(path_dir)


src_folder = "/Users/thejushp/Desktop/AMR/keras_amr/dataset/meters3_cropped"
target_root_folder = "/Users/thejushp/Desktop/AMR/keras_amr/dataset/augmented"

folders = [folder for folder in os.listdir(src_folder) if not folder.startswith('.')]
folders.sort()

for folder in folders:
    image_files = [image_file for image_file in os.listdir(src_folder+'/'+folder) if not image_file.startswith('.')]

    idx1 = 1
    for image_file in image_files:
        image = cv2.imread(src_folder+'/'+folder+'/'+image_file)
        checkdir(target_root_folder + '/' + folder)
        my_fun(image, target_root_folder + '/' + folder, idx1)
        idx1+=1
