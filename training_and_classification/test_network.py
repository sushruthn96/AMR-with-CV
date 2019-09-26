# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
# from src.new_setup import *


class Prediction:

    def __init__(self, model='/home/thejusp/Desktop/AMR/digits_training/try-model.model'):
        self.model = load_model(model)
        self.list = []
        for i in range(ord('0'), ord('9') + 1):
            self.list.append(chr(i))


    def get_prediction(self, images):

        for image in images:
            # Select the roi
            image_copy = image.copy()


            # cv2.imwrite('temp.jpg', image)
            # image = cv2.imread('temp.jpg')

            image = cv2.resize(image, (32, 32))
            image = image.astype("float") / 255.0
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            # load the trained convolutional neural network
            # print("[INFO] loading network...")

            results = self.model.predict(image)[0]
            # print(results)
            proba = max(results)
            max_index = np.argmax(results)
            label = self.list[max_index]
            # print(' result is ', label)

            # label = "{}: {:.2f}%".format(label, proba * 100)

            # draw the label on the image
            # output = imutils.resize(image_copy, width=400)
            # cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.7, (0, 255, 0), 2)

            return label
            # show the output image
            # cv2.imshow("Output", output)
            # cv2.waitKey(0)
            # cv2.destroyWindow("Output")


if __name__ == "__main__":
    path_to_test_image = '/home/thejusp/Desktop/AMR/dataset/sample/10.jpg'
    image = cv2.imread(path_to_test_image, 0)
    x, y, w, h = cv2.selectROI(image)
    image = image[y:y + h, x:x + w]
    bin_threshold = get_bin_threshold(image)
    ret, image = cv2.threshold(image, bin_threshold, 255, cv2.THRESH_BINARY_INV)
    # print(images)
    result = get_prediction([image])
    print(result)