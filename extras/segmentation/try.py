import cv2
import numpy as np


class Segment:
    def __init__(self, segments=3):
        # define number of segments, with default 3
        self.segments = segments

    def kmeans(self, image):
        image = cv2.GaussianBlur(image, (7, 7), 0)
        cv2.imshow("GaussianBlur", image)
        cv2.waitKey(0)

        vectorized = image.reshape(-1, 3)

        vectorized = np.float32(vectorized)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(vectorized, self.segments, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        res = center[label.flatten()]
        segmented_image = res.reshape(image.shape)
        return label.reshape((image.shape[0], image.shape[1])), segmented_image.astype(np.uint8)

    def extractComponent(self, image, label_image, label):
        component = np.zeros(image.shape, np.uint8)
        component[label_image == label] = image[label_image == label]
        return component


image = cv2.imread("0.jpg")
no_of_labels = 2

seg = Segment(no_of_labels)
label, result = seg.kmeans(image)

cv2.imshow("segmented", result)
cv2.waitKey(0)

for label_id in range(no_of_labels):
    result = seg.extractComponent(image, label, label_id)
    cv2.imshow("extracted label_id=%d" % label_id, result)
    cv2.waitKey(0)
