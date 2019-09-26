import cv2
import numpy as np
import operator
# from src.new_setup import get_bin_threshold
# from digits_training.test_network import get_prediction
filePath = ""


def get_numbers_images(img, bin_threshold):
    # image = cv2.imread('/home/thejusp/Desktop/AMR/dataset/sample/12.jpg')
    # cv2.imshow('image',img)
    segmented = []
    coords = []

    kernel = np.ones((3, 3), np.uint8)
    # thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    ret, thresh = cv2.threshold(img, bin_threshold, 255, cv2.THRESH_BINARY)  # | cv2.THRESH_OTSU)

    edges = cv2.Canny(thresh, 100, 150)
    _, cnts, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        # print(area)
        if (area > 5000):
            image = image[y:y + h, x:x + w]
            img = img[y:y + h, x:x + w]
            break

    # cv2.imshow('cropped image',image)

    thresh1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # cv2.imshow('otsu',thresh1)
    thresh2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 19, 1)

    # cv2.imshow('adaptive thresholding',thresh2)
    thresh = np.bitwise_and(thresh1, thresh2)
    edges = cv2.Canny(thresh, 110, 150, 200)

    # cv2.imshow('combined',thresh)

    _, cnts, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)

    ret, thresh = cv2.threshold(thresh, bin_threshold, 255, cv2.THRESH_BINARY_INV)  # | cv2.THRESH_OTSU)

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)

        area = cv2.contourArea(c)
        # areas.append((c, area))
        _, cnts, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)

        if (area > 8) and (h > 15) and (w > 10):
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # print ((x,y,w,h))
            coords.append((thresh[y:y + h, x:x + w], x))

    coords = sorted(coords, key=operator.itemgetter(1))
    numbers_images = []
    for ele in coords:
        # cv2.imwrite("segmented-%d.jpg"%count,ele[0])
        numbers_images.append(ele[0])
        # count += 1

    # rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # cv2.imshow('edges',edges)
    # cv2.imshow('bounding',image)

    # cv2.imwrite(filePath + "contour-canny%d.jpg" % 100, image)

    # cv2.imwrite(filePath + "canny%d.jpg" % 100, edges)
    # cv2.waitKey(0)
    return numbers_images


if __name__ == "__main__":
    img = cv2.imread('/home/thejusp/Desktop/AMR/dataset/sample/10.jpg', 0)
    bin_threshold = get_bin_threshold(img)
    numbers_images = get_numbers_images(img, bin_threshold)
    # print(numbers_images)
    for img in numbers_images:
        cv2.imshow("img", img)
        print(get_prediction([img]))
        cv2.waitKey(0)
