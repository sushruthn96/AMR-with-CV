import cv2
import json
import numpy as np
import math
from src.new_setup import draw_grid

params = {}
image_file = input("Enter the location of the image: ")
image = cv2.imread(image_file)
# image_copy = image.copy()
# height, width = image.shape[:2]

# x = 0
# y = 0
#
# while x < width:
#     x += int(width / 8)
#     y += int(height / 8)
#     cv2.line(image, (x, 0), (x, height), (0, 255, 0), 1)
#     cv2.line(image, (0, y), (width, y), (0, 255, 0), 1)

image_copy = image.copy()
x, y, w, h = cv2.selectROI(draw_grid(image_copy))
params.update({"xywh": (x, y, w, h)})

roi = image[y:y + h, x:x + w]

cv2.imshow("roi", roi)
cv2.waitKey(0)

# close all open windows
cv2.destroyAllWindows()

img_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
gaussian_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

# ret, thresh1 = cv2.threshold(img2, 160, 255, cv2.THRESH_BINARY)
# kernel = np.ones((5, 5), np.uint8)


def get_bin_threshold(gaussian_blur):
    global thresh
    bin_threshold = 160
    while True:
        ret, thresh = cv2.threshold(gaussian_blur, bin_threshold, 255, cv2.THRESH_BINARY)  # | cv2.THRESH_OTSU)
        print("threshold = ", bin_threshold)
        cv2.imshow('thresh', thresh)
        k = cv2.waitKey(0) & 0xFF
        # print(k)
        if k == ord('y'):
            break
        elif k == ord('w'):
            bin_threshold += 1
            bin_threshold %= 255
        elif k == ord('s'):
            bin_threshold -= 1
            bin_threshold %= 255
    cv2.destroyWindow('thresh1')
    return bin_threshold

bin_threshold = get_bin_threshold(gaussian_blur)
params.update({"bin_threshold": bin_threshold})

edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
# while True:
#     cv2.imshow("edges", edges)
#     k = cv2.waitKey(5) & 0xFF
#     if k == 27:
#         break
# cv2.destroyWindow('edges')


def get_hough_threshold(edges):
    global hough_threshold, mask
    hough_threshold = 26
    while True:
        lines = cv2.HoughLines(edges, 1, np.pi / 180, hough_threshold)
        print("Number of lines found is ", len(lines[0]), "; ", "hough_threshold = ", hough_threshold)
        x1 = None
        x2 = None
        y1 = None
        y2 = None
        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(roi, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.imshow('needle', roi)
        k = cv2.waitKey(0) & 0xFF

        if k == ord('y'):
            # mask = np.zeros(img_gray.shape, dtype="uint8")
            # cv2.line(mask, (x1, y1), (x2, y2), 255, 2)
            # while True:
            #     cv2.imshow('mask', mask)
            #     k = cv2.waitKey(0) & 0xFF
            #     if k == 27:
            #         break
            # print ("rho ", rho)

            # try:
            # print("Angle of line is ", 180 * math.atan((float(y2 - y1) / (x2 - x1))) / 3.14)
            # except:
            # print("Angle of line is ", 0)
            cv2.destroyWindow('needle')
            break

        elif k == ord('w'):
            hough_threshold += 1
        elif k == ord('s'):
            hough_threshold -= 1

    return hough_threshold


hough_threshold = get_hough_threshold(edges)
params.update({"hough_threshold": hough_threshold})



# res = cv2.bitwise_and(edges, edges, mask=mask)
# while 1:
#     cv2.imshow('res', res)
#     k = cv2.waitKey(0) & 0xFF
#     if k == 27:
#         break

cv2.destroyAllWindows()

'''
image_file1 = input("Enter the location of the image 1: ")
cv2.imshow("image1", image_file1)
val1 = input("Enter the reading 1")
cv2.destroyWindow("image1")

image_file2 = input("Enter the location of the image 1: ")
cv2.imshow("image1", image_file2)
val2 = input("Enter the reading 1")
cv2.destroyWindow("image1")

angle1 = 0
angle2 = 180
'''

# obj = {"refPt": roi, "bin_threshold": bin_threshold,
#        "hough_threshold": hough_threshold, "angle1":angle1, "angle2":angle2, "val1":val1, "val2":val2}
# print(obj)

print(params)
with open('data.json', 'w') as outfile:
    json.dump(params, outfile)
