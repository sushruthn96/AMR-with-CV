
# import the necessary packages
# import argparse
import cv2
import json
import numpy as np

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False

        # draw a rectangle around the region of interest
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", image)


'''
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
 '''
# load the image, clone it, and setup the mouse callback function
image_file = input("Enter the location of the image: ")
image = cv2.imread(image_file)
clone = image.copy()

height, width = image.shape[:2]
print(height, width)
x = 0
y = 0

while x < width:
    x += int(width / 8)
    y += int(height / 8)
    cv2.line(image, (x, 0), (x, height), (0, 0, 255), 1)
    cv2.line(image, (0, y), (width, y), (0, 0, 255), 1)

# height, width = image.shape[:2]
# image = cv2.resize(image, (813, 459))


cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        image = clone.copy()

    # if the 'c' key is pressed, break from the loop
    elif key == ord("c"):
        break

cv2.destroyWindow("image")

# if there are two reference points, then crop the region of interest
# from teh image and display it
if len(refPt) == 2:
    roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
    cv2.imshow("ROI", roi)
    # cv2.imwrite("cropped_img.jpg", roi);
    cv2.waitKey(0)

# close all open windows
cv2.destroyAllWindows()
img1 = roi

import math

# img1=cv2.imread('cropped_img.jpg')
img_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.GaussianBlur(img_gray, (5, 5), 0)

'''
while(1):
	cv2.imshow('img2',img2)
	k = cv2.waitKey(1) & 0xFF
	if k == 27:
		cv2.destroyWindow('img2')
		break
'''

# ret, thresh1 = cv2.threshold(img2, 160, 255, cv2.THRESH_BINARY)
kernel = np.ones((5, 5), np.uint8)

bin_threshold = 160
while True:
    ret, thresh1 = cv2.threshold(img2, bin_threshold, 255, cv2.THRESH_BINARY)  # | cv2.THRESH_OTSU)
    print("threshold = ", bin_threshold)
    cv2.imshow('thresh1', thresh1)
    k = cv2.waitKey(0) & 0xFF
    # print(k)
    if k == ord('y'):
        break
    elif k == ord('w'):
        bin_threshold += 1
        bin_threshold %= 255
    elif k == ord('s'):
        bin_threshold -= 2
        bin_threshold %= 255
cv2.destroyWindow('thresh1')

edges = cv2.Canny(thresh1, 50, 150, apertureSize=3)
while True:
    cv2.imshow("edges", edges)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyWindow('edges')

hough_threshold = 26
while (1):
    lines = cv2.HoughLines(edges, 1, np.pi / 180, hough_threshold)
    print("Number of lines found is ", len(lines[0]), "; ", "hough_threshold = ", hough_threshold)
    clone = image.copy()
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]], (x1, y1), (x2, y2), (255, 0, 0), 2)
    # print("Angle of line is ", 180* math.atan((float(y2-y1)/(x2-x1)))/3.14)

    cv2.imshow('clone', clone)
    k = cv2.waitKey(0) & 0xFF

    if k == ord('y'):
        mask = np.zeros(img_gray.shape, dtype="uint8")
        cv2.line(mask, (x1, y1), (x2, y2), 255, 2)
        print ("rho ", rho)

        # try:
        print("Angle of line is ", 180 * math.atan((float(y2 - y1) / (x2 - x1))) / 3.14)
        # except:
        # print("Angle of line is ", 0)
        break

    elif k == ord('w'):
        hough_threshold += 1
    elif k == ord('s'):
        hough_threshold -= 1
cv2.destroyWindow('clone')

while True:
    cv2.imshow('mask', mask)
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        break

res = cv2.bitwise_and(edges, edges, mask=mask)
while 1:
    cv2.imshow('res', res)
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        break

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

obj = {"refPt": refPt, "bin_threshold": bin_threshold,
       "hough_threshold": hough_threshold}  # ,"angle1":angle1, "angle2":angle2, "val1":val1, "val2":val2}
print(obj)

with open('data.txt', 'w') as outfile:
    json.dump(obj, outfile)
