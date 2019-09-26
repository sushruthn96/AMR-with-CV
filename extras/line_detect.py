import numpy as np
import cv2

img1 = cv2.imread('cropped_img.jpg')
img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.GaussianBlur(img2, (5, 5), 0)

while (1):
    cv2.imshow('img2', img2)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        cv2.destroyWindow('img2')
        break

ret, thresh1 = cv2.threshold(img2, 160, 255, cv2.THRESH_BINARY)  # | cv2.THRESH_OTSU)
kernel = np.ones((5, 5), np.uint8)

bin_threshold = 160
while (1):
    ret, thresh1 = cv2.threshold(img2, bin_threshold, 255, cv2.THRESH_BINARY)  # | cv2.THRESH_OTSU)
    print("threshold = ", bin_threshold)
    cv2.imshow('thresh1', thresh1)
    k = cv2.waitKey(0) & 0xFF
    # print(k)
    if k == ord('y'):
        break
    elif k == ord('w'):
        bin_threshold += 3
        bin_threshold %= 255
    elif k == ord('s'):
        bin_threshold -= 3
        bin_threshold %= 255
cv2.destroyWindow('thresh1')

edges = cv2.Canny(thresh1, 50, 150, apertureSize=3)
while (1):
    cv2.imshow("edges", edges)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyWindow('edges')

hough_threshold = 10
while (1):
    lines = cv2.HoughLines(edges, 1, np.pi / 180, hough_threshold)
    print("Number of lines found is ", len(lines[0]))
    clone = img2.copy()
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(clone, (x1, y1), (x2, y2), (255, 0, 0), 2)
    # print("Angle of line is ", 180* math.atan((float(y2-y1)/(x2-x1)))/3.14)

    cv2.imshow('clone', clone)
    k = cv2.waitKey(0) & 0xFF

    if k == ord('y'):
        img2 = clone
        break
    elif k == ord('w'):
        hough_threshold += 1
    elif k == ord('s'):
        hough_threshold -= 1
cv2.destroyWindow('clone')

while 1:
    cv2.imshow('final', img2)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
