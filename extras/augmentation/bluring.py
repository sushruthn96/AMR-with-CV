'''
 * Python script to demonstrate Gaussian blur.
 *
 * usage: python GaussBlur.py <filename> <kernel-size>
'''
import cv2, sys
import os

# get filename and kernel size from command line
# filename = sys.argv[1]
# k = int(sys.argv[2])
k = 3

# read and display original image

filename = os.listdir("../dataset/try_ring")[1]
print(filename)
img = cv2.imread("../dataset/try_ring"+'/'+filename)
# cv2.namedWindow("original", cv2.WINDOW_NORMAL)
cv2.imshow("original", img)
cv2.waitKey(0)

# apply Gaussian blur, creating a new image
gaussian_blur = cv2.GaussianBlur(img, (k, k), 0)
average_blur = cv2.blur(img,(k,k))
median_blur = cv2.medianBlur(img,5)
bilateral_filter = cv2.bilateralFilter(img,9,75,75)

# display blurred image
# cv2.namedWindow("blurred", cv2.WINDOW_NORMAL)
cv2.imshow("gaussian blurred", gaussian_blur)
cv2.imshow("average blurred", average_blur)
cv2.imshow("median blurred", median_blur)
cv2.imshow("bilateral blurred", bilateral_filter)

cv2.waitKey(0)