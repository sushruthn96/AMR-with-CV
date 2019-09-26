import numpy as np
import cv2
frame = cv2.imread("0.jpg", 0)
fgbg = cv2.createBackgroundSubtractorMOG2()

cv2.imshow('frame',frame)
cv2.waitKey(0)

fgmask = fgbg.apply(frame)
cv2.imshow('fgmask',fgmask)
cv2.waitKey(0)
