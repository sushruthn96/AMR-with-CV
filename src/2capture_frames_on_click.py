import numpy as np
import cv2

cap = cv2.VideoCapture(0)
index = 1
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', gray)

    k = cv2.waitKey(1)
    if k & 0xFF == ord('c'):
        cv2.imwrite("frames/%d.jpg"%index, frame)
        print("frame %d"%index + " captured.")
        index += 1
    elif k & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
