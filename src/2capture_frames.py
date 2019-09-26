import time
from threading import Thread
import cv2


def myfunc():
    print ("sleeping %f sec from thread" % interval)
    time.sleep(interval)
    global flag
    flag = True
    print("finished sleeping")


# cap = cv2.VideoCapture("clock.mp4")
cap = cv2.VideoCapture(1)
# cap.set(cv2.cv.CV_CAP_PROP_FPS, 30)

interval = 1

seq = 1
flag = True

while seq <= 5:
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', gray)

    # print "counter=%d "%counter, "seq=%d"%seq
    if flag:
        cv2.imwrite('frames/%d.jpg' % seq, gray)
        seq += 1
        t = Thread(target=myfunc)
        t.start()
        flag = False
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
