import cv2
cam = 1

cap = cv2.VideoCapture(cam)
ret, frame = cap.read()
height, width = frame.shape[:2]
print (height, width)

print ("Press q to quit")
while True:
    ret, frame = cap.read()
    x = 0
    y = 0

    move_right_dist = width / 8
    move_down_dist = height / 8

    while x < width:
        x += int(move_right_dist)
        y += int(move_down_dist)
        cv2.line(frame, (x, 0), (x, height), (0, 0, 255), 1)
        cv2.line(frame, (0, y), (width, y), (0, 0, 255), 1)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
