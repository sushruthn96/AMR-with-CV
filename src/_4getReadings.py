import cv2
import json
import numpy as np
import math
from threading import Thread
import time
import sys
import argparse
from digits_training.digit_segment import *
from digits_training.test_network import Prediction


def get_bin_threshold(gaussian_blur, bin_threshold=160):
    while True:
        ret, thresh = cv2.threshold(gaussian_blur, bin_threshold, 255, cv2.THRESH_BINARY)  # | cv2.THRESH_OTSU)
        print("threshold = ", bin_threshold)
        cv2.imshow('bin_thresh', thresh)
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
    cv2.destroyWindow('bin_thresh')
    return bin_threshold


def myfunc():
    time.sleep(interval)
    global flag
    flag = True


def getReading(frame, xywh, bin_threshold, hough_threshold):
    line_coord = None
    result = None
    if True:
        x, y, w, h = xywh
        roi = frame[y:y + h, x:x + w]

        gaussian_blur = cv2.GaussianBlur(roi, (5, 5), 0)
        ret, thresh = cv2.threshold(gaussian_blur, bin_threshold, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(thresh, 50, 150, apertureSize=3)

        lines = cv2.HoughLines(edges, 1, np.pi / 180, hough_threshold)

        if lines is not None:
            x1 = None
            y1 = None
            x2 = None
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
            line_coord = (x1, y1, x2, y2)

            try:
                result = 180 * math.atan((float(y2 - y1) / (x2 - x1))) / 3.14
            except:
                pass

            # print("No lines detected for %d.jpg" % seq)
    if result is not None and result < 0:
        result += 180
    return line_coord, result


def from_img(img_file):
    img1 = cv2.imread(img_file)
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    line_coord, result = getReading(gray, j["xywh"], j["bin_threshold"], j["hough_threshold"])
    return line_coord, result


def from_video(cam):
    cap = cv2.VideoCapture(cam)
    cap.set(cv2.CAP_PROP_FPS, 30)
    global interval
    global flag

    while cap.isOpened():
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', gray)

        if flag:
            refPt = j["xywh"]
            line_coord, result = getReading(gray, refPt, j["bin_threshold"], j["hough_threshold"])
            x1, y1, x2, y2 = line_coord
            x, y, w, h = refPt

            roi = frame[y:y + h, x:x + w]
            cv2.line(roi, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.imshow('roi', roi)
            if result is not None:
                value = (result - j["ref_angle"]) / j["sensitivity"]
                sys.stdout.write("\r" + "Voltage: " + str(round(value, 2)) + "volts")
                # requests.post(url, data = str(round(value, 2)))
            else:
                print("No reading")
            t = Thread(target=myfunc)

            t.start()
            flag = False
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.imshow('frame', gray)
    cap.release()


def digital(cam):
    cap = cv2.VideoCapture(cam)
    cap.set(cv2.CAP_PROP_FPS, 30)
    global interval
    global flag
    predict_obj = Prediction()

    while cap.isOpened():
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', gray)

        if flag:
            print()
            x, y, w, h = j["xywh"]
            # bin_threshold = get_bin_threshold(gray)
            cv2.imshow("roi", gray[y:y + h, x:x + w])

            numbers_images = get_numbers_images(gray[y:y + h, x:x + w], j['bin_threshold'])
            idx = 1
            for num in numbers_images[:3]:
                cv2.imshow(str(idx), num)
                idx += 1
            # print(numbers_images)
            val = 0
            for img in numbers_images[:3]:
                # cv2.imshow("img", img)
                res = predict_obj.get_prediction([img])
                val = val*10 + int(res)
                # print("result", str(res), end='')
                sys.stdout.write("\r" + "Current: " + str(val) + " mA")

                # cv2.waitKey(0)
            t = Thread(target=myfunc)

            t.start()
            flag = False
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", help="Camera option", nargs='?', const=0, default=1, type=int)
    args = parser.parse_args()
    cam = args.camera
    print(cam)
    interval = 0.2
    flag = True
    with open("data.json", "r") as f:
        j = json.loads(f.read())
    if j["type"] == "analog":
        from_video(cam=cam)
    elif j["type"] == "digital":
        digital(cam)
