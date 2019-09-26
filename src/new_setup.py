import cv2
import os
import json
import numpy as np
import math
import requests
import json
import src._4getReadings as get_readings


def draw_grid(frame):
    height, width = frame.shape[:2]
    x = 0
    y = 0

    move_right_dist = width / 8
    move_down_dist = height / 8

    while x < width:
        x += int(move_right_dist)
        y += int(move_down_dist)
        cv2.line(frame, (x, 0), (x, height), (0, 0, 255), 1)
        cv2.line(frame, (0, y), (width, y), (0, 0, 255), 1)
    return frame


def capture_frames(cam=1, location="frames/"):
    for image_file in os.listdir(location):
        os.remove(os.path.join(location, image_file))

    cap = cv2.VideoCapture(cam)
    _, frame = cap.read()

    index = 1
    # orientation and capture frames
    print("Press c to capture frame. Press q to quit")
    while True:
        _, frame = cap.read()
        frame_dup = frame.copy()
        cv2.imshow('frame', draw_grid(frame_dup))
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('c'):
            filename = os.path.join(location, "%d.jpg" % index)
            cv2.imwrite(filename, frame)
            cv2.imread(filename)
            print("Press y to save this frame, press n to recapture frame")

            frame_name = 'captured_frame_%d' % index
            cv2.imshow(frame_name, draw_grid(frame))
            k = cv2.waitKey(0) & 0xFF
            if k == ord('y'):
                print("frame saved as " + filename)
                index += 1
            elif k == ord('n'):
                os.remove(filename)
                print("Press c to recapture frame; q to quit")
            cv2.destroyWindow(frame_name)

    cap.release()
    cv2.destroyAllWindows()
    image_files = [os.path.join(location, file) for file in os.listdir(location) if file.endswith('.jpg')]
    return image_files


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


def get_hough_threshold(edges, hough_threshold=26):
    while True:
        lines = cv2.HoughLines(edges, 1, np.pi / 180, hough_threshold)
        print("Number of lines found is ", len(lines[0]), "; ", "hough_threshold = ", hough_threshold)

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
            cv2.destroyWindow('needle')
            break

        elif k == ord('w'):
            hough_threshold += 1
        elif k == ord('s'):
            hough_threshold -= 1

    return hough_threshold


def get_scaling_factors(image_files, params):
    scaling = []
    for image in image_files[:2]:
        frame = cv2.imread(image)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = frame[y:y + h, x:x + w]

        line_coord, result = get_readings.getReading(gray, params["xywh"], params["bin_threshold"],
                                                     params["hough_threshold"])
        if result is not None:
            print("result(angle): " + str(result))
        else:
            print("No reading")

        x1, y1, x2, y2 = line_coord
        cv2.line(roi, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.imshow('roi', roi)
        cv2.waitKey(0)
        val = float(input("Enter the corresponding reading"))

        scaling.append((result, val))

    sensitivity = (scaling[0][0] - scaling[1][0]) / (scaling[0][1] - scaling[1][1])
    ref_angle = scaling[0][0] - scaling[0][1] * sensitivity

    return ref_angle, sensitivity


if __name__ == "__main__":
    cam = 1

    # Capture 2 image frames at 2 different readings
    image_files = capture_frames(cam=1, location="frames")
    image = cv2.imread(image_files[0])

    image_copy = image.copy()

    # Select the roi
    x, y, w, h = cv2.selectROI(draw_grid(image_copy))

    roi = image[y:y + h, x:x + w]

    cv2.imwrite("roi.jpg", roi)
    files = {"image": open("roi.jpg", "rb")}
    response = requests.post("http://localhost:5000/predict", files=files)
    print(response.text)
    result = json.loads(response.text)
    type = sorted(result, key=lambda x: x[1], reverse=True)[0][0]

    cv2.imshow("roi", roi)
    cv2.waitKey(0)
    cv2.destroyWindow("roi")
    params = {"xywh": (x, y, w, h)}
    if type == 'analog':
        # Preprocessing

        # Apply Gaussian blur
        img_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gaussian_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

        # Binary threshold set manually
        bin_threshold = get_bin_threshold(gaussian_blur)
        ret, thresh = cv2.threshold(gaussian_blur, bin_threshold, 255, cv2.THRESH_BINARY)

        # Canny edge
        edges = cv2.Canny(thresh, 50, 150, apertureSize=3)

        # Find hough_threshold value manually
        hough_threshold = get_hough_threshold(edges)

        cv2.destroyAllWindows()

        # Find out the reference angle and the sensitivity of the analog meter

        params.update({"hough_threshold": hough_threshold, "bin_threshold": bin_threshold})
        ref_angle, sensitivity = get_scaling_factors(image_files, params)
        params.update({"ref_angle": ref_angle, "sensitivity": sensitivity})


    elif type == 'digital':
        img_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gaussian_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

        # Binary threshold set manually
        bin_threshold = get_bin_threshold(gaussian_blur)
        params.update({"bin_threshold": bin_threshold})

    params.update({"type": type})
    print(params)
    with open('data.json', 'w') as outfile:
        json.dump(params, outfile)
