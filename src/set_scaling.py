import cv2
import json
import src._4getReadings

image1 = input("Enter the location of image 1")
image2 = input("Enter the location of image 2")

scaling = []
with open("data.json") as f:
    j = json.loads(f.read())

x, y, w, h = j["xywh"]
print(x, y, w, h)

for image in [image1, image2]:
    frame = cv2.imread(image)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = frame[y:y + h, x:x + w]

    line_coord, result = src._4getReadings.getReading(gray, j["xywh"], j["bin_threshold"], j["hough_threshold"])
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

j.update({"ref_angle": ref_angle})
j.update({"sensitivity": sensitivity})

print(j)
with open('data2.json', 'w') as outfile:
    json.dump(j, outfile)
