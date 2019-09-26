# import the necessary packages
import argparse
import cv2
import os
 
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



# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="Path to the image")
# args = vars(ap.parse_args())
 
# load the image, clone it, and setup the mouse callback function

path_to_be_cropped = "meters-test"
path_cropped = "meters-test-cropped"
image_files = [image_file for image_file in os.listdir(path_to_be_cropped) if not image_file.startswith('.')]
image_files.sort()
print(image_files)

start_idx = 0
for idx in range(start_idx, len(image_files)):
    image = cv2.imread(path_to_be_cropped+'/'+image_files[idx], 0)
    # str = path_to_be_cropped+'/'+image_files[idx]
    # print str

    clone = image.copy()
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
        elif key == 32: #ord("c"):
            print(idx, "cropped")
            break

    # if there are two reference points, then crop the region of interest
    # from teh image and display it
    if len(refPt) == 2:
        roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        cv2.imshow("ROI", roi)
        # roi = cv2.resize(roi, (100, 100))
        cv2.imwrite(path_cropped+'/'+image_files[idx], roi)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
