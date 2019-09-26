# AMR-with-CV
Automated energy meter reading using Image processing and Machine learning

## Summary
1) Meters are classified as needle based and digit based using a CNN based model (LeNet architecture). Dataset for the same was collected using in-lab instruments. Around 250 images of each - analog meters with needle based display and digit based display were captured.
2) If meter with needle is detected from the above step, image preprocessing techniques are used to detect and measure the readings. Techniques like gaussain blur, binary thresholding, canny edge detector and hough transforms are used to get the sensitivity of the meter (value/angle). This sensitivity is then used to measure the meter value. 
3) If digit based meters are deteted, each digit is segmented using bounding box detection method. And then the digits are detected by a CNN based trained model. Chars74k dataset was used to train the model.

## Notes
* src/ contains the main source code.
* Dataset of analog meters [meters3_cropped](https://drive.google.com/file/d/1cOup5E6egYT229q4l4BJ5euf0bquJu77/view?usp=sharing)
* training_and_clasification/ has all the digit training and energy meter detection training models and code.
