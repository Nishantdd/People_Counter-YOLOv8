# People_Counter-YOLOv8\
This code is a Python script that uses the YOLO (You Only Look Once) object detection algorithm to detect people in a video and counts the number of people passing through two predefined regions of interest. The script uses the YOLOv8 implementation provided by the ultralytics package to perform object detection.

The script starts by importing necessary libraries such as OpenCV (cv2), cvzone, and the object detection model YOLOv8 from the ultralytics package. It then loads the input video and reads frames from it using OpenCV.

The script uses a pre-defined mask image to crop out the region of interest in the frame where people are expected to be present. The graphics on the image are then overlaid using the cvzone package.

The script then passes the cropped image to the YOLOv8 model for object detection, and the resulting detections are filtered to only include objects labeled as "person". The script uses the cvzone package to draw bounding boxes around the detected objects and to display the corresponding class and confidence scores.

The script then passes the detections to the SORT (Simple Online and Realtime Tracking) algorithm to track the objects across frames. The tracked objects are then checked if they have crossed two pre-defined regions of interest, and the number of people passing through these regions are counted and displayed on the frame.

The script terminates when all frames of the input video have been processed.


**Requirements :**
1. Python (version 3.6 or above)
2. OpenCV
3. cvzone
4. NumPy
5. PyTorch
6. Yolov8

You can install the dependencies using the following command:
 >pip install opencv-python cvzone numpy torch torchvision ultralytics
 
**Instructions :**\
Clone the repository:
>git clone https://github.com/samiksha-awate/YOLO-Person-Counter.git

Navigate to the directory:
>cd YOLO-Person-Counter/

Yolo configurations will  be automatically downloaded while running the code. If not the case, download them from [**here**](https://github.com/ultralytics/ultralytics) and store it in the Yolo Weights foler outside the YOLO-Person-Counter directory.

Download the mask_people.png and mask_graphics.png files from here and place them in the YOLO-PersonCounter directory.

Run the person_counter.py file using the following command:
 >python person_counter.py
This will start the person counting program, and you should see the output on your screen.

You can change the video path in line 9 of person_counter.py/
You can also adjust the region of interest by changing the limits variable in line 33 of person_counter.py

limits = [216, 562, 350, 641]
These values represent the top-left and bottom-right coordinates of the rectangular region of interest. You can adjust these values to suit your needs.

That's it! You should now be able to run the YOLO person counter program and count the number of people in the video.
