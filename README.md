Title: Automated traffic management system for emergency vehicles and two-wheeler rider helmet detection





Part 1: Automated traffic management system for emergency vehicles


Project Overview

This project consists of an automated traffic management system integrated with vehicle detection. The system manages traffic lights on a Raspberry Pi using GPIO pins and detects vehicles using a YOLOv5 model for real-time video analysis. The system also has special handling for emergency vehicles like ambulances, changing traffic light behavior accordingly.



Requirements

To run this project, you need to install the following Python libraries:

opencv-python==4.5.3.56
numpy==1.23.5
RPi.GPIO==0.7.1
torch==2.0.1
yolov5==6.2.0
pillow==9.3.0
matplotlib==3.7.1


Run:

pip install -r requirements.txt




Hardware Setup:

Connect the traffic light LEDs to the Raspberry Pi GPIO pins as specified in the code.
Ensure the Raspberry Pi is set up with the necessary hardware interfaces.
Model Weights:

Download the YOLOv5 weights and place them in the specified path. Update the yolov5_weight_file variable in the code with the path to your weights file.



Running the Project

Start the System:

Run the script using Python:


		python your_script_name.py

This will start the vehicle detection and traffic light management in separate threads.


Stop the System:

To stop the system, press q in the display window showing vehicle detection. The system will perform a cleanup of GPIO pins and close all windows.



Functionality

Traffic Lights Control:

The system manages four sets of traffic lights (s1, s2, s3, s4). Each set has red, yellow, and green lights controlled based on a timed cycle or emergency vehicle detection.



Vehicle Detection:

The system uses YOLOv5 for real-time vehicle detection via the webcam. Detected vehicles trigger changes in the traffic light operations.



Emergency Vehicle Handling:

When an emergency vehicle (ambulance) is detected, the traffic light system prioritizes the side where the ambulance is detected, allowing it to pass with a green light.


Cleanup
The cleanup function ensures that all GPIO pins are set to their initial state when the script exits. This is handled automatically on exit using the atexit module.





Part 2: Two-wheeler rider helmet detection


Overview

This project involves the development of a system that performs helmet detection on two-wheeler riders using YOLOv5 for object detection and a custom model for helmet classification. It also detects license plates from number plates using Optical Character Recognition (OCR). The system is designed to process video files or images, save results, and send email notifications with the processed data.




Features

Helmet Detection: Identifies whether a rider is wearing a helmet.
Object Detection: Detects objects such as heads, riders, and number plates using YOLOv5.
OCR: Extracts license plate numbers from detected number plates.
Email Notification: Sends email with attachments containing processed images and text files.
Video and Image Processing: Handles both video and image inputs, with options to save and display results.



Requirements

Python 3.x



Libraries:

opencv-python==4.5.3.56
torch==1.9.0
torchvision==0.10.0
numpy==1.21.1
pillow==8.3.1
pytesseract==0.3.8




Setup

Install Dependencies: Create a virtual environment (optional) and install the required libraries using the following command:

		pip install -r requirements.txt


Download YOLOv5 Weights: Download the YOLOv5 model weights and place them in the specified path (yolov5_weight_file).


Download Helmet Classifier Weights: Download the helmet classifier model weights and place them in the specified path (helmet_classifier_weight).


Configure Email Settings: Update the email settings in the code (e.g., sender email, password, SMTP server details) for sending notifications.

Set Paths: Update paths for video files, image files, and output directories in the code as needed.



Usage


Run the Code: 

Execute the script to start processing. The code will read from the video file or image file specified in source, perform object detection and helmet classification, and save results.



Video Processing:

The code will process each frame of the video, detecting objects and performing OCR.
Results will be saved as images and text files.
Optionally, a video file with processed frames can be saved.



Image Processing:

For image files, the code will process the image, detect objects, and perform OCR.
Results will be saved as images and text files.



View Results:

Processed images and text files will be saved in specified directories (riders_pictures, number_plates, numbers).
Email notifications with attachments will be sent to the specified recipient.




Code Structure

import statements: Import required libraries.

model setup: Load YOLOv5 and helmet classifier models.

img_classify(frame): Classifies images to check if the helmet is present.

object_detection(frame): Detects objects in the frame and identifies heads, riders, and number plates.

send_email(subject, body, to_email, attachment_paths, sender_email, sender_password, smtp_server, smtp_port): Sends an email with attachments.

process_image_and_ocr(num_img, time_stamp, conf_num): Processes the image for OCR and saves results.

inside_box(big_box, small_box): Checks if a bounding box is inside another bounding box.

while cap.isOpened(): Main loop for processing video frames or images.





Troubleshooting

Ensure all paths are correctly set in the code.
Verify that model weights and dependencies are correctly installed.
Check email settings for correct configuration.
