# Automated Traffic Management System

This project combines two powerful systems: an automated traffic management system for emergency vehicles and a two-wheeler rider helmet detection system. It leverages computer vision and machine learning techniques to enhance road safety and traffic flow.

## Part 1: Automated Traffic Management System for Emergency Vehicles

### Project Overview

This system manages traffic lights using a Raspberry Pi and detects vehicles in real-time using a YOLOv5 model. It includes special handling for emergency vehicles, adjusting traffic light behavior to prioritize their passage.

### Requirements

- Python libraries:
  ```
  opencv-python==4.5.3.56
  numpy==1.23.5
  RPi.GPIO==0.7.1
  torch==2.0.1
  yolov5==6.2.0
  pillow==9.3.0
  matplotlib==3.7.1
  ```
- Install dependencies:
  ```
  pip install -r requirements.txt
  ```

### Hardware Setup

- Connect traffic light LEDs to Raspberry Pi GPIO pins as specified in the code.
- Ensure Raspberry Pi is set up with necessary hardware interfaces.

### Model Weights

Download YOLOv5 weights and update the `yolov5_weight_file` variable in the code with the path to your weights file.

### Running the Project

1. Start the system:
   ```
   python your_script_name.py
   ```
2. Stop the system:
   Press 'q' in the display window showing vehicle detection.

### Functionality

- **Traffic Lights Control**: Manages four sets of traffic lights (s1, s2, s3, s4) based on timed cycles or emergency vehicle detection.
- **Vehicle Detection**: Uses YOLOv5 for real-time vehicle detection via webcam.
- **Emergency Vehicle Handling**: Prioritizes traffic flow for detected emergency vehicles.
- **Cleanup**: Automatically resets GPIO pins on script exit.

## Part 2: Two-Wheeler Rider Helmet Detection

### Overview

This system detects helmets on two-wheeler riders using YOLOv5 and a custom model. It also performs license plate detection and OCR, with capabilities to process videos or images and send email notifications.

### Features

- Helmet Detection
- Object Detection (heads, riders, number plates)
- License Plate OCR
- Email Notifications
- Video and Image Processing

### Requirements

- Python 3.x
- Libraries:
  ```
  opencv-python==4.5.3.56
  torch==1.9.0
  torchvision==0.10.0
  numpy==1.21.1
  pillow==8.3.1
  pytesseract==0.3.8
  ```

### Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Download YOLOv5 and helmet classifier weights.
3. Configure email settings in the code.
4. Update paths for video files, image files, and output directories.

### Usage

1. Run the code to start processing.
2. For video processing, the system will analyze each frame.
3. For image processing, it will analyze the provided image.
4. View results in specified output directories.

### Code Structure

- Model setup
- Image classification
- Object detection
- Email sending functionality
- OCR processing
- Main processing loop

### Troubleshooting

- Ensure all paths are correctly set.
- Verify model weights and dependencies are properly installed.
- Check email settings for correct configuration.

---

For more detailed information, please refer to the source code and comments within each script.
