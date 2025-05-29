# AI Traffic Management for Emergency Vehicles and Helmet Detection

## Overview
This project integrates an automated traffic management system for emergency vehicles and a helmet detection system for two-wheeler riders, enhancing road safety and traffic efficiency. The system uses YOLOv5 for real-time vehicle detection, including emergency vehicles and helmet compliance. License plate recognition is also incorporated using Optical Character Recognition (OCR), and notifications are sent via email for traffic violations.

## Features
- **Real-Time Traffic Control**: Detects emergency vehicles and dynamically controls traffic lights to prioritize their movement.
- **Helmet Detection**: Identifies two-wheeler riders without helmets using a YOLOv5-based detection system.
- **License Plate Recognition**: Utilizes PyTesseract OCR to recognize license plates from detected vehicles.
- **Automated Notifications**: Sends email notifications with helmet violation and license plate details.
- **Video/Image Processing**: Capable of processing live video streams or images for vehicle and helmet detection.
- **Hardware Integration**: Uses Raspberry Pi for GPIO control, enabling real-world traffic signal management.

## Model Overview
### YOLOv5
YOLOv5 is a state-of-the-art object detection model that detects emergency vehicles, helmets, and other objects in real time. It offers:
- **Real-Time Processing**: Capable of detecting objects from video streams in real-time, crucial for traffic management applications.
- **High Accuracy**: With fine-tuning, YOLOv5 provides accurate detection of emergency vehicles and two-wheeler helmets.
- **Pretrained Weights**: Leverages pre-trained YOLOv5 weights for faster deployment and fine-tuning on custom datasets.

### License Plate Recognition (OCR)
The PyTesseract library is used to recognize license plates:
- **Text Extraction**: Extracts text from detected license plate regions.
- **Email Integration**: Sends extracted plate numbers to designated authorities via email.

## PyTorch Framework
The deep learning models, including YOLOv5, are implemented using PyTorch. Key benefits include:
- **Dynamic Graphs**: PyTorch allows for flexible development and debugging during runtime.
- **GPU Acceleration**: Seamlessly uses GPU for training and inference to accelerate object detection tasks.
  
## Setup and Requirements

### Software Requirements
- **Python Version**: 3.x
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Hardware Requirements
- **Raspberry Pi**: For GPIO-based traffic signal control.
- **Camera Module**: For real-time video feed and detection.

### Python Libraries
- opencv-python==4.5.3.56
- torch==2.0.1
- yolov5==6.2.0
- pillow==9.3.0
- pytesseract==0.3.8
- RPi.GPIO==0.7.1
- numpy==1.23.5
- matplotlib==3.7.1

### Model Weights
- Download pre-trained YOLOv5 weights and helmet classifier weights. Update the paths in the code accordingly.

## Setup Instructions

### Install Dependencies
1. Install the required Python libraries using:
   ```bash
   pip install -r requirements.txt
   ```

### Prepare Data
- **Video/Image Files**: Place your video or image files in the specified directory.
- **Model Weights**: Download and place YOLOv5 weights in the designated directory. Update the paths in the code.

### Update Paths
- Modify paths in the code to point to your video files, images, output directories, and model weights.

## Usage

### Data Preparation
1. **Process Videos**: Extract frames from video files using the provided video processing function.
2. **Face Detection**: Utilize YOLOv5 to detect vehicles and helmets.
3. **License Plate Recognition**: Extract and recognize license plate numbers using PyTesseract.

### Traffic Signal Control
- Based on real-time emergency vehicle detection, the system dynamically controls traffic signals, allowing safe passage for emergency vehicles.

### Helmet Violation Detection
- Detects riders without helmets, captures license plate information, and sends violation reports via email.

## Code Structure

1. **Data Preparation**: Handles video frame extraction and YOLO-based object detection.
2. **Helmet Detection**: Uses YOLOv5 to detect helmets on two-wheeler riders.
3. **Traffic Signal Control**: Raspberry Pi GPIO control for managing real-world traffic lights.
4. **License Plate Recognition**: Performs OCR on detected vehicles for extracting license plate numbers.
5. **Email Notification**: Sends automated emails with helmet violation details and plate information.
6. **Main Functionality**: Combines all components for real-time traffic management and helmet detection.

## Troubleshooting

- **Ensure Correct Paths**: Verify that all file paths for weights, videos, images, and output directories are correct.
- **Model Weights**: Ensure YOLOv5 model weights are correctly loaded.
- **Library Dependencies**: Ensure that all required Python libraries are installed and compatible with the project.

For further details on code implementation and functionality, refer to the project source code and comments.
