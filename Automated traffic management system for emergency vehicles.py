# Import necessary libraries
import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
import threading
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression
import atexit

# Define sleep durations for traffic light states
s1greensleep = 15
s2greensleep = 15
s3greensleep = 15
s4greensleep = 15
ambulancetrafficsleep = 15
normalsleep = 7
yellowsleep = 2
s1count = 0
s2count = 0
s3count = 0
s4count = 0

# Define Raspberry Pi GPIO pins for traffic lights
s1red = 11
s1yellow = 13
s1green = 15
s2red = 22
s2yellow = 24
s2green = 26
s3red = 33
s3yellow = 35
s3green = 37
s4red = 36
s4yellow = 38
s4green = 40

# Set GPIO mode
GPIO.setmode(GPIO.BOARD)

# Setup GPIO pins as outputs
GPIO.setup(s1red, GPIO.OUT)
GPIO.setup(s2red, GPIO.OUT)
GPIO.setup(s3red, GPIO.OUT)
GPIO.setup(s4red, GPIO.OUT)
GPIO.setup(s1yellow, GPIO.OUT)
GPIO.setup(s2yellow, GPIO.OUT)
GPIO.setup(s3yellow, GPIO.OUT)
GPIO.setup(s4yellow, GPIO.OUT)
GPIO.setup(s1green, GPIO.OUT)
GPIO.setup(s2green, GPIO.OUT)
GPIO.setup(s3green, GPIO.OUT)
GPIO.setup(s4green, GPIO.OUT)

# Define operations for each traffic light state (red, yellow, green) for each side (s1, s2, s3, s4)
##### S1 INDIVIDUAL OPERATION #####
def s1yellowoperation():
    global s1red, s1yellow, s1green
    GPIO.output(s1red, 0)
    GPIO.output(s1yellow, 1)
    GPIO.output(s1green, 0)
       
def s1redoperation():
    global s1red, s1yellow, s1green
    GPIO.output(s1red, 1)
    GPIO.output(s1yellow, 0)
    GPIO.output(s1green, 0)
    
def s1greenoperation():
    global s1red, s1yellow, s1green
    GPIO.output(s1red, 0)
    GPIO.output(s1yellow, 0)
    GPIO.output(s1green, 1)

##### S2 INDIVIDUAL OPERATION #####
def s2yellowoperation():
    global s2red, s2yellow, s2green
    GPIO.output(s2red, 0)
    GPIO.output(s2yellow, 1)
    GPIO.output(s2green, 0)
    
def s2redoperation():
    global s2red, s2yellow, s2green
    GPIO.output(s2red, 1)
    GPIO.output(s2yellow, 0)
    GPIO.output(s2green, 0)
    
def s2greenoperation():
    global s2red, s2yellow, s2green
    GPIO.output(s2red, 0)
    GPIO.output(s2yellow, 0)
    GPIO.output(s2green, 1)

##### S3 INDIVIDUAL OPERATION #####
def s3yellowoperation():
    global s3red, s3yellow, s3green
    GPIO.output(s3red, 0)
    GPIO.output(s3yellow, 1)
    GPIO.output(s3green, 0)    
    
def s3redoperation():
    global s3red, s3yellow, s3green
    GPIO.output(s3red, 1)
    GPIO.output(s3yellow, 0)
    GPIO.output(s3green, 0)

def s3greenoperation():
    global s3red, s3yellow, s3green
    GPIO.output(s3red, 0)
    GPIO.output(s3yellow, 0)
    GPIO.output(s3green, 1)

##### S4 INDIVIDUAL OPERATION #####
def s4yellowoperation():
    global s4red, s4yellow, s4green
    GPIO.output(s4red, 0)
    GPIO.output(s4yellow, 1)
    GPIO.output(s4green, 0)    
    
def s4redoperation():
    global s4red, s4yellow, s4green
    GPIO.output(s4red, 1)
    GPIO.output(s4yellow, 0)
    GPIO.output(s4green, 0)
    
def s4greenoperation():
    global s4red, s4yellow, s4green
    GPIO.output(s4red, 0)
    GPIO.output(s4yellow, 0)
    GPIO.output(s4green, 1)

# Function to handle ambulance detection and traffic light management
def ambulance_detected():
    print("---- AMBULANCE DETECTED -----")
    print(" -------  SIDE 1 GREEN -------")
    s1greenoperation()
    s2redoperation()
    s3redoperation()
    s4redoperation()
    time.sleep(ambulancetrafficsleep)
    s1yellowoperation()
    s2yellowoperation()
    time.sleep(2)
    if vehicle_detected == 1:
        vehicle_detected = 0

# Function to control traffic lights
def trafficlights():
    global s1count, s2count, s3count, s4count
    global s1red, s1yellow, s1green
    global s2red, s2yellow, s2green
    global s3red, s3yellow, s3green
    global s4red, s4yellow, s4green
    global s1greensleep, s2greensleep, s3greensleep, s4greensleep
    global normalsleep, yellowsleep, ambulancetrafficsleep
    global vehicle_detected
    
    while True:
        if vehicle_detected == 0:
            print("------ S1 GREEN ----------")
            s1greenoperation()
            s2redoperation()
            s3redoperation()
            s4redoperation()
            time.sleep(normalsleep)
            s1yellowoperation()
            s2yellowoperation()
            time.sleep(2)
        else:
            ambulance_detected()
           
        if vehicle_detected == 0:
            print("------ S2 GREEN ----------")
            s1redoperation()
            s2greenoperation()
            s3redoperation()
            s4redoperation()
            time.sleep(normalsleep)
            s2yellowoperation()
            s3yellowoperation()
            time.sleep(2)
        else:
            ambulance_detected()
                
        if vehicle_detected == 0:
            print("------ S3 GREEN ----------")
            s1redoperation()
            s2redoperation()
            s3greenoperation()
            s4redoperation()
            time.sleep(normalsleep)
            s3yellowoperation()
            s4yellowoperation()
            time.sleep(2)
        else:
            ambulance_detected()
         
        if vehicle_detected == 0:
            print("------ S4 GREEN ----------")
            s1redoperation()
            s2redoperation()
            s3redoperation()
            s4greenoperation()
            time.sleep(normalsleep)
            s4yellowoperation()
            s1yellowoperation()
            time.sleep(2)
        else:
            ambulance_detected()

# Function to cleanup GPIO pins on exit
def cleanup():
    GPIO.output(s1red, 0)
    GPIO.output(s1yellow, 0)
    GPIO.output(s1green, 0)
    GPIO.output(s2red, 0)
    GPIO.output(s2yellow, 0)
    GPIO.output(s2green, 0)
    GPIO.output(s3red, 0)
    GPIO.output(s3yellow, 0)
    GPIO.output(s3green, 0)
    GPIO.output(s4red, 0)
    GPIO.output(s4yellow, 0)
    GPIO.output(s4green, 0)

# Load the YOLOv5 model
yolov5_weight_file = 'your weights path'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load(yolov5_weight_file, map_location=device)
model.conf = 0.45  # Set the confidence threshold

vehicle_detected = 0

# Main function for vehicle detection
def main_function():
    def detected(frame):
        global vehicle_detected
        
        # Preprocess the frame for inference
        frame = cv2.resize(frame, (800, 480))
        img = frame[:, :, ::-1]  # BGR to RGB
        img = img / 255.0  # Normalize to [0, 1]
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        img = torch.from_numpy(img).float().to(device).unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            results = model(img)
        
        results = non_max_suppression(results, 0.45, 0.45)

        # Check if results are valid
        if results[0] is None or len(results[0]) == 0:
            print("No results returned by model.")
            return frame

        # Get the results of vehicle detection
        result = results[0].cpu().numpy()
        
        # Draw bounding boxes around detected vehicles
        for *xyxy, conf, cls in result:
            if int(cls) == 0:  # Assuming vehicle class index is 0
                xmin, ymin, xmax, ymax = [int(x) for x in xyxy]

                # Draw bounding box
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

                # Display confidence score and label
                label = f'siren: {conf:.2f}'
                cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                # Set value of vehicle_detected based on confidence score
                if conf >= 0.45:
                    vehicle_detected = 1
                else:
                    vehicle_detected = 0

        return frame

    # Open webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Read frame from webcam
        ret, frame = cap.read()

        # Perform vehicle detection
        output_frame = detected(frame)

        # Display the output frame
        cv2.imshow('Vehicle Detection', output_frame)

        # Check for key press to exit
        if cv2.waitKey(1) == ord('q'):
            break

        # Print the value of vehicle_detected
        print(f"vehicle_detected: {vehicle_detected}")

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

# Register cleanup function to be called on exit
atexit.register(cleanup)

if __name__ == "__main__":
    # Start the main function and traffic lights control in separate threads
    t1 = threading.Thread(target=main_function)
    t2 = threading.Thread(target=trafficlights)
    t1.start()
    t2.start()
