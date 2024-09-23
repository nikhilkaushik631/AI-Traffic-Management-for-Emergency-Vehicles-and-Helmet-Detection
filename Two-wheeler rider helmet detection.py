# Import necessary libraries
import cv2  
import torch  
import torch.backends.cudnn as cudnn  # CUDNN for GPU acceleration
from models.experimental import attempt_load  # Function to load YOLOv5 model
from utils.general import non_max_suppression  
from torchvision import transforms  
from PIL import Image  # Python Imaging Library for image processing
import time  
import smtplib  # Library for sending emails
from email.mime.multipart import MIMEMultipart  # Email message container
from email.mime.text import MIMEText  # Email text content
from email.mime.base import MIMEBase  # Base class for email attachments
from email import encoders  # Encoder for email attachments
import os  
import pytesseract  # OCR library 
import numpy as np  

# Paths to model weights (change these to your actual file paths)
yolov5_weight_file = 'path_to_yolov5_weights.pt'
helmet_classifier_weight = 'path_to_helmet_classifier_weights.pth'
conf_set = 0.45  # Confidence threshold for YOLOv5
frame_size = (800, 480)  # Frame size for resizing video frames

head_classification_threshold = 3.0  # Threshold for helmet classification confidence
time_stamp = str(time.time())  # Generate a timestamp for file naming

# Set device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load YOLOv5 model
model = attempt_load(yolov5_weight_file, map_location=device)
cudnn.benchmark = True  # Enable benchmark mode in CUDNN for optimal performance
names = model.module.names if hasattr(model, 'module') else model.names  # Get class names

# Load helmet classification model
model2 = torch.load(helmet_classifier_weight, map_location=device)
model2.eval()  # Set model to evaluation mode

# Define image transformations for helmet classification
transform = transforms.Compose([
    transforms.Resize(144),  # Resize image to 144x144
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize([0.5], [0.5])  # Normalize image
])

def img_classify(frame):
    """Classify if the head image contains a helmet."""
    if frame.shape[0] < 46:  # Check if image is too small
        return [None, 0]

    frame = transform(Image.fromarray(frame))  # Apply transformations
    frame = frame.unsqueeze(0)  # Add batch dimension
    prediction = model2(frame)  # Make prediction
    result_idx = torch.argmax(prediction).item()  # Get predicted class index
    prediction_conf = sorted(prediction[0])  # Get sorted confidence scores

    # Calculate confidence score
    cs = (prediction_conf[-1] - prediction_conf[-2]).item()
    if cs > head_classification_threshold:
        return [True, cs] if result_idx == 0 else [False, cs]
    else:
        return [None, cs]

def object_detection(frame):
    """Perform object detection on the frame using YOLOv5."""
    img = torch.from_numpy(frame)  # Convert frame to PyTorch tensor
    img = img.permute(2, 0, 1).float().to(device)  # Change tensor shape and move to device
    img /= 255.0  # Normalize image
    if img.ndimension() == 3:
        img = img.unsqueeze(0)  # Add batch dimension

    pred = model(img, augment=False)[0]  # Make prediction
    pred = non_max_suppression(pred, conf_set, 0.30)  # Apply non-max suppression

    detection_result = []  # Initialize list to store detection results
    head_detected = False  # Flag to check if a head is detected

    for i, det in enumerate(pred):
        if len(det):  # If detections are found
            for d in det:
                x1 = int(d[0].item())  # x1 coordinate
                y1 = int(d[1].item())  # y1 coordinate
                x2 = int(d[2].item())  # x2 coordinate
                y2 = int(d[3].item())  # y2 coordinate
                conf = round(d[4].item(), 2)  # Confidence score
                c = int(d[5].item())  # Class index

                detected_name = names[c]  # Get class name
                print(f'Detected: {detected_name} conf: {conf}  bbox: x1:{x1} y1:{y1} x2:{x2} y2:{y2}')

                if detected_name == 'head':
                    frame = cv2.putText(frame, f'{names[c]} {str(conf)}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    detection_result.append([x1, y1, x2, y2, conf, c])
                    head_detected = True

                if detected_name == 'number' and head_detected:
                    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    frame = cv2.putText(frame, f'{names[c]} {str(conf)}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    detection_result.append([x1, y1, x2, y2, conf, c])

                elif detected_name == 'rider':
                    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    frame = cv2.putText(frame, f'{names[c]} {str(conf)}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    detection_result.append([x1, y1, x2, y2, conf, c])

    return frame, detection_result  # Return processed frame and detection results

def send_email(subject, body, to_email, attachment_paths, sender_email, sender_password, smtp_server, smtp_port):
    """Send an email with attachments."""
    msg = MIMEMultipart()  # Create message container
    msg['From'] = sender_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))  # Attach email body

    # Attach files to the email
    for attachment_path in attachment_paths:
        attachment = open(attachment_path, 'rb')
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f"attachment; filename= {os.path.basename(attachment_path)}")
        msg.attach(part)

    # Connect to SMTP server and send email
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_email, msg.as_string())

def process_image_and_ocr(num_img, time_stamp, conf_num):
    """Process the number plate image and perform OCR to extract text."""
    target_width = 600  # Target width for resizing image
    aspect_ratio = num_img.shape[1] / num_img.shape[0]  # Calculate aspect ratio
    target_height = int(target_width / aspect_ratio)  # Calculate target height
    resized_image = cv2.resize(num_img, (target_width, target_height))  # Resize image

    # Preprocess the image for OCR
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    denoised = cv2.fastNlMeansDenoising(gray, None, h=30, searchWindowSize=21)  # Denoise image
    blurred = cv2.GaussianBlur(denoised, (5, 5), 0)  # Apply Gaussian blur
    gamma = 2.3  # Gamma correction value
    adjusted_image = np.power(blurred / 255.0, gamma) * 255.0  # Apply gamma correction
    adjusted_image = adjusted_image.astype(np.uint8)
    _, thresholded = cv2.threshold(adjusted_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Threshold image
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(thresholded, kernel, iterations=1)  # Erode image
    dilated = cv2.dilate(eroded, kernel, iterations=1)  # Dilate image
    
    # Set the Tesseract OCR executable path
    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
    
    try:
        number_text = pytesseract.image_to_string(dilated)  # Perform OCR
    except:
        number_text = 'error'  # Set error message if OCR fails

    # Save the extracted text to a file
    with open(f'numbers/{time_stamp}.txt', 'w') as f:
        f.write(number_text)

def inside_box(rider_box, part_box):
    """Check if part_box is inside rider_box."""
    return all(rider_box[i] <= part_box[i] for i in range(2)) and all(rider_box[i + 2] >= part_box[i + 2] for i in range(2))

# Set input source and output options
input_path = 'path_to_input_video_or_image'  # Input video or image file
output_path = 'path_to_output_video'  # Output video file path
save_video = False  # Flag to save output video
show_video = True  # Flag to show output video
save_img = False  # Flag to save output image

# Open video capture
cap = cv2.VideoCapture(input_path)
if save_video:
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, frame_size)

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()  # Read a frame
    if ret:
        orifinal_frame = frame.copy()  # Copy the original frame

        # Perform object detection
        frame, detection_result = object_detection(frame)

        # Separate detections by class
        rider_list, head_list, number_list = [], [], []
        for result in detection_result:
            x1, y1, x2, y2, conf, clas = result
            if clas == 0:
                rider_list.append(result)
            elif clas == 1:
                head_list.append(result)
            elif clas == 2:
                number_list.append(result)

        # Find the highest confidence rider
        highest_confidence_rider = max(rider_list, key=lambda x: x[4]) if rider_list else None

        if highest_confidence_rider:
            x1r, y1r, x2r, y2r, cnfr, clasr = highest_confidence_rider
            for hd in head_list:
                x1h, y1h, x2h, y2h, cnfh, clash = hd
                if inside_box([x1r, y1r, x2r, y2r], [x1h, y1h, x2h, y2h]):  # Check if head is inside rider bbox
                    try:
                        head_img = orifinal_frame[y1h:y2h, x1h:x2h]
                        helmet_present = img_classify(head_img)
                    except:
                        helmet_present[0] = None

                    if helmet_present[0] == True:  # Helmet present
                        frame = cv2.rectangle(frame, (x1h, y1h), (x2h, y2h), (0, 255, 0), 1)
                        frame = cv2.putText(frame, f'{round(helmet_present[1], 1)}', (x1h, y1h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    elif helmet_present[0] == None:  # Poor prediction
                        frame = cv2.rectangle(frame, (x1h, y1h), (x2h, y2h), (0, 255, 255), 1)
                        frame = cv2.putText(frame, f'{round(helmet_present[1], 1)}', (x1h, y1h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    elif helmet_present[0] == False:  # Helmet absent
                        frame = cv2.rectangle(frame, (x1h, y1h), (x2h, y2h), (0, 0, 255), 1)
                        frame = cv2.putText(frame, f'{round(helmet_present[1], 1)}', (x1h, y1h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        try:
                            cv2.imwrite(f'riders_pictures/{time_stamp}.jpg', frame[y1r:y2r, x1r:x2r])
                        except:
                            print('could not save rider')

                        # Find the highest confidence number plate
                        highest_confidence_number = max(number_list, key=lambda x: x[4]) if number_list else None

                        if highest_confidence_number:
                            x1_num, y1_num, x2_num, y2_num, conf_num, clas_num = highest_confidence_number
                            if inside_box([x1r, y1r, x2r, y2r], [x1_num, y1_num, x2_num, y2_num]) and conf_num > 0.4:
                                try:
                                    num_img = orifinal_frame[y1_num:y2_num, x1_num:x2_num]
                                    cv2.imwrite(f'number_plates/{time_stamp}_{conf_num}.jpg', num_img)
                                    process_image_and_ocr(num_img, time_stamp, conf_num)
                                    
                                    # Collect file paths for attachments
                                    rider_image_path = f'riders/{time_stamp}.jpg'
                                    number_plate_image_path = f'plates/{time_stamp}_{conf_num}.jpg'
                                    number_text_path = f'license_numbers/{time_stamp}.txt'
                                    attachment_paths = [rider_image_path, number_plate_image_path, number_text_path]

                                    # Specify email details
                                    subject = "Helmet Detection Results"
                                    body = "Please find the attached images and text files for helmet detection results."
                                    to_email = "reciver mail"
                                    sender_email = "your mail"
                                    sender_password = "your password"
                                    smtp_server = "smtp.gmail.com"
                                    smtp_port = 587

                                    try:
                                        send_email(subject, body, to_email, attachment_paths, sender_email, sender_password, smtp_server, smtp_port)
                                    except Exception as e:
                                        print(f"Error sending email: {e}")

                                except:
                                    print('could not save number plate')

        if save_video:  # Save video
            out.write(frame)
        if save_img:  # Save image
            cv2.imwrite('saved_frame.jpg', frame)
        if show_video:  # Show video
            frame = cv2.resize(frame, (850, 480))  # Resize to fit the screen
            cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Break the loop if 'q' is pressed
            break

    else:
        break

cap.release()  # Release video capture
cv2.destroyAllWindows()  # Close all OpenCV windows
print('Execution completed')
