from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
import time

# Threshold for light intensity (adjust as needed)
light_threshold = 75

# Confidence threshold for both person and fan detection
confidence_threshold = 0.7

# Initialize Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height

# Load two different YOLO models
model1 = YOLO("../Yolo_weights/yolov8n.pt")  # Model for human detection
model2 = YOLO("../Yolo_weights/Fan.pt")  # Model for fan detection

# Class names for both models
classNames1 = ["person"]  # Class names for human detection
classNames2 = ["Fan"]  # Class names for fan detection

# Initialize time variables for FPS calculation
prev_frame_time = 0
new_frame_time = 0

# Variables to store fan status and debounce logic
fan_status = "Fan OFF"
fan_last_decision_time = 0  # Last time fan state was changed
fan_debounce_time = 5  # Time (in seconds) to debounce the fan decision

# Variables for Alarm status
alarm_status = "Alarm OFF"
last_person_detected_time = time.time()  # Timestamp of the last person detection
alarm_trigger_time = 10  # Time in seconds before triggering the alarm

def calculate_avg_intensity(image):
    """Calculates the average intensity of the frame."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    avg_intensity = np.mean(gray_image)
    return avg_intensity

def light_decision(avg_intensity):
    """Decides whether to turn the light on or off based on intensity."""
    if avg_intensity < light_threshold:
        return "Light ON"  # Light should be ON
    else:
        return "Light OFF"  # Light should be OFF

def is_fan_above_person(person_coords, fan_coords):
    """Check if the fan is directly above the person."""
    _, person_top_y, _, person_bottom_y = person_coords
    _, fan_top_y, _, fan_bottom_y = fan_coords
    
    # Fan should be above the person: Fan's bottom should be above the person's top
    return fan_bottom_y < person_top_y

while True:
    new_frame_time = time.time()
    success, img = cap.read()

    # Initialize variables to hold the coordinates of the person and fan
    person_coords = None
    fan_coords = None
    person_conf = 0  # To store the confidence of person detection
    fan_conf = 0     # To store the confidence of fan detection

    # ------------------------
    # Process with first model (for human detection)
    # ------------------------
    results1 = model1(img, stream=True)
    for r in results1:
        boxes = r.boxes
        for box in boxes:
            # Class name
            cls = int(box.cls[0])
            conf = box.conf[0]  # Get the confidence of the detection

            # Only process the "person" class (index 0) if confidence > threshold
            if cls == 0 and conf > confidence_threshold:
                # Bounding box for the person
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                person_coords = (x1, y1, x2, y2)  # Save person's coordinates
                person_conf = conf  # Save the confidence level
                
                # Draw bounding box and label for "person"
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h), colorR=(255, 0, 0))  # Red color for humans
                cvzone.putTextRect(img, f'{classNames1[cls]} {conf:.2f}', (max(0, x1), max(35, y1)), scale=1, thickness=1, colorR=(255, 0, 0))

    # ------------------------
    # Process with second model (for fan detection)
    # ------------------------
    results2 = model2(img, stream=True)
    for r in results2:
        boxes = r.boxes
        for box in boxes:
            # Class name
            cls = int(box.cls[0])
            conf = box.conf[0]  # Get the confidence of the detection

            # Only process the "fan" class if confidence > threshold
            if conf > confidence_threshold:
                # Bounding box for the fan
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                fan_coords = (x1, y1, x2, y2)  # Save fan's coordinates
                fan_conf = conf  # Save the confidence level

                # Draw bounding box and label for the fan
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h), colorR=(0, 255, 0))  # Green color for fan
                cvzone.putTextRect(img, f'{classNames2[cls]} {conf:.2f}', (max(0, x1), max(35, y1)), scale=1, thickness=1, colorR=(0, 255, 0))

    # ------------------------
    # Compare coordinates and decide "Fan ON" or "Fan OFF" with debounce logic
    # ------------------------
    current_time = time.time()
    if person_coords and fan_coords:
        last_person_detected_time = current_time  # Reset the last person detection time
        if is_fan_above_person(person_coords, fan_coords):
            if person_conf > confidence_threshold and fan_conf > confidence_threshold:
                if fan_status == "Fan OFF" and (current_time - fan_last_decision_time > fan_debounce_time):
                    fan_status = "Fan ON"  # Turn the fan on
                    fan_last_decision_time = current_time
        else:
            if fan_status == "Fan ON" and (current_time - fan_last_decision_time > fan_debounce_time):
                fan_status = "Fan OFF"  # Turn the fan off
                fan_last_decision_time = current_time

    # ------------------------
    # Check for Alarm status
    # ------------------------
    if current_time - last_person_detected_time > alarm_trigger_time:
        alarm_status = "Alarm ON"
    else:
        alarm_status = "Alarm OFF"

    # ------------------------
    # Calculate light intensity and decide "Light ON" or "Light OFF"
    # ------------------------
    avg_intensity = calculate_avg_intensity(img)
    light_status = light_decision(avg_intensity)

    # ------------------------
    # Display the fan, light, and alarm status
    # ------------------------
    cv2.putText(img, fan_status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(img, light_status, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(img, alarm_status, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # ------------------------
    # Calculate and display FPS
    # ------------------------
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the final output
    cv2.imshow("Camera Feed", img)

    # Wait for 1 millisecond for key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
