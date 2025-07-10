import json
import cv2
from ultralytics import YOLOv10
import numpy as np
import math
import re
import os
import sqlite3
from datetime import datetime, timedelta
from paddleocr import PaddleOCR

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Create a Video Capture Object
cap = cv2.VideoCapture(0)

# Initialize the YOLOv10 Model for Vehicle and Plate Detection
model = YOLOv10("weights/best.pt")  # Use appropriate weights for vehicle/plate detection

# Initialize the Paddle OCR
ocr = PaddleOCR(use_angle_cls=True, use_gpu=False)

# Tracking
recognized_plates = set()  # Set to store detected license plates
unrecognized_vehicles = {}  # Dict to track unrecognized vehicles and their last capture time


def paddle_ocr(frame, x1, y1, x2, y2):
    # Crop the detected region for OCR
    plate_region = frame[y1:y2, x1:x2]
    result = ocr.ocr(plate_region, det=False, rec=True, cls=False)
    text = ""
    for r in result:
        scores = r[0][1]
        if np.isnan(scores):
            scores = 0
        else:
            scores = int(scores * 100)
        if scores > 60:
            text = r[0][0]
    pattern = re.compile('[\W]')
    text = pattern.sub('', text)
    text = text.replace("???", "")
    text = text.replace("O", "0")
    text = text.replace("粤", "")
    return str(text)


def save_no_plate_data(vehicle_image, current_time):
    # Save the vehicle image without a plate
    no_plate_dir = "no_plate_images"
    os.makedirs(no_plate_dir, exist_ok=True)
    image_filename = f"{no_plate_dir}/vehicle_{current_time.strftime('%Y%m%d%H%M%S')}.jpg"
    cv2.imwrite(image_filename, vehicle_image)

    # Save to a JSON file
    no_plate_file = "json/NoPlateData.json"
    os.makedirs("json", exist_ok=True)
    no_plate_data = {
        "Time": current_time.isoformat(),
        "Image Path": image_filename
    }

    if os.path.exists(no_plate_file):
        with open(no_plate_file, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    existing_data.append(no_plate_data)

    with open(no_plate_file, 'w') as f:
        json.dump(existing_data, f, indent=2)


def save_recognized_data(license_plates, start_time, end_time):
    # Save recognized license plates
    interval_data = {
        "Start Time": start_time.isoformat(),
        "End Time": end_time.isoformat(),
        "License Plates": list(license_plates)
    }
    interval_file_path = "json/output_" + datetime.now().strftime("%Y%m%d%H%M%S") + ".json"
    os.makedirs("json", exist_ok=True)
    with open(interval_file_path, 'w') as f:
        json.dump(interval_data, f, indent=2)

    cummulative_file_path = "json/LicensePlateData.json"
    if os.path.exists(cummulative_file_path):
        with open(cummulative_file_path, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    existing_data.append(interval_data)

    with open(cummulative_file_path, 'w') as f:
        json.dump(existing_data, f, indent=2)


start_time = datetime.now()
license_plates = set()

while True:
    ret, frame = cap.read()
    if ret:
        current_time = datetime.now()

        results = model.predict(frame, conf=0.45)  # Adjust confidence as necessary

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Crop the vehicle region
                vehicle_image = frame[y1:y2, x1:x2]

                # Detect the license plate
                label = paddle_ocr(frame, x1, y1, x2, y2)

                if label:
                    # If plate is detected, save it
                    if label not in recognized_plates:
                        recognized_plates.add(label)
                        license_plates.add(label)
                else:
                    # If no plate is detected, handle unrecognized vehicles
                    vehicle_key = (x1, y1, x2, y2)
                    last_capture = unrecognized_vehicles.get(vehicle_key)
                    if not last_capture or (current_time - last_capture > timedelta(seconds=10)):
                        unrecognized_vehicles[vehicle_key] = current_time
                        save_no_plate_data(vehicle_image, current_time)

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                if label:
                    text_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                    c2 = x1 + text_size[0], y1 - text_size[1] - 3
                    cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
                    cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        # Save recognized data every 20 seconds
        if (current_time - start_time).seconds >= 10:
            end_time = current_time
            save_recognized_data(license_plates, start_time, end_time)
            start_time = current_time
            license_plates.clear()

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('1'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
