from flask import Flask, request, render_template, jsonify, send_file
import os
import cv2
from ultralytics import YOLO



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['RESULT_FOLDER'] = 'results/'

# Ensure the upload and result directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load your custom YOLO model for vest detection
model = YOLO("best.pt")  # Replace with the path to your trained model



# Enhanced vest detection function for images
def detect_vests(image_path):
    img = cv2.imread(image_path)
    result_image_path = os.path.join(app.config['RESULT_FOLDER'], 'result_' + os.path.basename(image_path))
    
    # Perform inference
    results = model(img)
    
    total_people = 0
    people_with_vests = 0
    
    # Process the results
    for result in results:
        boxes = result.boxes.xyxy
        confidences = result.boxes.conf
        class_ids = result.boxes.cls

        for bbox, conf, cls in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            confidence = conf.item()
            cls_id = int(cls.item())
            label = model.names[cls_id]

            total_people += 1
            if label == "vest" and confidence > 0.5:
                people_with_vests += 1
                # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # cv2.putText(img, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            elif label == "no-vest" and confidence > 0.5:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # cv2.putText(img, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    # Save the annotated image
    cv2.imwrite(result_image_path, img)
    
    # Generate summary text
    people_without_vests = total_people - people_with_vests
    text = 'are'

    if people_without_vests == 1:
        text = "is"
    if people_without_vests == 0:
        summary_text = "Pass, No potentioal risk detected, everyone is following the safety instructions."
    else:
        summary_text = f"Risk of injury is detected, {people_without_vests} person {text} not following the safety instructions, an action is needed."
    
    return result_image_path, summary_text

# Enhanced vest detection function for videos
def process_video(video_path):
    result_video_path = os.path.join(app.config['RESULT_FOLDER'], 'result_' + os.path.basename(video_path))
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(result_video_path, fourcc, 20.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    total_people = 0
    people_with_vests = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform inference
        results = model(frame)
        
        for result in results:
            boxes = result.boxes.xyxy
            confidences = result.boxes.conf
            class_ids = result.boxes.cls

            for bbox, conf, cls in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                confidence = conf.item()
                cls_id = int(cls.item())
                label = model.names[cls_id]

                total_people += 1
                if label == "vest" and confidence > 0.5:
                    people_with_vests += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                elif label == "no-vest" and confidence > 0.5:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    # cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        out.write(frame)
    
    cap.release()
    out.release()
    
    people_without_vests = total_people - people_with_vests
    if people_without_vests == 0:
        summary_text = "Pass, everyone is following the safety instructions."
    else:
        summary_text = f"Action needed, {people_without_vests} people are not following the safety instructions."
    
    return result_video_path, summary_text

