from flask import Flask, request, render_template, jsonify, send_file
import os
import cv2
from ultralytics import YOLO
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import json
import torch
import librosa
# Configure logging



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['RESULT_FOLDER'] = 'results/'

# Ensure the upload and result directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load your custom YOLO model for vest detection
model = YOLO("best.pt")  # Replace with the path to your trained model



# whisper_model = whisper.load_model("base")
# Load fine-tuned Whisper model and processor
whisper_processor = WhisperProcessor.from_pretrained("./resultsAudio")
whisper_model = WhisperForConditionalGeneration.from_pretrained("./resultsAudio")
whisper_model.eval()  # Set the model to evaluation mode

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
whisper_model.to(device)

# Words to highlight in the text
words_to_highlight = [
    "Risk", "cost", "delay", "started", "finished", "in progress",
    "caution", "situation", "time overrun", "cost overrun"
]

# Highlight specific words in text
def highlight_words(text, words_to_highlight):
    for word in words_to_highlight:
        text = text.replace(word, f'<span class="highlight">{word}</span>')
    return text


def load_audio(file_path, sampling_rate=16000):
    audio, sr = librosa.load(file_path, sr=sampling_rate)
    return audio

# Function to transcribe audio
def transcribe_audio(file_path):
    audio = load_audio(file_path)
    inputs = whisper_processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)

    # Generate transcription
    with torch.no_grad():
        generated_ids = whisper_model.generate(inputs)
    transcription = whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription


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
    if people_without_vests == 0:
        summary_text = "Pass, everyone is following the safety instructions."
    else:
        summary_text = f"Action needed, {people_without_vests} people are not following the safety instructions."
    
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
                    # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
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


# # Audio to text function using Whisper
# def audio_to_text(audio_path):
#     try:
#         result = whisper_model.transcribe(audio_path)
#         text = result["text"]
#     except Exception as e:
#         logging.error(f"An error occurred: {e}")
#         text = "An error occurred during transcription."

#     logging.debug(f"Extracted text: {text}")
#     return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            if file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                result_video_path, summary_text = process_video(filepath)
                return jsonify({'result_video': result_video_path, 'summary_text': summary_text})
            else:
                result_image_path, summary_text = detect_vests(filepath)
                return jsonify({'result_image': result_image_path, 'summary_text': summary_text})
    return render_template('detect_vest.html')

@app.route('/upload_audio', methods=['GET', 'POST'])
def upload_audio():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            text = transcribe_audio(filepath)
            highlighted_text = highlight_words(text, words_to_highlight)
            return render_template('result.html', results=[highlighted_text], title="Audio to Text Results")
    return render_template('audio_to_text.html')

@app.route('/display_image/<path:image_path>')
def display_image(image_path):
    return send_file(image_path, mimetype='image/jpeg')

@app.route('/display_video/<path:video_path>')
def display_video(video_path):
    return send_file(video_path, mimetype='video/mp4')

if __name__ == '__main__':
    app.run(debug=True)
