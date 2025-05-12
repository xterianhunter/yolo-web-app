1cR_hmCQsmm3p-DRxz00wm534zzzkGXb_from flask import Flask, render_template, request, jsonify, Response
import cv2
import os
from ultralytics import YOLO
import gdown
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['FRAME_FOLDER'] = 'static/frames'

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['FRAME_FOLDER'], exist_ok=True)

# Step 1: Download model from Google Drive if not exists
#model_path = 'yolov11.pt'
#drive_file_id = '1cR_hmCQsmm3p-DRxz00wm534zzzkGXb_'  # Replace with your Google Drive file ID
#download_url = f'https://drive.google.com/uc?id={drive_file_id}'

#if not os.path.exists(model_path):
#    print("Downloading model from Google Drive...")
 #   gdown.download(download_url, model_path, quiet=False)
  #  print("Download complete.")

# Step 2: Load model
model = YOLO("yolov11.pt")

# Webcam status flag
webcam_active = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No file part", 400
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Run detection
    results = model(file_path)
    result = results[0]
    annotated_frame = result.plot()
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], f"result_{file.filename}")
    cv2.imwrite(result_path, annotated_frame)

    return render_template('index.html', result_image=result_path)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    global webcam_active
    webcam_active = True
    return jsonify({'status': 'started'})

@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    global webcam_active
    webcam_active = False
    return jsonify({'status': 'stopped'})

def generate_frames():
    global webcam_active
    cap = cv2.VideoCapture(0)

    while webcam_active:
        success, frame = cap.read()
        if not success:
            break

        # Detect with YOLO
        results = model(frame)
        annotated = results[0].plot()

        # Save each frame
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(app.config['FRAME_FOLDER'], f"{timestamp}.jpg")
        cv2.imwrite(filename, annotated)

        # Encode to JPEG
        ret, buffer = cv2.imencode('.jpg', annotated)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

if __name__ == '__main__':
    app.run(debug=True)
