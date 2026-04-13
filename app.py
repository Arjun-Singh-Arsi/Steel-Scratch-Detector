import os
import cv2
import torch
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from utils import load_model, predict

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Ensure upload and static directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/css', exist_ok=True)
os.makedirs('static/js', exist_ok=True)
os.makedirs('static/predictions', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model (make sure weights are in root or update path)
MODEL_PATH = os.path.join('model', 'steel_defect_resunet_final (1).pth')
model = None

def get_model():
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH, device)
        else:
            print(f"Warning: Model weights not found at {MODEL_PATH}")
    return model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def handle_prediction():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        current_model = get_model()
        if current_model is None:
            return jsonify({'error': 'Model weights not found. Please train the model first.'}), 500
        
        # Run prediction
        overlay_image, defect_info = predict(filepath, current_model, device)
        
        # Save overlay image
        output_filename = f"pred_{filename}"
        output_path = os.path.join('static/predictions', output_filename)
        cv2.imwrite(output_path, cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))
        
        return jsonify({
            'original_image': filename,
            'processed_image': output_filename,
            'defects': defect_info
        })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
