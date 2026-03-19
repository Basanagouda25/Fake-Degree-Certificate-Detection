import os
import uuid
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request, jsonify, url_for

app = Flask(__name__)

# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

# Central model takes precedence, fallback to FL model
MODEL_PATH = os.path.join(PROJECT_DIR, 'fake_certificate_model.h5')
FL_MODEL_PATH = os.path.join(PROJECT_DIR, 'federated_learning', 'fl_global_model.h5')

# Global variable to hold the loaded model
app.config['MODEL'] = None

def load_model():
    if os.path.exists(MODEL_PATH):
        print(f"Loading central model from: {MODEL_PATH}")
        return tf.keras.models.load_model(MODEL_PATH)
    elif os.path.exists(FL_MODEL_PATH):
        print(f"Loading FL global model from: {FL_MODEL_PATH}")
        return tf.keras.models.load_model(FL_MODEL_PATH)
    else:
        print("Error: No models found at expected locations.")
        return None

print("Initializing model...")
app.config['MODEL'] = load_model()

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

IMG_SIZE = (224, 224)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if app.config['MODEL'] is None:
        return jsonify({'error': 'No trained model found. Please train a model first.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No image file uploaded.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty file selected.'}), 400
    
    # Save image
    ext = file.filename.split('.')[-1]
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        # Preprocess exactly as in fl_predict.py
        img = image.load_img(filepath, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        prediction = app.config['MODEL'].predict(img_array, verbose=0)
        score = float(prediction[0][0])
        
        # Interpret result
        if score >= 0.7:
            label = "REAL"
            confidence = score
            color = "success"
        elif score <= 0.3:
            label = "FAKE"
            confidence = 1.0 - score
            color = "danger"
        else:
            label = "UNCERTAIN"
            confidence = max(score, 1.0 - score)
            color = "warning"
            
        file_url = url_for('static', filename=f'uploads/{filename}')
        
        return jsonify({
            'success': True,
            'label': label,
            'score': score,
            'confidence': confidence,
            'color': color,
            'image_url': file_url
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)
