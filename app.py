
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os

# ===== CRITICAL: Set determinism BEFORE importing TensorFlow =====
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')  # Force CPU
os.environ.setdefault('TF_DETERMINISTIC_OPS', '1')
os.environ.setdefault('TF_CUDNN_DETERMINISTIC', '1')
os.environ.setdefault('PYTHONHASHSEED', '0')

# Set random seeds BEFORE any imports
import random
import numpy as np
random.seed(42)
np.random.seed(42)

import tensorflow as tf
tf.random.set_seed(42)

# Disable TensorFlow's internal GPU memory growth (helps with consistency)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, False)

from PIL import Image, ImageOps
import hashlib
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load manifest for Vite assets
import json

def load_manifest():
    manifest_path = os.path.join(app.static_folder, 'manifest.json')
    if not os.path.exists(manifest_path):
        return {}
    with open(manifest_path, 'r', encoding='utf-8') as fh:
        return json.load(fh)

_VITE_MANIFEST = load_manifest()

def get_manifest():
    global _VITE_MANIFEST
    if app.debug:
        try:
            manifest = load_manifest()
            if manifest:
                _VITE_MANIFEST = manifest
        except Exception:
            pass
    return _VITE_MANIFEST or {}

def asset_tags(entry='index.html'):
    manifest = get_manifest()
    tags = {'css': [], 'js': []}
    entry_meta = manifest.get(entry) or manifest.get('index.html')
    if not entry_meta:
        return tags
    for css in entry_meta.get('css', []):
        tags['css'].append(url_for('static', filename=css))
    file = entry_meta.get('file')
    if file:
        tags['js'].append(url_for('static', filename=file))
    return tags

# Load model with explicit inference configuration
print("Loading model...")
try:
    from keras.src.legacy.saving import legacy_h5_format
    model = legacy_h5_format.load_model_from_hdf5('traffic_model.h5')
    print("✅ Model loaded successfully using legacy format.")
except Exception as e:
    print(f"⚠️ Legacy load failed: {e}, trying with safe_mode=False...")
    model = tf.keras.models.load_model('traffic_model.h5', safe_mode=False)
    print("✅ Model loaded successfully with safe_mode=False.")

# Compile model to ensure consistent behavior
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Class labels and metadata
classes = ['Low', 'Medium', 'High']
colors = {'Low': 'success', 'Medium': 'warning', 'High': 'danger'}
icons = {'Low': '🚦', 'Medium': '🚗', 'High': '🚕🚙'}
sustain_messages = {
    'Low': "Traffic is low! Emissions are minimal. Keep it green! 🌿",
    'Medium': "Moderate congestion. Optimize routes to save 5-10% fuel.",
    'High': "High congestion detected! Consider signal timing or carpooling to reduce idle emissions by 15%. 🚗💨"
}
co2_savings = {
    'Low': "Low traffic: ~50 kg CO₂ saved monthly per intersection.",
    'Medium': "Moderate: ~80 kg CO₂ saved with minor optimizations.",
    'High': "High: Reducing by 10% saves ~120 kg CO₂ monthly."
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def _preprocess_image_for_model(filepath, size=(128, 128)):
   
    # Normalize EXIF orientation for consistency
    img = Image.open(filepath)
    img = ImageOps.exif_transpose(img)
    
    # Convert to RGB and resize with LANCZOS (deterministic resampling)
    image = img.convert('RGB').resize(size, resample=Image.LANCZOS)
    
    # Convert to float32 array and normalize
    image_array = np.array(image, dtype=np.float32) / 255.0
    
    # Create batch dimension
    input_tensor = np.expand_dims(image_array, axis=0)
    
    # Generate fingerprint for debugging
    fingerprint = hashlib.sha256(input_tensor.tobytes()).hexdigest()
    
    return input_tensor, fingerprint

def _predict_from_tensor(input_tensor):
   
    # CRITICAL: Use training=False to disable dropout/batch_norm randomness
    try:
        preds = model(input_tensor, training=False).numpy()
    except Exception:
        # Fallback to predict with batch_size=1
        preds = model.predict(input_tensor, batch_size=1, verbose=0)
    
    # Ensure shape is (num_classes,)
    preds = np.asarray(preds)
    if preds.ndim == 2 and preds.shape[0] == 1:
        preds = preds[0]
    
    return preds

def build_prediction_payload(preds, top_k=None, threshold=0.6):
   
    if top_k is None:
        top_k = len(classes)
    
    idx_sorted = np.argsort(preds)[::-1]
    topk = []
    for i in idx_sorted[:top_k]:
        topk.append({
            'label': classes[int(i)],
            'probability': float(preds[int(i)])
        })
    
    predicted_index = int(np.argmax(preds))
    predicted_class = classes[predicted_index]
    confidence = float(np.max(preds))
    uncertain = confidence < threshold
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'uncertain': uncertain,
        'topk': topk
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    image_path = None
    sustain_msg = None
    co2_msg = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(filepath)
            
            # Preprocess and predict
            input_tensor, fingerprint = _preprocess_image_for_model(filepath)
            preds = _predict_from_tensor(input_tensor)
            
            # Log for debugging
            print(f"[DEBUG] File: {filename}")
            print(f"[DEBUG] Fingerprint: {fingerprint}")
            print(f"[DEBUG] Raw predictions: {preds}")
            
            payload = build_prediction_payload(preds, top_k=len(classes), threshold=0.6)
            predicted_class = payload['predicted_class']
            confidence = round(payload['confidence'] * 100, 2)
            uncertain = payload['uncertain']
            
            prediction = predicted_class
            image_path = filepath
            sustain_msg = sustain_messages[predicted_class]
            co2_msg = co2_savings[predicted_class]
            topk = payload['topk']
            
            if uncertain:
                flash('Model is uncertain about this prediction (low confidence).')
            
            flash('Prediction successful!')
        else:
            flash('Invalid file type. Please upload .jpg or .png.')
    
    tags = asset_tags('index.html')
    return render_template('index.html', 
                           prediction=prediction, 
                           confidence=confidence, 
                           image_path=image_path, 
                           sustain_msg=sustain_msg, 
                           co2_msg=co2_msg,
                           colors=colors,
                           icons=icons,
                           topk=locals().get('topk', None),
                           uncertain=locals().get('uncertain', False),
                           asset_tags=tags)

@app.route('/<path:_any>')
def spa_fallback(_any):
    tags = asset_tags('index.html')
    return render_template('index.html', prediction=None, confidence=None, 
                           image_path=None, sustain_msg=None, co2_msg=None, 
                           colors=colors, icons=icons, asset_tags=tags)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predicting uploaded images with deterministic output."""
    if 'file' not in request.files:
        return jsonify({'error': 'no file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'no selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'invalid file type'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(filepath)
    
    input_tensor, fingerprint = _preprocess_image_for_model(filepath)
    preds = _predict_from_tensor(input_tensor)
    
    # Log for debugging
    print(f"[API DEBUG] File: {filename}")
    print(f"[API DEBUG] Fingerprint: {fingerprint}")
    print(f"[API DEBUG] Raw predictions: {preds}")
    
    payload = build_prediction_payload(preds, top_k=len(classes), threshold=0.6)
    payload['fingerprint'] = fingerprint
    
    return jsonify(payload)

if __name__ == '__main__':
    app.run(debug=True)