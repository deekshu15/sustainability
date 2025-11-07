# app.py (Flask version)

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os

# Force CPU inference to reduce nondeterminism (set before importing TensorFlow)
# This hides GPUs from TensorFlow; remove or change for GPU inference.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('TF_DETERMINISTIC_OPS', '1')

import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import hashlib
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  # For flash messages
app.config['UPLOAD_FOLDER'] = 'static/uploads/'  # Folder for uploaded images
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load Vite manifest to map entry points to hashed assets (will be written to static/manifest.json)
import json

def load_manifest():
    manifest_path = os.path.join(app.static_folder, 'manifest.json')
    if not os.path.exists(manifest_path):
        return {}
    with open(manifest_path, 'r', encoding='utf-8') as fh:
        return json.load(fh)

# Keep a cached manifest for production, but allow reload during debug for faster dev iteration
_VITE_MANIFEST = load_manifest()

def get_manifest():
    """Return the Vite manifest. Reload from disk when running in debug mode so rebuilds don't require restarting Flask."""
    global _VITE_MANIFEST
    if app.debug:
        # try to reload; if fails, fall back to cached
        try:
            manifest = load_manifest()
            if manifest:
                _VITE_MANIFEST = manifest
        except Exception:
            pass
    return _VITE_MANIFEST or {}


def asset_tags(entry='index.html'):
    """Return dict with lists: {'css': [...], 'js': [...]} for the given manifest entry."""
    manifest = get_manifest()
    tags = {'css': [], 'js': []}
    entry_meta = manifest.get(entry) or manifest.get('index.html')
    if not entry_meta:
        # manifest missing or malformed — return empty tags
        return tags
    # CSS files
    for css in entry_meta.get('css', []):
        tags['css'].append(url_for('static', filename=css))
    # JS entry file
    file = entry_meta.get('file')
    if file:
        tags['js'].append(url_for('static', filename=file))
    return tags

# Load the trained CNN model
import tensorflow as tf
from keras.src.legacy.saving import legacy_h5_format

try:
    model = legacy_h5_format.load_model_from_hdf5('traffic_model.h5')
    print("✅ Model loaded successfully using legacy format.")
except Exception as e:
    print("⚠️ Legacy load failed, trying with safe_mode=False...")
    model = tf.keras.models.load_model('traffic_model.h5', safe_mode=False)
    print("✅ Model loaded successfully with safe_mode=False.")

# Class labels and mappings
classes = ['Low', 'Medium', 'High']
colors = {'Low': 'success', 'Medium': 'warning', 'High': 'danger'}  # Bootstrap classes
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
    """Load image from filepath, convert to RGB, resize, normalize, and return input tensor and a fingerprint hash."""
    # normalize EXIF orientation so images with rotation metadata are consistent
    img = Image.open(filepath)
    img = ImageOps.exif_transpose(img)
    image = img.convert('RGB').resize(size, resample=Image.LANCZOS)
    image_array = np.array(image).astype('float32') / 255.0
    input_tensor = np.expand_dims(image_array, axis=0)
    # fingerprint for debugging: hash of the raw bytes
    fingerprint = hashlib.sha256(input_tensor.tobytes()).hexdigest()
    return input_tensor, fingerprint


def _predict_from_tensor(input_tensor):
    """Run the model in inference mode and return a flat probability vector."""
    try:
        preds = model(input_tensor, training=False).numpy()
    except Exception:
        preds = model.predict(input_tensor, batch_size=1)
    # ensure shape (num_classes,)
    preds = np.asarray(preds)
    if preds.ndim == 2 and preds.shape[0] == 1:
        preds = preds[0]
    return preds


def build_prediction_payload(preds, top_k=None, threshold=0.6):
    """Build a JSON-serializable payload with top-k probabilities, predicted class and uncertainty flag."""
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
            # Preprocess and predict using helper functions
            input_tensor, fingerprint = _preprocess_image_for_model(filepath)
            preds = _predict_from_tensor(input_tensor)
            # Log raw predictions and fingerprint for debugging
            print(f"[DEBUG] fingerprint={fingerprint} predictions={preds}")

            payload = build_prediction_payload(preds, top_k=len(classes), threshold=0.6)
            predicted_class = payload['predicted_class']
            confidence = round(payload['confidence'] * 100, 2)
            uncertain = payload['uncertain']
            
            prediction = predicted_class
            image_path = filepath
            sustain_msg = sustain_messages[predicted_class]
            co2_msg = co2_savings[predicted_class]
            # attach debug/topk to template context via flashing or direct variables
            topk = payload['topk']
            # Optionally flash if uncertain
            if uncertain:
                flash('Model is uncertain about this prediction (low confidence).')

            flash(f'Prediction successful!')
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


# Catch-all route to support client-side routing (BrowserRouter)
@app.route('/<path:_any>')
def spa_fallback(_any):
    tags = asset_tags('index.html')
    return render_template('index.html', prediction=None, confidence=None, image_path=None, sustain_msg=None, co2_msg=None, colors=colors, icons=icons, asset_tags=tags)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predicting an uploaded image. Returns JSON with top-k probabilities and fingerprint."""
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
    payload = build_prediction_payload(preds, top_k=len(classes), threshold=0.6)
    payload['fingerprint'] = fingerprint
    # convert probabilities to floats (already done) and return
    return jsonify(payload)

if __name__ == '__main__':
    app.run(debug=True)  # Run locally; set debug=False for production