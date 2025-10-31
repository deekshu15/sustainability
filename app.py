# app.py (Flask version)

from flask import Flask, render_template, request, redirect, url_for, flash
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  # For flash messages
app.config['UPLOAD_FOLDER'] = 'static/uploads/'  # Folder for uploaded images
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

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
            
            # Preprocess image
            image = Image.open(filepath).resize((128, 128))
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            
            # Predict
            predictions = model.predict(image_array)
            predicted_class = classes[np.argmax(predictions)]
            confidence = round(np.max(predictions) * 100, 2)
            
            prediction = predicted_class
            image_path = filepath
            sustain_msg = sustain_messages[predicted_class]
            co2_msg = co2_savings[predicted_class]
            
            flash(f'Prediction successful!')
        else:
            flash('Invalid file type. Please upload .jpg or .png.')
    
    return render_template('index.html', 
                           prediction=prediction, 
                           confidence=confidence, 
                           image_path=image_path, 
                           sustain_msg=sustain_msg, 
                           co2_msg=co2_msg,
                           colors=colors,
                           icons=icons)

if __name__ == '__main__':
    app.run(debug=True)  # Run locally; set debug=False for production