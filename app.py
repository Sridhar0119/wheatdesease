import os
import json
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_PATH = 'improved_wheat_disease_resnet50.h5'
UPLOAD_FOLDER = 'uploads'
# Class names matching the actual dataset classes the model was trained on
CLASS_NAMES = ['Healthy Wheat', 'Leaf Rust', 'Mildew', 'Wheat Loose Smut', 'Yellow Rust']


# Create uploads folder if not exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- USER DATABASE (Secure JSON Store) ---
USERS_FILE = 'secure_users_db.json'

def init_users_db():
    if not os.path.exists(USERS_FILE):
        default_users = {
            'admin': {'password': generate_password_hash('admin123'), 'role': 'admin'},
            'mrsrii': {'password': generate_password_hash('mrsrii123'), 'role': 'admin'},
            'user': {'password': generate_password_hash('user123'), 'role': 'user'}
        }
        with open(USERS_FILE, 'w') as f:
            json.dump(default_users, f)

init_users_db()

def load_users():
    with open(USERS_FILE, 'r') as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

from tensorflow.keras.layers import InputLayer
from tensorflow.keras.models import load_model

class CustomInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        if 'batch_shape' in kwargs:
            batch_shape = kwargs.pop('batch_shape')
            if batch_shape is not None and len(batch_shape) > 1:
                kwargs['input_shape'] = batch_shape[1:]
        super().__init__(*args, **kwargs)

# --- LOAD MODEL ---
print("Loading model... please wait.")
try:
    custom_objects = {'InputLayer': CustomInputLayer}
    model = load_model(MODEL_PATH, custom_objects=custom_objects)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback
    try:
        print("Trying to load from fixed model file...")
        model = load_model('fixed_wheat_disease_resnet50.h5', custom_objects=custom_objects)
        print("Model loaded successfully from fixed file!")
    except Exception as e2:
        print(f"Error loading fixed model: {e2}")
        raise e

def prepare_image(img_path):
    """
    Preprocesses the image to match ResNet50 requirements.
    """
    # 1. Load image and resize to 224x224 (Standard for ResNet)
    img = load_img(img_path, target_size=(224, 224))
    
    # 2. Convert to numpy array
    img_array = img_to_array(img)
    
    # 3. Add batch dimension (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # 4. Preprocess (ResNet expects specific normalization)
    # If you trained with rescaling 1./255, use: img_array = img_array / 255.0
    # If you used Transfer Learning defaults, use:
    img_array = preprocess_input(img_array)
    
    return img_array

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/api/register', methods=['POST'])
def register_user():
    data = request.json
    username = data.get('username', '').strip().lower()
    password = data.get('password', '')

    if not username or not password:
        return jsonify({'success': False, 'error': 'Missing credentials'})

    users = load_users()
    if username in users:
        return jsonify({'success': False, 'error': 'Username already exists'})

    # Secure: Force normal user role for all new registrations. 
    # Users CANNOT decide to be admin.
    users[username] = {
        'password': generate_password_hash(password),
        'role': 'user'
    }
    save_users(users)
    return jsonify({'success': True})

@app.route('/api/login', methods=['POST'])
def login_user():
    data = request.json
    username = data.get('username', '').strip().lower()
    password = data.get('password', '')

    users = load_users()
    if username in users and check_password_hash(users[username]['password'], password):
        role = users[username]['role']
        
        # Display logic format
        if username == 'admin':
            display_name = 'Admin'
        elif username == 'user':
            display_name = 'Guest User'
        elif username == 'mrsrii':
            display_name = 'mrsrii'
        else:
            display_name = username.capitalize()
            
        display_role = 'Admin Access' if role == 'admin' else 'User Access'
        initial = username[0].upper()
        
        return jsonify({
            'success': True, 
            'name': display_name, 
            'role': display_role, 
            'initial': initial,
            'username': username
        })
    else:
        return jsonify({'success': False, 'error': 'Invalid username or password'})

@app.route('/api/admin/users', methods=['GET'])
def get_all_users():
    users = load_users()
    # Return user list without passwords
    safe_users = [
        {'username': u, 'role': data['role']}
        for u, data in users.items()
    ]
    return jsonify({'success': True, 'users': safe_users})

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    if file:
        # Save file temporarily
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        try:
            # Check for human face / wrong upload
            img_bgr = cv2.imread(filepath)
            if img_bgr is not None:
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
                if len(faces) > 0:
                    return jsonify({'success': False, 'error': 'Wrong upload! Human face detected. Please capture a clear image of a wheat leaf.'})

                # Check for plant/crop colors (Green, Yellow, Brown, Red, Orange for rust)
                img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
                lower_plant = np.array([5, 20, 20]) # Expanded to cover red/orange rust pustules
                upper_plant = np.array([95, 255, 255])
                plant_mask = cv2.inRange(img_hsv, lower_plant, upper_plant)
                plant_ratio = cv2.countNonZero(plant_mask) / (img_bgr.shape[0] * img_bgr.shape[1])
                
                if plant_ratio < 0.015: # Less than 1.5% plant colors means it's likely a non-plant snap
                    return jsonify({'success': False, 'error': 'Wrong upload! This does not look like a crop. Please snap a clear photo of your wheat plant.'})

            # Process and Predict
            processed_img = prepare_image(filepath)
            predictions = model.predict(processed_img)
            
            # Get the highest probability class
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            confidence = float(np.max(predictions))
            
            # Safety check for class index
            if predicted_class_index < len(CLASS_NAMES):
                result = CLASS_NAMES[predicted_class_index]
            else:
                result = f"Class {predicted_class_index} (Unknown)"

            # Apply model correction logic for known false-positives
            if result == 'Wheat Loose Smut' and confidence < 0.60:
                result = 'Healthy Wheat'
                # Optionally, boost the correct healthy wheat probability visually
                predictions[0][CLASS_NAMES.index('Healthy Wheat')] = confidence
                
            if confidence < 0.35:
                result = 'Healthy Wheat'

            return jsonify({
                'class': result,
                'confidence': f"{confidence * 100:.2f}%",
                'all_probabilities': {
                    CLASS_NAMES[i]: float(predictions[0][i])
                    for i in range(len(CLASS_NAMES))
                },
                'success': True
            })
        except Exception as e:
            return jsonify({'error': str(e)})
        finally:
            # Clean up: remove the uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')