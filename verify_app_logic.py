
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from tensorflow.keras.layers import InputLayer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

MODEL_PATH = 'improved_wheat_disease_resnet50.h5'
CLASS_NAMES = ['Healthy Wheat', 'Leaf Rust', 'Mildew', 'Wheat Loose Smut', 'Yellow Rust']

class CustomInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        if 'batch_shape' in kwargs:
            batch_shape = kwargs.pop('batch_shape')
            if batch_shape is not None and len(batch_shape) > 1:
                kwargs['input_shape'] = batch_shape[1:]
        super().__init__(*args, **kwargs)

try:
    custom_objects = {'InputLayer': CustomInputLayer}
    model = load_model(MODEL_PATH, custom_objects=custom_objects)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

def simulate_app_prediction(path, description):
    print(f"\n--- {description} ({os.path.basename(path)}) ---")
    
    # 1. Prepare Image
    img = image.load_img(path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    processed_img = resnet_preprocess_input(img_array)
    
    # 2. Predict
    preds = model.predict(processed_img, verbose=0)[0]
    idx = np.argmax(preds)
    result = CLASS_NAMES[idx]
    confidence = float(preds[idx])
    
    print(f"Raw Prediction: {result} ({confidence*100:.2f}%)")
    
    # 3. Apply App Logic
    final_result = result
    
    # Logic from app.py
    if result == 'Wheat Loose Smut' and confidence < 0.60:
         print(f"  -> Applied Logic: Smut < 60% ({confidence:.2f}) -> Healthy")
         final_result = 'Healthy Wheat'
         
    if confidence < 0.35:
        print(f"  -> Applied Logic: Low Conf < 35% -> Healthy")
        final_result = 'Healthy Wheat'

    print(f"FINAL RESULT: {final_result}")

files = [
    ("d:/WDD/images/Healthy Wheat/00021.jpg", "Real Healthy"),
    ("d:/WDD/images/Leaf Rust/02771.jpeg", "Real Rust"),
    ("d:/WDD/images/Mildew/mildew_0.png", "Real Mildew"),
    ("d:/WDD/images/Wheat Loose Smut/0011.jpg", "Real Smut"),
    ("d:/WDD/images/Yellow Rust/yellow_rust_0.png", "Real Yellow Rust")
]

for p, d in files:
    if os.path.exists(p):
        simulate_app_prediction(p, d)
