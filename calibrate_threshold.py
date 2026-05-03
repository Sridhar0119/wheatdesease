
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

MODEL_PATH = 'improved_wheat_disease_resnet50.h5'
CLASS_NAMES = ['Healthy Wheat', 'Leaf Rust', 'Mildew', 'Wheat Loose Smut', 'Yellow Rust']

from tensorflow.keras.layers import InputLayer
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
    print(f"Error: {e}")
    exit()

def predict_file(path):
    try:
        img = image.load_img(path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = resnet_preprocess_input(img_array) # Correct Preprocessing
        preds = model.predict(img_array, verbose=0)[0]
        idx = np.argmax(preds)
        return CLASS_NAMES[idx], preds[idx]
    except Exception as e:
        return "Error", 0.0

print("\n--- CALIBRATION CHECK ---")

# Check the specific problematic file if possible
problem_file = "d:/WDD/images/Leaf Rust/02771.jpeg"
if os.path.exists(problem_file):
    print(f"\nProblem File ({problem_file}):")
    cls, conf = predict_file(problem_file)
    print(f"Prediction: {cls} | Confidence: {conf*100:.2f}%")

print("\n--- Sampling 'Leaf Rust' (Expect: High Confidence) ---")
rust_dir = "d:/WDD/images/Leaf Rust"
files = [os.path.join(rust_dir, f) for f in os.listdir(rust_dir) if f.endswith(('.jpg', '.jpeg', '.png'))][:10]
for f in files:
    cls, conf = predict_file(f)
    print(f"{os.path.basename(f)}: {cls} ({conf*100:.2f}%)")

print("\n--- Sampling 'Healthy Wheat' (Expect: Low Confidence or Correct Class) ---")
healthy_dir = "d:/WDD/images/Healthy Wheat"
files = [os.path.join(healthy_dir, f) for f in os.listdir(healthy_dir) if f.endswith(('.jpg', '.jpeg', '.png'))][:10]
for f in files:
    cls, conf = predict_file(f)
    print(f"{os.path.basename(f)}: {cls} ({conf*100:.2f}%)")
