
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

MODEL_PATH = 'fixed_wheat_disease_resnet50.h5'
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

def analyze_file(path, description):
    print(f"\n--- Analyzing {description} ({os.path.basename(path)}) ---")
    try:
        img = image.load_img(path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = resnet_preprocess_input(img_array)
        preds = model.predict(img_array, verbose=0)[0]
        
        # Print all
        for i, prob in enumerate(preds):
            print(f"{CLASS_NAMES[i]}: {prob*100:.4f}%")
            
        winner_idx = np.argmax(preds)
        print(f"Winner: {CLASS_NAMES[winner_idx]} ({preds[winner_idx]*100:.2f}%)")
        
    except Exception as e:
        print(f"Error: {e}")

# Real Disease (Problematic Low Confidence)
analyze_file("d:/WDD/images/Leaf Rust/02771.jpeg", "Real Disease (Rust)")

# Real Healthy (Problematic High False Positive)
analyze_file("d:/WDD/images/Healthy Wheat/00021.jpg", "Real Healthy (False Smut)")
analyze_file("d:/WDD/images/Healthy Wheat/00041.jpg", "Real Healthy (False Rust High Conf)")
