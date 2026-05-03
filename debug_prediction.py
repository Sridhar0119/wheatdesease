
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Model setup
MODEL_PATH = 'improved_wheat_disease_resnet50.h5'
CLASS_NAMES = ['Healthy Wheat', 'Leaf Rust', 'Mildew', 'Wheat Loose Smut', 'Yellow Rust']

# Fix for batch_shape compatibility (copy from app.py)
from tensorflow.keras.layers import InputLayer
class CustomInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        if 'batch_shape' in kwargs:
            batch_shape = kwargs.pop('batch_shape')
            if batch_shape is not None and len(batch_shape) > 1:
                kwargs['input_shape'] = batch_shape[1:]
        super().__init__(*args, **kwargs)

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # MATCHING APP.PY
    return img_array

print(f"Loading {MODEL_PATH}...")
custom_objects = {'InputLayer': CustomInputLayer}
try:
    model = load_model(MODEL_PATH, custom_objects=custom_objects)
except Exception as e:
    print(f"Failed to load model: {e}")
    exit()

# Predict
img_path = 'd:/WDD/images/Healthy Wheat/00021.jpg'
if not os.path.exists(img_path):
    print("Debug image not found.")
    exit()

print(f"Predicting on {img_path}...")
processed_img = prepare_image(img_path)
predictions = model.predict(processed_img, verbose=0)[0]

print("\n--- Raw Probabilities ---")
for i, prob in enumerate(predictions):
    print(f"{CLASS_NAMES[i]}: {prob*100:.4f}%")

predicted_index = np.argmax(predictions)
print(f"\nWinner: {CLASS_NAMES[predicted_index]} ({predictions[predicted_index]*100:.2f}%)")

# Heuristic Test: Penalize Healthy
print("\n--- With Healthy Penalized (0.2x) ---")
mod_preds = predictions.copy()
mod_preds[0] *= 0.2 # Penalize healthy
# Renormalize
mod_preds = mod_preds / np.sum(mod_preds)

for i, prob in enumerate(mod_preds):
    print(f"{CLASS_NAMES[i]}: {prob*100:.4f}%")

predicted_index = np.argmax(mod_preds)
print(f"New Winner: {CLASS_NAMES[predicted_index]} ({mod_preds[predicted_index]*100:.2f}%)")
