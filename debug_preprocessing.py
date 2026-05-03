
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

MODEL_PATH = 'best_wheat_disease_resnet50.h5'
CLASS_NAMES = ['Healthy Wheat', 'Leaf Rust', 'Mildew', 'Wheat Loose Smut', 'Yellow Rust']

from tensorflow.keras.layers import InputLayer
class CustomInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        if 'batch_shape' in kwargs:
            batch_shape = kwargs.pop('batch_shape')
            if batch_shape is not None and len(batch_shape) > 1:
                kwargs['input_shape'] = batch_shape[1:]
        super().__init__(*args, **kwargs)

def prepare_image_resnet(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = resnet_preprocess_input(img_array) # Original method
    return img_array

def prepare_image_rescale(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 # New method
    return img_array

print(f"Loading {MODEL_PATH}...")
custom_objects = {'InputLayer': CustomInputLayer}
try:
    model = load_model(MODEL_PATH, custom_objects=custom_objects)
except Exception as e:
    print(f"Failed to load model: {e}")
    exit()

img_path = 'd:/WDD/images/Healthy Wheat/00021.jpg' # Usage healthy image

print("\n--- Test 1: ResNet Preprocessing ---")
p1 = model.predict(prepare_image_resnet(img_path), verbose=0)[0]
i1 = np.argmax(p1)
print(f"Winner: {CLASS_NAMES[i1]} ({p1[i1]*100:.2f}%)")
print(f"Healthy: {p1[0]*100:.2f}%")

print("\n--- Test 2: Rescale 1/255 ---")
p2 = model.predict(prepare_image_rescale(img_path), verbose=0)[0]
i2 = np.argmax(p2)
print(f"Winner: {CLASS_NAMES[i2]} ({p2[i2]*100:.2f}%)")
print(f"Healthy: {p2[0]*100:.2f}%")
