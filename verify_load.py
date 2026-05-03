
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

print(f"TensorFlow Version: {tf.__version__}")
MODEL_PATH = 'fixed_wheat_disease_resnet50.h5'

print(f"Attempting to load {MODEL_PATH} using load_model...")
try:
    model = load_model(MODEL_PATH)
    print("Success! Model loaded with load_model.")
    model.summary()
except Exception as e:
    print(f"Failed to load model: {e}")
