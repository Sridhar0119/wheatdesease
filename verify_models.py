import os
import tensorflow as tf
from tensorflow.keras.models import load_model

print("TensorFlow Version:", tf.__version__)

RESNET_PATH = 'final_wheat_disease_resnet50.h5'
EFFNET_PATH = 'final_wheat_disease_efficientnet.h5'

print(f"\nChecking ResNet at {RESNET_PATH}")
if os.path.exists(RESNET_PATH):
    print(f"File size: {os.path.getsize(RESNET_PATH)} bytes")
    try:
        model = load_model(RESNET_PATH, compile=False)
        print("ResNet Loaded Successfully")
    except Exception as e:
        print(f"ResNet Load Failed: {e}")
else:
    print("ResNet File not found")

print(f"\nChecking EfficientNet at {EFFNET_PATH}")
if os.path.exists(EFFNET_PATH):
    print(f"File size: {os.path.getsize(EFFNET_PATH)} bytes")
    try:
        model = load_model(EFFNET_PATH, compile=False)
        print("EfficientNet Loaded Successfully")
    except Exception as e:
        print(f"EfficientNet Load Failed: {e}")
else:  
    print("EfficientNet File not found")

try:
    from tensorflow.keras.applications.efficientnet import preprocess_input
    print("\nEfficientNet Preprocess Import: SUCCESS")
except ImportError as e:
    print(f"\nEfficientNet Preprocess Import: FAILED ({e})")
