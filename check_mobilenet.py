
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

MODEL_PATH = 'mobilenet_wheat.h5'
CLASS_NAMES = ['Healthy Wheat', 'Leaf Rust', 'Mildew', 'Wheat Loose Smut', 'Yellow Rust']

def analyze_file(path, description):
    print(f"\n=== Analyzing {description} ({os.path.basename(path)}) ===")
    if not os.path.exists(path):
        print(f"  File {path} not found.")
        return

    # Load image
    img = image.load_img(path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess (MobileNetV2 style: -1 to 1)
    processed_img = preprocess_input(img_array)
    
    if not os.path.exists(MODEL_PATH):
        print(f"  Model {MODEL_PATH} not found yet.")
        return

    try:
        model = load_model(MODEL_PATH)
        preds = model.predict(processed_img, verbose=0)[0]
        
        # Show Top 3
        top_indices = preds.argsort()[-3:][::-1]
        for i in top_indices:
            print(f"    {CLASS_NAMES[i]}: {preds[i]*100:.4f}%")
            
        winner_idx = np.argmax(preds)
        print(f"  > WINNER: {CLASS_NAMES[winner_idx]}")
        
    except Exception as e:
        print(f"    Error: {e}")

if __name__ == "__main__":
    files = [
        ("d:/WDD/images/Healthy Wheat/00021.jpg", "Real Healthy"),
        ("d:/WDD/images/Leaf Rust/02771.jpeg", "Real Leaf Rust"),
        ("d:/WDD/images/Mildew/mildew_0.png", "Real Mildew"),
        ("d:/WDD/images/Wheat Loose Smut/0011.jpg", "Real Smut"),
        ("d:/WDD/images/Yellow Rust/yellow_rust_0.png", "Real Yellow Rust")
    ]
    
    for p, d in files:
        analyze_file(p, d)
