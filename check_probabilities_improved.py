
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.layers import InputLayer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

MODELS = [
    'improved_wheat_disease_resnet50.h5',
    'final_wheat_disease_resnet50.h5',
    'improved_wheat_disease_resnet50.h5',
    'fixed_wheat_disease_resnet50.h5'
]
CLASS_NAMES = ['Healthy Wheat', 'Leaf Rust', 'Mildew', 'Wheat Loose Smut', 'Yellow Rust']

class CustomInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        if 'batch_shape' in kwargs:
            batch_shape = kwargs.pop('batch_shape')
            if batch_shape is not None and len(batch_shape) > 1:
                kwargs['input_shape'] = batch_shape[1:]
        super().__init__(*args, **kwargs)

def analyze_file(path, description):
    print(f"\n=== Analyzing {description} ({os.path.basename(path)}) ===")
    if not os.path.exists(path):
        print(f"  File {path} not found.")
        return

    # Load image once
    img = image.load_img(path, target_size=(224, 224))
    img_array_base = image.img_to_array(img) # 0-255 RGB
    img_array_base = np.expand_dims(img_array_base, axis=0)
    
    modes = {
        "ResNet Default (caffe)": lambda x: resnet_preprocess(x.copy()),
        "Scale 0-1 (/255)": lambda x: x.copy() / 255.0,
        "Scale -1 to 1": lambda x: (x.copy() / 127.5) - 1.0
    }

    for model_path in MODELS:
        if not os.path.exists(model_path): continue
        print(f"  -- Model: {model_path} --")
        
        try:
            custom_objects = {'InputLayer': CustomInputLayer}
            model = load_model(model_path, custom_objects=custom_objects, compile=False)
            
            for mode_name, func in modes.items():
                processed_img = func(img_array_base)
                preds = model.predict(processed_img, verbose=0)[0]
                winner_idx = np.argmax(preds)
                print(f"    [{mode_name}] Winner: {CLASS_NAMES[winner_idx]} ({preds[winner_idx]*100:.2f}%)")
                
        except Exception as e:
            print(f"    Error: {e}")

if __name__ == "__main__":
    analyze_file("d:/WDD/images/Healthy Wheat/00021.jpg", "Real Healthy")
    analyze_file("d:/WDD/images/Leaf Rust/02771.jpeg", "Real Leaf Rust")
    analyze_file("d:/WDD/images/Mildew/mildew_0.png", "Real Mildew")
    analyze_file("d:/WDD/images/Wheat Loose Smut/0011.jpg", "Real Smut")
    analyze_file("d:/WDD/images/Yellow Rust/yellow_rust_0.png", "Real Yellow Rust")
