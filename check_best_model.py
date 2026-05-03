
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
import h5py

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

MODEL_PATHS = ['best_wheat_disease_resnet50.h5', 'wheat_disease_resnet152.h5', 'final_wheat_disease_resnet50.h5', 'wheat_disease_mobilenetv3.h5']
CLASS_NAMES = ['Healthy Wheat', 'Leaf Rust', 'Mildew', 'Wheat Loose Smut', 'Yellow Rust']

from tensorflow.keras.layers import InputLayer

class CustomInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        if 'batch_shape' in kwargs:
            batch_shape = kwargs.pop('batch_shape')
            if batch_shape is not None and len(batch_shape) > 1:
                kwargs['input_shape'] = batch_shape[1:]
        super().__init__(*args, **kwargs)

# Mock DTypePolicy
class DTypePolicy:
    def __init__(self, *args, **kwargs):
        self.name = 'float32'
        self._name = 'float32'
        self.compute_dtype = 'float32'
        self.variable_dtype = 'float32'
        self.compute_dtype = 'float32'
        self.variable_dtype = 'float32'
    @classmethod
    def from_config(cls, config):
        return cls()
    def get_config(self):
        return {}

from tensorflow.keras.layers import BatchNormalization
class CustomBatchNormalization(BatchNormalization):
    def __init__(self, *args, **kwargs):
        if 'synchronized' in kwargs:
            kwargs.pop('synchronized')
        super().__init__(*args, **kwargs)

def load_with_h5py_hack(model_path):
    # Try loading with custom object
    custom_objects = {
        'InputLayer': CustomInputLayer,
        'DTypePolicy': DTypePolicy,
        'BatchNormalization': CustomBatchNormalization
    }
    try:
        return load_model(model_path, custom_objects=custom_objects)
    except Exception as e:
        print(f"Standard load failed: {e}")
        return None

for model_path in MODEL_PATHS:
    print(f"\nChecking model: {model_path}")
    if not os.path.exists(model_path):
        print(f"Skipping {model_path} (not found)")
        continue
        
    model = load_with_h5py_hack(model_path)

    if model:
        print("SUCCESS: Loaded model.")
        
        def predict_file(path, label):
            try:
                img = image.load_img(path, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                
                # Try ResNet preprocess first
                p_resnet = model.predict(resnet_preprocess_input(img_array.copy()), verbose=0)[0]
                
                # Try Rescale first
                img_rescale = img_array.copy() / 255.0
                p_rescale = model.predict(img_rescale, verbose=0)[0]
                
                print(f"\n--- {label} ---")
                i_res = np.argmax(p_resnet)
                print(f"ResNet Preprocess: {CLASS_NAMES[i_res]} ({p_resnet[i_res]*100:.2f}%)")
                
                i_scl = np.argmax(p_rescale)
                print(f"Rescale Preprocess: {CLASS_NAMES[i_scl]} ({p_rescale[i_scl]*100:.2f}%)")
                
            except Exception as e:
                print(f"Err: {e}")

        predict_file("d:/WDD/images/Leaf Rust/02771.jpeg", "Real Disease (Rust)")
        predict_file("d:/WDD/images/Healthy Wheat/00021.jpg", "Real Healthy")
    else:
        print(f"FATAL: Could not load {model_path}.")
