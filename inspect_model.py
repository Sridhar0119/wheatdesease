import h5py
import json
import numpy as np

MODEL_PATH = 'best_wheat_disease_resnet50.h5'

try:
    with h5py.File(MODEL_PATH, 'r') as f:
        if 'model_config' in f.attrs:
            config_str = f.attrs['model_config']
            if isinstance(config_str, bytes):
                config_str = config_str.decode('utf-8')
            config = json.loads(config_str)
            
            if 'config' in config and 'layers' in config['config']:
                layers = config['config']['layers']
                print(f"Total layers: {len(layers)}")
                
                print("Last 5 layers:")
                for layer in layers[-5:]:
                    print(json.dumps(layer, indent=2))
            elif 'layers' in config:
                 layers = config['layers']
                 print(f"Total layers: {len(layers)}")
                 print("Last 5 layers:")
                 for layer in layers[-5:]:
                    print(json.dumps(layer, indent=2))
        else:
            print("No model_config found.")
except Exception as e:
    print(f"Error: {e}")
