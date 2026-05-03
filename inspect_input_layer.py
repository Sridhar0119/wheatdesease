
import h5py
import json

MODEL_PATH = 'best_wheat_disease_resnet50.h5'

try:
    with h5py.File(MODEL_PATH, 'r') as f:
        if 'model_config' in f.attrs:
            config_str = f.attrs['model_config']
            if isinstance(config_str, bytes):
                config_str = config_str.decode('utf-8')
            config = json.loads(config_str)
            
            # Find input layer
            layers = config['config']['layers']
            for layer in layers:
                if layer['class_name'] == 'InputLayer':
                    print("--- InputLayer Config ---")
                    print(json.dumps(layer, indent=2))
        else:
            print("No model_config")
except Exception as e:
    print(f"Error: {e}")
