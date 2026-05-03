import h5py
import json
import shutil
import os

INPUT_MODEL = 'best_wheat_disease_resnet50.h5'
OUTPUT_MODEL = 'fixed_wheat_disease_resnet50.h5'

print(f"Copying {INPUT_MODEL} to {OUTPUT_MODEL}...")
shutil.copy2(INPUT_MODEL, OUTPUT_MODEL)

def fix_config(config):
    # Handle both Sequential and Functional
    layers = config['config']['layers'] if 'config' in config else config.get('layers', [])
    
    for layer in layers:
        layer_config = layer['config']
        
        # Fix InputLayer batch_shape -> batch_input_shape
        if layer['class_name'] == 'InputLayer':
            if 'batch_shape' in layer_config:
                print(f"Fixing InputLayer: {layer['name']} (batch_shape -> batch_input_shape)")
                layer_config['batch_input_shape'] = layer_config.pop('batch_shape')
        
        # Fix DTypePolicy -> string
        if 'dtype' in layer_config and isinstance(layer_config['dtype'], dict):
             if layer_config['dtype'].get('class_name') == 'DTypePolicy':
                 # print(f"Fixing DTypePolicy for layer: {layer['name']}")
                 layer_config['dtype'] = layer_config['dtype']['config']['name']

        # Fix BatchNormalization synchronized argument
        if layer['class_name'] == 'BatchNormalization':
            if 'synchronized' in layer_config:
                print(f"Fixing BatchNormalization: {layer['name']} (removing synchronized)")
                layer_config.pop('synchronized')

        # Fix inbound_nodes format
        if 'inbound_nodes' in layer and layer['inbound_nodes']:
            new_inbound_nodes = []
            for node in layer['inbound_nodes']:
                # Check if it's the new format (dict)
                if isinstance(node, dict) and 'args' in node:
                    # print(f"Fixing inbound_node for layer: {layer['name']}")
                    args = node.get('args', [])
                    converted_args = []
                    for arg in args:
                        if isinstance(arg, list):
                             # Handle list of tensors (e.g. for Add layer)
                             for sub_arg in arg:
                                 if isinstance(sub_arg, dict) and sub_arg.get('class_name') == '__keras_tensor__':
                                     history = sub_arg['config']['keras_history']
                                     converted_args.append(history + [{}])
                        elif isinstance(arg, dict) and arg.get('class_name') == '__keras_tensor__':
                            history = arg['config']['keras_history']
                            converted_args.append(history + [{}])
                    
                    # CRITICAL FIX: Flatten if single input to avoid UnboundLocalError in process_node
                    if not converted_args:
                        if args:
                            print(f"Warning: Layer {layer['name']} has args but no converted_args. Args: {args}")
                        # If empty, do not append empty list to avoid [[]]
                    elif len(converted_args) == 1:
                        new_inbound_nodes.append(converted_args[0])
                    else:
                        new_inbound_nodes.append(converted_args)
                else:
                    new_inbound_nodes.append(node)
            
            layer['inbound_nodes'] = new_inbound_nodes
            
    return config

try:
    with h5py.File(OUTPUT_MODEL, 'r+') as f:
        if 'model_config' in f.attrs:
            config_str = f.attrs['model_config']
            if isinstance(config_str, bytes):
                config_str = config_str.decode('utf-8')
            config = json.loads(config_str)
            
            fixed_config = fix_config(config)
            
            f.attrs['model_config'] = json.dumps(fixed_config).encode('utf-8')
            print("Model config fixed and saved.")
        else:
            print("No model_config found.")
        
        # Fix optimizer config - remove unsupported weight_decay parameter
        if 'training_config' in f.attrs:
            training_config_str = f.attrs['training_config']
            if isinstance(training_config_str, bytes):
                training_config_str = training_config_str.decode('utf-8')
            training_config = json.loads(training_config_str)
            
            if 'optimizer_config' in training_config:
                optimizer_config = training_config['optimizer_config']
                if 'config' in optimizer_config:
                    opt_params = optimizer_config['config']
                    # List of parameters not supported in TF 2.10.0
                    # These are from newer Keras 3.x / TF 2.16+ versions
                    unsupported_params = [
                        'weight_decay', 'use_ema', 'ema_momentum', 'ema_overwrite_frequency', 
                        'loss_scale_factor', 'gradient_accumulation_steps'
                    ]
                    removed_params = []
                    for param in unsupported_params:
                        if param in opt_params:
                            opt_params.pop(param)
                            removed_params.append(param)
                    
                    if removed_params:
                        print(f"Removing unsupported optimizer parameters: {removed_params}")
                        f.attrs['training_config'] = json.dumps(training_config).encode('utf-8')
                        print("Optimizer config fixed and saved.")
except Exception as e:
    print(f"Error: {e}")
