import os
import tensorflow as tf
import numpy as np
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

# Fix for reproducible results
tf.random.set_seed(42)
np.random.seed(42)

# --- Configuration ---
BATCH_SIZE = 16 
IMG_SIZE = (224, 224)
DATA_DIR = 'images'
NUM_CLASSES = 5
EPOCHS_HEAD = 5         
EPOCHS_FINE_TUNE = 10   
MODEL_NAME = 'final_wheat_disease_efficientnet.h5'

def create_efficientnet_model(num_classes):
    input_shape = IMG_SIZE + (3,)
    try:
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    except:
        print("Warning: Could not load ImageNet weights. Using random initialization.")
        base_model = EfficientNetB0(weights=None, include_top=False, input_shape=input_shape)
    
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.4)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

def main():
    print("="*50)
    print(f"Training New Algorithm: EfficientNetB0 (Resume Supported)")
    print("="*50)
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} not found.")
        return

    # Data Augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    valid_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2
    )

    print("Loading datasets...")
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = valid_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    # Calculate Weights
    print("Computing class weights...")
    class_weights_array = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.arange(NUM_CLASSES),
        y=train_generator.classes
    )
    class_weights = {i: float(w) for i, w in enumerate(class_weights_array)}
    if 0 in class_weights: class_weights[0] *= 2.5 
    print(f"Class Weights: {class_weights}")

    # --- RESUME LOGIC ---
    initial_phase_done = False
    
    if os.path.exists(MODEL_NAME):
        print(f"\n[INFO] Found existing model: {MODEL_NAME}")
        try:
            from tensorflow.keras.models import load_model
            model = load_model(MODEL_NAME)
            print("Successfully loaded checkpoint. Skipping Phase 1.")
            initial_phase_done = True
            
            # Re-compile to ensure optimizer state is fresh/correct
            model.compile(optimizer=Adam(learning_rate=1e-5), 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
            base_model = model.layers[0] # The base is usually the first layer
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting fresh.")
            model, base_model = create_efficientnet_model(NUM_CLASSES)
    else:
        print("[INFO] No checkpoint found. Starting fresh.")
        model, base_model = create_efficientnet_model(NUM_CLASSES)

    # Compile Phase 1 (Only if starting fresh)
    if not initial_phase_done:
        print("\nPhase 1: Training Classification Head...")
        model.compile(optimizer=Adam(learning_rate=0.001), 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])
        
        for epoch in range(EPOCHS_HEAD):
            print(f"Epoch {epoch+1}/{EPOCHS_HEAD}")
            model.fit(
                train_generator,
                epochs=1,
                validation_data=validation_generator,
                class_weight=class_weights,
                verbose=1
            )
            try:
                 model.save(MODEL_NAME)
                 print(f"Saved model after epoch {epoch+1}")
            except Exception as e:
                 print(f"Error saving model: {e}")

    # Phase 2: Fine Tuning
    print("\nPhase 2: Fine Tuning EfficientNet Layers...")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
        
    # Lower learning rate for fine tuning
    model.compile(optimizer=Adam(learning_rate=1e-5), 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
                  
    print(f"Starting Fine Tuning for {EPOCHS_FINE_TUNE} Epochs...")
    for epoch in range(EPOCHS_FINE_TUNE):
        print(f"Fine Tune Epoch {epoch+1}/{EPOCHS_FINE_TUNE}")
        model.fit(
            train_generator,
            epochs=1,
            validation_data=validation_generator,
            class_weight=class_weights,
            verbose=1
        )
        try:
             model.save(MODEL_NAME)
             print(f"Saved model after fine tune epoch {epoch+1}")
        except Exception as e:
             print(f"Error saving model: {e}")
    
    print(f"\nTraining Complete. Model saved to {MODEL_NAME}")

if __name__ == '__main__':
    main()
