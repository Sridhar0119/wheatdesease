
import os
import tensorflow as tf
import numpy as np
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
from tensorflow.keras.applications.resnet import preprocess_input

# Focal loss implementation for handling class imbalance
def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        # standard categorical cross‑entropy
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        pt = tf.exp(-ce)
        focal = alpha * tf.pow(1 - pt, gamma) * ce
        return focal
    return loss

# Set random seed for reproducibility
tf.random.set_seed(42)

# Configuration
BATCH_SIZE = 16 # Reduce batch size for larger model
IMG_SIZE = (224, 224)
DATA_DIR = 'd:/WDD/images' # Absolute path
NUM_CLASSES = 5
EPOCHS_HEAD = 10
EPOCHS_FINE_TUNE = 60  # Increased for deeper model
MODEL_NAME = 'wheat_disease_resnet152.h5'

def create_model(num_classes):
    # Load ResNet152 base model with ImageNet weights
    try:
        base_model = ResNet152(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    except Exception as e:
        print(f"Warning: Could not load ImageNet weights due to {e}. Using randomly initialized weights.")
        base_model = ResNet152(weights=None, include_top=False, input_shape=IMG_SIZE + (3,))
    
    # Freeze the base model initially
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # Added L2 regularization
    x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.5)(x)
    # Another dense layer for better feature extraction
    x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

def main():
    print("Initializing ResNet152 Training Pipeline for 100% ACCURACY...")
    
    # 1. Advanced Data Augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=45,        
        width_shift_range=0.25,
        height_shift_range=0.25,
        shear_range=0.25,
        zoom_range=0.35,           
        horizontal_flip=True,
        vertical_flip=True,       
        fill_mode='nearest',
        brightness_range=[0.6, 1.4], 
        validation_split=0.2
    )

    valid_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2
    )

    print(f"Loading Data from {DATA_DIR}...")
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory {DATA_DIR} not found!")
        return

    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
    
    # Robust Class Weight Calculation
    print("Computing class weights...")
    class_weights_array = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.arange(NUM_CLASSES),
        y=train_generator.classes
    )
    
    class_weights = dict(enumerate(class_weights_array))
    print(f"Base Balanced Weights: {class_weights}")
    
    # Additional boost for Healthy Wheat (Index 0)
    # Assuming Healthy Wheat is index 0 based on folder sorting, but let's be careful.
    # Class indices are generated alphabetically. 
    # 'Healthy Wheat' will be 0 if folders are: ['Healthy Wheat', 'Leaf Rust', 'Mildew', 'Wheat Loose Smut', 'Yellow Rust']
    # If using absolute path, verify indices.
    print(f"Class Indices: {train_generator.class_indices}")
    
    healthy_idx = train_generator.class_indices.get('Healthy Wheat')
    if healthy_idx is not None:
         class_weights[healthy_idx] = class_weights[healthy_idx] * 2.0 # Higher boost
    
    print(f"Final Class Weights: {class_weights}")

    validation_generator = valid_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    # 2. Build Model
    model, base_model = create_model(NUM_CLASSES)
    
    # 3. Compile Model (Phase 1: Head only)
    print("Compiling model for Phase 1 (Training Head)...")
    
    model.compile(optimizer=Adam(learning_rate=0.001),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=['accuracy'])

    # Callbacks
    checkpoint = ModelCheckpoint(MODEL_NAME, monitor='val_accuracy', save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=1e-7, verbose=1)

    # 4. Train Head
    print("Starting Phase 1 Training (Frozen Base)...")
    history_head = model.fit(
        train_generator,
        epochs=EPOCHS_HEAD,
        validation_data=validation_generator,
        callbacks=[checkpoint, reduce_lr],
        class_weight=class_weights
    )

    # 5. Fine Tuning (Phase 2: Unfreeze top layers)
    print("Unfreezing top layers for Phase 2 (Fine Tuning)...")
    
    base_model.trainable = True
    # Unfreeze top 100 layers for ResNet152 (it has >150 layers)
    # ResNet152 is deep. Let's unfreeze the last conv block.
    # A safe bet is unfreezing the last 60 layers.
    for layer in base_model.layers[:-60]:
        layer.trainable = False
        
    print("Compiling model for Phase 2 (Lower Learning Rate)...")
    model.compile(optimizer=Adam(learning_rate=1e-5), 
                  loss=focal_loss(gamma=2.0, alpha=0.25), 
                  metrics=['accuracy'])
                  
    print("Starting Phase 2 Training...")
    history_fine = model.fit(
        train_generator,
        epochs=EPOCHS_FINE_TUNE,
        initial_epoch=history_head.epoch[-1],
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stop, reduce_lr],
        class_weight=class_weights
    )
    
    print(f"Training Complete. Ultimate model saved as {MODEL_NAME}")

if __name__ == "__main__":
    main()
