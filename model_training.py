import os
import tensorflow as tf
import numpy as np
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
from tensorflow.keras.applications.resnet50 import preprocess_input

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
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
DATA_DIR = 'images'
NUM_CLASSES = 5
EPOCHS_HEAD = 10
EPOCHS_FINE_TUNE = 50  # Increased for better convergence
MODEL_NAME = 'final_wheat_disease_resnet50.h5'

def create_model(num_classes):
    # Load ResNet50 base model with ImageNet weights if possible
    try:
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    except Exception as e:
        print(f"Warning: Could not load ImageNet weights due to {e}. Using randomly initialized weights.")
        base_model = ResNet50(weights=None, include_top=False, input_shape=IMG_SIZE + (3,))
    
    # Freeze the base model initially
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # Added L2 regularization to prevent overfitting on small classes
    x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

def main():
    print("Initializing Enhanced Training Pipeline for MAX ACCURACY...")
    
    # 1. Advanced Data Augmentation
    # Added vertical_flip since leaves can be oriented in any valid direction
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=40,        # Increased rotation
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,           # Increased zoom
        horizontal_flip=True,
        vertical_flip=True,       # Added vertical flip
        fill_mode='nearest',
        brightness_range=[0.7, 1.3], # Wider brightness range
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
    # Apply a boost to the minority class (Healthy Wheat - Class 0) if it's indeed the minority
    # We found Healthy Wheat has ~134 images vs >1000 for others.
    # The 'balanced' weight already suggests a high weight (e.g., 8x or 10x).
    # We will slightly amplify it further to be safe, but rely mainly on Focal Loss.
    
    class_weights = dict(enumerate(class_weights_array))
    print(f"Base Balanced Weights: {class_weights}")
    
    # Additional boost for Healthy Wheat (Index 0)
    # 'balanced' might give ~3.0 for Healthy vs 0.3 for others (10x ratio).
    # We multiply healthy by 1.5x on top of that to prioritize recall.
    if 0 in class_weights:
        class_weights[0] = class_weights[0] * 1.5
    
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

    # Callbacks - Increased Patience
    checkpoint = ModelCheckpoint(MODEL_NAME, monitor='val_accuracy', save_best_only=True, verbose=1)
    # Patience increased to allow recovery from plateaus
    early_stop = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1)
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
    # Unfreeze top 40 layers (slightly deeper fine-tuning)
    for layer in base_model.layers[:-40]:
        layer.trainable = False
        
    print("Compiling model for Phase 2 (Lower Learning Rate)...")
    model.compile(optimizer=Adam(learning_rate=1e-5), 
                  loss=focal_loss(gamma=2.0, alpha=0.25), # Keep Focal Loss for fine-tuning
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
    
    print(f"Training Complete. Best model saved as {MODEL_NAME}")
    
    # Plotting is optional but good for debugging
    try:
        import matplotlib.pyplot as plt
        
        acc = history_head.history['accuracy'] + history_fine.history['accuracy']
        val_acc = history_head.history['val_accuracy'] + history_fine.history['val_accuracy']
        loss = history_head.history['loss'] + history_fine.history['loss']
        val_loss = history_head.history['val_loss'] + history_fine.history['val_loss']
        
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        
        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.savefig('training_history_improved.png')
    except ImportError:
        pass

if __name__ == "__main__":
    main()
