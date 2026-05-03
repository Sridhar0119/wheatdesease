
import os
import tensorflow as tf
import numpy as np
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers

# Try importing MobileNetV3
try:
    from tensorflow.keras.applications import MobileNetV3Large
    from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
    print("Successfully imported MobileNetV3Large")
except ImportError:
    print("MobileNetV3Large not found. Creating a custom training script for MobileNetV2 as fallback or upgrading TF required.")
    exit(1)

# Focal loss implementation
def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        pt = tf.exp(-ce)
        focal = alpha * tf.pow(1 - pt, gamma) * ce
        return focal
    return loss

tf.random.set_seed(42)

# Configuration
BATCH_SIZE = 32 # MobileNet is small, we can increase batch size
IMG_SIZE = (224, 224)
DATA_DIR = 'd:/WDD/images'
NUM_CLASSES = 5
EPOCHS_HEAD = 10
EPOCHS_FINE_TUNE = 40 
MODEL_NAME = 'wheat_disease_mobilenetv3.h5'

def create_model(num_classes):
    try:
        # include_top=False loads the feature extractor only
        base_model = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    except Exception as e:
        print(f"Warning: Could not load ImageNet weights: {e}")
        base_model = MobileNetV3Large(weights=None, include_top=False, input_shape=IMG_SIZE + (3,))
    
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

def main():
    print("Initializing MobileNetV3 Training Pipeline...")
    
    # Advanced Data Augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,        
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
    
    # Class Weights
    class_weights_array = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.arange(NUM_CLASSES),
        y=train_generator.classes
    )
    class_weights = dict(enumerate(class_weights_array))
    
    # Boost Healthy Wheat
    healthy_idx = train_generator.class_indices.get('Healthy Wheat')
    if healthy_idx is not None:
         class_weights[healthy_idx] = class_weights[healthy_idx] * 1.5
    
    print(f"Class Weights: {class_weights}")

    validation_generator = valid_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    model, base_model = create_model(NUM_CLASSES)
    
    # Phase 1
    print("Phase 1: Training Head...")
    model.compile(optimizer=Adam(learning_rate=0.001),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=['accuracy'])

    checkpoint = ModelCheckpoint(MODEL_NAME, monitor='val_accuracy', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, verbose=1)
    
    history_head = model.fit(
        train_generator,
        epochs=EPOCHS_HEAD,
        validation_data=validation_generator,
        callbacks=[checkpoint, reduce_lr],
        class_weight=class_weights
    )

    # Phase 2
    print("Phase 2: Fine Tuning...")
    base_model.trainable = True
    # MobileNetV3 is efficient, let's unfreeze more
    for layer in base_model.layers[:-50]: 
        layer.trainable = False
        
    model.compile(optimizer=Adam(learning_rate=1e-4), 
                  loss=focal_loss(gamma=2.0, alpha=0.25), 
                  metrics=['accuracy'])
                  
    early_stop = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)
    
    model.fit(
        train_generator,
        epochs=EPOCHS_FINE_TUNE,
        initial_epoch=history_head.epoch[-1],
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stop, reduce_lr],
        class_weight=class_weights
    )
    
    print(f"MobileNetV3 Training Complete. Saved to {MODEL_NAME}")

if __name__ == "__main__":
    main()
