
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Config
DATA_DIR = 'images'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MODEL_FILE = 'mobilenet_wheat.h5'

def train():
    # 1. Data Generators
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    print("Loading Data...")
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    valid_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    # 2. Model Setup
    print("Building MobileNetV2 Model...")
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=IMG_SIZE + (3,)
    )
    
    # Freeze base
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(train_generator.num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # 3. Train Head
    print("Training Head (Phase 1)...")
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(
        train_generator,
        epochs=3, # Fast initial training
        validation_data=valid_generator
    )

    # 4. Fine Tune
    print("Fine Tuning (Phase 2)...")
    base_model.trainable = True
    # Fine-tune only top layers
    for layer in base_model.layers[:-40]:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(
        train_generator,
        epochs=3, # Short fine-tuning
        validation_data=valid_generator
    )

    # 5. Save
    print(f"Saving to {MODEL_FILE}...")
    model.save(MODEL_FILE)
    print("Done!")

if __name__ == '__main__':
    train()
