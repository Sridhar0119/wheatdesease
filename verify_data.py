
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

DATA_DIR = 'images'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print(f"Checking data in {os.path.abspath(DATA_DIR)}")

if not os.path.exists(DATA_DIR):
    print("Data directory not found!")
    exit()

datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print("Class Indices:", generator.class_indices)
print("Samples:", generator.samples)
print("Classes found:", len(generator.class_indices))

# Check counts per class
from collections import Counter
counts = Counter(generator.classes)
print("Counts per class index:")
for idx, count in counts.items():
    class_name = [k for k, v in generator.class_indices.items() if v == idx][0]
    print(f"  {idx} ({class_name}): {count}")

