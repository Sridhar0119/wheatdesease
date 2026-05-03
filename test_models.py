import os
import numpy as np
from app import prepare_image, build_model

import tensorflow as tf

MODEL_PATH = 'best_wheat_disease_resnet50.h5'
model = build_model()
model.load_weights(MODEL_PATH)

def test_img(path, label):
    img = prepare_image(path)
    preds = model.predict(img, verbose=0)[0]
    idx = np.argmax(preds)
    print(f"{label} -> predicted index {idx}, probs: {np.round(preds, 3)}")

print(f"Testing {MODEL_PATH}")
test_img('d:/WDD/images/Healthy Wheat/00021.jpg', 'Real Healthy')
test_img('d:/WDD/images/Leaf Rust/02771.jpeg', 'Real Rust')
test_img('d:/WDD/images/Mildew/mildew_0.png', 'Real Mildew')
test_img('d:/WDD/images/Wheat Loose Smut/0011.jpg', 'Real Smut')
test_img('d:/WDD/images/Yellow Rust/yellow_rust_0.png', 'Real Yellow Rust')
