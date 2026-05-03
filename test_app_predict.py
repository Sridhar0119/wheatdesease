import os
import io
import requests
from app import app

client = app.test_client()

img_path = 'd:/WDD/images/Leaf Rust/02771.jpeg'
with open(img_path, 'rb') as f:
    img_bytes = f.read()

data = {
    'file': (io.BytesIO(img_bytes), 'image.jpeg')
}

print("Sending request to /predict...")
response = client.post('/predict', data=data, content_type='multipart/form-data')

print("Status Code:", response.status_code)
print("Response Data:", response.get_json() or response.data)

