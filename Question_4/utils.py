# QAI_Interview/Question_4/utils.py

from PIL import Image
import numpy as np
import os
from flask import current_app

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    with Image.open(image_path) as img:
        img = img.convert('L')  # convert to grayscale
        img = img.resize((28, 28))  # resize to 28x28 pixels
        img_data = np.array(img, dtype=np.float32).flatten() / 255.0
        img_data = np.insert(img_data, 0, 1)  # add bias term
    return img_data.reshape(1, -1)
