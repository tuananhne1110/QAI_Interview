import os
from flask import Flask, request, jsonify, render_template, url_for
from PIL import Image
import numpy as np
import io
import pickle
from preproc import PreProc  # Import PreProc từ preproc.py
from model import NeuralNetwork  # Import NeuralNetwork từ model.py

app = Flask(__name__)

# Load the pre-trained neural network model
def load_model():
    with open('./Question_4/weights/neural_network.pkl', 'rb') as f:
        return pickle.load(f)

nn = load_model()

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image = np.array(image)  # Convert to NumPy array
    image = image / 255.0  # Normalize pixel values to [0, 1]
    image = image.flatten()  # Flatten the image to match the input shape
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        # Đọc file ảnh và chuyển đổi thành đối tượng PIL Image
        image = Image.open(file)
        processed_image = preprocess_image(image)
        
        # Đảm bảo thư mục static tồn tại
        if not os.path.exists('./Question_4/static'):
            os.makedirs('./Question_4/static')
        
        # Xác định đường dẫn để lưu hình ảnh
        image_path = os.path.join('./Question_4/static', 'uploaded_image.png')
        
        try:
            # Lưu hình ảnh với định dạng PNG
            image.save(image_path, format='PNG')
            
            # Dự đoán kết quả
            nn.forward(processed_image)
            prediction = np.argmax(nn.out, axis=1)[0]
            
            # Xác định URL hình ảnh
            image_url = url_for('static', filename='uploaded_image.png')
            
            # Hiển thị kết quả dự đoán
            return render_template('result.html', prediction=int(prediction), image_url=image_url)
        except Exception as e:
            return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
