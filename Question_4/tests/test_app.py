import sys
import os
import pytest

# Đưa thư mục gốc vào PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app  # Import từ app.py

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_index(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Upload an Image' in response.data

def test_predict(client):
    with open('./Question_4/test_image.png', 'rb') as img:
        response = client.post('/predict', data={'file': img})
    assert response.status_code == 200
    assert b'Predicted Digit' in response.data
