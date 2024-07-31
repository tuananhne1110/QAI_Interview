import sys
import os
import pytest

# Add the directory containing `app.py` to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app  # Import the Flask app instance

@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    with app.test_client() as client:
        yield client

def test_homepage(client):
    """Test the homepage renders correctly."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Upload an Image' in response.data

def test_upload_image(client):
    """Test uploading an image and receiving a prediction."""
    # Note: Make sure to replace 'test_image.png' with a valid image path
    with open('./Question_4/test_image.png', 'rb') as img:
        response = client.post('/', data={'file': (img, 'test_image.png')})
    assert response.status_code == 200
    assert b'Prediction' in response.data  # Check that the result is part of the response
