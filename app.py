from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import requests

app = Flask(__name__)

# URL to the .h5 model file stored on GitHub or GitHub Releases
MODEL_URL = "https://github.com/username/repo/releases/download/v1.0.0/MAIN_MUZZLE.h5"
MODEL_PATH = "MAIN_MUZZLE.h5"  # Local path to save the model
INPUT_SHAPE = (71, 71, 3)  # Update to match your model's input shape

def download_model():
    """Download the model from the provided URL if not already present."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        response = requests.get(MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Model downloaded successfully.")
        else:
            raise Exception(f"Failed to download model. HTTP Status: {response.status_code}")

try:
    # Download and load the model
    download_model()
    model = load_model(MODEL_PATH)
    print('Model loaded successfully.')
    
    # Get the penultimate layer's output
    penultimate_layer_model = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)
except Exception as e:
    print(f'Error loading model: {str(e)}')
    exit(1)

# Function to preprocess image
def preprocess_image(img_path):
    """Preprocess image according to your model's requirements."""
    img = image.load_img(img_path, target_size=INPUT_SHAPE[:2])
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale the image
    return img_array

# Function to extract features using the penultimate layer
def extract_features(img_array):
    """Extract features using the penultimate layer of the model."""
    features = penultimate_layer_model.predict(img_array)
    features = features.flatten()  # Flatten to a 1D array
    return features[:256]  # Extract the first 256 features

@app.route('/extract_features', methods=['POST'])
def extract_features_api():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file provided'}), 400

    # Define temporary paths for saving files
    temp_dir = 'temp_uploads'
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.filename)

    try:
        # Save the file temporarily
        file.save(temp_path)

        # Preprocess the image and extract features
        img_array = preprocess_image(temp_path)
        features = extract_features(img_array)

        return jsonify({
            'message': 'Features extracted successfully',
            'features': features.tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    # Run on localhost (127.0.0.1) for local network access
    app.run(host='127.0.0.1', port=5000, debug=True)
