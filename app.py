from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Define URL and local path for the Keras model
MODEL_PATH = os.getenv('MODEL_PATH', 'MAIN_MUZZLE.h5')
INPUT_SHAPE = (71, 71, 3)

# Initialize model globally
model = None
penultimate_layer_model = None

def init_model():
    global model, penultimate_layer_model
    try:
        model = load_model(MODEL_PATH)
        print('Model loaded successfully.')
        penultimate_layer_model = tf.keras.models.Model(
            inputs=model.input, 
            outputs=model.layers[-2].output
        )
    except Exception as e:
        print(f'Error loading model: {str(e)}')
        raise

# Initialize model when app starts
init_model()

def preprocess_image(img_path):
    """Preprocess image according to your model's requirements."""
    img = image.load_img(img_path, target_size=INPUT_SHAPE[:2])
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale the image
    return img_array

def extract_features(img_array):
    """Extract features using the penultimate layer of the model."""
    features = penultimate_layer_model.predict(img_array)
    features = features.flatten()  # Flatten to a 1D array
    return features[:256]  # Extract the first 256 features

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

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
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
