from flask import Flask, request, jsonify
import numpy as np
import joblib
import os

app = Flask(__name__)

# Define model path (use joblib instead of h5)
MODEL_PATH = os.getenv('MODEL_PATH', 'model.joblib')

# Initialize model globally
model = None
penultimate_layer_model = None

def init_model():
    global model, penultimate_layer_model
    try:
        # Load the joblib model
        model = joblib.load(MODEL_PATH)
        print('Model loaded successfully.')

        # Assuming your model supports extracting a specific layer output
        # For scikit-learn models, penultimate_layer_model might not apply
        penultimate_layer_model = model
    except Exception as e:
        print(f'Error loading model: {str(e)}')
        raise

# Initialize the model when the app starts
init_model()

def preprocess_image(img_path):
    """Preprocess image for scikit-learn models or other non-TensorFlow models."""
    # Add preprocessing steps as per your model's requirements
    # For image-based scikit-learn models, consider resizing and normalizing the image
    from PIL import Image
    img = Image.open(img_path).resize((71, 71))
    img_array = np.array(img).astype(np.float32) / 255.0  # Normalize the image
    img_array = img_array.flatten()  # Flatten for scikit-learn models
    return img_array.reshape(1, -1)  # Reshape to match input requirements

def extract_features(img_array):
    """Extract features using the model."""
    try:
        # Directly predict if using scikit-learn or equivalent
        features = penultimate_layer_model.predict(img_array)
        return features.flatten()[:256]  # Extract the first 256 features
    except Exception as e:
        raise ValueError(f"Error during feature extraction: {str(e)}")

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

