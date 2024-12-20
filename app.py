from flask import Flask, request, jsonify
import numpy as np
import joblib
import os
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Define model path
MODEL_PATH = os.getenv('MODEL_PATH', 'model.joblib')  # Using .joblib file for scikit-learn model
INPUT_SHAPE = (71, 71, 3)

# Global variables for models
model = None

def init_model():
    """Initialize the model"""
    global model
    try:
        # Load the joblib model
        model = joblib.load(MODEL_PATH)
        print('Model loaded successfully.')
    except Exception as e:
        print(f'Error loading model: {str(e)}')
        raise

# Initialize model when app starts
init_model()

def preprocess_image(img_path):
    """Preprocess image according to model requirements."""
    img = image.load_img(img_path, target_size=INPUT_SHAPE[:2])
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def extract_features(img_array):
    """Extract features for scikit-learn models."""
    # If your model is a traditional machine learning model like SVM, RandomForest, etc.,
    # you can extract features directly from the image or through a preprocessing step.
    # This example assumes the model accepts raw pixel data, which might not be true for your case.
    # You may need to perform additional feature extraction or transformation depending on your model.
    
    # Flatten image and extract features
    img_array = img_array.flatten().reshape(1, -1)
    
    # If you're using a model like a RandomForest, for instance, you can get predictions directly.
    features = model.predict(img_array)  # This is a simple example, adapt to your model
    
    return features

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
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
