from flask import Flask, request, jsonify
import numpy as np
import joblib
import os
from PIL import Image

app = Flask(__name__)

# Define model path
MODEL_PATH = os.getenv('MODEL_PATH', 'model.joblib')

# Initialize model globally
model = None

def init_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print('Model loaded successfully.')
    except Exception as e:
        print(f'Error loading model: {str(e)}')
        raise

# Initialize the model when the app starts
init_model()

def preprocess_image(img_path):
    """Preprocess image according to model requirements."""
    try:
        # Open and resize image
        img = Image.open(img_path).convert('RGB')
        img = img.resize((71, 71))
        
        # Convert to numpy array and normalize
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Flatten the image for scikit-learn model
        img_array = img_array.reshape(1, -1)
        
        return img_array
    except Exception as e:
        raise ValueError(f"Error during preprocessing: {str(e)}")

def extract_features(img_array):
    """Extract features using the model."""
    try:
        # For scikit-learn models, we might use predict_proba or decision_function
        # depending on the model type
        if hasattr(model, 'predict_proba'):
            features = model.predict_proba(img_array)
        elif hasattr(model, 'decision_function'):
            features = model.decision_function(img_array)
        else:
            features = model.predict(img_array)
            
        return features.flatten()[:256]  # Return first 256 features
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
        return jsonify({'error': 'No file selected'}), 400

    # Define temporary paths for saving files
    temp_dir = 'temp_uploads'
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.filename)

    try:
        # Save the file temporarily
        file.save(temp_path)
        
        # Preprocess the image
        img_array = preprocess_image(temp_path)
        
        # Extract features
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
    # Use the port specified in the Dockerfile
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
