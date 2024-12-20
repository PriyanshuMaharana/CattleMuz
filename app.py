from flask import Flask, request, jsonify
import numpy as np
import joblib
import os

app = Flask(__name__)

# Define model path
MODEL_PATH = os.getenv('MODEL_PATH', 'MAIN_MUZZLE.joblib')  # Your joblib model path

# Global variables for models
model = None

def init_model():
    """Initialize the model"""
    global model
    try:
        # Load the model using joblib
        model = joblib.load(MODEL_PATH)
        print('Model loaded successfully.')
    except Exception as e:
        print(f'Error loading model: {str(e)}')
        raise

# Initialize model when app starts
init_model()

def preprocess_image(img_path):
    """Preprocess image for scikit-learn models."""
    from PIL import Image
    img = Image.open(img_path).convert('RGB')
    img = img.resize((71, 71))  # Resize to the required dimensions
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize the image
    return img_array.flatten().reshape(1, -1)

def extract_features(img_array):
    """Extract features for scikit-learn models."""
    # Use the model to make predictions or extract features
    prediction = model.predict(img_array)  # Modify as needed for your model
    return prediction

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

@app.route('/predict', methods=['POST'])
def predict_api():
    """API to make predictions from the uploaded image."""
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

        # Preprocess the image and make predictions
        img_array = preprocess_image(temp_path)
        prediction = extract_features(img_array)

        return jsonify({
            'message': 'Prediction made successfully',
            'prediction': prediction.tolist()
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
