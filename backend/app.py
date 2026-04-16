import flask
from flask import request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np

app = flask.Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

try:
    model = tf.keras.models.load_model('model.h5', compile=False)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Map index to sign label natively matching the loaded ASL model
labels_map = [
    '_', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'bad', 'bye', 
    'C', 'D', 'E', 'F', 'G', 'good', 'H', 'hello', 'help', 'hungry', 'I', 'J', 'K', 
    'L', 'love', 'M', 'N', 'no', 'O', 'P', 'please', 'Q', 'R', 'ready', 'S', 'sorry', 
    'stop', 'T', 'thank_you', 'U', 'V', 'W', 'X', 'Y', 'yes', 'Z'
]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    if not data or 'sequence' not in data:
        return jsonify({'error': 'No sequence data provided'}), 400

    try:
        # sequence is a list of 30 frames, each frame is a list of 1662 keypoints
        input_sequence = data['sequence']
        
        if len(input_sequence) != 30:
            return jsonify({'error': 'Sequence must be exactly 30 frames'}), 400

        prediction_text = ""
        if model is not None:
            # Inference expects (1, 30, 1662)
            input_data = np.expand_dims(input_sequence, axis=0)
            res = model.predict(input_data, verbose=0)[0]
            
            # Ignore the '_' class unconditionally so it always guesses a physical sign
            res[0] = 0.0
            idx = np.argmax(res)
            prediction_text = labels_map[idx] if idx < len(labels_map) else ""
            probability = float(res[idx] * 100) # Convert to percentage
            
        return jsonify({'prediction': prediction_text, 'probability': probability})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Prediction failed'}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
