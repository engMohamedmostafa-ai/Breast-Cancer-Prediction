from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

# Load the saved models
svm_model = joblib.load('svm_model.pkl')
rf_model = joblib.load('random_forest_model.pkl')

# Create a Flask app
app = Flask(_name_)
CORS(app)  # Enable CORS for all routes

# Load the saved scaler
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        model_type = data.get('model', 'svm').lower()
        raw_features = np.array(data['features']).reshape(1, -1)
        
        # Validate feature count
        if raw_features.shape[1] != 30:
            return jsonify({'error': 'Exactly 30 features are required.'}), 400

        # Standardize the input features using the loaded scaler
        standardized_features = scaler.transform(raw_features)

        # Perform prediction based on the selected model
        if model_type == 'svm':
            prediction = svm_model.predict(standardized_features)
            return jsonify({'model': 'SVM', 'prediction': int(prediction[0])})
        elif model_type == 'random_forest':
            prediction = rf_model.predict(standardized_features)
            return jsonify({'model': 'Random Forest', 'prediction': int(prediction[0])})
        else:
            return jsonify({'error': 'Invalid model type. Use "svm" or "random_forest".'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Handle favicon requests to avoid 404 errors in logs
@app.route('/favicon.ico')
def favicon():
    return '', 204

# Run the app
if _name_ == '_main_':
    app.run(debug=True)