from flask import Flask, jsonify, request
import sys
import os

# Add project root to Python path to allow imports from other modules if necessary
# This assumes your main project directory is two levels up from src/api/
# Adjust if your structure is different or if app.py doesn't need to import from outside src/api
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

# Example: If you had feature extraction logic in main.py or src/feature_extraction
# from main import extract_image_features # This would require main.py to be importable
# from src.feature_extraction.extract_features import extract_image_features_from_path

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the AI Card Grading API!"})

@app.route('/grade', methods=['POST'])
def grade_card_api():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    image_file = request.files['image']
    
    # --- Placeholder for your actual image processing and model prediction ---
    # 1. Save the image temporarily (or process in memory if possible)
    # temp_image_path = os.path.join(PROJECT_ROOT, "temp_uploads", image_file.filename)
    # os.makedirs(os.path.join(PROJECT_ROOT, "temp_uploads"), exist_ok=True)
    # image_file.save(temp_image_path)
    
    # 2. Extract features (you'll need to import or define your feature extraction logic)
    # features = extract_image_features_from_path(temp_image_path) # Example call
    
    # 3. Load your trained model (ensure it's accessible)
    # model = load_my_keras_model() # Example call
    
    # 4. Make a prediction
    # prediction = model.predict(features)
    
    # 5. Clean up temporary file
    # os.remove(temp_image_path)
    
    # For now, just returning a success message with the filename
    grade_result = {
        "filename": image_file.filename,
        "grade_prediction": "Awaiting model integration",
        "confidence": 0.0 # Placeholder
    }
    # --- End Placeholder ---
    
    return jsonify(grade_result)

if __name__ == '__main__':
    # Port 5000 is conventional for Flask apps
    # debug=True is useful for development, should be False in production
    app.run(host='0.0.0.0', port=5000, debug=True)