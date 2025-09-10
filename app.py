# app.py
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
import os
import time

# Adjust imports for the new structure
from src.inference import InferenceEngine
from src.config import ExtractionConfig

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Initialize Model ---
print("Loading model for Flask app...")
config = ExtractionConfig.from_json_file('config.json')
engine = InferenceEngine(model_path='./models', config=config)
print("Model loaded successfully.")

# --- Your HTML Template (keep as is) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
... (your full HTML code) ...
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/extract', methods=['POST'])
def extract_api():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        start_time = time.time()
        results = engine.extract_from_image(filepath)
        processing_time = time.time() - start_time
        
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'extracted_fields': results['extracted_fields'],
            'processing_time': round(processing_time, 2)
        })
    except Exception as e:
        # Log the full exception for debugging
        print(f"Error during extraction: {e}")
        return jsonify({'success': False, 'error': 'An internal error occurred during processing.'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)