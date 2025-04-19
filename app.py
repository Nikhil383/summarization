import os
import torch
from flask import Flask, request, render_template, jsonify
import logging
import time

# Import custom modules
from utils.text_processor import clean_text, count_words, calculate_compression_ratio
from utils.model_manager import summarize_text, get_model_info, MODEL_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_url_path='/static')
app.secret_key = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # Limit uploads to 1MB

@app.route('/', methods=['GET'])
def index():
    # Get model info
    model_info = get_model_info()
    return render_template('index.html', model_info=model_info)

@app.route('/summarize', methods=['POST'])
def summarize():
    start_time = time.time()

    # Get form data
    text = request.form.get('text', '')

    # Clean the text
    cleaned_text = clean_text(text)

    # Generate summary
    summary = summarize_text(cleaned_text)

    # Calculate statistics
    original_word_count = count_words(cleaned_text)
    summary_word_count = count_words(summary)
    compression_ratio = calculate_compression_ratio(cleaned_text, summary)
    processing_time = round(time.time() - start_time, 2)

    # Get model info
    model_info = get_model_info()

    # Log the request
    logger.info(f"Summarized text with {original_word_count} words using {model_info['name']} model")

    return render_template(
        'result.html',
        summary=summary,
        original_text=cleaned_text,
        original_word_count=original_word_count,
        summary_word_count=summary_word_count,
        compression_ratio=compression_ratio,
        processing_time=processing_time,
        model_name=model_info['name']
    )

@app.route('/api/summarize', methods=['POST'])
def api_summarize():
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'Invalid JSON'}), 400

        if 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        # Get parameters
        text = data['text']

        # Clean the text
        cleaned_text = clean_text(text)

        # Generate summary
        summary = summarize_text(cleaned_text)

        # Calculate statistics
        original_word_count = count_words(cleaned_text)
        summary_word_count = count_words(summary)
        compression_ratio = calculate_compression_ratio(cleaned_text, summary)

        return jsonify({
            'summary': summary,
            'original_word_count': original_word_count,
            'summary_word_count': summary_word_count,
            'compression_ratio': round(compression_ratio * 100),
            'model': MODEL_CONFIG['name']
        })
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model', methods=['GET'])
def api_model():
    """Return information about the model"""
    return jsonify({
        'model': get_model_info()
    })

@app.errorhandler(413)
def request_entity_too_large(error):
    return render_template('error.html', error="File too large. Maximum size is 1MB."), 413

@app.errorhandler(500)
def server_error(error):
    return render_template('error.html', error="An internal server error occurred."), 500

@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', error="Page not found."), 404

if __name__ == '__main__':
    # Log startup information
    logger.info(f"Starting Text Summarization App with BART-large-CNN model")
    logger.info(f"Running on {'GPU' if torch.cuda.is_available() else 'CPU'}")

    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))