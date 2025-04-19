# BART-CNN Text Summarization

A Flask web application that uses the state-of-the-art BART-large-CNN model to generate high-quality summaries of text content.

## Features

- **Powerful Summarization Model**:
  - Uses BART-large-CNN, specifically fine-tuned for summarization tasks
  - Optimized for news articles and general content
  - Produces coherent and readable summaries

- **Enhanced User Experience**:
  - Clean, responsive interface with dark mode support
  - Loading indicators during processing
  - Copy-to-clipboard functionality
  - Print-friendly results

- **Advanced Functionality**:
  - Text preprocessing and cleaning
  - Model caching for improved performance
  - Detailed statistics (word count, compression ratio, processing time)
  - Error handling and comprehensive logging

- **Developer-Friendly**:
  - RESTful API endpoints for programmatic access
  - Modular code structure for easy maintenance
  - Comprehensive documentation

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv env
   ```
3. Activate the virtual environment:
   - Windows: `env\Scripts\activate`
   - Linux/Mac: `source env/bin/activate`
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the application:
   ```
   python app.py
   ```
2. Open your browser and navigate to `http://127.0.0.1:5000`
3. Enter or paste the text you want to summarize
4. Click "Generate Summary" and wait for the result

## Project Structure

```
├── app.py                 # Main Flask application
├── requirements.txt       # Project dependencies
├── static/                # Static assets
│   ├── styles.css         # CSS styles
│   └── script.js          # JavaScript functionality
├── templates/             # HTML templates
│   ├── index.html         # Main page
│   ├── result.html        # Results page
│   └── error.html         # Error page
└── utils/                 # Utility modules
    ├── text_processor.py  # Text processing utilities
    └── model_manager.py   # Model management utilities
```

## API Usage

### Generate a Summary

```python
import requests
import json

url = "http://127.0.0.1:5000/api/summarize"
headers = {"Content-Type": "application/json"}
data = {
    "text": "Your long text to summarize goes here..."
}

response = requests.post(url, headers=headers, data=json.dumps(data))
result = response.json()

print(f"Summary: {result['summary']}")
print(f"Compression: {result['compression_ratio']}%")
```

### Get Model Information

```python
import requests

url = "http://127.0.0.1:5000/api/model"
response = requests.get(url)
model_info = response.json()

print(f"Model: {model_info['model']}")
```

## Model Information

This application uses the BART-large-CNN model from Facebook AI, which is a sequence-to-sequence model pre-trained on a large corpus of text and fine-tuned specifically for summarization tasks on the CNN/Daily Mail dataset.

BART (Bidirectional and Auto-Regressive Transformers) combines the bidirectional encoder from BERT with the auto-regressive decoder from GPT, making it particularly effective for text generation tasks like summarization.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.