# Advanced Text Summarization

A Flask web application that uses state-of-the-art NLP models to generate high-quality summaries of text content. This application offers multiple summarization models, each optimized for different types of content.

![Text Summarization App](https://i.imgur.com/example.png)

## Features

- **Multiple Summarization Models**:
  - BART-large-CNN: Optimized for news articles and general content
  - PEGASUS-XSUM: Excellent for extreme summarization with high compression
  - BART-large-XSUM: Fine-tuned for concise, single-sentence summaries

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
3. Select a summarization model based on your needs
4. Enter or paste the text you want to summarize
5. Click "Generate Summary" and wait for the result

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




## Model Information

This application provides multiple summarization models:

1. **BART-large-CNN**: A sequence-to-sequence model from Facebook AI, fine-tuned on CNN/Daily Mail dataset. Best for news articles and general content.

2. **PEGASUS-XSUM**: Google's model specifically trained for extreme summarization. Creates very concise summaries with high compression rates.

3. **BART-large-XSUM**: Facebook's BART model fine-tuned on the XSUM dataset. Optimized for generating single-sentence summaries.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.