"""
Model management utilities for the summarization app.
"""
import torch
import logging
from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = "facebook/bart-large-cnn"
MODEL_CONFIG = {
    "name": "BART-large-CNN",
    "description": "Optimized for news articles and general summarization",
    "max_length": 142,
    "min_length": 30,
    "num_beams": 4,
    "no_repeat_ngram_size": 3,
    "early_stopping": True
}

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@lru_cache(maxsize=1)
def get_model_and_tokenizer():
    """
    Load and cache the BART-large-CNN model and tokenizer.

    Returns:
        tuple: (tokenizer, model) tuple
    """
    logger.info(f"Loading model {MODEL_NAME} on {DEVICE}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
        return tokenizer, model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def get_model_info():
    """
    Get information about the model.

    Returns:
        dict: Model information
    """
    return {
        "name": MODEL_CONFIG["name"],
        "description": MODEL_CONFIG["description"]
    }

def summarize_text(text):
    """
    Generate a summary for the given text using BART-large-CNN.

    Args:
        text (str): The text to summarize

    Returns:
        str: The generated summary
    """
    if not text or len(text.strip()) == 0:
        return "Please provide text to summarize."

    try:
        # Get model and tokenizer
        tokenizer, model = get_model_and_tokenizer()

        # Preprocess text
        inputs = tokenizer(text, max_length=1024, return_tensors="pt", truncation=True).to(DEVICE)

        # Generate summary
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=MODEL_CONFIG["max_length"],
            min_length=MODEL_CONFIG["min_length"],
            num_beams=MODEL_CONFIG["num_beams"],
            no_repeat_ngram_size=MODEL_CONFIG["no_repeat_ngram_size"],
            early_stopping=MODEL_CONFIG["early_stopping"]
        )

        # Convert summary to text
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary
    except Exception as e:
        logger.error(f"Error during summarization: {str(e)}")
        return f"An error occurred during summarization: {str(e)}"
