"""
Model management utilities for the summarization app.
"""
import os
import torch
import logging
from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)

# Available models with their configurations
AVAILABLE_MODELS = {
    "facebook/bart-large-cnn": {
        "name": "BART-large-CNN",
        "description": "Optimized for news articles and general summarization",
        "max_length": 142,
        "min_length": 30,
        "num_beams": 4,
        "no_repeat_ngram_size": 3,
        "early_stopping": True
    },
    "google/pegasus-xsum": {
        "name": "PEGASUS-XSUM",
        "description": "Excellent for extreme summarization with high compression",
        "max_length": 128,
        "min_length": 20,
        "num_beams": 8,
        "no_repeat_ngram_size": 3,
        "early_stopping": True
    },
    "facebook/bart-large-xsum": {
        "name": "BART-large-XSUM",
        "description": "Fine-tuned for concise, single-sentence summaries",
        "max_length": 62,
        "min_length": 15,
        "num_beams": 6,
        "no_repeat_ngram_size": 3,
        "early_stopping": True
    }
}

# Default model
DEFAULT_MODEL = "facebook/bart-large-cnn"

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@lru_cache(maxsize=2)  # Cache up to 2 models
def get_model_and_tokenizer(model_name=DEFAULT_MODEL):
    """
    Load and cache model and tokenizer.
    
    Args:
        model_name (str): The model identifier
        
    Returns:
        tuple: (tokenizer, model) tuple
    """
    if model_name not in AVAILABLE_MODELS:
        logger.warning(f"Model {model_name} not found in available models. Using default.")
        model_name = DEFAULT_MODEL
    
    logger.info(f"Loading model {model_name} on {DEVICE}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
        return tokenizer, model
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        # Fallback to default model if the requested one fails
        if model_name != DEFAULT_MODEL:
            logger.info(f"Falling back to default model {DEFAULT_MODEL}")
            return get_model_and_tokenizer(DEFAULT_MODEL)
        else:
            raise

def get_model_config(model_name=DEFAULT_MODEL):
    """
    Get the configuration for a specific model.
    
    Args:
        model_name (str): The model identifier
        
    Returns:
        dict: Model configuration
    """
    if model_name not in AVAILABLE_MODELS:
        return AVAILABLE_MODELS[DEFAULT_MODEL]
    
    return AVAILABLE_MODELS[model_name]

def summarize_text(text, model_name=DEFAULT_MODEL):
    """
    Generate a summary for the given text using the specified model.
    
    Args:
        text (str): The text to summarize
        model_name (str): The model to use for summarization
        
    Returns:
        str: The generated summary
    """
    if not text or len(text.strip()) == 0:
        return "Please provide text to summarize."
    
    try:
        # Get model and tokenizer
        tokenizer, model = get_model_and_tokenizer(model_name)
        
        # Get model configuration
        config = get_model_config(model_name)
        
        # Preprocess text
        inputs = tokenizer(text, max_length=1024, return_tensors="pt", truncation=True).to(DEVICE)
        
        # Generate summary
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=config["max_length"],
            min_length=config["min_length"],
            num_beams=config["num_beams"],
            no_repeat_ngram_size=config["no_repeat_ngram_size"],
            early_stopping=config["early_stopping"]
        )
        
        # Convert summary to text
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return summary
    except Exception as e:
        logger.error(f"Error during summarization: {str(e)}")
        return f"An error occurred during summarization: {str(e)}"
