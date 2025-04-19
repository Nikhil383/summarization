"""
Text processing utilities for the summarization app.
"""
import re
import logging

logger = logging.getLogger(__name__)

def clean_text(text):
    """
    Clean and preprocess text before summarization.
    
    Args:
        text (str): The input text to clean
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove URLs (simple pattern)
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    return text

def truncate_text(text, max_length=1024):
    """
    Truncate text to a maximum length while preserving complete sentences.
    
    Args:
        text (str): The input text to truncate
        max_length (int): Maximum length in characters
        
    Returns:
        str: Truncated text
    """
    if not text or len(text) <= max_length:
        return text
    
    # Try to truncate at sentence boundary
    truncated = text[:max_length]
    last_period = truncated.rfind('.')
    
    if last_period > max_length * 0.7:  # Only truncate at sentence if we're not losing too much
        return truncated[:last_period + 1]
    
    return truncated

def count_words(text):
    """
    Count the number of words in a text.
    
    Args:
        text (str): The input text
        
    Returns:
        int: Number of words
    """
    if not text:
        return 0
    
    # Split by whitespace and count non-empty words
    words = [word for word in text.split() if word.strip()]
    return len(words)

def calculate_compression_ratio(original_text, summary):
    """
    Calculate the compression ratio between original text and summary.
    
    Args:
        original_text (str): The original text
        summary (str): The generated summary
        
    Returns:
        float: Compression ratio (0-1)
    """
    if not original_text or not summary:
        return 0
    
    original_word_count = count_words(original_text)
    summary_word_count = count_words(summary)
    
    if original_word_count == 0:
        return 0
    
    return 1 - (summary_word_count / original_word_count)
