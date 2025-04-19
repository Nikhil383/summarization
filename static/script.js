// Main JavaScript for Text Summarization App

document.addEventListener('DOMContentLoaded', function() {
    // Form submission handling
    const form = document.getElementById('summarize-form');
    if (form) {
        form.addEventListener('submit', function() {
            document.getElementById('loading').style.display = 'block';
            document.getElementById('submit-btn').disabled = true;
        });
    }

    // Copy to clipboard functionality
    const copyButtons = document.querySelectorAll('.copy-btn');
    copyButtons.forEach(button => {
        button.addEventListener('click', function() {
            const textToCopy = this.previousElementSibling.textContent;
            navigator.clipboard.writeText(textToCopy).then(() => {
                // Change button text temporarily
                const originalText = this.textContent;
                this.textContent = 'Copied!';
                setTimeout(() => {
                    this.textContent = originalText;
                }, 2000);
            }).catch(err => {
                console.error('Failed to copy text: ', err);
            });
        });
    });

    // Model selection handling
    const modelSelect = document.getElementById('model-select');
    if (modelSelect) {
        modelSelect.addEventListener('change', function() {
            const modelInfo = document.getElementById('model-info');
            if (modelInfo) {
                const selectedModel = this.value;
                
                // Update model info based on selection
                if (selectedModel === 'facebook/bart-large-cnn') {
                    modelInfo.textContent = 'BART-large-CNN: Optimized for news articles and general summarization';
                } else if (selectedModel === 'google/pegasus-xsum') {
                    modelInfo.textContent = 'PEGASUS-XSUM: Excellent for extreme summarization with high compression';
                } else if (selectedModel === 'facebook/bart-large-xsum') {
                    modelInfo.textContent = 'BART-large-XSUM: Fine-tuned for concise, single-sentence summaries';
                }
            }
        });
    }
});
