from transformers import T5Tokenizer, T5ForConditionalGeneration
from flask import Flask, request, render_template, jsonify

# Load pre-trained model tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')

def summarize_text(text):
    # Preprocess text
    input_text = "summarize: " + text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Generate summary
    output = model.generate(input_ids, max_length=50, min_length=30, num_beams=4, no_repeat_ngram_size=2)

    # Convert summary to text
    summary = tokenizer.decode(output[0], skip_special_tokens=True)

    return summary

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.form['text']
    summary = summarize_text(text)
    return render_template('result.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)