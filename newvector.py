from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import re

app = Flask(__name__)

# Load the pre-trained model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def preprocess_text(text):
    """Preprocess text by removing special characters and standardizing format."""
    if text is None:
        return ""
    text = str(text)  # Ensure the input is a string
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

@app.route('/embed_search_query', methods=['POST'])
def embed_search_query():
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "No search query provided"}), 400

    query = data['query']

    # Preprocess the query
    preprocessed_query = preprocess_text(query)

    # Generate embedding
    embedding = model.encode([preprocessed_query])[0].tolist()

    return jsonify({"embedding": embedding, "processed_query": preprocessed_query})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/model_info', methods=['GET'])
def model_info():
    return jsonify({
        "model_name": model.get_sentence_embedding_dimension(),
        "embedding_dimension": model.get_sentence_embedding_dimension()
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)