# Semantic Search with Sentence Transformers

A simple semantic search application that converts text documents into 384-dimensional embeddings and retrieves the most semantically relevant documents based on user queries using cosine similarity.

## Requirements

Install the required Python packages:

```bash
pip install numpy sentence-transformers
```

### Dependencies
- **NumPy:** Numerical operations and similarity score calculations.
- **Sentence Transformers:** Loading the embedding model and generating text embeddings.

### Model
`paraphrase-multilingual-MiniLM-L12-v2`

The model is automatically downloaded from Hugging Face on the first execution.