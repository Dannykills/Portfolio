
# 🔍 Semantic Product Search with BERT-Based Embeddings

This project implements an embedding-based semantic retrieval system for e-commerce product search. Instead of relying on keyword matching, the system encodes product descriptions and user queries into a shared vector space and ranks products based on semantic similarity. The focus is on tokenization, encoding consistency, and system-level design decisions that enable intent-aware and scalable retrieval.

## 📁 What’s in This Folder

- `semantic_search.ipynb` — Main notebook implementing tokenization, embedding generation, and retrieval.
- `data/` — Product descriptions and synthetic user queries.
- `src/` — Helper functions for tokenization, encoding, and similarity search.
- `requirements.txt` — Python dependencies for the project.

## 💡 What It Actually Does

- Tokenizes product descriptions and user queries using BERT-compatible WordPiece tokenization.
- Preserves special tokens (`[CLS]`, `[SEP]`) to maintain alignment with pretraining conventions.
- Splits long inputs into fixed-size chunks to respect model sequence-length limits.
- Encodes each chunk with a pretrained BERT-based model and extracts `[CLS]` embeddings.
- Aggregates chunk-level embeddings into a single representation per product.
- Precomputes and stores product embeddings for efficient retrieval.
- Ranks products in real time using cosine similarity between query and product embeddings.

## 🧠 Why Tokenization Matters Here

Tokenization is treated as shared infrastructure rather than preprocessing. Consistent handling of subword units, special tokens, and sequence boundaries ensures alignment between offline embedding generation and online query encoding. Chunking and aggregation avoid silent truncation while producing stable representations for semantic similarity search.

## 🛠 Requirements

- Python 3.x
- PyTorch
- Hugging Face Transformers
- Sentence Transformers
- NumPy, Pandas
- scikit-learn
- tqdm

Install dependencies:
```bash
pip install -r requirements.txt
```


## 🚀 How to Run

- Open the notebook in Jupyter or VS Code.
- Ensure dependencies are installed.
- Load or replace the sample product dataset
- Run the notebook to generate embeddings.
- Enter a query to retrieve semantically relevant products.

## 📝 Notes

The dataset includes synthetic user queries generated with ChatGPT to simulate realistic search behavior. The retrieval pipeline is dataset-agnostic and can be applied to any product catalog with comparable textual fields.

This project emphasizes how encoding and aggregation decisions propagate into downstream retrieval behavior, rather than model training or hyperparameter tuning.

## 🔮 Future Work

- Alternative aggregation strategies (attention-weighted pooling)
- Approximate nearest neighbor search (FAISS)
- Multilingual tokenization support
- Token-level diagnostics and observability tools
