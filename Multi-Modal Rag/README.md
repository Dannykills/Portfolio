# 🧠 Multi-Modal RAG for Nutrition Intelligence

A multimodal Retrieval-Augmented Generation (RAG) system that answers
nutrition-related questions using both structured medical documentation
and visual food-source data.

This project combines:

-   📄 A 350-page PDF of human vitamin and mineral requirements\
-   🖼 Curated images of common dietary vitamin sources\
-   🔎 Persistent vector database (ChromaDB)\
-   🧬 OpenCLIP multimodal embeddings\
-   🎯 Cross-encoder re-ranking (MS MARCO MiniLM)\
-   💬 Interactive interface (Gradio)

------------------------------------------------------------------------

## 🚀 Project Overview

This project explores how multimodal RAG systems can enhance educational
AI tools.

Instead of relying purely on text retrieval, this system:

1.  Embeds images and text into a shared semantic space\
2.  Stores embeddings in a persistent vector database\
3.  Retrieves relevant content using similarity search\
4.  Re-ranks results using a cross-encoder\
5.  Generates context-aware responses

------------------------------------------------------------------------

## 🏗 Architecture

User Query\
↓\
Vector Search (OpenCLIP embeddings)\
↓\
Top-K Retrieval\
↓\
Cross-Encoder Re-ranking (MS MARCO)\
↓\
Context Assembly\
↓\
Response Generation

------------------------------------------------------------------------

## 📂 Project Structure

Multi-Modal RAG/\
│\
├── sources/ \# Vitamin image sources\
├── Vitamin_and_minerals.pdf \# Nutritional requirements reference\
├── Multi-Modal_RAG.ipynb\
├── requirements.txt\
├── README.md\
└── .gitignore

------------------------------------------------------------------------

## ⚙️ Installation

### 1️⃣ Clone the repository

``` bash
git clone <your-repo-url>
cd Multi-Modal-RAG
```

### 2️⃣ Install dependencies

``` bash
pip install -r requirements.txt
```

### 3️⃣ Run the notebook

``` bash
jupyter notebook
```

Execute `Multi-Modal_RAG.ipynb`. The vector database will be created
automatically on first run.

------------------------------------------------------------------------

## 🧪 Key Components

-   **ChromaDB (PersistentClient)** -- Stores vector embeddings
    locally.\
-   **OpenCLIPEmbeddingFunction** -- Generates image embeddings in a
    shared semantic space.\
-   **HuggingFaceCrossEncoder (ms-marco-MiniLM-L-6-v2)** -- Re-ranks
    retrieved documents.\
-   **Gradio Interface** -- Enables interactive querying.

------------------------------------------------------------------------

## 📈 Why This Matters

Traditional RAG systems operate on text alone. This project
demonstrates:

-   Cross-modal retrieval (text ↔ image)\
-   Persistent local vector storage\
-   Re-ranking for improved relevance\
-   Practical multimodal AI deployment patterns

------------------------------------------------------------------------

## 🔮 Future Improvements

-   Integrate structured nutrient tables for quantitative reasoning\
-   Add evaluation metrics (precision@k, MRR)\
-   Deploy as a hosted web app\
-   Expand into a full dietary planning assistant

------------------------------------------------------------------------

## 👤 Author

Daniel Killorin\
Ocean Engineer \| AI Systems Builder
