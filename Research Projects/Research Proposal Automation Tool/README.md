# 🧠 AI NIH Proposal Assistant

This tool uses large language models (LLMs) to help researchers align their past work with new NIH funding opportunities (NOFOs). It reads the NOFO, scans a publication archive, suggests aligned research directions, and drafts proposal sections — like having a smart research co-pilot during grant season.

## 📁 What’s in This Folder

- `Research Proposal Automation Tool.ipynb` — Main Jupyter Notebook for the assistant.
- `config_template.json` — Fill this in with your OpenAI API key and base URL.
- `NOFO.pdf` — A sample Notice of Funding Opportunity (NIMH).
- `Research Proposal Template.pdf` — A guide to NIH formatting and content structure.
- `ResearchPapers/` — Example past publications to simulate archive matching.
- `For Reference - Generated Sample Research Proposals/` — Example generated proposal outputs.
- `README.md` — This file.

## 💡 What It Actually Does

- Loads a NOFO document and a body of past research (PDF or plain text).
- Uses embeddings and semantic search to match past work with NOFO priorities.
- Suggests possible research directions aligned with the funding call.
- Drafts key NIH proposal sections like Significance, Innovation, and Approach.
- Can cite relevant prior work to justify the proposal’s foundation.

## 🛠 Requirements

- Python 3.x  
- Jupyter Notebook or Google Colab

Install dependencies (Colab/Jupyter):

```bash
pip install langchain huggingface-hub openai chromadb langchain-community langchain-openai \
lark rank_bm25 numpy scipy scikit-learn transformers pypdf markdown-pdf tiktoken sentence-transformers
```

If using a GPU, install PyTorch (adjust for your CUDA version if needed):

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

## 🔒 API Config

Use the `config_template.json` file to insert your own OpenAI API credentials. Replace the placeholder string with your actual key.

```json
{
  "API_KEY": "your-api-key-here",
  "OPENAI_API_BASE": "https://api.openai.com/v1"
}
```

## 🚀 How to Run

1. Open the notebook in Jupyter or Google Colab.
2. Fill out your config_template.json file with valid credentials.
3. Upload a NOFO PDF and your past work (or use the samples provided).
4. Run each cell — the assistant will analyze the NOFO, match it to your archive, and generate proposal drafts.
5. Review, refine, and submit high-quality grant text with less stress.