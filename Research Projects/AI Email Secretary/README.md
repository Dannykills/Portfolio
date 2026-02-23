# ✉️ AI Email Secretary

This tool uses OpenAI's language models to help you manage your inbox by summarizing yesterday’s emails and drafting thoughtful replies to the most important ones. It’s like a personal assistant for your daily email catch-up — especially useful after a busy day or long weekend.

## 📁 What’s in This Folder

- `AI_Email_Secretary.ipynb` — The main notebook that runs the whole email assistant workflow.
- `Alex_emails_march_04.csv` — A sample inbox dataset with emails from the previous day.
- `config_template.json` — A blank config where you can insert your OpenAI API key and base URL.

## 💡 What It Actually Does

- Loads a CSV of emails received the previous day (you can replace it with your own).
- Extracts high-level summaries of each email using GPT.
- Ranks messages based on urgency and importance.
- Drafts AI-generated responses to the top few emails (based on your chosen limit).
- Offers easy access to both subject line suggestions and full reply drafts.

## 🛠 Requirements

- Python 3.x
- `openai` Python package
- Jupyter Notebook

Install dependencies:
```bash
pip install openai
```

## 🔒 API Config

Use the `config_template.json` file to insert your own OpenAI API credentials. Replace the placeholder string with your actual key (do **not** share the real key in public repos).

```json
{
  "API_KEY": "your-api-key-here",
  "OPENAI_API_BASE": "https://api.openai.com/v1"
}
```

## 🚀 How to Run

1. Open the notebook in Jupyter or Google Colab.
2. Make sure your config_template.json is filled out and accessible.
3. Upload your inbox CSV or use the sample provided.
4. Run each cell — the notebook will process the emails, rank them, and draft responses.
5. Read, copy, or save the suggested replies however you like.
