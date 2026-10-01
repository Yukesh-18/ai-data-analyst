# 📊 AI Data Analyst Agent

An AI-powered data analysis tool that combines classic machine learning 
(Random Forest churn prediction) with a locally-run LLM (Llama 3.2 via 
Ollama) to generate plain-English insights from any uploaded dataset — 
no data ever leaves your machine.

## Features
- 📁 Upload any CSV and get instant summary statistics
- 🤖 ML model predicts customer churn with ~78% accuracy
- 📈 Automatic feature importance analysis
- 💬 Ask natural-language questions about your data
- 🔒 Runs 100% locally — private, free, no API costs
- 🎨 Clean, interactive web interface (Streamlit)

## Tech Stack
`Python` · `pandas` · `scikit-learn` · `Ollama` · `Llama 3.2` · `Streamlit`

## How it works
1. **EDA** (`eda.py`) — explores and cleans the dataset
2. **Model training** (`train_model.py`) — trains a Random Forest classifier to predict customer churn
3. **AI explanation** (`explain.py`) — uses a local LLM to explain the model's feature importances in plain English
4. **Web app** (`app.py`) — a Streamlit interface where anyone can upload a CSV and ask questions about it

## Setup

```bash
git clone https://github.com/Yukesh-18/ai-data-analyst.git
cd ai-data-analyst
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Install [Ollama](https://ollama.com/download) and pull the model:
```bash
ollama pull llama3.2
```

Run the app:
```bash
python -m streamlit run app.py
```

## Dataset
This project uses the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) 
from Kaggle. Download it and place it as `data.csv` in the project folder.

## Example Results
- Model accuracy: ~78.5%
- Top predictive features: Total Charges, Monthly Charges, Customer Tenure

---
Built as a Data Science + Generative AI portfolio project.
