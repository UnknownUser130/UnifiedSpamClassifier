# Unified Spam Classifier

A Streamlit-based personal project for spam detection across SMS and email data. It combines:

- SMS classification using a Naive Bayes model
- Email classification using a TensorFlow ANN model
- A graph-based spam filter for email metadata
- A login/signup page for personal-project usage

## Project structure

- `interface.py` – main app entry and authentication screen
- `app.py` – core spam classification dashboard
- `Graph_Email.py` – graph-based email spam filter implementation
- `build_graph_filter.py` – rebuilds the graph model artifact
- `data/` – sample text inputs
- `users.db` – SQLite database for local login storage

## Run locally

```bash
streamlit run interface.py
```

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Notes

- This project is designed as a personal portfolio/demo app.
- The default auth database is local SQLite and should be treated as a simple demo setup.
- The app expects model artifact files such as `vectorizer_super.pkl`, `model_bnb_super.pkl`, and `ann_model.keras` to be present in the project root.

## Deployment idea

This app can be deployed to:

- Streamlit Community Cloud
- Render
- Railway
- Hugging Face Spaces

For cloud deployment, use `interface.py` as the app entry file and keep file paths portable.
