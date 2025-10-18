# Codex

Streamlit application for interacting with an OpenAI Agent Builder agent.

## Project structure

```
.
├── app/
│   └── streamlit_app.py
├── requirements.txt
├── runtime.txt
└── README.md
```

## Local development

1. Create and activate a Python 3.11 virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Export the required environment variables:
   ```bash
   export OPENAI_API_KEY="sk-your-key"
   export OPENAI_AGENT_ID="agent_xxx"
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app/streamlit_app.py
   ```

## Deployment to Streamlit Cloud

1. Push this repository to GitHub.
2. Create a new Streamlit Cloud app pointing to `app/streamlit_app.py`.
3. In the Streamlit Cloud app settings, configure the following secrets:
   ```toml
   OPENAI_API_KEY = "sk-your-key"
   OPENAI_AGENT_ID = "agent_xxx"
   ```
4. Deploy the app. Streamlit will automatically install dependencies from
   `requirements.txt` and use the Python version specified in `runtime.txt`.
