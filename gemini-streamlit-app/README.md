# ✨ Gemini Chat — Streamlit

A production-ready **Streamlit** chat UI for **Google's Gemini** via the official `google-generativeai` SDK.

## Features
- Chat-style interface with message history
- Model selection: `gemini-1.5-flash`, `gemini-1.5-pro`
- Tunable params: temperature, top-p, top-k, max tokens
- Optional System Prompt to steer behavior
- Image upload for multimodal prompts (vision)
- Streaming responses
- API key via Streamlit secrets or environment variable

---

## Quickstart (Local)

```bash
# 1) Clone or unzip this repo
cd gemini-streamlit-app

# 2) Create a virtual env (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Set your Gemini API key
export GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"   # Windows (Powershell): $Env:GOOGLE_API_KEY="..."

# 5) Run
streamlit run app.py
```

Open the URL printed by Streamlit (usually http://localhost:8501).

### Using Streamlit Secrets
Create `.streamlit/secrets.toml`:

```toml
GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY"
```

---

## Deploy to GitHub (CLI)

With the [GitHub CLI](https://cli.github.com/) installed and authenticated:

```bash
# From the project folder
gh repo create gemini-streamlit-app --public --source=. --remote=origin --push
```

That creates a new repo and pushes the code in one command.

### Manual Git Commands
```bash
git init
git add .
git commit -m "Initial commit: Gemini Streamlit app"
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/gemini-streamlit-app.git
git push -u origin main
```

---

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub.
2. Go to https://share.streamlit.io (or https://streamlit.io/cloud) → **New app**.
3. Pick your repo and `app.py` as the entry point.
4. Set the secret `GOOGLE_API_KEY` under **App Settings → Secrets**.
5. Deploy.

---

## Docker (Optional)

```Dockerfile
# Build
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=8501
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
```

Build & run:
```bash
docker build -t gemini-streamlit-app .
docker run -p 8501:8501 -e GOOGLE_API_KEY=YOUR_GEMINI_API_KEY gemini-streamlit-app
```

---

## Configuration
- **API Key**: `GOOGLE_API_KEY` environment variable or Streamlit secrets.
- **Models**: Choose `gemini-1.5-flash` for speed/cost; `gemini-1.5-pro` for complex reasoning.

---

## License
[MIT](LICENSE)
