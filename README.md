# YouTube → Case Study PDF

Paste a YouTube URL, get a case‑study PDF named `Agency-Product-Campaign_Commercial-Director.pdf`.

## Local Run
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-5.1"  # optional
python app.py
# open http://127.0.0.1:5000
```

## Deploy to Render (Docker)
1) Push this folder to GitHub.
2) On Render: **New → Web Service** → connect your repo.
3) Add env var `OPENAI_API_KEY` (required). Optional: `OPENAI_MODEL`, `OUT_DIR=/tmp/out`.
4) Deploy. Render detects Dockerfile, builds, and serves on port 8080.

## Notes
- Uses oEmbed + transcript API; works without YouTube Data API key.
- WeasyPrint renders HTML → PDF. Container includes required libs.
- Storage is ephemeral; PDFs are streamed back on each request.
