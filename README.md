# YouTube → Case Study PDF

Paste a **YouTube URL** (and optionally a **Transcript**) → get a polished **PDF**:
`Agency-Product-Campaign_Commercial-Director.pdf`

## Local Run

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."             # required
export OPENAI_MODEL="gpt-4o"               # optional, defaults to gpt-4o
export OUT_DIR="/tmp/out"                  # optional
python app.py                              # serves on http://127.0.0.1:8080
