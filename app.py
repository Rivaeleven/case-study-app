import os, re, json
from urllib.parse import urlparse, parse_qs
from typing import Optional, Dict

from flask import Flask, request, render_template_string, send_file
from youtube_transcript_api import YouTubeTranscriptApi
from jinja2 import Template
from pydantic import BaseModel, Field
from weasyprint import HTML
import requests

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.1")
OUT_DIR = os.getenv("OUT_DIR", "out")
os.makedirs(OUT_DIR, exist_ok=True)

import openai
openai.api_key = OPENAI_API_KEY

app = Flask(__name__)

def video_id_from_url(url: str) -> str:
    q = urlparse(url)
    if q.hostname in ("youtu.be",):
        return q.path.lstrip("/")
    if q.hostname and "youtube.com" in q.hostname:
        vid = parse_qs(q.query).get("v", [None])[0]
        if vid:
            return vid
    raise ValueError("Could not extract YouTube video id from URL.")

def safe_token(s: str) -> str:
    s = (s or "").strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_\-]", "", s)

def fetch_basic_metadata(video_id: str) -> Dict[str, str]:
    try:
        r = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"},
            timeout=15,
        )
        if r.ok:
            j = r.json()
            return {"title": j.get("title", ""), "author": j.get("author_name", "")}
    except Exception:
        pass
    return {"title": "", "author": ""}

def fetch_transcript(video_id: str) -> str:
    try:
        trs = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            t = trs.find_transcript(["en", "en-US"]).fetch()
            return " ".join(seg["text"] for seg in t if seg["text"].strip())
        except Exception:
            pass
        t = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join(seg["text"] for seg in t if seg["text"].strip())
    except Exception:
        return ""

class Naming(BaseModel):
    agency: str = Field("Unknown")
    product: str = Field("Unknown")
    campaign: str = Field("Unknown")
    commercial: str = Field("Unknown")
    director: str = Field("Unknown")
    def filename(self) -> str:
        return (
            f"{safe_token(self.agency)}-"
            f"{safe_token(self.product)}-"
            f"{safe_token(self.campaign)}_"
            f"{safe_token(self.commercial)}-"
            f"{safe_token(self.director)}.pdf"
        )

ANALYSIS_SYSTEM = (
    "You are a creative ad analyst. Produce a thorough case study of a video campaign.\n"
    "Structure strictly as:\n"
    "1. Content Summary and Structure\n"
    "2. What Makes It Compelling & Unique\n"
    "3. Creative Strategy Applied\n"
    "4. Campaign Objectives, Execution & Reach (use a 2-column table)\n"
    "5. Performance & Audience Impact (include quant data if present; otherwise mark as 'indicative')\n"
    "6. Why It’s Award‑Worthy\n"
    "7. Core Insight (The 'Big Idea')\n"
    "Include one reference link to the provided YouTube URL at the end.\n"
    "Tone: professional, clear, no emojis."
)

NAMING_SYSTEM = (
    "Extract (or reasonably infer) the following fields for naming: Agency, Product, Campaign, Commercial, Director.\n"
    "Return ONLY compact JSON with keys: agency, product, campaign, commercial, director.\n"
    "Be conservative with proper nouns. If uncertain, use 'Unknown'."
)

ANALYSIS_USER_TEMPLATE = Template(
    """YouTube URL: {{ url }}\nVideo Title: {{ title }}\nChannel/Author: {{ author }}\n\nTranscript (may be partial):\n{{ transcript }}"""
)

def llm_json(system: str, user: str) -> Dict:
    resp = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
    )
    txt = resp.choices[0].message["content"]
    try:
        start = txt.find("{")
        end = txt.rfind("}")
        return json.loads(txt[start : end + 1])
    except Exception:
        return {}

def llm_text(system: str, user: str) -> str:
    resp = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.5,
    )
    return resp.choices[0].message["content"].strip()

PDF_HTML = Template(
    """
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>{{ file_title }}</title>
  <style>
    body { font: 12pt/1.55 -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; color:#111; margin: 28px; }
    h1 { font-size: 20pt; margin: 0 0 8px; }
    h2 { font-size: 14pt; margin: 22px 0 10px; }
    hr { border: none; border-top: 1px solid #e5e7eb; margin: 14px 0; }
    .meta { color:#555; font-size:10pt; margin-bottom: 16px; }
    table { border-collapse: collapse; width:100%; font-size:10.5pt }
    th, td { border:1px solid #e5e7eb; padding:8px; vertical-align:top }
    .ref { margin-top:18px; font-size:10pt; color:#374151 }
  </style>
</head>
<body>
  <h1>{{ heading }}</h1>
  <div class=\"meta\">\n    <strong>Agency:</strong> {{ naming.agency }} &nbsp;|&nbsp;
    <strong>Product:</strong> {{ naming.product }} &nbsp;|&nbsp;
    <strong>Campaign:</strong> {{ naming.campaign }} &nbsp;|&nbsp;
    <strong>Commercial:</strong> {{ naming.commercial }} &nbsp;|&nbsp;
    <strong>Director:</strong> {{ naming.director }}
  </div>
  <hr/>
  {{ body_html }}
  <div class=\"ref\"><strong>Reference:</strong> <a href=\"{{ url }}\">{{ url }}</a></div>
</body>
</html>
    """
)

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>YouTube → Case Study PDF</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; color:#111; margin: 24px; }
    .card { max-width: 860px; margin: 0 auto; border: 1px solid #e5e7eb; border-radius: 14px; padding: 20px; box-shadow: 0 6px 20px rgba(0,0,0,.05); }
    label { display:block; margin: 10px 0 6px; font-weight: 600; }
    input[type=text] { width:100%; padding:10px 12px; border:1px solid #d1d5db; border-radius:10px; font-size:15px; }
    .grid { display:grid; grid-template-columns: repeat(auto-fit,minmax(180px,1fr)); gap: 10px; }
    .btn { background:#111827; color:#fff; border:none; padding: 10px 14px; border-radius: 10px; cursor:pointer; }
    .btn:disabled { opacity:.5; cursor:not-allowed; }
    .muted { color:#6b7280; font-size: 13px; }
  </style>
</head>
<body>
  <div class=\"card\">
    <h2>YouTube → Case Study PDF</h2>
    <p class=\"muted\">Paste a YouTube URL. Optionally provide naming overrides. We’ll return a PDF in the format <code>Agency-Product-Campaign_Commercial-Director.pdf</code>.</p>
    <form method=\"post\" action=\"/generate\">\n      <label>YouTube URL</label>
      <input name=\"url\" type=\"text\" placeholder=\"https://www.youtube.com/watch?v=...\" required />
      <div class=\"grid\">\n        <div><label>Agency (optional)</label><input name=\"agency\" type=\"text\" /></div>
        <div><label>Product (optional)</label><input name=\"product\" type=\"text\" /></div>
        <div><label>Campaign (optional)</label><input name=\"campaign\" type=\"text\" /></div>
        <div><label>Commercial (optional)</label><input name=\"commercial\" type=\"text\" /></div>
        <div><label>Director (optional)</label><input name=\"director\" type=\"text\" /></div>
      </div>
      <p style=\"margin-top:14px\"><button class=\"btn\" type=\"submit\">Generate PDF</button></p>
    </form>
  </div>
</body>
</html>
"""

def build_case_study(url: str, overrides: Optional[Dict[str, str]] = None) -> str:
    vid = video_id_from_url(url)
    meta = fetch_basic_metadata(vid)
    transcript = fetch_transcript(vid)

    user_block = ANALYSIS_USER_TEMPLATE.render(
        url=url,
        title=meta.get("title", ""),
        author=meta.get("author", ""),
        transcript=transcript[:12000],
    )

    raw_naming = llm_json(NAMING_SYSTEM, user_block) or {}
    overrides = overrides or {}
    raw_naming.update({k: v for k, v in overrides.items() if v})
    naming = Naming(
        agency=raw_naming.get("agency", "Unknown"),
        product=raw_naming.get("product", "Unknown"),
        campaign=raw_naming.get("campaign", "Unknown"),
        commercial=raw_naming.get("commercial", meta.get("title", "Unknown")),
        director=raw_naming.get("director", "Unknown"),
    )

    body = llm_text(ANALYSIS_SYSTEM, user_block)

    heading = f"{naming.agency} – {naming.product} – {naming.campaign} – {naming.commercial} – {naming.director}"
    html = PDF_HTML.render(
        file_title=naming.filename().replace(".pdf", ""),
        heading=heading,
        naming=naming,
        body_html=body.replace("\n", "<br/>"),
        url=url,
    )
    pdf_path = os.path.join(OUT_DIR, naming.filename())
    HTML(string=html).write_pdf(pdf_path)
    return pdf_path

@app.get("/")
def index():
    return render_template_string(INDEX_HTML)

@app.post("/generate")
def generate():
    url = request.form.get("url", "").strip()
    overrides = {
        "agency": request.form.get("agency") or None,
        "product": request.form.get("product") or None,
        "campaign": request.form.get("campaign") or None,
        "commercial": request.form.get("commercial") or None,
        "director": request.form.get("director") or None,
    }
    try:
        pdf_path = build_case_study(url, overrides=overrides)
        return send_file(pdf_path, as_attachment=True, download_name=os.path.basename(pdf_path))
    except Exception as e:
        return f"<pre>Error: {e}</pre>", 400

if __name__ == "__main__":
    app.run(debug=True)
