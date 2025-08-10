import os, re, json
from urllib.parse import urlparse, parse_qs
from typing import Optional, Dict, List

from flask import Flask, request, render_template_string, send_file
from youtube_transcript_api import YouTubeTranscriptApi
from jinja2 import Template
from pydantic import BaseModel, Field
from weasyprint import HTML
import requests

# ---- OpenAI v1 SDK ----
from openai import OpenAI

# Lazy client to avoid boot-time failures
def get_client():
    return OpenAI()  # reads OPENAI_API_KEY from env

# ---- Env ----
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OUT_DIR = os.getenv("OUT_DIR", "out")
os.makedirs(OUT_DIR, exist_ok=True)

app = Flask(__name__)

# ----------------- Utilities -----------------

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
    """Lightweight metadata via YouTube oEmbed (no API key)."""
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


def _format_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(round(seconds % 60))
    return f"{m:02d}:{s:02d}"


def fetch_transcript_segments(video_id: str) -> List[Dict]:
    """Return list of {'start': float, 'text': str}. Empty list on failure."""
    try:
        trs = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            t = trs.find_transcript(["en", "en-US"]).fetch()
        except Exception:
            t = YouTubeTranscriptApi.get_transcript(video_id)
        return [
            {"start": float(seg.get("start", 0.0)), "text": seg.get("text", "")}
            for seg in t if seg.get("text", "").strip()
        ]
    except Exception:
        return []

# ----------------- Naming model -----------------
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

# ----------------- Prompts -----------------
ANALYSIS_SYSTEM = (
    "You are a creative ad analyst. Produce a rigorous, concrete breakdown of a TVC/online video ad.
"
    "OUTPUT A (for humans): VALID HTML only (no Markdown). Use semantic tags (<h2>, <ol>, <ul>, <table>, <p>).
"
    "Sections (exact titles as <h2>):
"
    "1. Scene-by-Scene Description (Timecoded)
"
    "   - Use an ordered list. For each beat: [mm:ss], WHAT WE SEE (framing/camera/mise-en-scène), WHAT WE HEAR (dialog/VO/SFX/music), any on-screen text/supers, and the narrative purpose.
"
    "2. Annotated Script
"
    "   - Provide an HTML <table> with headers: Time | Script (verbatim) | Director’s Notes | Brand/Strategy Note.
"
    "   - Director’s Notes: shot type (WS/MS/CU/ECU), movement (lock-off, push-in, handheld), pacing; notable color/light/composition.
"
    "   - Brand/Strategy Note: how this beat serves message, product role, or distinctive brand asset.
"
    "3. What Makes It Compelling & Unique
"
    "4. Creative Strategy Applied
"
    "5. Campaign Objectives, Execution & Reach (2-column HTML table)
"
    "6. Performance & Audience Impact (real quant if present; otherwise label as 'indicative')
"
    "7. Why It’s Award‑Worthy
"
    "8. Core Insight (The 'Big Idea')
"
    "Rules: Prefer precise, observable details over adjectives. Use [inferred] when you add connective tissue beyond transcript.
"
    "Also output a single hyperlink at the end back to the YouTube URL.
"
)

ANALYSIS_USER_TEMPLATE = Template(
    """YouTube URL: {{ url }}
Video Title: {{ title }}
Channel/Author: {{ author }}

Timecoded transcript (first lines):
{{ timecoded }}

Plain transcript (may be partial):
{{ transcript }}

Instructions:
- Use the timecoded lines to anchor each beat and script quotes.
- Quote dialogue/VO verbatim where possible; mark any fabricated bridging lines as [inferred].
- If on-screen text/supers appear, capture them verbatim and timecode them."""
)

NAMING_SYSTEM = (
    "Extract (or reasonably infer) the following fields for naming: Agency, Product, Campaign, Commercial, Director.
"
    "Return ONLY compact JSON with keys: agency, product, campaign, commercial, director.
"
    "Be conservative with proper nouns. If uncertain, set 'Unknown'."
)

STRUCTURED_JSON_SPEC = (
    "Produce ONLY a JSON object with these top-level keys:
"
    "{
"
    "  \"video\": {\"url\": string, \"title\": string, \"channel\": string},
"
    "  \"scenes\": [
"
    "    {
"
    "      \"start\": \"mm:ss\",
"
    "      \"end\": \"mm:ss\" (optional),
"
    "      \"visual\": string,
"
    "      \"audio\": string,
"
    "      \"on_screen_text\": [string],
"
    "      \"purpose\": string
"
    "    }
"
    "  ],
"
    "  \"script\": [
"
    "    {
"
    "      \"time\": \"mm:ss\",
"
    "      \"speaker\": string|null,
"
    "      \"line\": string,
"
    "      \"directors_notes\": string,
"
    "      \"strategy_note\": string
"
    "    }
"
    "  ],
"
    "  \"brand\": {
"
    "    \"product\": string,
"
    "    \"distinctive_assets\": [string],
"
    "    \"claims\": [string],
"
    "    \"ctas\": [string]
"
    "  }
"
    "}
"
    "Rules: return STRICT JSON (no trailing commas, no comments, no markdown). Use values from transcript; if you must infer, add ' [inferred]' at the end of the value."
)

# ----------------- LLM helpers -----------------

def llm_json(system: str, user: str) -> Dict:
    client = get_client()
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
    )
    txt = resp.choices[0].message.content or ""
    try:
        start = txt.find("{")
        end = txt.rfind("}")
        return json.loads(txt[start:end+1])
    except Exception:
        return {}


def llm_html(system: str, user: str) -> str:
    client = get_client()
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.5,
    )
    return (resp.choices[0].message.content or "").strip()


def llm_json_structured(system: str, user: str) -> Dict:
    client = get_client()
    # Try strict JSON mode first
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
        )
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception:
        # Fallback: normal completion + best-effort JSON extraction
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
        )
        txt = resp.choices[0].message.content or "{}"
        try:
            start = txt.find("{")
            end = txt.rfind("}")
            return json.loads(txt[start:end+1])
        except Exception:
            return {}

# ----------------- HTML for PDF Shell -----------------
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
    ol { padding-left: 18px; }
    ul { padding-left: 18px; }
    .ref { margin-top:18px; font-size:10pt; color:#374151 }
    a { color:#0f62fe; text-decoration: none; }
  </style>
</head>
<body>
  <h1>{{ heading }}</h1>
  <div class=\"meta\">
    <strong>Agency:</strong> {{ naming.agency }} &nbsp;|&nbsp;
    <strong>Product:</strong> {{ naming.product }} &nbsp;|&nbsp;
    <strong>Campaign:</strong> {{ naming.campaign }} &nbsp;|&nbsp;
    <strong>Commercial:</strong> {{ naming.commercial }} &nbsp;|&nbsp;
    <strong>Director:</strong> {{ naming.director }}
  </div>
  <hr/>
  {{ body_html | safe }}
  <div class=\"ref\"><strong>Reference:</strong> <a href=\"{{ url }}\">{{ url }}</a></div>
</body>
</html>
    """
)

# ----------------- Core build logic -----------------

def build_case_study(url: str, overrides: Optional[Dict[str, str]] = None) -> str:
    vid = video_id_from_url(url)
    meta = fetch_basic_metadata(vid)

    segments = fetch_transcript_segments(vid)
    timecoded = "
".join(
        f"{_format_time(s['start'])}  {s['text']}" for s in segments[:120]
    )
    transcript = " ".join(s["text"] for s in segments)[:12000]

    user_block = ANALYSIS_USER_TEMPLATE.render(
        url=url,
        title=meta.get("title", ""),
        author=meta.get("author", ""),
        timecoded=timecoded,
        transcript=transcript,
    )

    # 1) Naming (LLM + overrides)
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

    # 2a) Human-friendly HTML body for PDF
    body_html = llm_html(ANALYSIS_SYSTEM, user_block)

    # 2b) Machine-readable JSON sidecar for downstream agents
    json_obj = llm_json_structured(STRUCTURED_JSON_SPEC, user_block)
    json_path = os.path.join(OUT_DIR, naming.filename().replace(".pdf", ".json"))
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_obj, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # 3) HTML → PDF
    heading = f"{naming.agency} – {naming.product} – {naming.campaign} – {naming.commercial} – {naming.director}"
    html = PDF_HTML.render(
        file_title=naming.filename().replace(".pdf", ""),
        heading=heading,
        naming=naming,
        body_html=body_html,
        url=url,
    )
    pdf_path = os.path.join(OUT_DIR, naming.filename())
    HTML(string=html).write_pdf(pdf_path)
    return pdf_path

# ----------------- Minimal UI -----------------
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
    <p class=\"muted\">Paste a YouTube URL. Optionally provide naming overrides. You’ll get a PDF with a detailed <strong>scene-by-scene</strong>, an <strong>annotated script</strong>, and a JSON sidecar for training.</p>
    <form method=\"post\" action=\"/generate\">
      <label>YouTube URL</label>
      <input name=\"url\" type=\"text\" placeholder=\"https://www.youtube.com/watch?v=...\" required />
      <div class=\"grid\">
        <div><label>Agency (optional)</label><input name=\"agency\" type=\"text\" /></div>
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

# ----------------- Routes -----------------
@app.get("/")
def index():
    return render_template_string(INDEX_HTML)


@app.get("/health")
def health():
    return "ok", 200


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
