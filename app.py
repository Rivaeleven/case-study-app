import os, re, json
from urllib.parse import urlparse, parse_qs
from typing import Optional, Dict, List

from flask import Flask, request, render_template_string, send_file
from youtube_transcript_api import YouTubeTranscriptApi
from jinja2 import Template
from pydantic import BaseModel, Field
import requests

# ───────────────────────── ENV ─────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # accurate, available
OUT_DIR = os.getenv("OUT_DIR", "out")
os.makedirs(OUT_DIR, exist_ok=True)

# ───────────────────── OpenAI (lazy) ───────────────────
from openai import OpenAI

def get_client():
    return OpenAI()  # reads OPENAI_API_KEY from env

# ───────────────────────── APP ─────────────────────────
app = Flask(__name__)

# ────────────────────── Utilities ──────────────────────
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

def _format_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(round(seconds % 60))
    return f"{m:02d}:{s:02d}"

def fetch_transcript_segments(video_id: str) -> List[Dict]:
    """Return list of segments with start times and text; empty list if missing."""
    try:
        trs = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            t = trs.find_transcript(["en", "en-US"]).fetch()
        except Exception:
            t = YouTubeTranscriptApi.get_transcript(video_id)
        return [
            {"start": float(seg.get("start", 0.0)), "text": seg.get("text", "")}
            for seg in t
            if seg.get("text", "").strip()
        ]
    except Exception:
        return []

# ────────────────────── Naming model ───────────────────
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

# ─────────────────────── Prompts ───────────────────────
NAMING_SYSTEM = """
Extract (or infer cautiously) the work's naming fields:
- Agency (creative agency behind the commercial)
- Product (brand/product)
- Campaign (campaign/series name if stated; else 'Unknown')
- Commercial (spot title; default to video title if unclear)
- Director (if known in video or description; else 'Unknown')
Return ONLY compact JSON with keys: agency, product, campaign, commercial, director.
Be conservative; if a proper noun is uncertain, use 'Unknown'.
""".strip()

STRUCTURED_JSON_SPEC = """
You will receive a YouTube URL, title, channel, and a timecoded transcript. 
Return ONLY a STRICT JSON object (no markdown) with keys:

{
  "video": {"url": string, "title": string, "channel": string},
  "naming": {"agency": string, "product": string, "campaign": string, "commercial": string, "director": string},
  "scenes": [
    {
      "start": "mm:ss",
      "end": "mm:ss" | null,
      "visual": string,             # WHAT WE SEE (framing/action/mise-en-scène)
      "audio": string,              # WHAT WE HEAR (dialog/VO/SFX/music) – quote verbatim if in transcript
      "on_screen_text": [string],   # exact supers/lower-thirds if present
      "purpose": string             # narrative/strategic purpose of the beat
    }
  ],
  "script": [
    {
      "time": "mm:ss",
      "speaker": string | null,     # e.g., "VO", "Woman", "Crowd"
      "line": string,               # verbatim if in transcript; add ' [inferred]' if not verbatim
      "directors_notes": string,    # shot type (WS/MS/CU/ECU), movement (lock-off, push-in, handheld), pacing, notable color/light/composition
      "strategy_note": string       # how this beat lands message/product/brand
    }
  ],
  "brand": {
    "product": string,
    "distinctive_assets": [string], # pack, color, mnemonic, sonic logo
    "claims": [string],
    "ctas": [string]
  }
}

Rules:
- Anchor timestamps to the provided timecoded transcript; use mm:ss.
- Do NOT invent lines; use transcript verbatim where possible. If you need connective tissue, append ' [inferred]'.
- If the transcript lacks on-screen text/supers, use an empty array.
- Keep strings concise and concrete.
""".strip()

ANALYSIS_FROM_JSON = """
You will receive a JSON describing scenes and script extracted from a YouTube ad, plus the plain transcript text.
Write VALID HTML (no markdown) for the following sections ONLY, drawing facts strictly from the JSON/transcript. 
If you deduce something, mark it [inferred]. Use precise language and avoid generic adjectives.

Sections:
<h2>What Makes It Compelling & Unique</h2>
<h2>Creative Strategy Applied</h2>
<h2>Campaign Objectives, Execution & Reach</h2>  <!-- 2-column HTML table -->
<h2>Performance & Audience Impact</h2>           <!-- note if metrics are not public -->
<h2>Why It’s Award‑Worthy</h2>
<h2>Core Insight (The 'Big Idea')</h2>
""".strip()

ANALYSIS_USER_TEMPLATE = Template(
    """YouTube URL: {{ url }}
Video Title: {{ title }}
Channel/Author: {{ author }}

Timecoded transcript (first lines):
{{ timecoded }}

Plain transcript (may be partial):
{{ transcript }}"""
)

# ───────────────────── LLM Helpers ─────────────────────
def llm_json(system: str, user: str) -> Dict:
    client = get_client()
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
    )
    txt = resp.choices[0].message.content or ""
    try:
        start = txt.find("{"); end = txt.rfind("}")
        return json.loads(txt[start:end+1])
    except Exception:
        return {}

def llm_json_structured(system: str, user: str) -> Dict:
    client = get_client()
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
        )
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception:
        # Fallback: normal mode + best-effort JSON extraction
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
        )
        txt = resp.choices[0].message.content or "{}"
        try:
            start = txt.find("{"); end = txt.rfind("}")
            return json.loads(txt[start:end+1])
        except Exception:
            return {}

def llm_html_from_json(system: str, json_payload: Dict, transcript_text: str) -> str:
    client = get_client()
    user_blob = json.dumps({"json": json_payload, "transcript": transcript_text}, ensure_ascii=False)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_blob},
        ],
        temperature=0.4,
    )
    return (resp.choices[0].message.content or "").strip()

# ─────────────────── HTML Templates ────────────────────
PDF_HTML = Template("""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
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
    ol { padding-left: 18px; }
    .pill { display:inline-block; background:#f3f4f6; padding:2px 8px; border-radius:999px; font-size:10pt; margin-left:6px; }
  </style>
</head>
<body>
  <h1>{{ heading }}</h1>
  <div class="meta">
    <strong>Agency:</strong> {{ naming.agency }} &nbsp;|&nbsp;
    <strong>Product:</strong> {{ naming.product }} &nbsp;|&nbsp;
    <strong>Campaign:</strong> {{ naming.campaign }} &nbsp;|&nbsp;
    <strong>Commercial:</strong> {{ naming.commercial }} &nbsp;|&nbsp;
    <strong>Director:</strong> {{ naming.director }}
  </div>
  <hr/>

  <h2>Scene-by-Scene Description (Timecoded)</h2>
  <ol>
    {% for sc in scenes %}
      <li>
        <strong>[{{ sc.start }}{% if sc.end %}–{{ sc.end }}{% endif %}]</strong>
        <div><em>What we see:</em> {{ sc.visual }}</div>
        <div><em>What we hear:</em> {{ sc.audio }}</div>
        {% if sc.on_screen_text and sc.on_screen_text|length %}
          <div><em>On-screen text:</em> {{ ", ".join(sc.on_screen_text) }}</div>
        {% endif %}
        <div><em>Purpose:</em> {{ sc.purpose }}</div>
      </li>
    {% endfor %}
  </ol>

  <h2>Annotated Script</h2>
  <table>
    <thead>
      <tr><th>Time</th><th>Script (verbatim)</th><th>Director’s Notes</th><th>Brand/Strategy Note</th></tr>
    </thead>
    <tbody>
      {% for line in script %}
      <tr>
        <td>{{ line.time }}</td>
        <td>{{ line.line }}</td>
        <td>{{ line.directors_notes }}</td>
        <td>{{ line.strategy_note }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>

  {{ analysis_html | safe }}

  <div class="ref"><strong>Reference:</strong> <a href="{{ url }}">{{ url }}</a></div>
</body>
</html>
""")

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>YouTube → Case Study PDF</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; color:#111; margin: 24px; }
    .card { max-width: 860px; margin: 0 auto; border: 1px solid #e5e7eb; border-radius: 14px; padding: 20px; box-shadow: 0 6px 20px rgba(0,0,0,.05); }
    label { display:block; margin: 10px 0 6px; font-weight: 600; }
    input[type=text] { width:100%; padding:10px 12px; border:1px solid #d1d5db; border-radius:10px; font-size:15px; }
    .grid { display:grid; grid-template-columns: repeat(auto-fit,minmax(180px,1fr)); gap: 10px; }
    .btn { background:#111827; color:#fff; border:none; padding: 10px 14px; border-radius: 10px; cursor:pointer; }
    .muted { color:#6b7280; font-size: 13px; }
  </style>
</head>
<body>
  <div class="card">
    <h2>YouTube → Case Study PDF</h2>
    <p class="muted">Paste a YouTube URL. Optionally provide naming overrides. We’ll return a PDF and save a JSON sidecar named <code>Agency-Product-Campaign_Commercial-Director.json</code>.</p>
    <form method="post" action="/generate">
      <label>YouTube URL</label>
      <input name="url" type="text" placeholder="https://www.youtube.com/watch?v=..." required />
      <div class="grid">
        <div><label>Agency (optional)</label><input name="agency" type="text" /></div>
        <div><label>Product (optional)</label><input name="product" type="text" /></div>
        <div><label>Campaign (optional)</label><input name="campaign" type="text" /></div>
        <div><label>Commercial (optional)</label><input name="commercial" type="text" /></div>
        <div><label>Director (optional)</label><input name="director" type="text" /></div>
      </div>
      <p style="margin-top:14px"><button class="btn" type="submit">Generate PDF</button></p>
    </form>
  </div>
</body>
</html>
"""

# ─────────────────── Builder pipeline ──────────────────
def build_case_study(url: str, overrides: Optional[Dict[str, str]] = None) -> str:
    vid = video_id_from_url(url)
    meta = fetch_basic_metadata(vid)
    segments = fetch_transcript_segments(vid)
    timecoded = "\n".join(f"{_format_time(s['start'])}  {s['text']}" for s in segments[:180])
    transcript_text = " ".join(s["text"] for s in segments)[:16000]

    user_block = ANALYSIS_USER_TEMPLATE.render(
        url=url, title=meta.get("title",""), author=meta.get("author",""),
        timecoded=timecoded, transcript=transcript_text
    )

    # 1) Naming (cautious) + overrides
    raw_naming = llm_json(NAMING_SYSTEM, user_block) or {}
    overrides = overrides or {}
    raw_naming.update({k:v for k,v in overrides.items() if v})
    naming = Naming(
        agency=raw_naming.get("agency","Unknown"),
        product=raw_naming.get("product","Unknown"),
        campaign=raw_naming.get("campaign","Unknown"),
        commercial=raw_naming.get("commercial", meta.get("title","Unknown")),
        director=raw_naming.get("director","Unknown"),
    )

    # 2) Structured JSON (scenes + script + brand)
    json_payload = llm_json_structured(STRUCTURED_JSON_SPEC, user_block)
    # Ensure required keys exist
    json_payload.setdefault("video", {"url": url, "title": meta.get("title",""), "channel": meta.get("author","")})
    json_payload.setdefault("naming", {
        "agency": naming.agency, "product": naming.product, "campaign": naming.campaign,
        "commercial": naming.commercial, "director": naming.director
    })
    json_payload.setdefault("scenes", [])
    json_payload.setdefault("script", [])
    json_payload.setdefault("brand", {
        "product": naming.product, "distinctive_assets": [], "claims": [], "ctas": []
    })

    # 3) Analysis sections strictly from JSON + transcript
    analysis_html = llm_html_from_json(ANALYSIS_FROM_JSON, json_payload, transcript_text)

    # Save sidecar JSON
    json_path = os.path.join(OUT_DIR, Naming(**json_payload["naming"]).filename().replace(".pdf", ".json"))
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # 4) Render PDF from JSON (lazy WeasyPrint import)
    from weasyprint import HTML as WEASY_HTML

    heading = f"{naming.agency} – {naming.product} – {naming.campaign} – {naming.commercial} – {naming.director}"
    html = PDF_HTML.render(
        file_title=naming.filename().replace(".pdf",""),
        heading=heading,
        naming=naming,
        scenes=json_payload["scenes"],
        script=json_payload["script"],
        analysis_html=analysis_html,
        url=url,
    )
    pdf_path = os.path.join(OUT_DIR, naming.filename())
    WEASY_HTML(string=html).write_pdf(pdf_path)
    return pdf_path

# ─────────────────────── Routes ────────────────────────
@app.get("/health")
def health():
    return "ok", 200

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
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=True)

