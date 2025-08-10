import os, re, json
from urllib.parse import urlparse, parse_qs
from typing import Optional, Dict, List

from flask import Flask, request, render_template_string, send_file
from youtube_transcript_api import YouTubeTranscriptApi
from jinja2 import Template
from pydantic import BaseModel, Field
import requests

# ───────────────────────── ENV ─────────────────────────
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-5")  # default to GPT‑5 (you confirmed access)
OUT_DIR        = os.getenv("OUT_DIR", "out")
MAX_TOKENS     = int(os.getenv("MAX_TOKENS", "2000"))  # allow longer answers for accuracy
REPAIR_PASSES  = int(os.getenv("REPAIR_PASSES", "2"))  # up to 2 repair attempts
os.makedirs(OUT_DIR, exist_ok=True)

# ───────────────────── OpenAI (lazy) ───────────────────
from openai import OpenAI

def get_client():
    # Lazy init so the app boots even if the key is missing; errors occur at call time instead.
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
You will receive a YouTube URL, title, channel, a timecoded transcript, and optional user 'hints'.
Return ONLY a STRICT JSON object with keys:

{
  "video": {"url": string, "title": string, "channel": string},
  "naming": {"agency": string, "product": string, "campaign": string, "commercial": string, "director": string},
  "scenes": [
    {"start": "mm:ss", "end": "mm:ss" | null, "visual": string, "audio": string, "on_screen_text": [string], "purpose": string}
  ],
  "script": [
    {"time": "mm:ss", "speaker": string | null, "line": string, "directors_notes": string, "strategy_note": string}
  ],
  "brand": {"product": string, "distinctive_assets": [string], "claims": [string], "ctas": [string]}
}

Rules:
- Anchor timestamps to the timecoded transcript.
- 'line' must be a verbatim substring of either the transcript OR the hints; if not possible, keep it short and append ' [inferred]'.
- Prefer 8–14 scenes and 8–14 script rows for a ~30s ad (merge micro-beats if needed).
- Keep SUPERS exact if present; otherwise empty array.
- Be concise and concrete.
""".strip()

ANALYSIS_FROM_JSON = """
You will receive a JSON describing scenes and script extracted from a YouTube ad, plus the plain transcript text.
Write VALID HTML (no markdown) for the following sections ONLY, drawing facts strictly from the JSON/transcript. 
If you deduce something, mark it [inferred]. Use precise language and avoid generic adjectives.

Sections:
<h2>What Makes It Compelling & Unique</h2>
<h2>Creative Strategy Applied</h2>
<h2>Campaign Objectives, Execution & Reach</h2>
<h2>Performance & Audience Impact</h2>
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
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=MAX_TOKENS,
        )
        txt = resp.choices[0].message.content or ""
        start = txt.find("{"); end = txt.rfind("}")
        return json.loads(txt[start:end+1])
    except Exception as e:
        return {"_error": f"llm_json failed: {e}"}

def llm_json_structured(system: str, user: str) -> Dict:
    client = get_client()
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=MAX_TOKENS,
        )
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception:
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=MAX_TOKENS,
            )
            txt = resp.choices[0].message.content or "{}"
            start = txt.find("{"); end = txt.rfind("}")
            return json.loads(txt[start:end+1])
        except Exception as e:
            return {"_error": f"llm_json_structured failed: {e}"}

def llm_html_from_json(system: str, json_payload: Dict, transcript_text: str) -> str:
    client = get_client()
    try:
        user_blob = json.dumps({"json": json_payload, "transcript": transcript_text}, ensure_ascii=False)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user_blob}],
            max_tokens=MAX_TOKENS,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"<p><em>analysis generation failed:</em> {e}</p>"

# ─────────────────── Validation / Repair ───────────────
def _is_substring_loosely(s: str, blob: str) -> bool:
    s_norm = re.sub(r"\s+", " ", (s or "").strip().lower())
    blob_norm = re.sub(r"\s+", " ", (blob or "").strip().lower())
    return s_norm and (s_norm in blob_norm)

def validate_json(payload: Dict, transcript_blob: str, hints_blob: str) -> List[str]:
    errs = []
    scenes = payload.get("scenes") or []
    script = payload.get("script") or []
    if len(scenes) < 6:
        errs.append(f"too_few_scenes:{len(scenes)}")
    if len(script) < 6:
        errs.append(f"too_few_script_lines:{len(script)}")
    for i, row in enumerate(script):
        line = (row or {}).get("line", "")
        if line and not _is_substring_loosely(line, transcript_blob) and not _is_substring_loosely(line, hints_blob):
            errs.append(f"non_verbatim_script_line_at_{i}")
    return errs

def repair_json_with_model(bad_payload: Dict, transcript_blob: str, hints_blob: str) -> Dict:
    client = get_client()
    instruction = {
        "task": "repair",
        "issues": validate_json(bad_payload, transcript_blob, hints_blob),
        "rules": [
            "Ensure 8–14 scenes and 8–14 script lines.",
            "Every 'line' must be a substring of transcript OR hints; otherwise append ' [inferred]' and keep it short.",
            "Preserve timestamps and supers where possible.",
        ],
        "transcript": transcript_blob,
        "hints": hints_blob,
        "json": bad_payload,
    }
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role":"system","content":"You repair JSON to satisfy validation rules using only transcript/hints."},
                {"role":"user","content":json.dumps(instruction, ensure_ascii=False)}
            ],
            max_tokens=MAX_TOKENS,
        )
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception:
        return bad_payload

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
    input[type=text], textarea { width:100%; padding:10px 12px; border:1px solid #d1d5db; border-radius:10px; font-size:15px; }
    textarea { min-height: 110px; }
    .grid { display:grid; grid-template-columns: repeat(auto-fit,minmax(180px,1fr)); gap: 10px; }
    .btn { background:#111827; color:#fff; border:none; padding: 10px 14px; border-radius: 10px; cursor:pointer; }
    .muted { color:#6b7280; font-size: 13px; }
  </style>
</head>
<body>
  <div class="card">
    <h2>YouTube → Case Study PDF</h2>
    <p class="muted">Paste a YouTube URL. Optionally add naming overrides and Hints (credits, key lines, beats, supers). We’ll return a PDF and save a JSON sidecar.</p>
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
      <label style="margin-top:12px">Hints / Credits / Key Lines (optional)</label>
      <textarea name="hints" placeholder=""></textarea>
      <p style="margin-top:14px"><button class="btn" type="submit">Generate PDF</button></p>
    </form>
  </div>
</body>
</html>
"""

# ─────────────────── Builder pipeline ──────────────────
def build_case_study(url: str, overrides: Optional[Dict[str, str]] = None) -> str:
    html = ""  # for debug write if we fail after building HTML
    try:
        vid = video_id_from_url(url)
        meta = fetch_basic_metadata(vid)
        segments = fetch_transcript_segments(vid)
        timecoded = "\n".join(f"{_format_time(s['start'])}  {s['text']}" for s in segments[:220])
        transcript_text = " ".join(s["text"] for s in segments)[:22000]

        # Hints from UI (credits, key lines, beats)
        hints = (overrides or {}).get("_hints", "")

        # Prepare prompt blocks
        user_block = ANALYSIS_USER_TEMPLATE.render(
            url=url, title=meta.get("title",""), author=meta.get("author",""),
            timecoded=timecoded, transcript=transcript_text
        )
        json_user_block = user_block + "\n\nHINTS (optional user-provided facts):\n" + hints

        # 1) Naming (cautious) + overrides
        raw_naming = llm_json(NAMING_SYSTEM, user_block) or {}
        overrides = overrides or {}
        raw_naming.update({k:v for k,v in overrides.items() if v and k != "_hints"})
        naming = Naming(
            agency=raw_naming.get("agency","Unknown"),
            product=raw_naming.get("product","Unknown"),
            campaign=raw_naming.get("campaign","Unknown"),
            commercial=raw_naming.get("commercial", meta.get("title","Unknown")),
            director=raw_naming.get("director","Unknown"),
        )

        # 2) Structured JSON (scenes + script + brand)
        json_payload = llm_json_structured(STRUCTURED_JSON_SPEC, json_user_block)
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

        # 3) Validation + limited repair passes for verbatim accuracy / coverage
        for _ in range(REPAIR_PASSES):
            issues = validate_json(json_payload, transcript_text, hints)
            if not issues:
                break
            json_payload = repair_json_with_model(json_payload, transcript_text, hints)

        # 4) Analysis sections strictly from JSON + transcript
        analysis_html = llm_html_from_json(ANALYSIS_FROM_JSON, json_payload, transcript_text)

        # Save sidecar JSON
        json_path = os.path.join(OUT_DIR, Naming(**json_payload["naming"]).filename().replace(".pdf", ".json"))
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        # 5) Render PDF from JSON (lazy WeasyPrint import)
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

    except Exception as e:
        # Write debug HTML for post-mortem and raise
        try:
            debug_path = os.path.join(OUT_DIR, "error_debug.html")
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(f"<h1>Generation failed</h1><pre>{e}</pre><hr/><h2>HTML at failure</h2>{html}")
        except Exception:
            pass
        raise

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
        "_hints": request.form.get("hints") or "",
    }
    try:
        pdf_path = build_case_study(url, overrides=overrides)
        return send_file(pdf_path, as_attachment=True, download_name=os.path.basename(pdf_path))
    except Exception as e:
        return f"<pre>Internal error while generating PDF:\n{e}\n\nCheck Render logs for stack trace. If it persists, try setting OPENAI_MODEL=gpt-4o or lowering MAX_TOKENS.</pre>", 500

if __name__ == "__main__":
    # Local dev
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=True)

