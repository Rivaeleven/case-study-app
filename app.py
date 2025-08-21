import os, re, json
from urllib.parse import urlparse, parse_qs
from typing import Dict, List, Optional, Tuple

from flask import Flask, request, render_template_string, send_from_directory, abort
from jinja2 import Template
from youtube_transcript_api import YouTubeTranscriptApi
import requests

# ───────────────────────── ENV ─────────────────────────
OUT_DIR = os.getenv("OUT_DIR", "out")
os.makedirs(OUT_DIR, exist_ok=True)

# Use vision-capable model for thumbnails + strong reasoning for grounding passes
OPENAI_MODEL_VISION = os.getenv("OPENAI_MODEL_VISION", "gpt-4o")
OPENAI_MODEL_TEXT   = os.getenv("OPENAI_MODEL_TEXT", "gpt-4o")

# ───────────────────── OpenAI client ───────────────────
from openai import OpenAI
def get_client():
    return OpenAI()  # reads OPENAI_API_KEY

# ───────────────────── Flask app ───────────────────────
app = Flask(__name__)

# ───────────────────── Utilities ───────────────────────
def safe_token(s: str) -> str:
    s = (s or "").strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_\-]", "", s)

def video_id_from_url(url: str) -> str:
    q = urlparse(url)
    if q.hostname in ("youtu.be",):
        return q.path.lstrip("/")
    if q.hostname and "youtube.com" in q.hostname:
        vid = parse_qs(q.query).get("v", [None])[0]
        if vid:
            return vid
    raise ValueError("Could not extract YouTube video id from URL.")

def _format_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(round(seconds % 60))
    return f"{m:02d}:{s:02d}"

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

# Pull a few high-res thumbnails so the model can “see” what’s actually on screen
def get_video_thumbnails(youtube_url: str, max_imgs: int = 3) -> List[str]:
    try:
        import yt_dlp
        ydl_opts = {"quiet": True, "skip_download": True, "extract_flat": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
        thumbs = info.get("thumbnails") or []
        thumbs = sorted(thumbs, key=lambda t: (t.get("height", 0) * t.get("width", 0)), reverse=True)
        urls, seen = [], set()
        for t in thumbs:
            u = t.get("url")
            if u and u not in seen:
                urls.append(u)
                seen.add(u)
            if len(urls) >= max_imgs:
                break
        return urls
    except Exception:
        return []

# ───────────────── HTML shells ─────────────────────────
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
    textarea { min-height: 140px; }
    .btn { background:#111827; color:#fff; border:none; padding: 10px 14px; border-radius: 10px; cursor:pointer; }
    .muted { color:#6b7280; font-size: 13px; }
  </style>
</head>
<body>
  <div class="card">
    <h2>YouTube → Case Study PDF</h2>
    <p class="muted">Paste a YouTube URL and (optionally) a full transcript. We’ll generate a detailed, grounded breakdown and auto-download the PDF.</p>
    <form method="post" action="/generate">
      <label>YouTube URL</label>
      <input name="url" type="text" placeholder="https://www.youtube.com/watch?v=..." required />
      <label style="margin-top:12px">Transcript (optional — paste if you already have it)</label>
      <textarea name="transcript" placeholder="Paste full transcript here. If empty, we’ll try to fetch captions automatically."></textarea>
      <p style="margin-top:14px"><button class="btn" type="submit">Generate PDF</button></p>
    </form>
  </div>
</body>
</html>
"""

SUCCESS_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Generation Successful</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; color:#111; margin: 24px; }
    .card { max-width: 860px; margin: 0 auto; border: 1px solid #e5e7eb; border-radius: 14px; padding: 20px; box-shadow: 0 6px 20px rgba(0,0,0,.05); text-align: center; }
    .btn { background:#111827; color:#fff; border:none; padding: 10px 14px; border-radius: 10px; cursor:pointer; text-decoration: none; }
    .muted { color:#6b7280; font-size: 13px; }
  </style>
</head>
<body>
  <div class="card">
    <h2>Success!</h2>
    <p>Your PDF is ready — it should auto-download. If not, use the button below.</p>
    <p><a href="{{ pdf_url }}" class="btn" download>Download PDF: {{ pdf_filename }}</a></p>
    <p style="margin-top:20px"><a href="/">Generate another video case study</a></p>
  </div>
  <script>
    // auto-download
    (function(){ const a = document.createElement('a'); a.href = "{{ pdf_url }}"; a.download = "{{ pdf_filename }}"; document.body.appendChild(a); a.click(); a.remove(); })();
  </script>
</body>
</html>
"""

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
    a { color:#1f2937; text-decoration: none; border-bottom: 1px dotted #9ca3af; }
    table { border-collapse: collapse; width:100%; font-size:10.5pt }
    th, td { border:1px solid #e5e7eb; padding:8px; vertical-align:top }
    ol { padding-left: 18px; }
    .ref { margin-top:18px; font-size:10pt; color:#374151 }
  </style>
</head>
<body>
  <h1>{{ heading }}</h1>
  <div class="meta"><strong>Channel:</strong> {{ channel }} &nbsp;|&nbsp; <strong>URL:</strong> <a href="{{ url }}">{{ url }}</a></div>
  <hr/>
  {{ analysis_html | safe }}
  <div class="ref"><strong>Reference:</strong> <a href="{{ url }}">{{ url }}</a></div>
</body>
</html>
""")

# ─────────────── LLM passes (EVIDENCE → WRITE → SCRUB) ───────────────
def call_json(system: str, user_content, model: str, temperature: float = 0.2) -> Dict:
    client = get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":system},{"role":"user","content":user_content}],
        response_format={"type":"json_object"},
        temperature=temperature,
    )
    try:
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception:
        return {}

def call_text(system: str, user_content, model: str, temperature: float = 0.3, max_tokens: int = 3000) -> str:
    client = get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":system},{"role":"user","content":user_content}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    txt = resp.choices[0].message.content or ""
    return txt.replace("```html","").replace("```","")

def build_evidence(youtube_url: str, transcript_text: str, title: str, channel: str, image_urls: List[str]) -> Dict:
    sys = (
        "You extract **ground truth evidence** from thumbnails + transcript. "
        "Only report what is visible in thumbnails or explicit in transcript. Be conservative."
    )
    user_payload = []
    for u in image_urls:
        user_payload.append({"type":"image_url","image_url":{"url":u}})
    user_text = f"""
Title: {title}
Channel: {channel}
URL: {youtube_url}

Transcript (canonical, may lack visuals):
{transcript_text}

Return STRICT JSON:
{{
  "people_visible": boolean,
  "voiceover_only_likely": boolean,        // true if transcript is VO without visible speaker
  "settings": [string],                     // e.g., "kitchen", "studio", "abstract graphics"
  "product_shot_types": [string],           // e.g., "pack shot", "beauty pour", "logo end card"
  "on_screen_text": [string],               // exact SUPERS if legible; else empty
  "notable_props": [string],
  "num_people_estimate": integer,           // 0 if none seen
  "confidence": number                      // 0.0–1.0
}}
Rules:
- If thumbnails show only product/graphics, people_visible=false.
- If no face/mouth-on-camera evidence and transcript is disembodied VO, set voiceover_only_likely=true.
- If you are not sure, choose the safer (more restrictive) option.
"""
    user_payload.append({"type":"text","text":user_text})
    return call_json(sys, user_payload, OPENAI_MODEL_VISION, temperature=0.1)

def write_scene_html_from_evidence(youtube_url: str, transcript_text: str, title: str, channel: str, evidence: Dict) -> str:
    sys = """You are an award-winning advertising critic and storyboard analyst.

STRICT RULES:
• Do NOT invent humans, rooms, clothing, or props not supported by EVIDENCE. If evidence.people_visible=false or evidence.voiceover_only_likely=true, **no on-camera spokesperson**.
• Prefer product-only visuals when evidence is sparse.
• Any deduction must end with " [inferred]".
• Use concrete filmic nouns/verbs. Ban generic phrases (“people react”, “someone”, “various settings”).
• Keep sentences short.

OUTPUT: VALID HTML (no Markdown) with EXACTLY these sections:
<h2>Scene-by-Scene Description (Timecoded)</h2>
— 12–18 scenes for ~30s; each item has [start–end], What we see, What we hear, On-screen text (exact supers), Purpose.
<h2>Annotated Script</h2>
— Table: Time | Script (verbatim) | Director’s Notes | Brand/Strategy Note
<h2>What Makes It Compelling & Unique</h2>
<h2>Creative Strategy Applied</h2>
<h2>Campaign Objectives, Execution & Reach</h2>
<h2>Performance & Audience Impact</h2>
<h2>Why It’s Award-Worthy</h2>
<h2>Core Insight (The 'Big Idea')</h2>
"""
    user = f"""
EVIDENCE (ground truth):
{json.dumps(evidence, ensure_ascii=False, indent=2)}

Title: {title}
Channel: {channel}
URL: {youtube_url}

Transcript (canonical; prefer verbatim for Script column):
{transcript_text}

Write the HTML now. Remember: stay within evidence bounds. If people_visible=false, do NOT add humans; lean on product shots and graphics.
"""
    return call_text(sys, user, OPENAI_MODEL_TEXT, temperature=0.25, max_tokens=3500)

def scrub_hallucinations(html: str, evidence: Dict, transcript_text: str) -> str:
    sys = """You are a hallucination scrubber. Remove or rewrite any lines that assert visuals not supported by evidence.
Rules:
- If evidence.people_visible=false, delete any mention of on-camera people, rooms, clothing, faces, or dialogue delivered by visible speakers.
- Only keep on-screen text that is in evidence.on_screen_text.
- Keep product shots, logos, pours, end cards described in evidence.product_shot_types.
- Preserve structure and headings; keep timecodes; tighten wording.
- If uncertain, omit rather than guess.
Output VALID HTML only."""
    user = f"""EVIDENCE:
{json.dumps(evidence, ensure_ascii=False, indent=2)}

TRANSCRIPT:
{transcript_text}

HTML TO SCRUB:
{html}
"""
    return call_text(sys, user, OPENAI_MODEL_TEXT, temperature=0.0, max_tokens=3500)

# Main analysis orchestrator
def analyze_transcript_grounded(youtube_url: str, transcript_text: str, video_title: str, channel: str) -> str:
    # 1) thumbnails
    image_urls = get_video_thumbnails(youtube_url, max_imgs=3)

    # 2) evidence (conservative)
    evidence = build_evidence(youtube_url, transcript_text, video_title, channel, image_urls)

    # 3) write under constraints
    raw_html = write_scene_html_from_evidence(youtube_url, transcript_text, video_title, channel, evidence)

    # 4) scrub for any unsupported claims
    clean_html = scrub_hallucinations(raw_html, evidence, transcript_text)

    return clean_html

# ─────────────────── Builder pipeline ──────────────────
def build_pdf_from_analysis(url: str, provided_transcript: Optional[str] = None) -> str:
    """
    Builds a grounded, director-level case study PDF from a YouTube URL and (optionally) a pasted transcript.
    Returns the absolute path to the generated PDF file.
    """
    # 1) Basic metadata
    vid = video_id_from_url(url)
    meta = fetch_basic_metadata(vid)
    title = meta.get("title", "") or "Untitled Spot"
    channel = meta.get("author", "") or "Unknown Channel"

    # 2) Transcript (prefer user-provided; else fetch captions)
    if provided_transcript and provided_transcript.strip():
        transcript_text = provided_transcript.strip()[:30000]
    else:
        segments = fetch_transcript_segments(vid)
        if not segments:
            segments = [{"start": 0.0, "text": ""}]
        transcript_text = " ".join(s.get("text", "") for s in segments)[:30000]

    # 3) Grounded analysis (vision + transcript with evidence and scrub)
    analysis_html = analyze_transcript_grounded(
        youtube_url=url,
        transcript_text=transcript_text,
        video_title=title,
        channel=channel,
    )

    # 4) Render PDF
    from weasyprint import HTML as WEASY_HTML
    base_name = safe_token(title) or f"case_study_{safe_token(vid)}"
    heading = f"{title} — Case Study"
    html_shell = PDF_HTML.render(
        file_title=base_name,
        heading=heading,
        channel=channel,
        url=url,
        analysis_html=analysis_html,
    )
    pdf_path = os.path.join(OUT_DIR, f"{base_name}.pdf")
    WEASY_HTML(string=html_shell, base_url=".").write_pdf(pdf_path)
    return pdf_path

# ─────────────────────── Routes ────────────────────────
@app.get("/health")
def health():
    return "ok", 200

@app.get("/")
def index():
    return render_template_string(INDEX_HTML)

@app.get("/out/<path:filename>")
def get_file(filename):
    try:
        # as_attachment=True so browsers treat it as a download
        return send_from_directory(OUT_DIR, filename, as_attachment=True)
    except Exception:
        abort(404)

@app.post("/generate")
def generate():
    url = request.form.get("url", "").strip()
    transcript_text = (request.form.get("transcript") or "").strip()
    try:
        pdf_path = build_pdf_from_analysis(url, provided_transcript=transcript_text or None)
        pdf_filename = os.path.basename(pdf_path)
        return render_template_string(
            SUCCESS_HTML,
            pdf_url=f"/out/{pdf_filename}",
            pdf_filename=pdf_filename,
        )
    except Exception as e:
        try:
            with open(os.path.join(OUT_DIR, "last_error.txt"), "w", encoding="utf-8") as f:
                f.write(str(e))
        except Exception:
            pass
        return f"<pre>Error generating case study:\n{e}\nCheck Logs and /out/last_error.txt for details.</pre>", 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=True)
