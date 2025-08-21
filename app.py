import os, re, json
from urllib.parse import urlparse, parse_qs
from typing import Dict, List, Optional

from flask import Flask, request, render_template_string, send_from_directory, abort
from jinja2 import Template
from youtube_transcript_api import YouTubeTranscriptApi
import requests

# ───────────────────────── ENV ─────────────────────────
OUT_DIR = os.getenv("OUT_DIR", "out")
os.makedirs(OUT_DIR, exist_ok=True)

# Models: vision for thumbnails; text for structured writing
OPENAI_MODEL_VISION = os.getenv("OPENAI_MODEL_VISION", "gpt-4o")
OPENAI_MODEL_TEXT   = os.getenv("OPENAI_MODEL_TEXT", "gpt-4o")

# ───────────────────── OpenAI client ───────────────────
from openai import OpenAI
def get_client():
    return OpenAI()  # needs OPENAI_API_KEY

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

def get_video_thumbnails(youtube_url: str, max_imgs: int = 3) -> List[str]:
    """Grab a few high-res thumbnail URLs (no download)."""
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
                urls.append(u); seen.add(u)
            if len(urls) >= max_imgs: break
        return urls
    except Exception:
        return []

# ───────────── Tiny helpers for formatting ─────────────
def compact_transcript(transcript_text: str, max_chars=1800) -> str:
    t = re.sub(r"\s+", " ", transcript_text or "").strip()
    return t[:max_chars]

def contains_generic(text: str) -> bool:
    if not text: return False
    bad = ["abstract graphics", "various settings", "people react", "someone", "person", "consumers", "generic"]
    tl = text.lower()
    return any(b in tl for b in bad)

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
    <p class="muted">Paste a YouTube URL and (optionally) a full transcript + any Director’s Hints. We’ll generate a clean, detailed breakdown and auto-download the PDF.</p>
    <form method="post" action="/generate">
      <label>YouTube URL</label>
      <input name="url" type="text" placeholder="https://www.youtube.com/watch?v=..." required />
      <label style="margin-top:12px">Transcript (optional)</label>
      <textarea name="transcript" placeholder="Paste full transcript here. If empty, we’ll try to fetch captions automatically."></textarea>
      <label style="margin-top:12px">Director’s Hints (optional)</label>
      <textarea name="hints" placeholder="Describe what’s on screen that you know is true (e.g., living-room watch party; VO only; no on-camera spokesperson; montage of NO → YES reactions; final product end card)."></textarea>
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
    p, li { font-size: 11.5pt; }
    ul { margin: 0 0 0 20px; }
    a { color:#1f2937; text-decoration: none; border-bottom: 1px dotted #9ca3af; }
    .ref { margin-top:18px; font-size:10pt; color:#374151 }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
  </style>
</head>
<body>
  <h1 class="mono">{{ heading }}</h1>
  <div style="color:#555;margin-bottom:16px"><strong>Channel:</strong> {{ channel }} &nbsp;|&nbsp; <strong>URL:</strong> <a href="{{ url }}">{{ url }}</a></div>
  <hr/>
  {{ analysis_html | safe }}
  <div class="ref"><strong>Reference:</strong> <a href="{{ url }}">{{ url }}</a></div>
</body>
</html>
""")

# ───────────── LLM calls ─────────────
def call_text(system: str, user, model: str, temperature: float, max_tokens: int = 3200) -> str:
    client = get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return (resp.choices[0].message.content or "").replace("```html","").replace("```","")

def analyze_to_template(youtube_url: str, transcript_text: str, title: str, channel: str, hints: str) -> str:
    """
    Produce a STRICT, minimal template:
    - Big Idea (tight)
    - Scene-by-Scene Breakdown (timecoded, 12–18 beats, ultra concrete)
    - Why It Works (3 bullets)
    """
    # Compact transcript for context; we still keep it canonical for verbatim lines
    transcript_compact = compact_transcript(transcript_text, max_chars=2200)
    # Pull thumbnails
    thumbs = get_video_thumbnails(youtube_url, max_imgs=3)

    # System prompt forces the exact structure you want
    system = """You are an advertising analyst. Your job is to write a concise, concrete breakdown in a fixed template.

Hard rules:
- Use the transcript for verbatim VO/dialogue lines only (short snippets).
- Use Director’s Hints as ground truth for visuals if present; otherwise infer conservatively from transcript and thumbnails.
- No on-camera spokesperson unless clearly supported by hints or thumbnails.
- Ban generic phrases: “abstract graphics”, “people react”, “someone”, “various settings”, “consumers”.
- Use short filmic nouns/verbs (crowd, couch, confetti, cooler, TV, end card, pack shot).
- If a visual is not in the hints or thumbnails, append “ [inferred]”.

Output MUST be VALID HTML with EXACTLY these sections and styles:
<h2 class="mono">Big Idea</h2>
<p>2–5 tight sentences.</p>

<h2 class="mono">Scene-by-Scene Breakdown (Timecoded)</h2>
<ol>
  <li>
    <div class="mono">0:00 — Beat name.</div>
    <div><strong>VO/Dialogue:</strong> "verbatim line..."</div>
    <div><strong>Visual:</strong> concrete, specific action (append “ [inferred]” if not grounded)</div>
  </li>
  ... 12–18 items total ...
</ol>

<h2 class="mono">Why It Works</h2>
<ul>
  <li>Rhythmic tension & release …</li>
  <li>Protecting tradition while innovating …</li>
  <li>Simple, repeatable device …</li>
</ul>
"""

    # User payload with thumbnails + text context
    user_payload = []
    for u in thumbs:
        user_payload.append({"type":"image_url","image_url":{"url":u}})
    user_text = f"""
TITLE: {title}
CHANNEL: {channel}
URL: {youtube_url}

DIRECTOR_HINTS (authoritative if plausible):
{hints or "(none)"}

TRANSCRIPT (canonical; use for verbatim quotes):
{transcript_compact}

Now write the HTML exactly in the requested structure and tone.
"""
    user_payload.append({"type":"text","text":user_text})

    html = call_text(system, user_payload, OPENAI_MODEL_VISION, temperature=0.15, max_tokens=3300)

    # If the model drifted into generic wording, run a scrubber pass
    if contains_generic(html):
        scrub_sys = """You are a scrubber. Replace vague or generic visuals with concrete actions,
without inventing on-camera spokespersons. If not grounded by hints/thumbnails, append “ [inferred]”.
Keep the SAME HTML structure/tags/timecodes."""
        html = call_text(scrub_sys, html, OPENAI_MODEL_TEXT, temperature=0.0, max_tokens=3300)

    return html

# ─────────────────── Builder pipeline ──────────────────
def build_pdf_from_template(url: str, provided_transcript: Optional[str], hints: Optional[str]) -> str:
    vid = video_id_from_url(url)
    meta = fetch_basic_metadata(vid)
    title = meta.get("title", "") or "Untitled Spot"
    channel = meta.get("author", "") or "Unknown Channel"

    # Transcript
    if provided_transcript and provided_transcript.strip():
        transcript_text = provided_transcript.strip()[:30000]
    else:
        segments = fetch_transcript_segments(vid)
        if not segments:
            segments = [{"start": 0.0, "text": ""}]
        transcript_text = " ".join(s.get("text", "") for s in segments)[:30000]

    # Analysis in the strict template
    analysis_html = analyze_to_template(
        youtube_url=url,
        transcript_text=transcript_text,
        title=title,
        channel=channel,
        hints=hints or "",
    )

    # Render PDF
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
        return send_from_directory(OUT_DIR, filename, as_attachment=True)
    except Exception:
        abort(404)

@app.post("/generate")
def generate():
    url = request.form.get("url", "").strip()
    transcript_text = (request.form.get("transcript") or "").strip()
    hints = (request.form.get("hints") or "").strip()
    try:
        pdf_path = build_pdf_from_template(url, provided_transcript=transcript_text or None, hints=hints or None)
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
