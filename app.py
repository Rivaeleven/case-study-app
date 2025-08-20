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

# Prefer the larger vision-capable model for detail (you can override via env)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# ───────────────────────── APP ─────────────────────────
app = Flask(__name__)

# ───────────────────── OpenAI client ───────────────────
from openai import OpenAI
def get_client():
    return OpenAI()  # uses OPENAI_API_KEY

# ───────────────────── Helpers / Utils ─────────────────
def safe_token(s: str) -> str:
    """Filesystem-safe slug."""
    s = (s or "").strip()
    s = re.sub(r"\s+", "_", s)
    return re.sub(r"[^A-Za-z0-9_\-]", "", s) or "file"

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
    """YouTube oEmbed for title/author (quick, no API key)."""
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

# ───────────── Thumbnails (vision grounding, no download) ──────────
def get_video_thumbnails(youtube_url: str, max_imgs: int = 3) -> List[str]:
    try:
        import yt_dlp
        ydl_opts = {"quiet": True, "skip_download": True, "extract_flat": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
        thumbs = info.get("thumbnails") or []
        # prefer highest resolution, keep top N distinct URLs
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

# ───────────── Vague → Concrete pass (post-rewrite) ────────────────
def needs_more_concrete(detail_html: str) -> bool:
    generic_markers = [
        "people react", "someone", "person", "consumers", "smiling faces",
        "various settings", "general audience", "generic", "celebrating wildly"
    ]
    text = (detail_html or "").lower()
    return any(g in text for g in generic_markers)

def rewrite_more_concrete(detail_html: str) -> str:
    client = get_client()
    system = (
        "You fix vague scene descriptions into concrete, filmic actions with specific subjects, "
        "props, and motion. Keep HTML structure and timecodes identical; rewrite only the visual text."
    )
    user = f"""
Rules:
- Each scene's “What we see” must include a specific subject (e.g., “grandma”, “teen”, “line cook”, “dog”) and a strong action verb.
- Remove generic phrasing like “people react”, “someone smiles”, “various settings”.
- Keep the same number of scenes and the same timecodes; only improve the wording with concrete, filmic description.

HTML to rewrite:
{detail_html}
"""
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.5,
        max_tokens=1800
    )
    html = resp.choices[0].message.content or ""
    return html.replace("```html", "").replace("```", "")

# ───────────── LLM: vision + transcript detailed analysis ──────────
def analyze_transcript(youtube_url: str, transcript_text: str, video_title: str, channel: str) -> str:
    """
    Produces a full HTML breakdown (scenes + annotated script + strategy sections),
    with vivid ‘ad critic’ detail and chaos montage micro-beats, using transcript + 2–3 thumbnails.
    """
    client = get_client()
    system = """You are an award-winning advertising critic and storyboard analyst.

When writing scene-by-scene breakdowns:
• Do NOT just repeat dialogue.
• Vividly describe the VISUAL ACTIONS — sets, props, crowd reactions, comedic chaos, product shots, smash-cuts, spit-takes, pratfalls, drywall dives, pets howling, table flips, objects breaking.
• Expand sparse beats into cinematic detail, especially absurd, escalating “freak-out” montage moments common in Super Bowl spots.
• Use precise, filmic nouns and action verbs (e.g., “grandma spits coffee across the table,” “guy dives head-first through drywall,” “dog howls at blender,” “chili crock pot splashes”).
• Use ad-trade language where helpful: “freak-out montage,” “comic escalation,” “beauty shot,” “button.”
• Keep sentences short and concrete. Ban generic phrases: “people react,” “someone,” “person,” “consumers.”

Output: VALID HTML (no Markdown). Include ONLY these sections:
<h2>Scene-by-Scene Description (Timecoded)</h2>
— 16–22 scenes for ~30s; each item has [start–end], What we see, What we hear, On-screen text (exact supers if known), Purpose (why the beat exists).
<h2>Annotated Script</h2>
— A table with Time, Script (verbatim), Director’s Notes (blocking / camera / performance), Brand/Strategy Note.
<h2>What Makes It Compelling & Unique</h2>
<h2>Creative Strategy Applied</h2>
<h2>Campaign Objectives, Execution & Reach</h2>
<h2>Performance & Audience Impact</h2>
<h2>Why It’s Award-Worthy</h2>
<h2>Core Insight (The 'Big Idea')</h2>

Rules:
• If any visual detail is inferred beyond transcript or thumbnails, append “ [inferred]”.
• “Script (verbatim)” must be literal substrings when possible; if not certain, keep very short + “ [inferred]”.
• Prefer tight, specific language over generalities.
"""
    # Prepare vision inputs
    image_urls = get_video_thumbnails(youtube_url, max_imgs=3)
    vision_parts = []
    for u in image_urls:
        vision_parts.append({"type": "image_url", "image_url": {"url": u}})

    # Build user content with both text and images
    user_content = []
    if vision_parts:
        user_content.extend(vision_parts)
    user_text = f"""Video Title: {video_title}
Channel: {channel}
URL: {youtube_url}

Transcript (canonical; may be sparse on visuals):
{transcript_text}

Please produce the HTML now (no Markdown)."""
    user_content.append({"type": "text", "text": user_text})

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content}
        ],
        temperature=0.8,     # a touch more creative for richer beats
        max_tokens=3500      # allow density
    )
    html = resp.choices[0].message.content or ""
    html = html.replace("```html", "").replace("```", "")

    # Post pass: enforce concrete visuals if needed
    if needs_more_concrete(html):
        html = rewrite_more_concrete(html)

    return html

# ─────────────────── HTML Templates ────────────────────
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
    <p class="muted">Paste a YouTube URL and (optionally) a full transcript. We’ll generate a detailed director-level breakdown and auto-download the PDF.</p>
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
  </style>
</head>
<body>
  <h1>{{ heading }}</h1>
  <div class="meta">
    <strong>Title:</strong> {{ title }} &nbsp;|&nbsp;
    <strong>Channel:</strong> {{ channel }}
  </div>
  <hr/>
  {{ analysis_html | safe }}
  <div style="margin-top:18px"><strong>Reference:</strong> <a href="{{ url }}">{{ url }}</a></div>
</body>
</html>
""")

# ─────────────────── Builder pipeline ──────────────────
def build_pdf_from_analysis(url: str, provided_transcript: Optional[str] = None) -> str:
    """
    Builds a director-level case study PDF from a YouTube URL and (optionally) a pasted transcript.
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
            # keep the pipeline alive even if no captions exist
            segments = [{"start": 0.0, "text": ""}]
        transcript_text = " ".join(s.get("text", "") for s in segments)[:30000]

    # 3) Ask the model for a richly detailed HTML analysis (vision + transcript)
    analysis_html = analyze_transcript(
        youtube_url=url,
        transcript_text=transcript_text,
        video_title=title,
        channel=channel,
    )

    # 4) PDF shell (simple, clean stylesheet)
    html_shell = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{title} — Case Study</title>
  <style>
    body {{ font: 12pt/1.55 -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; color:#111; margin: 28px; }}
    h1 {{ font-size: 20pt; margin: 0 0 8px; }}
    h2 {{ font-size: 14pt; margin: 22px 0 10px; }}
    hr {{ border: none; border-top: 1px solid #e5e7eb; margin: 14px 0; }}
    .meta {{ color:#555; font-size:10pt; margin-bottom: 16px; }}
    table {{ border-collapse: collapse; width:100%; font-size:10.5pt }}
    th, td {{ border:1px solid #e5e7eb; padding:8px; vertical-align:top }}
    ol {{ padding-left: 18px; }}
    a {{ color: #1f2937; text-decoration: none; border-bottom: 1px dotted #9ca3af; }}
    .ref {{ margin-top:18px; font-size:10pt; color:#374151 }}
  </style>
</head>
<body>
  <h1>{title} — Case Study</h1>
  <div class="meta"><strong>Channel:</strong> {channel} &nbsp;|&nbsp; <strong>URL:</strong> <a href="{url}">{url}</a></div>
  <hr/>
  {analysis_html}
  <div class="ref"><strong>Reference:</strong> <a href="{url}">{url}</a></div>
</body>
</html>"""

    # 5) Write PDF to OUT_DIR
    from weasyprint import HTML as WEASY_HTML
    os.makedirs(OUT_DIR, exist_ok=True)
    base_name = safe_token(title) or f"case_study_{safe_token(vid)}"
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
