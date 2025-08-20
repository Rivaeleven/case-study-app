import os, re, json
from urllib.parse import urlparse, parse_qs
from typing import Dict, List, Optional, Tuple

from flask import Flask, request, render_template_string, send_from_directory, abort
from jinja2 import Template
from youtube_transcript_api import YouTubeTranscriptApi
import requests

# ───────────────────────── ENV ─────────────────────────
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OUT_DIR = os.getenv("OUT_DIR", "out")
os.makedirs(OUT_DIR, exist_ok=True)

app = Flask(__name__)

# ───────────────────── OpenAI client ───────────────────
from openai import OpenAI
def get_client():
    return OpenAI()  # uses OPENAI_API_KEY

# --- LLM JSON helper (ensures structured case study output) ---
def llm_json(prompt: str, schema: dict = None, model: str = OPENAI_MODEL) -> dict:
    """
    Calls the OpenAI API and tries to return a structured JSON response.
    Falls back gracefully if parsing fails.
    """
    client = get_client()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an advertising strategist who outputs structured case studies in JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        content = resp.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print("LLM JSON error:", e)
        return {"error": str(e)}

# --- LLM helper that produces richly-detailed HTML analysis ---
def analyze_transcript(transcript_text: str, video_title: str, channel: str, url: str) -> str:
    """
    Produces a full HTML breakdown (scenes + annotated script + strategy sections),
    with vivid ‘ad critic’ detail and chaos montage micro-beats.
    """
    client = get_client()
    system = """You are an award-winning advertising critic and storyboard analyst.

When expanding transcripts into scene-by-scene breakdowns:
• Do NOT just repeat dialogue.
• Vividly describe the VISUAL ACTIONS on screen — sets, props, crowd reactions, comedic chaos, product shots, smash-cuts, spit-takes, pratfalls, drywall dives, objects breaking, pets howling, table flips, etc.
• Expand sparse beats into cinematic description, especially absurd or over-the-top reactions common in Super Bowl spots.
• Use precise, filmic nouns and action verbs (e.g., “grandma spits coffee across the table,” “guy dives head-first through drywall,” “dog howls at blender,” “chili crock pot splashes”).
• Use ad-trade language where helpful (e.g., “freak-out montage,” “comic escalation,” “beauty shot,” “button”).
• Keep sentences short and concrete. Ban generic phrases like “people react,” “someone,” “person,” “consumers.”

Output format: VALID HTML only (no Markdown, no triple backticks). Include these sections:
<h2>Scene-by-Scene Description (Timecoded)</h2>
— 12–20 scenes; each item must have [start–end], What we see, What we hear, On-screen text (if any), Purpose (why the beat exists).
<h2>Annotated Script</h2>
— A table with Time, Script (verbatim), Director’s Notes (blocking / camera / performance), Brand/Strategy Note (communication role).
<h2>What Makes It Compelling & Unique</h2>
<h2>Creative Strategy Applied</h2>
<h2>Campaign Objectives, Execution & Reach</h2>
<h2>Performance & Audience Impact</h2>
<h2>Why It’s Award-Worthy</h2>
<h2>Core Insight (The 'Big Idea')</h2>

Rules:
• If a detail is inferred beyond the transcript, append “ [inferred]” to that sentence.
• Keep “Script (verbatim)” lines as literal substrings from the transcript where possible; if not certain, keep very short + “ [inferred]”.
• Prefer tight, specific language over generalities.
"""
    user = f"""Video Title: {video_title}
Channel: {channel}
URL: {url}

Transcript (canonical, may be sparse on visuals):
{transcript_text}

Please produce the HTML now (no Markdown)."""

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=0.7,
        max_tokens=3000
    )
    html = resp.choices[0].message.content or ""
    # Safety: strip any accidental code fences if model inserted them
    html = html.replace("```html", "").replace("```", "")
    return html

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
def build_pdf_from_analysis(url: str, provided_transcript: Optional[str]) -> str:
    # 1) Extract metadata
    vid = video_id_from_url(url)
    meta = fetch_basic_metadata(vid)
    title = meta.get("title", "") or "Untitled"
    channel = meta.get("author", "") or "Unknown Channel"

    # 2) Transcript (prefer user-provided)
    if provided_transcript and provided_transcript.strip():
        transcript_text = provided_transcript.strip()[:24000]
    else:
        segments = fetch_transcript_segments(vid)
        if not segments:
            segments = [{"start": 0.0, "text": ""}]
        transcript_text = " ".join(s["text"] for s in segments)[:24000]

    # 3) Get detailed HTML analysis from the model
    analysis_html = analyze_transcript(transcript_text, title, channel, url)

    # 4) Render HTML → PDF
    from weasyprint import HTML as WEASY_HTML
    heading = f"{title} — Case Study"
    html = PDF_HTML.render(
        file_title=title,
        heading=heading,
        title=title,
        channel=channel,
        analysis_html=analysis_html,
        url=url,
    )
    safe_name = re.sub(r"[^A-Za-z0-9_\-]+", "_", title).strip("_") or "case_study"
    pdf_path = os.path.join(OUT_DIR, f"{safe_name}.pdf")
    WEASY_HTML(string=html, base_url=".").write_pdf(pdf_path)
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
