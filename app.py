import os, re, json, base64, subprocess, shutil, tempfile
from urllib.parse import urlparse, parse_qs
from typing import Dict, List, Optional, Tuple

from flask import Flask, request, render_template_string, send_from_directory, abort
from jinja2 import Template
from youtube_transcript_api import YouTubeTranscriptApi
import requests

# ───────────────────────── ENV ─────────────────────────
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # vision-capable model
OUT_DIR = os.getenv("OUT_DIR", "out")
os.makedirs(OUT_DIR, exist_ok=True)

# Optional: search keys (recommended for best results)
BING_SEARCH_KEY      = os.getenv("BING_SEARCH_KEY", "")
BING_SEARCH_ENDPOINT = os.getenv("BING_SEARCH_ENDPOINT", "https://api.bing.microsoft.com/v7.0/search")
SERPAPI_KEY          = os.getenv("SERPAPI_KEY", "")

# ───────────────────────── APP ─────────────────────────
app = Flask(__name__)

# ───────────────────── OpenAI client ───────────────────
from openai import OpenAI
def get_client():
    return OpenAI()  # uses OPENAI_API_KEY

# ────────────────────── Utilities ──────────────────────
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

# ──────────────── Trade-press helpers ─────────────────
PUBLISHER_WHITELIST = [
    "adage.com","adweek.com","campaignlive.com","campaignlive.co.uk",
    "lbbonline.com","shots.net","shootonline.com","thedrum.com","musebycl.io",
    "ispot.tv","adsoftheworld.com","adforum.com","businesswire.com","prnewswire.com"
]

def _host_ok(url: str) -> bool:
    try:
        host = urlparse(url).hostname or ""
        return any(dom in host for dom in PUBLISHER_WHITELIST)
    except Exception:
        return False

def http_get_readable(url: str, timeout=12) -> str:
    # Try Jina Reader proxy first (clean text), then raw
    try:
        r = requests.get(f"https://r.jina.ai/{url}", timeout=timeout)
        if r.ok and len(r.text) > 400:
            return r.text
    except Exception:
        pass
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent":"Mozilla/5.0"})
        if r.ok:
            return r.text
    except Exception:
        pass
    return ""

def web_search(query: str, limit: int = 6) -> List[Dict[str,str]]:
    results = []
    try:
        if BING_SEARCH_KEY:
            r = requests.get(
                BING_SEARCH_ENDPOINT,
                params={"q": query, "count": limit},
                headers={"Ocp-Apim-Subscription-Key": BING_SEARCH_KEY},
                timeout=10
            )
            if r.ok:
                for i in r.json().get("webPages", {}).get("value", []):
                    results.append({"title": i.get("name",""), "url": i.get("url","")})
        elif SERPAPI_KEY:
            r = requests.get(
                "https://serpapi.com/search.json",
                params={"engine":"google","q":query,"num":limit,"api_key":SERPAPI_KEY},
                timeout=10
            )
            if r.ok:
                for i in r.json().get("organic_results", []):
                    results.append({"title": i.get("title",""), "url": i.get("link","")})
    except Exception:
        pass
    # Keep only first hits from trusted domains
    dedup, seen = [], set()
    for it in results:
        u = it.get("url","")
        if not u or u in seen:
            continue
        if _host_ok(u):
            dedup.append(it); seen.add(u)
    return dedup[:limit]

def extract_trade_snippets(title: str) -> Tuple[List[str], List[str]]:
    """
    Pull short supportive quotes (1–2 sentences) from ad trades about the spot.
    Return (snippets, citations).
    """
    if not title:
        return [], []
    queries = [
        f'{title} ad credits', f'{title} director', f'{title} Super Bowl ad',
        f'{title} shootonline', f'{title} adweek', f'{title} adage', f'{title} lbbonline'
    ]
    pages = []
    seen = set()
    for q in queries:
        for res in web_search(q, limit=5):
            u = res.get("url","")
            if not u or u in seen: 
                continue
            if not _host_ok(u):
                continue
            seen.add(u)
            pages.append((u, http_get_readable(u)))

    snippets, cites = [], []
    for url, text in pages:
        if not text:
            continue
        for m in re.finditer(r"([^\n\r]{60,260})", text):
            s = m.group(1).strip()
            if any(k in s.lower() for k in ["super bowl","stunt","window","credits","director","agency","spot","commercial","yes","no"]):
                if len(snippets) < 6:
                    snippets.append(s)
                    cites.append(url)
            if len(snippets) >= 6:
                break
    # dedupe citations
    uniq = []
    for c in cites:
        if c not in uniq:
            uniq.append(c)
    return snippets, uniq[:6]

# ───────────── Keyframe extraction (ffmpeg) ────────────
def extract_keyframes(youtube_url: str, num_frames: int = 6) -> List[str]:
    """
    Download a short chunk of the video and extract ~num_frames JPGs.
    Returns a list of data URLs (base64-encoded) suitable for OpenAI vision input.
    """
    tmpdir = tempfile.mkdtemp(prefix="frames_")
    data_urls: List[str] = []
    try:
        # 1) Download a short mp4 (best available) — fast start
        mp4_path = os.path.join(tmpdir, "clip.mp4")
        ytdlp_cmd = [
            "yt-dlp",
            "-f", "mp4",
            "--no-warnings",
            "--quiet",
            "--no-call-home",
            "--max-filesize", "50M",    # be kind to quotas
            "--output", mp4_path,
            youtube_url
        ]
        subprocess.run(ytdlp_cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not os.path.exists(mp4_path) or os.path.getsize(mp4_path) == 0:
            return []  # fallback will be trade-only + transcript

        # 2) Probe duration
        probe = subprocess.run(
            ["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1", mp4_path],
            capture_output=True, text=True
        )
        try:
            dur = float((probe.stdout or "0").strip())
        except Exception:
            dur = 30.0

        # 3) Sample N frames evenly across first min(dur, 30s)
        window = min(dur, 30.0)
        if window <= 0:
            window = 30.0
        step = window / (num_frames + 1)

        frame_paths = []
        for i in range(1, num_frames+1):
            ts = step * i
            out_jpg = os.path.join(tmpdir, f"frame_{i:02d}.jpg")
            # Extract a single frame at timestamp ts
            ff = [
                "ffmpeg","-ss", f"{ts:.2f}","-i", mp4_path, "-frames:v","1",
                "-q:v","3", out_jpg, "-y"
            ]
            subprocess.run(ff, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(out_jpg) and os.path.getsize(out_jpg) > 0:
                frame_paths.append(out_jpg)

        # 4) Convert frames to data URLs
        for p in frame_paths:
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
                data_urls.append(f"data:image/jpeg;base64,{b64}")

    finally:
        # Remove temp folder
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass
    return data_urls

# ───────── Concrete-language post pass (optional) ───────
def needs_more_concrete(detail_html: str) -> bool:
    generic_markers = [
        "people react", "someone", "person", "consumers",
        "various settings", "smiling faces", "celebrating wildly", "abstract graphics"
    ]
    text = detail_html.lower()
    return any(g in text for g in generic_markers)

def rewrite_more_concrete(detail_html: str) -> str:
    client = get_client()
    system = (
        "You fix vague scene descriptions into concrete, filmic actions with specific subjects, "
        "props, and motion. Keep HTML structure and any time labels identical; rewrite only the visual text."
    )
    user = f"""
Rules:
- Each scene's “Visual” must include a specific subject (e.g., “grandma”, “teen”, “line cook”, “dog”) and a strong action verb.
- Remove generic phrasing like “people react”, “someone smiles”, “various settings”.
- Keep the same number of scenes and the same time labels; only improve the wording with concrete, filmic description.

HTML to rewrite:
{detail_html}
"""
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.3,
        max_tokens=1800
    )
    html = resp.choices[0].message.content or ""
    return html.replace("```html","").replace("```","")

# ───────── LLM: build the HTML analysis (vision+trades) ─────────
def analyze_transcript(youtube_url: str, transcript_text: str, video_title: str, channel: str) -> str:
    """
    Produces a tight HTML breakdown with:
    • Big Idea
    • Beat-by-beat actions (what we actually see) — ACTION LEDGER fused into readable bullets
    • Why It Works
    Evidence-only: uses transcript + sampled keyframes + trade snippets. No invention.
    """
    client = get_client()

    # 1) Evidence: frames (vision) + trades (text) + transcript
    frame_urls = extract_keyframes(youtube_url, num_frames=6)  # data URLs
    trade_snips, trade_cites = extract_trade_snippets(video_title)

    # 2) Evidence clauses
    evidence_clause = (
        "You are EVIDENCE-LOCKED. Only describe what is visible in the sampled frames and what is literally said in the transcript. "
        "If a detail is from trade-press snippets, cite it. If you are not sure, write “UNKNOWN” or add “ [inferred]” conservatively.\n"
    )

    # 3) System prompt (Accuracy Playbook condensed)
    system = (
        "You are a fact-locked ad analyst. Extract ONLY what is supported by evidence.\n"
        "RULES:\n"
        "- Do not guess. Unknown → write “UNKNOWN”.\n"
        "- Separate VISUAL actions from VO lines and press facts.\n"
        "- Prefer on-screen evidence. Never invent props, clothing, or identities.\n"
        "- Timecode dialogue with transcript; keep visuals tied to frames, not fantasies.\n"
        + evidence_clause
        + "Style: short, concrete, filmic nouns & verbs. Ban generic phrases like “people react”.\n"
    )

    # 4) Build the user message (multimodal)
    content: List[Dict] = []

    # Frames first (if any)
    for u in frame_urls:
        content.append({"type":"image_url","image_url":{"url":u}})

    # Then the textual block
    trades_block = ""
    if trade_snips:
        trades_block = "Trade snippets (for facts only; do not invent visuals from these):\n" + "\n".join(f"• {s}" for s in trade_snips)
        if trade_cites:
            trades_block += "\nCitations:\n" + "\n".join(f"- {u}" for u in trade_cites)

    user_text = f"""Video Title: {video_title}
Channel: {channel}
URL: {youtube_url}

{trades_block if trades_block else ""}

Transcript (canonical):
{transcript_text}

OUTPUT EXACTLY THIS HTML SHAPE (no Markdown):

<h2>Big Idea</h2>
<p>2–4 tight sentences.</p>

<h2>Beat-by-beat actions (what we actually see)</h2>
<ul>
  <!-- 8–16 items, each should be an action or VO line backed by evidence. -->
  <!-- For visuals describe concrete actions; for VO use exact lines. Add “ [inferred]” when unsure. -->
</ul>

<h2>Why It Works</h2>
<ul>
  <li>Point 1</li>
  <li>Point 2</li>
  <li>Point 3</li>
</ul>

<h3>Sources</h3>
<ul>
  <!-- Trade URLs if used (deduped). -->
</ul>
"""
    content.append({"type":"text","text":user_text})

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role":"system","content":system},{"role":"user","content":content}],
        temperature=0.2,      # accuracy > flourish
        top_p=0.9,
        max_tokens=2500
    )
    html = (resp.choices[0].message.content or "").replace("```html","").replace("```","")

    # 5) Post-pass to toughen language if still vague
    if needs_more_concrete(html):
        html = rewrite_more_concrete(html)

    # 6) Ensure sources section shows the trade URLs if the model forgot
    if trade_cites and ("<h3>Sources" not in html and "<h2>Sources" not in html):
        links = "".join(f'<li><a href="{u}">{u}</a></li>' for u in trade_cites)
        html += f'\n<h3>Sources</h3>\n<ul>{links}</ul>\n'

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
    <p class="muted">Paste a YouTube URL and (optionally) a full transcript. We’ll sample frames + pull trade snippets for an evidence-true breakdown, then auto-download the PDF.</p>
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
    ol { padding-left: 18px; }
    table { border-collapse: collapse; width:100%; font-size:10.5pt }
    th, td { border:1px solid #e5e7eb; padding:8px; vertical-align:top }
  </style>
</head>
<body>
  <h1>{{ title }} — Case Study</h1>
  <div class="meta">
    <strong>Channel:</strong> {{ channel }} &nbsp;|&nbsp;
    <strong>URL:</strong> <a href="{{ url }}">{{ url }}</a>
  </div>
  <hr/>
  {{ analysis_html | safe }}
  <div style="margin-top:18px"><strong>Reference:</strong> <a href="{{ url }}">{{ url }}</a></div>
</body>
</html>
""")

# ─────────────────── Builder pipeline ──────────────────
def build_pdf_from_analysis(url: str, provided_transcript: Optional[str] = None) -> str:
    # 1) Basic metadata
    vid = video_id_from_url(url)
    meta = fetch_basic_metadata(vid)
    title = meta.get("title", "") or "Untitled Spot"
    channel = meta.get("author", "") or "Unknown Channel"

    # 2) Transcript (prefer user-provided; else fetch captions)
    if provided_transcript and provided_transcript.strip():
        transcript_text = provided_transcript.strip()[:30000]
    else:
        segs = fetch_transcript_segments(vid)
        if not segs:
            segs = [{"start": 0.0, "text": ""}]
        transcript_text = " ".join(s.get("text", "") for s in segs)[:30000]

    # 3) Ask the model for an evidence-true HTML analysis (vision + trades)
    analysis_html = analyze_transcript(
        youtube_url=url,
        transcript_text=transcript_text,
        video_title=title,
        channel=channel,
    )

    # 4) Render PDF
    from weasyprint import HTML as WEASY_HTML
    os.makedirs(OUT_DIR, exist_ok=True)
    base_name = safe_token(title) or f"case_study_{safe_token(vid)}"
    pdf_path = os.path.join(OUT_DIR, f"{base_name}.pdf")
    html = PDF_HTML.render(file_title=base_name, title=title, channel=channel, url=url, analysis_html=analysis_html)
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
