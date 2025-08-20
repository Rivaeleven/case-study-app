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

# Prefer the larger vision-capable model for detail
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# ───────────────────── OpenAI client ───────────────────
from openai import OpenAI
def get_client():
    return OpenAI()  # uses OPENAI_API_KEY

app = Flask(__name__)

# ─────────────────────── Helpers ───────────────────────
def safe_token(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", "_", s)
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "", s) or "file"

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

# ───────── Thumbnails (visual evidence for 4o) ─────────
def get_video_thumbnails(youtube_url: str, max_imgs: int = 6) -> List[str]:
    try:
        import yt_dlp
        ydl_opts = {"quiet": True, "skip_download": True, "extract_flat": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
        thumbs = info.get("thumbnails") or []
        thumbs = sorted(thumbs, key=lambda t: (t.get("height",0)*t.get("width",0)), reverse=True)
        urls, seen = [], set()
        for t in thumbs:
            u = t.get("url")
            if u and u not in seen:
                urls.append(u); seen.add(u)
            if len(urls) >= max_imgs: break
        return urls
    except Exception:
        return []

# ───────────── Research enrichment (optional) ──────────
PUBLISHER_WHITELIST = [
    "adage.com","adweek.com","campaignlive.com","campaignlive.co.uk","lbbonline.com",
    "shots.net","shootonline.com","ispot.tv","adforum.com","adsoftheworld.com",
    "businesswire.com","prnewswire.com"
]

def _is_whitelisted(u: str) -> bool:
    try:
        host = urlparse(u).hostname or ""
        return any(d in host for d in PUBLISHER_WHITELIST)
    except Exception:
        return False

def research_enrich(title: str, brand_hint: str = "") -> dict:
    """
    Return {'facts': '...multi-line...', 'citations': [urls...] }
    Works if BING_SEARCH_KEY or SERPAPI_KEY is set. Otherwise returns empty.
    """
    key = os.getenv("BING_SEARCH_KEY","")
    serp = os.getenv("SERPAPI_KEY","")
    if not (key or serp):
        return {"facts": "", "citations": []}

    queries = [
        f"{title} credits", f"{title} director", f"{title} voiceover", f"{title} agency",
    ]
    if brand_hint:
        queries += [f"{brand_hint} Super Bowl ad director", f"{brand_hint} Super Bowl ad voiceover", f"{brand_hint} ad agency"]

    urls = []
    if key:
        for q in queries:
            try:
                r = requests.get(
                    "https://api.bing.microsoft.com/v7.0/search",
                    params={"q": q, "count": 6},
                    headers={"Ocp-Apim-Subscription-Key": key},
                    timeout=10
                )
                for it in (r.json().get("webPages",{}) or {}).get("value",[])[:6]:
                    u = it.get("url","")
                    if u and _is_whitelisted(u) and u not in urls:
                        urls.append(u)
            except Exception:
                pass
    elif serp:
        for q in queries:
            try:
                r = requests.get(
                    "https://serpapi.com/search.json",
                    params={"engine":"google","q": q,"num":10,"api_key": serp},
                    timeout=10
                )
                for it in r.json().get("organic_results",[])[:8]:
                    u = it.get("link","")
                    if u and _is_whitelisted(u) and u not in urls:
                        urls.append(u)
            except Exception:
                pass

    VO_PAT = r"(?:Voice[- ]?over|Narration|Narrator|VO)\s*[:\-]\s*([^\n,;|]+)"
    DIR_PAT = r"(?:Director|Directed by)\s*[:\-]\s*([^\n,;|]+)"
    AGENCY_PAT = r"(?:Agency|Creative Agency)\s*[:\-]\s*([^\n,;|]+)"

    page_texts = []
    for u in urls[:8]:
        try:
            prox = f"https://r.jina.ai/{u}"
            txt = requests.get(prox, timeout=10).text
            if not txt or len(txt) < 300:
                txt = requests.get(u, timeout=10).text
            page_texts.append((u, txt))
        except Exception:
            pass

    def _pick(pat):
        bag = {}
        for u, txt in page_texts:
            for m in re.findall(pat, txt or "", flags=re.IGNORECASE):
                name = m.strip()
                if name:
                    bag.setdefault(name, set()).add(u)
        if not bag: return "Unknown", []
        best = max(bag.items(), key=lambda kv: len(kv[1]))
        return best[0], sorted(best[1])

    vo, vo_src = _pick(VO_PAT)
    dr, dr_src = _pick(DIR_PAT)
    ag, ag_src = _pick(AGENCY_PAT)

    facts_lines = []
    if vo != "Unknown": facts_lines.append(f"- VO/Narrator: {vo}")
    if ag != "Unknown": facts_lines.append(f"- Agency: {ag}")
    if dr != "Unknown": facts_lines.append(f"- Director: {dr}")

    return {"facts": "\n".join(facts_lines), "citations": sorted(set(vo_src + dr_src + ag_src))}

# ───── Anti-generic quality gate (self-critique & fix) ─────
def critique_and_fix(detail_html: str) -> str:
    generic_markers = ["people react","someone","person","consumers","various settings","smiling faces"]
    if not any(g in (detail_html or "").lower() for g in generic_markers):
        return detail_html

    client = get_client()
    system = "You are a ruthless script supervisor who fixes vague blocking into specific, filmic action."
    checklist = """Fix the HTML below without changing structure or timecodes.

REQUIREMENTS:
- Each 'What we see' must name a specific subject (e.g., "office manager", "teen gamer", "golden retriever") and a strong action verb + prop/setting.
- No generic phrases like 'people react', 'someone smiles', 'various settings'.
- Keep existing sections and order. Only rewrite the vague text into concrete action.
- If a detail isn’t in transcript or thumbnails, append ' [inferred]' at end of that sentence.
"""
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role":"system","content":system},{"role":"user","content": checklist + "\n\nHTML:\n" + (detail_html or "")}],
        temperature=0.5,
        max_tokens=1800
    )
    html = resp.choices[0].message.content or ""
    return html.replace("```html","").replace("```","")

# ───── Detailed analysis (vision + transcript + facts) ─────
def analyze_transcript(youtube_url: str, transcript_text: str, video_title: str, channel: str, facts_context: str = "") -> str:
    """
    Produces a full HTML breakdown (scenes + annotated script + strategy sections),
    with vivid ‘ad critic’ detail and chaos montage micro-beats, using transcript + thumbnails + verified facts.
    """
    client = get_client()
    system = "You are an award-winning advertising critic and storyboard analyst."
    rules = f"""
Use the verified facts below exactly if present; never contradict them.
{facts_context.strip()}

When writing scene-by-scene breakdowns:
• Do NOT just repeat dialogue.
• Vividly describe the VISUAL ACTIONS — sets, props, crowd reactions, comedic chaos, product shots, smash-cuts, spit-takes, drywall dives, pets howling, table flips, objects breaking.
• Enumerate 18–22 distinct micro-beats for a ~30s spot (merge micro-beats only if necessary).
• Each 'What we see' must include a concrete subject + strong verb + prop/setting.
• Mark any visual deduction beyond transcript/thumbs with ' [inferred]'.
• 'Script (verbatim)' lines must be literal substrings of transcript when possible; otherwise keep very short + ' [inferred]'.
• Ban generic phrases: "people react", "someone", "person", "consumers", "various settings".
Output: VALID HTML with ONLY these sections:
<h2>Scene-by-Scene Description (Timecoded)</h2>
<h2>Annotated Script</h2>
<h2>What Makes It Compelling & Unique</h2>
<h2>Creative Strategy Applied</h2>
<h2>Campaign Objectives, Execution & Reach</h2>
<h2>Performance & Audience Impact</h2>
<h2>Why It’s Award-Worthy</h2>
<h2>Core Insight (The 'Big Idea')</h2>
"""

    # Vision evidence: thumbnails
    image_urls = get_video_thumbnails(youtube_url, max_imgs=6)

    # Build a multimodal message for the chat.completions API
    # We pass the rules as one text part, then transcript and metadata, then images.
    user_parts = [{"type":"text","text": f"Video Title: {video_title}\nChannel: {channel}\nURL: {youtube_url}\n\nTranscript (canonical):\n{transcript_text}\n"}]
    for u in image_urls:
        user_parts.append({"type":"image_url","image_url":{"url":u}})

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,  # use 4o (vision) for better detail
        messages=[
            {"role":"system","content":system},
            {"role":"user","content":[{"type":"text","text":rules}, *user_parts]}
        ],
        temperature=0.85,
        max_tokens=4800
    )
    html = (resp.choices[0].message.content or "").replace("```html","").replace("```","")
    html = critique_and_fix(html)
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
    (function(){
      const a = document.createElement('a');
      a.href = "{{ pdf_url }}";
      a.download = "{{ pdf_filename }}";
      document.body.appendChild(a); a.click(); a.remove();
    })();
  </script>
</body>
</html>
"""

# (We keep the analysis HTML as the body content itself; simple PDF shell.)
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
            segments = [{"start": 0.0, "text": ""}]  # keep pipeline alive
        transcript_text = " ".join(s.get("text", "") for s in segments)[:30000]

    # 3) Optional: enrich credits/facts from ad trades (requires keys)
    enrich = research_enrich(title, brand_hint="")
    facts_context = ""
    if enrich.get("facts"):
        facts_context = "Verified Spot Credits:\n" + enrich["facts"]
        if enrich.get("citations"):
            facts_context += "\n\nCitations:\n" + "\n".join(enrich["citations"])

    # 4) Ask the model for a richly detailed HTML analysis (vision + transcript + facts)
    analysis_html = analyze_transcript(
        youtube_url=url,
        transcript_text=transcript_text,
        video_title=title,
        channel=channel,
        facts_context=facts_context
    )

    # 5) PDF shell (clean stylesheet)
    facts_block = f'<div class="meta"><pre style="white-space:pre-wrap">{facts_context}</pre></div>' if facts_context else ""
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
  {facts_block}
  <hr/>
  {analysis_html}
  <div class="ref"><strong>Reference:</strong> <a href="{url}">{url}</a></div>
</body>
</html>"""

    # 6) Write PDF to OUT_DIR
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
