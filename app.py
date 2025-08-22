import os, re, json, html
from urllib.parse import urlparse, parse_qs
from typing import Dict, List, Optional, Tuple

from flask import Flask, request, render_template_string, send_from_directory, abort
from jinja2 import Template
from youtube_transcript_api import YouTubeTranscriptApi
import requests

# ───────────────────────── ENV ─────────────────────────
# Vision-capable model for any image/thumbnails usage; JSON model for structured outputs
OPENAI_MODEL       = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_JSON_MODEL  = os.getenv("OPENAI_JSON_MODEL", "gpt-4o-mini")
OUT_DIR            = os.getenv("OUT_DIR", "out")
STRICT_VISUALS     = os.getenv("STRICT_VISUALS", "0") == "1"  # if true, only show actions with confidence==3

# Optional: search keys (recommended for best accuracy from trades)
BING_SEARCH_KEY      = os.getenv("BING_SEARCH_KEY", "")
BING_SEARCH_ENDPOINT = os.getenv("BING_SEARCH_ENDPOINT", "https://api.bing.microsoft.com/v7.0/search")
SERPAPI_KEY          = os.getenv("SERPAPI_KEY", "")  # alternative to Bing

os.makedirs(OUT_DIR, exist_ok=True)
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

def _format_time_hhmmss_mmm(seconds: float) -> str:
    # to HH:MM:SS.mmm
    ss = max(0.0, float(seconds))
    h = int(ss // 3600); rem = ss - h*3600
    m = int(rem // 60); s = rem - m*60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

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

# ───────────── Thumbnails for vision context (optional) ───────────
def get_video_thumbnails(youtube_url: str, max_imgs: int = 3) -> List[str]:
    """Try yt-dlp thumbnails, else fall back to standard YouTube thumbnail URLs."""
    urls: List[str] = []
    try:
        import yt_dlp
        ydl_opts = {"quiet": True, "skip_download": True, "extract_flat": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
        thumbs = info.get("thumbnails") or []
        thumbs = sorted(thumbs, key=lambda t: (t.get("height", 0) * t.get("width", 0)), reverse=True)
        seen = set()
        for t in thumbs:
            u = t.get("url")
            if u and u not in seen:
                urls.append(u); seen.add(u)
            if len(urls) >= max_imgs:
                break
    except Exception:
        pass
    # Fallback patterns if none found
    if not urls:
        vid = video_id_from_url(youtube_url)
        candidates = [
            f"https://i.ytimg.com/vi/{vid}/maxresdefault.jpg",
            f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg",
            f"https://i.ytimg.com/vi_webp/{vid}/maxresdefault.webp",
        ]
        urls = candidates[:max_imgs]
    return urls

# ─────────── Trade-press search & fetching ─────────────
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
    # Jina Reader proxy first (clean text), then raw
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

def collect_trade_pages(title_or_query: str, brand_hint: str = "") -> List[Tuple[str, str]]:
    """Return a small list of (url, cleaned_text) from trusted ad trades."""
    base = title_or_query or brand_hint
    if not base:
        return []
    queries = [
        f"{base} credits", f"{base} director", f"{base} voiceover",
        f"{base} Super Bowl ad", f"{base} ad breakdown", f"{base} behind the scenes",
        f"{base} shootonline", f"{base} adage", f"{base} adweek", f"{base} lbbonline", f"{base} shots"
    ]
    seen, pages = set(), []
    for q in queries:
        for res in web_search(q, limit=5):
            u = res.get("url","")
            if not u or u in seen: 
                continue
            if not _host_ok(u):
                continue
            seen.add(u)
            pages.append((u, http_get_readable(u)))
            if len(pages) >= 8:
                break
        if len(pages) >= 8:
            break
    # Trim text length to be safe
    cleaned = []
    for u, t in pages:
        if t:
            cleaned.append((u, re.sub(r"\s+", " ", t)[:8000]))
    return cleaned

# ───────────── Accuracy Playbook: System Prompt ─────────────
ACCURACY_SYSTEM = """
You are a fact-locked ad analyst. Extract ONLY what is supported by evidence.

RULES
- Do not guess. If unknown, write "UNKNOWN".
- Separate VISUAL actions from VO lines and external sources.
- Every non-trivial claim must include an evidence array.
- Prefer on-screen evidence (video/transcript); never invent clothing/names/locations.
- Timecode all actions/lines using the provided transcript segments.

TASKS
1) ACTION_LEDGER: list of atomic on-screen actions:
   { "t_start":"HH:MM:SS.mmm","t_end":"HH:MM:SS.mmm","actor":"generic label",
     "verb":"concrete verb","object":"thing being acted on",
     "evidence":[{"type":"onscreen","locator":"HH:MM:SS.mmm"}], "confidence": 0|1|2|3 }
2) DIALOG_LEDGER: timecoded exact lines heard:
   { "time":"HH:MM:SS.mmm","text":"verbatim snippet",
     "evidence":[{"type":"audio","locator":"HH:MM:SS.mmm"}] }
3) BEAT_MAP: 3–6 beats; each has { "title":string, "indexes":{"actions":[ints], "dialog":[ints]} }.
4) CASE_STUDY: { "big_idea": string (<=80 words), "why_it_works": [<=3 bullets] }
5) CITATIONS: unique list of external sources used (URLs) for facts that are not strictly on-screen/audio.

OUTPUT JSON ONLY:
{ "action_ledger":[], "dialog_ledger":[], "beat_map":[], "case_study":{}, "citations":[] }

STYLE
- Present tense. Concrete verbs (“dunks”, “crashes”, “howls”), not abstractions (“reacts”).
- No paraphrase for dialog; if uncertain, omit.
"""

# ───────────── LLM JSON helper (robust) ─────────────
def llm_structured(prompt: str, sys: str, model: str = None) -> Dict:
    """
    Call OpenAI with JSON mode; fall back to best-effort parsing.
    """
    client = get_client()
    model = model or OPENAI_JSON_MODEL
    try:
        resp = client.chat.completions.create(
            model=model,
            response_format={"type":"json_object"},
            messages=[
                {"role":"system","content":sys},
                {"role":"user","content":prompt}
            ],
            temperature=0.2,
            max_tokens=2200
        )
        content = resp.choices[0].message.content or "{}"
        return json.loads(content)
    except Exception:
        # permissive fallback parse
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role":"system","content":sys},{"role":"user","content":prompt}],
                temperature=0.2,
                max_tokens=2200
            )
            txt = resp.choices[0].message.content or "{}"
            start = txt.find("{"); end = txt.rfind("}")
            if start != -1 and end != -1:
                return json.loads(txt[start:end+1])
        except Exception:
            pass
    return {"action_ledger":[],"dialog_ledger":[],"beat_map":[],"case_study":{"big_idea":"","why_it_works":[]},"citations":[]}

# ───────────── Build ledgers from transcript + trades ─────────────
def build_ledgers_from_evidence(transcript_segments: List[Dict],
                                transcript_text: str,
                                trade_pages: List[Tuple[str,str]]) -> Dict:
    """
    Returns the JSON object:
    { action_ledger, dialog_ledger, beat_map, case_study, citations }
    """
    # Normalize transcript segments to HH:MM:SS.mmm
    norm = []
    for s in (transcript_segments or []):
        ss = float(s.get("start", 0.0))
        norm.append({
            "time": _format_time_hhmmss_mmm(ss),
            "text": s.get("text","")
        })

    # Compact evidence payload for the model (trade pages)
    evidence = []
    for (u, t) in (trade_pages or [])[:8]:
        evidence.append({"url": u, "text": t})

    user = {
        "transcript_segments": norm[:260],           # enough for ~30s spots
        "transcript_text": (transcript_text or "")[:12000],  # cap for safety
        "evidence_pages": evidence
    }

    result = llm_structured(
        prompt=json.dumps(user, ensure_ascii=False),
        sys=ACCURACY_SYSTEM,
        model=OPENAI_JSON_MODEL
    )

    # Basic shape guardrails
    for k in ["action_ledger","dialog_ledger","beat_map","citations"]:
        result.setdefault(k, [])
    result.setdefault("case_study", {"big_idea":"", "why_it_works":[]})
    return result

# ───────────── Render HTML from ledgers (clean & tight) ─────────────
def render_from_ledgers(url: str, title: str, channel: str, ledgers: Dict) -> str:
    big = (ledgers.get("case_study", {}) or {}).get("big_idea","").strip()
    why = (ledgers.get("case_study", {}) or {}).get("why_it_works",[])[:3]
    acts = ledgers.get("action_ledger", []) or []
    dlog = ledgers.get("dialog_ledger", []) or []
    beats = ledgers.get("beat_map", []) or []
    cites = ledgers.get("citations", []) or []

    # Optional: strict filter to only show actions with top confidence
    if STRICT_VISUALS:
        acts_view = []
        idx_map = {}  # old index -> new index
        for i, a in enumerate(acts):
            if int(a.get("confidence", 0)) == 3:
                idx_map[i] = len(acts_view)
                acts_view.append(a)
        # remap beat indexes
        new_beats = []
        for b in beats:
            a_idx = []
            for i in (b.get("indexes",{}).get("actions",[]) or []):
                if i in idx_map: a_idx.append(idx_map[i])
            new_beats.append({"title": b.get("title","Beat"),
                              "indexes":{"actions":a_idx, "dialog":b.get("indexes",{}).get("dialog",[])}})
        acts = acts_view
        beats = new_beats

    # Build beat-by-beat block with on-screen ACTIONS only
    beat_html = []
    for b in beats:
        title_b = b.get("title","Beat")
        a_idx = (b.get("indexes",{}) or {}).get("actions",[]) or []
        items = []
        for i in a_idx:
            if i < 0 or i >= len(acts): continue
            a = acts[i] or {}
            t0 = (a.get("t_start","00:00:00.000") or "00:00:00.000")[3:8]  # show MM:SS
            verb = a.get("verb","").strip()
            actor = a.get("actor","").strip()
            obj = a.get("object","").strip()
            conf = int(a.get("confidence",0))
            badge = "" if conf == 3 else f" <span style='color:#6b7280'>(conf {conf})</span>"
            line = f"<li><strong>{html.escape(t0)}</strong> — {html.escape(actor)} {html.escape(verb)} {html.escape(obj)}{badge}</li>"
            items.append(line)
        beat_html.append(f"<h3>{html.escape(title_b)}</h3><ul>{''.join(items) or '<li><em>No on-screen actions with evidence</em></li>'}</ul>")

    why_html = "".join(f"<li>{html.escape(w)}</li>" for w in why)

    src_html = ""
    if cites:
        src_html = "<h3>Sources</h3><ul>" + "".join(
            f"<li><a href='{html.escape(u)}'>{html.escape(u)}</a></li>" for u in cites
        ) + "</ul>"

    return f"""
<h2>Big Idea</h2>
<p>{html.escape(big) if big else 'UNKNOWN'}</p>

<h2>Beat-by-beat actions (what we actually see)</h2>
{''.join(beat_html)}

<h2>Why It Works</h2>
<ul>{why_html or '<li>UNKNOWN</li>'}</ul>

{src_html}
"""

# ─────────────────── HTML Templates (UI) ────────────────────
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
    <p class="muted">Paste a YouTube URL and (optionally) a full transcript. We’ll generate an evidence-locked breakdown and auto-download the PDF.</p>
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
    """
    Builds an evidence-locked case study PDF from a YouTube URL and (optionally) a pasted transcript.
    Returns the absolute path to the generated PDF file.
    """
    # 1) Basic metadata
    vid = video_id_from_url(url)
    meta = fetch_basic_metadata(vid)
    title = meta.get("title", "") or "Untitled Spot"
    channel = meta.get("author", "") or "Unknown Channel"

    # 2) Transcript segments (for timecodes) — always fetch if possible
    segments = fetch_transcript_segments(vid)
    if not segments:
        segments = [{"start": 0.0, "text": ""}]

    # Preferred transcript text
    if provided_transcript and provided_transcript.strip():
        transcript_text = provided_transcript.strip()[:30000]
    else:
        transcript_text = " ".join(s.get("text", "") for s in segments)[:30000]

    # 3) Gather trade pages (Adweek, Ad Age, SHOOT, etc.) for factual context
    trade_pages = collect_trade_pages(title, brand_hint=title)

    # 4) Build ledgers (action/dialog) with evidence + beat map + citations
    ledgers = build_ledgers_from_evidence(
        transcript_segments=segments,
        transcript_text=transcript_text,
        trade_pages=trade_pages
    )

    # 5) Render concise HTML from ledgers (Big Idea, beat-by-beat actions, Why it works, Sources)
    analysis_html = render_from_ledgers(url=url, title=title, channel=channel, ledgers=ledgers)

    # 6) Write PDF
    from weasyprint import HTML as WEASY_HTML
    os.makedirs(OUT_DIR, exist_ok=True)
    base_name = safe_token(title) or f"case_study_{safe_token(vid)}"
    pdf_path = os.path.join(OUT_DIR, f"{base_name}.pdf")
    html_doc = PDF_HTML.render(file_title=base_name, title=title, channel=channel, url=url, analysis_html=analysis_html)
    WEASY_HTML(string=html_doc, base_url=".").write_pdf(pdf_path)
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
