import os, re, json, html
from urllib.parse import urlparse, parse_qs
from typing import Dict, List, Optional, Tuple

from flask import Flask, request, render_template_string, send_from_directory, abort
from jinja2 import Template
from youtube_transcript_api import YouTubeTranscriptApi
import requests

# ───────────────────────── ENV ─────────────────────────
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # vision-capable
OUT_DIR = os.getenv("OUT_DIR", "out")
STRICT_VISUALS = os.getenv("STRICT_VISUALS", "0") == "1"  # if 1, hide press-sourced actions
MAX_TRADE_PAGES = int(os.getenv("MAX_TRADE_PAGES", "6"))

# Optional: search keys (recommended for best results)
BING_SEARCH_KEY      = os.getenv("BING_SEARCH_KEY", "")
BING_SEARCH_ENDPOINT = os.getenv("BING_SEARCH_ENDPOINT", "https://api.bing.microsoft.com/v7.0/search")
SERPAPI_KEY          = os.getenv("SERPAPI_KEY", "")  # if you prefer SerpAPI instead of Bing

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

def transcript_text_only(segments: List[Dict]) -> str:
    return " ".join(s.get("text", "") for s in (segments or []))

# ───────────── Thumbnails for vision context ───────────
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
        try:
            vid = video_id_from_url(youtube_url)
            candidates = [
                f"https://i.ytimg.com/vi/{vid}/maxresdefault.jpg",
                f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg",
                f"https://i.ytimg.com/vi_webp/{vid}/maxresdefault.webp",
            ]
            urls = candidates[:max_imgs]
        except Exception:
            urls = []
    return urls

# ─────────── Trade-press search & snippets ─────────────
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

def _extract_lines(text: str, max_chars=260) -> List[str]:
    # Take 1–2 sentence chunks around interesting words
    chunks = []
    for m in re.finditer(r"([^\r\n]{60,260})", text):
        s = m.group(1).strip()
        if any(k in s.lower() for k in ["director","agency","voice","vo","super bowl","spot","commercial","credits","cast","crew","stunt","chaos","gag","window"]):
            chunks.append(s[:max_chars])
        if len(chunks) >= 6:
            break
    return chunks

def enrich_from_trades_for_prompt(title: str, brand_hint: str = "") -> Dict[str, List[str]]:
    """
    Returns small verbatim snippets + any credits we can spot, from trusted trades.
    { "snippets":[...], "citations":[...], "credits":{"voiceover":[],"director":[],"agency":[]} }
    """
    base = title or brand_hint
    if not base:
        return {"snippets":[],"citations":[],"credits":{"voiceover":[],"director":[],"agency":[]}}
    queries = [
        f"{base} credits", f"{base} director", f"{base} VO", f"{base} voiceover",
        f"{base} Super Bowl ad", f"{base} ad breakdown", f"{base} shootonline",
        f"{base} adage", f"{base} adweek", f"{base} lbbonline", f"{base} shots"
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
            if len(pages) >= MAX_TRADE_PAGES:
                break
        if len(pages) >= MAX_TRADE_PAGES:
            break

    snippets, cites = [], []
    credit_bag = {"voiceover":{}, "director":{}, "agency":{}}
    for url, text in pages:
        if not text:
            continue
        # collect small verbatim chunks
        for sn in _extract_lines(text):
            if len(snippets) < 6:
                snippets.append(sn)
                cites.append(url)
        # try simple credit regex
        for key, pat in {
            "voiceover": r"(?:Voice[- ]?over|VO|Narration|Narrator)\s*[:\-]\s*([^\n\r;|,]+)",
            "director" : r"(?:Director|Directed by)\s*[:\-]\s*([^\n\r;|,]+)",
            "agency"   : r"(?:Agency|Creative Agency)\s*[:\-]\s*([^\n\r;|,]+)",
        }.items():
            for hit in re.findall(pat, text, flags=re.IGNORECASE):
                name = hit.strip()
                if name:
                    credit_bag[key].setdefault(name, []).append(url)

    def top_keys(d: Dict[str, List[str]]) -> List[str]:
        if not d: return []
        return [k for k,_ in sorted(d.items(), key=lambda kv: len(kv[1]), reverse=True)][:2]

    credits = {k: top_keys(v) for k,v in credit_bag.items()}
    # dedupe & trim citations in same order as snippets
    citations = []
    for c in cites:
        if c not in citations:
            citations.append(c)
    return {"snippets": snippets, "citations": citations[:MAX_TRADE_PAGES], "credits": credits}


# ───────── Accuracy-locked system prompt ─────────
ACCURACY_SYSTEM = """
You are a fact-locked ad analyst. Extract ONLY what is supported by evidence.

EVIDENCE SOURCES (priority)
1) On-screen/audio: transcript segments + timing
2) Press/trades: evidence_pages (Adweek, Ad Age, SHOOT, LBB, etc.)
3) Everything else: ignore

RULES
- Do not guess. If unknown, write "UNKNOWN".
- Separate VISUAL actions from VO lines and external sources.
- Every non-trivial claim must include an evidence array.
- Prefer on-screen evidence; never invent clothing/names/locations.
- Timecode all actions/lines from transcript when possible.
- If an action is described in a trade-press article and you cannot confirm the timecode,
  include it with evidence type "press" and confidence=2, and use "UNKNOWN" for t_start/t_end.

IF TRANSCRIPT/FRAMES ARE SPARSE
- Mine evidence_pages for concrete scene descriptions (“guest dives through window”, “spit take”, “dog howls”).
- Include those actions with evidence=[{"type":"press","locator":"URL"}], confidence=2.
- Better to include press-verified actions than to output empty beats.

TASKS
1) ACTION_LEDGER: list atomic on-screen actions:
   { "t_start":"HH:MM:SS.mmm"|"UNKNOWN",
     "t_end":"HH:MM:SS.mmm"|"UNKNOWN",
     "actor":"generic label (e.g., guest_male, grandma, dog)",
     "verb":"concrete verb",
     "object":"thing acted on (optional)",
     "evidence":[{"type":"onscreen"|"audio"|"press","locator":"HH:MM:SS.mmm or URL"}],
     "confidence": 0|1|2|3 }
   - confidence=3: clearly visible/audible via transcript timing
   - confidence=2: described by credible press/trade source
   - confidence=1: weak press hint (avoid)
   - confidence=0: inference (forbid)

2) DIALOG_LEDGER:
   { "time":"HH:MM:SS.mmm",
     "text":"verbatim snippet",
     "evidence":[{"type":"audio","locator":"HH:MM:SS.mmm"}] }

3) BEAT_MAP: 3–6 beats; each:
   { "title":string,
     "indexes":{"actions":[ints], "dialog":[ints]} }
   >>> Do NOT leave beats empty. Every beat must include >=1 action OR >=1 dialog index.

4) CASE_STUDY: { "big_idea": <=80 words, "why_it_works": [<=3 bullets] }

5) CITATIONS: unique URLs used from evidence_pages.

OUTPUT JSON ONLY:
{ "action_ledger":[], "dialog_ledger":[], "beat_map":[], "case_study":{}, "citations":[] }

STYLE
- Present tense. Concrete verbs (“dunks”, “crashes”, “howls”), not abstractions (“reacts”).
- No paraphrase for dialog; if uncertain, omit.
"""


# ───────── LLM: build ledgers (vision+trades) ─────────
def build_ledgers(youtube_url: str, transcript_text: str, video_title: str, channel: str) -> Dict:
    """
    Produces the accuracy-locked JSON ledgers using:
    • transcript_text
    • 2–3 thumbnails (vision context)
    • evidence_pages from trade press (URL + excerpt)
    """
    client = get_client()

    # Vision inputs
    image_urls = get_video_thumbnails(youtube_url, max_imgs=3)
    vision_parts = [{"type":"image_url","image_url":{"url":u}} for u in image_urls]

    # Evidence pages (trades) with short excerpts
    trade = enrich_from_trades_for_prompt(video_title, brand_hint=video_title)
    pages = []
    for idx, url in enumerate(trade.get("citations", [])):
        excerpt = trade.get("snippets", [])[idx] if idx < len(trade.get("snippets", [])) else ""
        pages.append({"url": url, "excerpt": excerpt})

    # Build strictness notice
    strict_note = "Only include actions that are visible or in dialog/audio; do NOT use press-derived actions." if STRICT_VISUALS else \
                  "If visuals are sparse, you MAY include press-derived actions as confidence=2 with evidence type 'press'."

    user_payload = {
        "video": {"title": video_title, "channel": channel, "url": youtube_url},
        "transcript": transcript_text,
        "evidence_pages": pages,
        "instructions": strict_note
    }

    content = []
    if vision_parts:
        content.extend(vision_parts)
    content.append({"type":"text","text": json.dumps(user_payload, ensure_ascii=False)})

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":ACCURACY_SYSTEM},{"role":"user","content":content}],
            temperature=0.35,   # accuracy > creativity
            max_tokens=2600,
            response_format={"type":"json_object"}
        )
        raw = resp.choices[0].message.content or "{}"
        data = json.loads(raw)
    except Exception:
        # Permissive fallback (try without forced JSON)
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role":"system","content":ACCURACY_SYSTEM},{"role":"user","content":content}],
                temperature=0.4,
                max_tokens=2600
            )
            raw = resp.choices[0].message.content or "{}"
            start = raw.find("{"); end = raw.rfind("}")
            data = json.loads(raw[start:end+1]) if start != -1 and end != -1 else {}
        except Exception:
            data = {}

    # Ensure required keys exist
    data.setdefault("action_ledger", [])
    data.setdefault("dialog_ledger", [])
    data.setdefault("beat_map", [])
    data.setdefault("case_study", {"big_idea":"", "why_it_works":[]})
    data.setdefault("citations", trade.get("citations", []))
    return data


# ───────── Render HTML from ledgers ─────────
def render_from_ledgers(url: str, title: str, channel: str, ledgers: Dict) -> str:
    big = (ledgers.get("case_study", {}) or {}).get("big_idea","").strip()
    why = (ledgers.get("case_study", {}) or {}).get("why_it_works",[])[:3]
    acts = ledgers.get("action_ledger", []) or []
    dlog = ledgers.get("dialog_ledger", []) or []
    beats = ledgers.get("beat_map", []) or []
    cites = ledgers.get("citations", []) or []

    # Helper to display mm:ss from HH:MM:SS.mmm or UNKNOWN
    def mmss(t: str) -> str:
        if not t or t.upper() == "UNKNOWN":
            return "??:??"
        try:
            parts = t.split(":")
            m = int(parts[-2])
            s = int(float(parts[-1]))
            return f"{m:02d}:{s:02d}"
        except Exception:
            return "??:??"

    # If a beat is empty, backfill with a dialog line so nothing renders empty
    for b in beats:
        idx = b.get("indexes") or {}
        if not idx.get("actions") and not idx.get("dialog"):
            if dlog:
                idx["dialog"] = [0]
            b["indexes"] = idx

    # Build beat-by-beat list items
    beat_html = []
    for b in beats:
        title_b = b.get("title","Beat")
        idx = b.get("indexes",{}) or {}
        a_idx = idx.get("actions",[]) or []
        d_idx = idx.get("dialog",[]) or []

        items = []

        # Actions (show source tags; hide press if STRICT_VISUALS=1)
        for i in a_idx:
            if i < 0 or i >= len(acts): continue
            a = acts[i] or {}
            # detect source
            srcs = a.get("evidence",[]) or []
            src_tag = "on-screen"
            is_press = False
            for ev in srcs:
                if (ev.get("type") or "").lower() == "press":
                    src_tag = "press"
                    is_press = True
                    break
            if STRICT_VISUALS and is_press:
                continue  # hide press-derived actions if strict mode enabled

            t0 = mmss(a.get("t_start","UNKNOWN"))
            verb = (a.get("verb","") or "").strip()
            actor = (a.get("actor","") or "").strip()
            obj = (a.get("object","") or "").strip()
            conf = int(a.get("confidence",0))
            badge = f" <span style='color:#6b7280'>(conf {conf}, {src_tag})</span>"
            obj_part = f" {html.escape(obj)}" if obj else ""
            line = f"<li><strong>{html.escape(t0)}</strong> — {html.escape(actor)} {html.escape(verb)}{obj_part}{badge}</li>"
            items.append(line)

        # Dialog (always allowed)
        for j in d_idx[:2]:  # at most two lines per beat
            if j < 0 or j >= len(dlog): continue
            dl = dlog[j] or {}
            t = mmss(dl.get("time","UNKNOWN"))
            txt = (dl.get("text","") or "").strip()
            items.append(f"<li><strong>{html.escape(t)}</strong> — VO/Dialogue: {html.escape(txt)}</li>")

        beat_html.append(f"<h3>{html.escape(title_b)}</h3><ul>{''.join(items) or '<li><em>No evidence-backed items</em></li>'}</ul>")

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


# ───────── Orchestrate → HTML → PDF ─────────
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
            segments = [{"start": 0.0, "text": ""}]
        transcript_text = transcript_text_only(segments)[:30000]

    # 3) Build evidence-locked ledgers (vision + trades)
    ledgers = build_ledgers(
        youtube_url=url,
        transcript_text=transcript_text,
        video_title=title,
        channel=channel,
    )

    # 4) Render HTML from ledgers
    analysis_html = render_from_ledgers(url, title, channel, ledgers)

    # 5) Render PDF
    from weasyprint import HTML as WEASY_HTML
    os.makedirs(OUT_DIR, exist_ok=True)
    base_name = safe_token(title) or f"case_study_{safe_token(vid)}"
    pdf_path = os.path.join(OUT_DIR, f"{base_name}.pdf")

    shell = PDF_HTML.render(file_title=base_name, title=title, channel=channel, url=url, analysis_html=analysis_html)
    WEASY_HTML(string=shell, base_url=".").write_pdf(pdf_path)
    return pdf_path


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
    <p class="muted">Paste a YouTube URL and (optionally) a full transcript. We’ll generate an evidence-backed breakdown and auto-download the PDF.</p>
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
