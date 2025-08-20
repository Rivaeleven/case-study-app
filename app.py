import os, re, json
from urllib.parse import urlparse, parse_qs, quote_plus
from typing import Optional, Dict, List, Tuple

from flask import Flask, request, render_template_string, send_from_directory, abort
from youtube_transcript_api import YouTubeTranscriptApi
from jinja2 import Template
from pydantic import BaseModel, Field
import requests

# ───────────────────────── ENV ─────────────────────────
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o")  # use a real default
OUT_DIR         = os.getenv("OUT_DIR", "out")
REPAIR_PASSES   = int(os.getenv("REPAIR_PASSES", "2"))
OPENAI_FALLBACK = os.getenv("OPENAI_FALLBACK", "1") == "1"
os.makedirs(OUT_DIR, exist_ok=True)

# Optional search keys (either works; Bing preferred if present)
BING_SEARCH_KEY      = os.getenv("BING_SEARCH_KEY", "")
BING_SEARCH_ENDPOINT = os.getenv("BING_SEARCH_ENDPOINT", "https://api.bing.microsoft.com/v7.0/search")
SERPAPI_KEY          = os.getenv("SERPAPI_KEY", "")

# ───────────────────── OpenAI (lazy) ───────────────────
from openai import OpenAI
def get_client():
    return OpenAI()  # reads OPENAI_API_KEY from env

def _create_with_fallback(**create_kwargs):
    client = get_client()
    try:
        return client.chat.completions.create(**create_kwargs)
    except Exception:
        if OPENAI_FALLBACK and "gpt-5" in create_kwargs.get("model", ""):
            try:
                alt = dict(create_kwargs, model="gpt-4o")
                return client.chat.completions.create(**alt)
            except Exception:
                pass
        raise

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

# ─────────── Readable fetch, whitelist, search ─────────
PUBLISHER_WHITELIST = [
    "adage.com", "adweek.com", "campaignlive.co.uk", "campaignlive.com",
    "shots.net", "lbbonline.com", "thedrum.com", "musebycl.io", "ispot.tv",
    "prnewswire.com", "businesswire.com"
]

def is_whitelisted(url: str) -> bool:
    try:
        host = urlparse(url).hostname or ""
        return any(d in host for d in PUBLISHER_WHITELIST)
    except Exception:
        return False

def http_get_readable(url: str, timeout=15):
    """Try Jina reader proxy first for clean text, then direct."""
    try:
        u = urlparse(url)
        proxy = f"https://r.jina.ai/http://{u.netloc}{u.path}"
        r = requests.get(proxy, timeout=timeout)
        if r.ok and len(r.text) > 500:
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

def web_search(query: str, limit: int = 6) -> List[Dict[str, str]]:
    """Use Bing Web Search if available; else SerpAPI; else empty."""
    results = []
    try:
        if BING_SEARCH_KEY:
            r = requests.get(
                BING_SEARCH_ENDPOINT,
                params={"q": query, "count": limit},
                headers={"Ocp-Apim-Subscription-Key": BING_SEARCH_KEY},
                timeout=12
            )
            if r.ok:
                for i in r.json().get("webPages", {}).get("value", []):
                    results.append({"title": i.get("name",""), "url": i.get("url","")})
        elif SERPAPI_KEY:
            r = requests.get(
                "https://serpapi.com/search.json",
                params={"engine":"google", "q": query, "num": limit, "api_key": SERPAPI_KEY},
                timeout=12
            )
            if r.ok:
                for i in r.json().get("organic_results", []):
                    results.append({"title": i.get("title",""), "url": i.get("link","")})
    except Exception:
        pass
    # De-dup + whitelist preference
    dedup = []
    seen = set()
    for it in results:
        u = it.get("url","")
        if not u or u in seen:
            continue
        if is_whitelisted(u):
            dedup.append(it); seen.add(u)
    return dedup[:limit]

# ─────────────── YouTube description (yt-dlp) ──────────
def yt_description(video_url: str) -> str:
    """Use yt-dlp to fetch description reliably (when available)."""
    try:
        import yt_dlp
        ydl_opts = {"quiet": True, "skip_download": True, "extract_flat": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            return (info.get("description") or "")[:20000]
    except Exception:
        return ""

# ───────────── Simple credit extraction helpers ────────
JINA = "https://r.jina.ai/http://"

def fetch_youtube_description(video_id: str) -> str:
    """Best‑effort: readable watch page; often contains description text."""
    try:
        url = f"{JINA}www.youtube.com/watch?v={video_id}"
        r = requests.get(url, timeout=15)
        if r.ok:
            return r.text[:20000]
    except Exception:
        pass
    return ""

def _extract_credit_fields(text: str) -> Dict[str, str]:
    """Heuristic regex extraction for Agency/Director/Product/Campaign from text."""
    credits = {}
    try:
        t = re.sub(r"[ \t]+", " ", text or "")
        patterns = {
            "agency": r"(?:Agency|Creative Agency)\s*:\s*([^\n\r;|]+)",
            "director": r"(?:Director)\s*:\s*([^\n\r;|]+)",
            "product": r"(?:Brand|Client|Product)\s*:\s*([^\n\r;|]+)",
            "campaign": r"(?:Campaign|Title)\s*:\s*([^\n\r;|]+)",
        }
        for k, pat in patterns.items():
            m = re.search(pat, t, flags=re.IGNORECASE)
            if m:
                credits[k] = m.group(1).strip()
    except Exception:
        pass
    return credits

def scrape_ispot_by_title(title: str) -> Tuple[str, str]:
    """Try iSpot.tv: return (ad_page_url, page_text) for first likely ad."""
    try:
        q = quote_plus(title or "")
        if not q:
            return "", ""
        search_url = f"https://www.ispot.tv/search?q={q}"
        rs = http_get_readable(search_url)
        if not rs:
            return "", ""
        ad_paths = re.findall(r"/ad/\w+/[a-z0-9\-]+", rs, flags=re.IGNORECASE)
        if not ad_paths:
            return "", ""
        ad_url = f"https://www.ispot.tv{ad_paths[0]}"
        page = http_get_readable(ad_url)
        return ad_url, page
    except Exception:
        return "", ""

def scrape_shots_by_title(title: str) -> Tuple[str, str]:
    """Try shots.net search page and return (url, text)."""
    try:
        q = quote_plus(title or "")
        if not q:
            return "", ""
        url = f"https://www.shots.net/search?q={q}"
        page = http_get_readable(url)
        return url, page
    except Exception:
        return "", ""

def consolidate_credits(candidate_pages: List[Tuple[str, str]]) -> Dict[str, Dict[str, List[str]]]:
    """
    candidate_pages: list of (url, page_text).
    Returns: { field: { value: [urls...] } }
    """
    bag: Dict[str, Dict[str, List[str]]] = {}
    for url, text in candidate_pages:
        if not text:
            continue
        fields = _extract_credit_fields(text)
        for k, v in fields.items():
            v_norm = v.strip()
            if not v_norm:
                continue
            bag.setdefault(k, {}).setdefault(v_norm, []).append(url)
    return bag

def pick_credit(bag: Dict[str, Dict[str, List[str]]], key: str, min_support: int = 2) -> Tuple[str, List[str]]:
    options = bag.get(key, {})
    if not options:
        return "Unknown", []
    best_val = max(options.items(), key=lambda kv: len(kv[1]))
    if len(best_val[1]) >= min_support:
        return best_val[0], best_val[1]
    return "Unknown", []

# ─────────────────────── Prompts ───────────────────────
NAMING_SYSTEM = """
Extract (or infer cautiously) the work's naming fields from the provided text and any user-provided hints.
Prefer the user hints for proper nouns (agency, director, campaign) when present; if hints are missing or conflict with transcript/metadata, use 'Unknown'.
- Agency (creative agency behind the commercial)
- Product (brand/product)
- Campaign (campaign/series name if stated; else 'Unknown')
- Commercial (spot title; default to video title if unclear)
- Director (if known in video or description; else 'Unknown')
Return ONLY compact JSON with keys: agency, product, campaign, commercial, director.
Be conservative; if a proper noun is uncertain, use 'Unknown'.
""".strip()

STRUCTURED_JSON_SPEC = """
You will receive a YouTube URL, title, channel, a timecoded transcript (if available), a plain transcript, and optional user 'hints'.
Treat the provided transcript as canonical. If other metadata contradicts it, prefer the transcript. If a line is not verbatim, append ' [inferred]'.
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
- Anchor timestamps to the timecoded transcript if present; if not available, estimate conservatively and mark [inferred] where needed.
- 'line' must be a verbatim substring of the transcript OR the hints; otherwise keep it short and append ' [inferred]'.
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

# ─────────────────── LLM Helpers ─────────────────────
def llm_json(system: str, user: str) -> Dict:
    try:
        resp = _create_with_fallback(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        txt = resp.choices[0].message.content or ""
        start = txt.find("{"); end = txt.rfind("}")
        return json.loads(txt[start:end+1])
    except Exception as e:
        return {"agency":"Unknown","product":"Unknown","campaign":"Unknown","commercial":"Unknown","director":"Unknown","_error": str(e)}

def llm_json_structured(system: str, user: str) -> Dict:
    try:
        resp = _create_with_fallback(
            model=OPENAI_MODEL,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception:
        try:
            resp = _create_with_fallback(
                model=OPENAI_MODEL,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            txt = resp.choices[0].message.content or "{}"
            start = txt.find("{"); end = txt.rfind("}")
            return json.loads(txt[start:end+1])
        except Exception:
            return {}

def llm_html_from_json(system: str, json_payload: Dict, transcript_text: str) -> str:
    try:
        user_blob = json.dumps({"json": json_payload, "transcript": transcript_text}, ensure_ascii=False)
        resp = _create_with_fallback(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user_blob}],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"<p><em>analysis generation failed:</em> {e}</p>"

# ─────────────────── Validation / Repair ───────────────
def _is_substring_loosely(s: str, blob: str) -> bool:
    s_norm = re.sub(r"[^\w\s]", "", (s or "").strip().lower())
    blob_norm = re.sub(r"[^\w\s]", "", (blob or "").strip().lower())
    return s_norm and (s_norm in blob_norm)

def _looks_like_time(ts: str) -> bool:
    return bool(re.match(r"^\d{2}:\d{2}$", (ts or "").strip()))

def validate_json(payload: Dict, transcript_blob: str, hints_blob: str) -> List[str]:
    errs = []
    scenes = payload.get("scenes") or []
    script = payload.get("script") or []
    if len(scenes) < 6:
        errs.append(f"too_few_scenes:{len(scenes)}")
    if len(script) < 6:
        errs.append(f"too_few_script_lines:{len(script)}")
    for i, sc in enumerate(scenes):
        if not _looks_like_time(sc.get("start", "")):
            errs.append(f"bad_scene_time_at_{i}")
    for i, row in enumerate(script):
        if not _looks_like_time(row.get("time", "")):
            errs.append(f"bad_script_time_at_{i}")
        line = (row or {}).get("line", "")
        if line and not _is_substring_loosely(line, transcript_blob) and not _is_substring_loosely(line, hints_blob):
            errs.append(f"non_verbatim_script_line_at_{i}")
    return errs

def repair_json_with_model(bad_payload: Dict, transcript_blob: str, hints_blob: str) -> Dict:
    instruction = {
        "task": "repair",
        "issues": validate_json(bad_payload, transcript_blob, hints_blob),
        "rules": [
            "Ensure 8–14 scenes and 8–14 script lines.",
            "Every 'line' must be a substring of transcript OR hints; otherwise append ' [inferred]' and keep it short.",
            "Preserve timestamps and supers where possible.",
            "Prefer user hints for proper nouns; if hints are missing or conflict, leave 'Unknown'."
        ],
        "transcript": transcript_blob,
        "hints": hints_blob,
        "json": bad_payload,
    }
    try:
        resp = _create_with_fallback(
            model=OPENAI_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role":"system","content":"You repair JSON to satisfy validation rules using only transcript/hints."},
                {"role":"user","content":json.dumps(instruction, ensure_ascii=False)}
            ],
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
    a { color: #1f2937; text-decoration: none; border-bottom: 1px dotted #9ca3af; }
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

  {% if sources and sources|length %}
  <h2>Sources</h2>
  <ol>
    {% for src in sources %}
      <li><a href="{{ src }}">{{ src }}</a></li>
    {% endfor %}
  </ol>
  {% endif %}

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
    <p class="muted">Paste a YouTube URL and, if you have it, the full <strong>Transcript</strong>. We’ll return a PDF.</p>
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
    <p>Your PDF is ready. Download it below.</p>
    <p><a href="{{ pdf_url }}" class="btn" download>Download PDF: {{ pdf_filename }}</a></p>
    <p style="margin-top:20px"><a href="/">Generate another</a></p>
  </div>
</body>
</html>
"""

# ─────────────────── Builder pipeline ──────────────────
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

def _fill_from_title_if_possible(naming: Naming, title: str) -> Naming:
    if not title:
        return naming
    if naming.product == "Unknown" or naming.commercial == "Unknown":
        if " / " in title:
            parts = title.split(" / ", 1)
            if naming.product == "Unknown":
                naming.product = parts[0].strip() or "Unknown"
            if naming.commercial == "Unknown":
                naming.commercial = parts[1].strip() or naming.commercial
        elif " – " in title:
            parts = title.split(" – ", 1)
            if naming.product == "Unknown":
                naming.product = parts[0].strip() or "Unknown"
            if naming.commercial == "Unknown":
                naming.commercial = parts[1].strip() or naming.commercial
    return naming

def build_case_study(url: str, overrides: Optional[Dict[str, str]] = None, provided_transcript: Optional[str] = None) -> str:
    """
    Returns: path_to_pdf
    """
    # Extract and fetch metadata
    vid = video_id_from_url(url)
    meta = fetch_basic_metadata(vid)

    # Transcript handling (prefer user-provided)
    if provided_transcript:
        transcript_text = provided_transcript[:22000]
        preview_lines = transcript_text.splitlines()[:20]
        timecoded = "\n".join(preview_lines)
    else:
        segments = fetch_transcript_segments(vid)
        if not segments:
            segments = [{"start": 0.0, "text": ""}]
        timecoded = "\n".join(f"{_format_time(s['start'])}  {s['text']}" for s in segments[:220])
        transcript_text = " ".join(s["text"] for s in segments)[:22000]

    # Optional provenance scraping (kept minimal)
    candidate_pages: List[Tuple[str, str]] = []
    desc_txt = yt_description(f"https://www.youtube.com/watch?v={vid}") or fetch_youtube_description(vid)
    if desc_txt:
        candidate_pages.append((f"https://www.youtube.com/watch?v={vid}", desc_txt))

    bag = consolidate_credits(candidate_pages)
    auto_agency, agency_srcs = pick_credit(bag, "agency", min_support=1)
    auto_director, director_srcs = pick_credit(bag, "director", min_support=1)
    auto_product, product_srcs = pick_credit(bag, "product", min_support=1)
    auto_campaign, campaign_srcs = pick_credit(bag, "campaign", min_support=1)

    # Build naming via LLM (then merge any auto/user overrides)
    user_block = ANALYSIS_USER_TEMPLATE.render(
        url=url, title=meta.get("title",""), author=meta.get("author",""),
        timecoded=timecoded, transcript=transcript_text
    )
    raw_naming = llm_json(NAMING_SYSTEM, user_block) or {}
    overrides = overrides or {}

    auto_overrides = {}
    if auto_agency != "Unknown": auto_overrides["agency"] = auto_agency
    if auto_director != "Unknown": auto_overrides["director"] = auto_director
    if auto_product != "Unknown": auto_overrides["product"] = auto_product
    if auto_campaign != "Unknown": auto_overrides["campaign"] = auto_campaign

    for k, v in auto_overrides.items():
        if v:
            raw_naming[k] = v
    for k, v in overrides.items():
        if k != "_hints" and v:
            raw_naming[k] = v

    naming = Naming(
        agency=raw_naming.get("agency","Unknown"),
        product=raw_naming.get("product","Unknown"),
        campaign=raw_naming.get("campaign","Unknown"),
        commercial=raw_naming.get("commercial", meta.get("title","Unknown")),
        director=raw_naming.get("director","Unknown"),
    )
    naming = _fill_from_title_if_possible(naming, meta.get("title",""))

    # Structured JSON (scenes + script + brand)
    hints = overrides.get("_hints", "") if overrides else ""
    json_user_block = user_block + ("\n\nHINTS:\n" + hints if hints else "")
    json_payload = llm_json_structured(STRUCTURED_JSON_SPEC, json_user_block)
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
    # (No JSON sidecar saved)

    # Validation and light repair
    for _ in range(REPAIR_PASSES):
        issues = validate_json(json_payload, transcript_text, hints)
        if not issues:
            break
        json_payload = repair_json_with_model(json_payload, transcript_text, hints)

    # Analysis strictly from JSON + transcript
    analysis_html = llm_html_from_json(ANALYSIS_FROM_JSON, json_payload, transcript_text)

    # Render PDF
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
        sources=[],  # still supported by template; we’re not adding any list here
    )
    pdf_path = os.path.join(OUT_DIR, naming.filename())
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
        return send_from_directory(OUT_DIR, filename, as_attachment=False)
    except Exception:
        abort(404)

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
    transcript_text = (request.form.get("transcript") or "").strip()
    try:
        pdf_path = build_case_study(url, overrides=overrides, provided_transcript=(transcript_text or None))
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
