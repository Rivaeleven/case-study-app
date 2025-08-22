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
os.makedirs(OUT_DIR, exist_ok=True)

# Optional: search keys (recommended for best results)
BING_SEARCH_KEY      = os.getenv("BING_SEARCH_KEY", "")
BING_SEARCH_ENDPOINT = os.getenv("BING_SEARCH_ENDPOINT", "https://api.bing.microsoft.com/v7.0/search")
SERPAPI_KEY          = os.getenv("SERPAPI_KEY", "")  # if you prefer SerpAPI instead of Bing

app = Flask(__name__)

# ───────────────────── OpenAI client ───────────────────
from openai import OpenAI
def _llm_client():
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

# ───────────── Thumbnails for vision context ───────────
def get_video_thumbnails(youtube_url: str, max_imgs: int = 4) -> List[str]:
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

CREDIT_PATTERNS = {
    "voiceover": r"(?:Voice[- ]?over|VO|Narration|Narrator)\s*[:\-]\s*([^\n\r;|,]+)",
    "director" : r"(?:Director|Directed by)\s*[:\-]\s*([^\n\r;|,]+)",
    "agency"   : r"(?:Agency|Creative Agency)\s*[:\-]\s*([^\n\r;|,]+)",
}

def _extract_lines(text: str, max_chars=260):
    # Take 1–2 sentence chunks around interesting words
    chunks = []
    for m in re.finditer(r"([^\r\n]{60,260})", text):
        s = m.group(1).strip()
        if any(k in s.lower() for k in ["director","agency","voice","vo","super bowl","spot","commercial","credits","cast","crew"]):
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
        for key, pat in CREDIT_PATTERNS.items():
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
    return {"snippets": snippets, "citations": citations[:6], "credits": credits}

# ───────── LLM: JSON with visuals_montage_sourced (strict) ─────────
SOURCE_PRIORITY_PROMPT = """
You are a fact-locked ad analyst. Output ONLY JSON.

EVIDENCE PRIORITY (strict):
1) SOURCE_VERIFIED_VISUALS = things clearly visible in the provided thumbnails (and consistent with transcript audio).
2) transcript_audio        = exact lines heard in the transcript/captions you receive.
3) trade_press             = facts quoted from the provided research snippets/links (if any).
4) inferred_low            = only if weakly suggested; avoid when unsure.

RULES:
- If it’s not clearly supported by evidence, DO NOT include it.
- For `visuals_montage_sourced`, include ONLY on-screen actions you can confidently verify from the thumbnails/transcript. If you are not sure an action appears on-screen, DO NOT list it.
- Never invent props, wardrobe, names, or counts.
- Timecode everything you can. Dialog lines MUST be substrings of the transcript; if not 100% certain, omit them.
- Use short, concrete, filmic wording. No generic phrases like “people react”.

OUTPUT JSON SCHEMA (return exactly this shape):
{
  "meta": {
    "title": "string",
    "channel": "string",
    "url": "string"
  },
  "big_idea": "string",
  "beat_map": [
    {
      "label": "string",
      "time_start": "MM:SS",
      "time_end": "MM:SS",
      "vo_or_dialog": [ "verbatim lines from transcript only" ],
      "visual": "what we SEE (short, concrete). If not visible, omit.",
      "provenance": ["source_verified_visuals" | "transcript_audio" | "trade_press" | "inferred_low"]
    }
  ],
  "dialogs": [
    { "time": "MM:SS", "line": "verbatim from transcript" }
  ],
  "visuals_montage_sourced": [
    {
      "description": "on-screen action (short, concrete)",
      "provenance": ["source_verified_visuals"]
    }
  ],
  "sources": [
    "optional trade-press urls you actually used"
  ]
}

VALIDATION:
- `visuals_montage_sourced` must include ONLY actions you can SEE in thumbnails (and that do not contradict transcript). If you can’t verify a stunt is on-screen, leave it out.
- If thumbnails don’t show enough, the array can be empty.
- No Markdown, no commentary, JSON only.
""".strip()

def _vision_user_payload(youtube_url: str, video_title: str, channel: str,
                         transcript_text: str, image_urls: List[str],
                         research_snips: List[str], research_urls: List[str]) -> List[dict]:
    # Vision + text content for GPT-4o
    parts: List[dict] = []
    for u in image_urls:
        parts.append({"type": "image_url", "image_url": {"url": u}})
    text = [
        f"Title: {video_title}",
        f"Channel: {channel}",
        f"URL: {youtube_url}",
        "",
    ]
    if research_snips:
        text.append("Trade-press snippets (for factual context only):")
        for s in research_snips[:8]:
            text.append(f"• {s}")
        if research_urls:
            text.append("Links:")
            for u in research_urls[:8]:
                text.append(f"- {u}")
        text.append("")
    text.append("Transcript (verbatim):")
    text.append(transcript_text)
    parts.append({"type": "text", "text": "\n".join(text)})
    return parts

def _gpt_json(system_prompt: str, user_payload: List[dict]) -> dict:
    client = _llm_client()
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", OPENAI_MODEL),
        response_format={"type": "json_object"},
        messages=[
            {"role":"system","content": system_prompt},
            {"role":"user","content": user_payload},
        ],
        temperature=0.3,   # accuracy first
        max_tokens=2200,
    )
    txt = resp.choices[0].message.content or "{}"
    try:
        return json.loads(txt)
    except Exception:
        start = txt.find("{"); end = txt.rfind("}")
        return json.loads(txt[start:end+1]) if start != -1 and end != -1 else {}

def build_debranded_json(youtube_url: str, provided_transcript: Optional[str]) -> dict:
    """
    Build the JSON payload with a hard-required `visuals_montage_sourced` list that only
    contains on-screen, source-verified actions.
    """
    # 1) Basic meta
    vid = video_id_from_url(youtube_url)
    meta = fetch_basic_metadata(vid)
    title = meta.get("title", "") or "Untitled Spot"
    channel = meta.get("author", "") or "Unknown Channel"

    # 2) Transcript
    if provided_transcript and provided_transcript.strip():
        transcript_text = provided_transcript.strip()[:30000]
    else:
        segs = fetch_transcript_segments(vid)
        if not segs:
            segs = [{"start": 0.0, "text": ""}]
        transcript_text = " ".join(s.get("text","") for s in segs)[:30000]

    # 3) Thumbnails (visual evidence)
    thumbs = get_video_thumbnails(youtube_url, max_imgs=4)

    # 4) Light trade-press context (optional)
    trade = enrich_from_trades_for_prompt(title, brand_hint=title)
    research_snips = trade.get("snippets", [])
    research_urls  = trade.get("citations", [])

    # 5) Ask model for JSON
    user_payload = _vision_user_payload(
        youtube_url=youtube_url,
        video_title=title,
        channel=channel,
        transcript_text=transcript_text,
        image_urls=thumbs,
        research_snips=research_snips,
        research_urls=research_urls,
    )
    data = _gpt_json(SOURCE_PRIORITY_PROMPT, user_payload)

    # 6) Post-validate minimal fields and constrain montage provenance
    data.setdefault("meta", {}).update({"title": title, "channel": channel, "url": youtube_url})
    data.setdefault("big_idea", "")
    data.setdefault("beat_map", [])
    data.setdefault("dialogs", [])
    data.setdefault("visuals_montage_sourced", [])
    # attach any gathered sources (if model omitted)
    if "sources" not in data or not isinstance(data.get("sources"), list):
        data["sources"] = research_urls[:8]
    elif research_urls:
        # de-dupe merge
        sset = set(data["sources"])
        for u in research_urls[:8]:
            if u not in sset:
                data["sources"].append(u); sset.add(u)

    # Enforce montage items to be strictly source_verified_visuals or drop them
    clean_montage = []
    for item in data.get("visuals_montage_sourced", []):
        desc = (item or {}).get("description", "").strip()
        prov = [p for p in (item or {}).get("provenance", []) if p == "source_verified_visuals"]
        if desc and prov:
            clean_montage.append({"description": desc, "provenance": ["source_verified_visuals"]})
    data["visuals_montage_sourced"] = clean_montage

    # Lightweight hallucination guard on montage (ban these words without verified visuals)
    banned = re.compile(r"\b(wearing|shirt|tie|named|boyfriend|girlfriend|manager|influencer|exactly\s*\d+)\b", re.I)
    data["visuals_montage_sourced"] = [
        it for it in data["visuals_montage_sourced"]
        if not banned.search(it.get("description",""))
    ]

    # 7) Stable id for file name
    data["id"] = safe_token(f"{title or 'case_study'}_{vid}")
    return data

# ─────────────────── HTML Templates ────────────────────
INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>YouTube → JSON Case Study</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; color:#111; margin: 24px; }
    .card { max-width: 860px; margin: 0 auto; border: 1px solid #e5e7eb; border-radius: 14px; padding: 20px; box-shadow: 0 6px 20px rgba(0,0,0,.05); }
    label { display:block; margin: 10px 0 6px; font-weight: 600; }
    input[type=text], textarea, select { width:100%; padding:10px 12px; border:1px solid #d1d5db; border-radius:10px; font-size:15px; }
    textarea { min-height: 140px; }
    .row { display:flex; gap:12px; align-items:center; }
    .row > div { flex:1; }
    .btn { background:#111827; color:#fff; border:none; padding: 10px 14px; border-radius: 10px; cursor:pointer; }
    .muted { color:#6b7280; font-size: 13px; }
  </style>
</head>
<body>
  <div class="card">
    <h2>YouTube → JSON Case Study</h2>
    <p class="muted">Paste a YouTube URL and (optionally) a full transcript. Choose your download format: raw <code>.txt</code> (JSON) or a <code>.pdf</code> that embeds the JSON.</p>
    <form method="post" action="/generate">
      <label>YouTube URL</label>
      <input name="url" type="text" placeholder="https://www.youtube.com/watch?v=..." required />
      <label style="margin-top:12px">Transcript (optional — paste if you already have it)</label>
      <textarea name="transcript" placeholder="Paste full transcript here. If empty, we’ll try to fetch captions automatically."></textarea>
      <div class="row" style="margin-top:12px">
        <div>
          <label>Download format</label>
          <select name="format">
            <option value="txt" selected>.txt (raw JSON)</option>
            <option value="pdf">.pdf (JSON pretty-printed)</option>
          </select>
        </div>
      </div>
      <p style="margin-top:14px"><button class="btn" type="submit">Generate</button></p>
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
    code { background: #f3f4f6; padding: 2px 6px; border-radius: 6px; }
  </style>
</head>
<body>
  <div class="card">
    <h2>Success!</h2>
    <p>Your file should auto-download. If not, click the button below.</p>
    <p><a href="{{ file_url }}" class="btn" download>Download: <code>{{ file_name }}</code></a></p>
    <p style="margin-top:20px"><a href="/">Generate another</a></p>
  </div>
  <script>
    (function(){ const a = document.createElement('a'); a.href = "{{ file_url }}"; a.download = "{{ file_name }}"; document.body.appendChild(a); a.click(); a.remove(); })();
  </script>
</body>
</html>
"""

PDF_WRAPPER = Template("""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{{ title }}</title>
  <style>
    body { font: 12pt/1.55 -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; color:#111; margin: 28px; }
    h1 { font-size: 18pt; margin: 0 0 12px; }
    pre { padding: 14px; border: 1px solid #e5e7eb; border-radius: 10px; background: #f9fafb; white-space: pre-wrap; word-wrap: break-word; }
    .meta { color:#555; font-size:10pt; margin-bottom: 16px; }
  </style>
</head>
<body>
  <h1>{{ heading }}</h1>
  <div class="meta">{{ url }}</div>
  <pre>{{ json_text }}</pre>
</body>
</html>
""")

# ─────────────────── Builder + writers ──────────────────
def build_and_write_json_file(url: str, provided_transcript: Optional[str], fmt: str) -> Tuple[str, str]:
    """
    Build JSON and write either .txt (raw JSON) or .pdf (pretty JSON wrapped).
    Returns (abs_path, file_name).
    """
    data = build_debranded_json(url, provided_transcript)
    file_id = data.get("id") or safe_token("case_study")
    pretty = json.dumps(data, ensure_ascii=False, indent=2)

    if fmt == "pdf":
        from weasyprint import HTML as WEASY_HTML
        html_doc = PDF_WRAPPER.render(
            title=file_id,
            heading=data.get("meta", {}).get("title", file_id),
            url=data.get("meta", {}).get("url", url),
            json_text=html.escape(pretty),
        )
        pdf_path = os.path.join(OUT_DIR, f"{file_id}.pdf")
        WEASY_HTML(string=html_doc, base_url=".").write_pdf(pdf_path)
        return pdf_path, f"{file_id}.pdf"

    # default: txt
    txt_path = os.path.join(OUT_DIR, f"{file_id}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(pretty)
    return txt_path, f"{file_id}.txt"

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
    fmt = (request.form.get("format") or "txt").strip().lower()
    if fmt not in ("txt","pdf"):
        fmt = "txt"
    try:
        abs_path, file_name = build_and_write_json_file(url, provided_transcript=transcript_text or None, fmt=fmt)
        return render_template_string(
            SUCCESS_HTML,
            file_url=f"/out/{file_name}",
            file_name=file_name,
        )
    except Exception as e:
        try:
            with open(os.path.join(OUT_DIR, "last_error.txt"), "w", encoding="utf-8") as f:
                f.write("Error: " + str(e))
        except Exception:
            pass
        return f"<pre>Error generating JSON:\n{e}\nCheck /out/last_error.txt for details.</pre>", 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=True)
