import os, re, json, datetime
from urllib.parse import urlparse, parse_qs
from typing import Dict, List, Optional

from flask import Flask, request, render_template_string, send_from_directory, abort
from jinja2 import Template
from youtube_transcript_api import YouTubeTranscriptApi
import requests

# ───────────────────────── ENV ─────────────────────────
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # vision-capable
OUT_DIR = os.getenv("OUT_DIR", "out")
os.makedirs(OUT_DIR, exist_ok=True)

# Optional search keys (to pull small trade-press snippets)
BING_SEARCH_KEY      = os.getenv("BING_SEARCH_KEY", "")
BING_SEARCH_ENDPOINT = os.getenv("BING_SEARCH_ENDPOINT", "https://api.bing.microsoft.com/v7.0/search")
SERPAPI_KEY          = os.getenv("SERPAPI_KEY", "")

# Debranding map — extend as needed
DEBRAND_RULES = [
    {"from": r"(?i)\bReese['’]s\b", "to": "candy"},
    {"from": r"(?i)\bReese['’]s Peanut Butter Cups\b", "to": "candy Peanut Butter Cups"},
]

# ───────────────────── App & OpenAI ───────────────────
app = Flask(__name__)
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
        if vid: return vid
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

def fetch_transcript_text(video_id: str) -> str:
    try:
        trs = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            t = trs.find_transcript(["en","en-US"]).fetch()
        except Exception:
            t = YouTubeTranscriptApi.get_transcript(video_id)
        txt = " ".join((seg.get("text") or "").strip() for seg in t if (seg.get("text") or "").strip())
        return txt[:32000]
    except Exception:
        return ""

def get_video_thumbnails(youtube_url: str, max_imgs: int = 3) -> List[str]:
    """Try yt-dlp, else fall back to static thumbnail URLs."""
    urls: List[str] = []
    try:
        import yt_dlp
        ydl_opts = {"quiet": True, "skip_download": True, "extract_flat": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
        thumbs = info.get("thumbnails") or []
        thumbs = sorted(thumbs, key=lambda t: (t.get("height",0)*t.get("width",0)), reverse=True)
        seen = set()
        for t in thumbs:
            u = t.get("url")
            if u and u not in seen:
                urls.append(u); seen.add(u)
            if len(urls) >= max_imgs: break
    except Exception:
        pass
    if not urls:
        vid = video_id_from_url(youtube_url)
        urls = [
            f"https://i.ytimg.com/vi/{vid}/maxresdefault.jpg",
            f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg",
            f"https://i.ytimg.com/vi_webp/{vid}/maxresdefault.webp",
        ][:max_imgs]
    return urls

# ─────────── Trade-press search (optional) ────────────
PUBLISHER_WHITELIST = [
    "adage.com","adweek.com","campaignlive.com","campaignlive.co.uk","lbbonline.com",
    "shots.net","shootonline.com","thedrum.com","musebycl.io","ispot.tv",
    "adsoftheworld.com","adforum.com","businesswire.com","prnewswire.com"
]

def _host_ok(url: str) -> bool:
    try:
        host = urlparse(url).hostname or ""
        return any(dom in host for dom in PUBLISHER_WHITELIST)
    except Exception:
        return False

def http_get_readable(url: str, timeout=10) -> str:
    try:
        r = requests.get(f"https://r.jina.ai/{url}", timeout=timeout)
        if r.ok and len(r.text) > 400: return r.text
    except Exception:
        pass
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent":"Mozilla/5.0"})
        if r.ok: return r.text
    except Exception:
        pass
    return ""

def web_search(query: str, limit: int = 5) -> List[Dict[str,str]]:
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
    # prefer whitelist
    dedup, seen = [], set()
    for it in results:
        u = it.get("url","")
        if not u or u in seen: continue
        if _host_ok(u):
            dedup.append(it); seen.add(u)
    return dedup[:limit]

def gather_trade_snippets(title: str) -> Dict[str, List[str]]:
    if not title: return {"snippets":[], "citations":[]}
    queries = [
        f"{title} Super Bowl ad credits",
        f"{title} director credits",
        f"{title} ad breakdown",
        f"{title} site:adage.com",
        f"{title} site:adweek.com",
        f"{title} site:shootonline.com",
    ]
    pages, cites = [], []
    for q in queries:
        for r in web_search(q, limit=4):
            u = r.get("url","")
            if not u or not _host_ok(u): continue
            txt = http_get_readable(u)
            if not txt: continue
            pages.append((u, txt))
            cites.append(u)
    # Pull a handful of 1–2 sentence snippets
    snips = []
    for (_, txt) in pages:
        for m in re.finditer(r"([^\n\r]{80,240})", txt):
            s = m.group(1).strip()
            if any(k in s.lower() for k in ["director","agency","stunt","gag","montage","spot","super bowl","credits"]):
                snips.append(s[:240])
                if len(snips) >= 6: break
        if len(snips) >= 6: break
    # dedupe citations
    dedup_cites = []
    for c in cites:
        if c not in dedup_cites: dedup_cites.append(c)
    return {"snippets": snips, "citations": dedup_cites[:8]}

# ─────────────── LLM JSON builder (STRICT) ─────────────
SYSTEM_JSON = """
You are a fact-locked ad analyst. Return ONLY valid JSON for a case study with this schema:
{
  "id": string,                       // slug (a-z0-9-_)
  "category": string,                 // generic category (e.g., "Candy")
  "title": string,                    // debranded title
  "source_url": string,               // the YouTube URL
  "summary": string,                  // 2-3 sentences
  "breakdown_beats": [                // 4-10 beats
    {
      "timecode": "m:ss m:ss",
      "dialogue": [string],
      "visuals": string,
      "provenance": ["transcript"|"source_verified_visuals"|"press"],
      "notes": string
    }
  ],
  "visuals_montage_sourced": [        // only if supported by thumbnails or press snippets
    { "description": string, "provenance": ["source_verified_visuals"|"press"] }
  ],
  "why_its_good": {
    "strategy": [string], "craft": [string], "communication": [string]
  },
  "how_to_reuse_big_idea": [string],
  "verbatim_transcript": [ { "timecode": "m:ss", "text": string } ],
  "debranding_rules": [ { "from": string, "to": string } ],
  "schema_version": 1,
  "generated_at": ISO_8601_string
}

RULES:
- Do not guess. If visual actions are not clearly implied by transcript or thumbnails, omit them.
- Keep "provenance" honest: "transcript" for VO lines; "source_verified_visuals" only if the visual is evident in thumbnails; "press" only if supported by the provided press snippets.
- Debrand: Replace brand/product names with generic category terms given in the prompt.
- Output JSON only; no markdown, no commentary.
"""

def build_debranded_json(youtube_url: str, pasted_transcript: Optional[str]) -> Dict:
    client = get_client()
    vid = video_id_from_url(youtube_url)
    meta = fetch_basic_metadata(vid)
    title = meta.get("title","") or "Untitled"
    channel = meta.get("author","") or "Unknown"

    # Transcript
    transcript_text = (pasted_transcript or "").strip()
    if not transcript_text:
        transcript_text = fetch_transcript_text(vid)

    # Thumbnails for light vision grounding
    thumbs = get_video_thumbnails(youtube_url, max_imgs=3)
    vision_parts = [{"type":"image_url","image_url":{"url":u}} for u in thumbs]

    # Optional trade press snippets (as textual evidence only)
    trade = gather_trade_snippets(title)
    trade_snips = trade.get("snippets", [])
    trade_cites = trade.get("citations", [])

    # Debranding rules we want the model to respect
    debrand_rules = [{"from": r["from"], "to": r["to"]} for r in DEBRAND_RULES]

    # Choose a generic category + debranded title seed
    generic_category = "Candy"
    # crude debrand seed for title
    de_title = title
    for rule in DEBRAND_RULES:
        de_title = re.sub(rule["from"], rule["to"], de_title)
    if de_title == title:
        de_title = f"{generic_category} — {title}"

    # Build user content (multimodal)
    research_block = ""
    if trade_snips:
        research_block = "Press snippets (for provenance='press' only):\n" + "\n".join(f"• {s}" for s in trade_snips)
        if trade_cites:
            research_block += "\nCitations:\n" + "\n".join(f"- {u}" for u in trade_cites)

    user_text = f"""
CATEGORY: {generic_category}
DEBRANDED_TITLE_SEED: {de_title}
SOURCE_URL: {youtube_url}
CHANNEL: {channel}

TRANSCRIPT (verbatim/auto; may be imperfect):
{transcript_text[:20000]}

{research_block}

DEBRANDING_RULES (apply in wording):
{json.dumps(debrand_rules, ensure_ascii=False)}
"""

    content = []
    if vision_parts: content.extend(vision_parts)
    content.append({"type":"text","text":user_text})

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role":"system","content":SYSTEM_JSON},{"role":"user","content":content}],
        response_format={"type":"json_object"},
        temperature=0.2,     # accuracy-first
        max_tokens=2200
    )
    data = json.loads(resp.choices[0].message.content or "{}")

    # Post-process: ensure debranding substitutions across text fields
    def _debrand_inplace(obj):
        if isinstance(obj, dict):
            for k,v in obj.items():
                obj[k] = _debrand_inplace(v)
        elif isinstance(obj, list):
            return [_debrand_inplace(x) for x in obj]
        elif isinstance(obj, str):
            s = obj
            for rule in DEBRAND_RULES:
                s = re.sub(rule["from"], rule["to"], s)
            return s
        return obj

    data = _debrand_inplace(data)

    # Ensure required top fields
    if not data.get("id"):
        data["id"] = safe_token(f"{generic_category}_{vid}")[:80]
    if not data.get("category"):
        data["category"] = generic_category
    if not data.get("title"):
        data["title"] = de_title
    if not data.get("source_url"):
        data["source_url"] = youtube_url
    data["schema_version"] = 1
    data["generated_at"] = datetime.datetime.utcnow().isoformat() + "Z"

    # Attach the debranding rules we actually used (human-readable)
    data["debranding_rules"] = [{"from": r["from"], "to": r["to"]} for r in DEBRAND_RULES]

    # Attach trade citations if missing but we used snippets
    if trade_cites and not data.get("why_its_good"):
        data["why_its_good"] = {"strategy":[],"craft":[],"communication":[]}
    if trade_cites:
        data.setdefault("_citations", trade_cites)  # aux field (optional)
    return data

# ─────────────────── HTML Templates ────────────────────
INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>YouTube → De-Branded JSON</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; color:#111; margin: 24px; }
    .card { max-width: 860px; margin: 0 auto; border: 1px solid #e5e7eb; border-radius: 14px; padding: 20px; box-shadow: 0 6px 20px rgba(0,0,0,.05); }
    label { display:block; margin: 10px 0 6px; font-weight: 600; }
    input[type=text], textarea { width:100%; padding:10px 12px; border:1px solid #d1d5db; border-radius:10px; font-size:15px; }
    textarea { min-height: 140px; }
    .btn { background:#111827; color:#fff; border:none; padding: 10px 14px; border-radius: 10px; cursor:pointer; }
    pre { background:#0b1020; color:#d1e7ff; padding:12px; border-radius:10px; overflow:auto; max-height:380px; }
  </style>
</head>
<body>
  <div class="card">
    <h2>YouTube → De-Branded Case Study (JSON)</h2>
    <p>Paste a YouTube URL and optionally a transcript. We’ll return strict JSON for your case study, de-branded to generic names.</p>
    <form method="post" action="/generate_json">
      <label>YouTube URL</label>
      <input name="url" type="text" placeholder="https://www.youtube.com/watch?v=..." required />
      <label style="margin-top:12px">Transcript (optional — paste if you already have it)</label>
      <textarea name="transcript" placeholder="If empty, we’ll try to fetch captions automatically."></textarea>
      <p style="margin-top:14px"><button class="btn" type="submit">Generate JSON</button></p>
    </form>
  </div>
</body>
</html>
"""

RESULT_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>JSON Ready</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; color:#111; margin: 24px; }
    .card { max-width: 980px; margin: 0 auto; border: 1px solid #e5e7eb; border-radius: 14px; padding: 20px; box-shadow: 0 6px 20px rgba(0,0,0,.05); }
    .btn { background:#111827; color:#fff; border:none; padding: 10px 14px; border-radius: 10px; cursor:pointer; text-decoration:none; }
    pre { background:#0b1020; color:#d1e7ff; padding:12px; border-radius:10px; overflow:auto; max-height:520px; }
  </style>
</head>
<body>
  <div class="card">
    <h2>De-Branded JSON generated</h2>
    <p><a class="btn" href="{{ json_url }}" download>Download JSON: {{ json_filename }}</a></p>
    <pre>{{ json_preview }}</pre>
    <p style="margin-top:16px"><a href="/">Generate another</a></p>
  </div>
  <script>(function(){const a=document.createElement('a');a.href="{{ json_url }}";a.download="{{ json_filename }}";document.body.appendChild(a);a.click();a.remove();})();</script>
</body>
</html>
"""
# ─────────────────────── Routes ────────────────────────
@app.get("/health")
def health():
    return "ok", 200

# Update the form to post to /generate (see INDEX_HTML in your file if needed)
@app.get("/")
def index():
    return render_template_string(INDEX_HTML.replace('action="/generate_json"', 'action="/generate"'))

def _handle_generate_json():
    url = request.form.get("url","").strip()
    transcript_text = (request.form.get("transcript") or "").strip()
    data = build_debranded_json(url, transcript_text or None)
    base = data.get("id") or safe_token("case_study")
    fname = f"{base}.json"
    fpath = os.path.join(OUT_DIR, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return render_template_string(
        RESULT_HTML,
        json_url=f"/out/{fname}",
        json_filename=fname,
        json_preview=json.dumps(data, ensure_ascii=False, indent=2)[:100000]
    )

# Accept BOTH endpoints so old links/buttons still work
@app.post("/generate")
def generate_route():
    try:
        return _handle_generate_json()
    except Exception as e:
        try:
            with open(os.path.join(OUT_DIR, "last_error.txt"), "w", encoding="utf-8") as f:
                f.write(str(e))
        except Exception:
            pass
        return f"<pre>Error generating JSON:\n{e}\nCheck /out/last_error.txt for details.</pre>", 400

@app.post("/generate_json")
def generate_json_route():
    try:
        return _handle_generate_json()
    except Exception as e:
        try:
            with open(os.path.join(OUT_DIR, "last_error.txt"), "w", encoding="utf-8") as f:
                f.write(str(e))
        except Exception:
            pass
        return f"<pre>Error generating JSON:\n{e}\nCheck /out/last_error.txt for details.</pre>", 400

# Nice-to-have: GETs on these endpoints just bounce back to the form
@app.get("/generate")
@app.get("/generate_json")
def get_generate_redirect():
    return ('<meta http-equiv="refresh" content="0; url=/" />', 302, {"Location": "/"})

@app.get("/out/<path:filename>")
def get_file(filename):
    try:
        return send_from_directory(OUT_DIR, filename, as_attachment=True)
    except Exception:
        abort(404)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT","8080")), debug=True)
