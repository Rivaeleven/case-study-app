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

# ───────────────────── LLM JSON GEN (with montage visuals) ─────────────────────
from openai import OpenAI
def _llm_client():
    return OpenAI()  # uses OPENAI_API_KEY

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
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
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
        # permissive fallback
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
    data.setdefault("sources", research_urls[:8])

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
