import os, re, json, html, shutil, subprocess, tempfile, glob
from urllib.parse import urlparse, parse_qs
from typing import Dict, List, Optional, Tuple

from flask import Flask, request, render_template_string, send_from_directory, abort, url_for
from jinja2 import Template
from youtube_transcript_api import YouTubeTranscriptApi
import requests

# ───────────────────────── ENV ─────────────────────────
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # vision-capable
OUT_DIR = os.getenv("OUT_DIR", "out")
os.makedirs(OUT_DIR, exist_ok=True)

# Optional web search keys (helpful but not required)
BING_SEARCH_KEY      = os.getenv("BING_SEARCH_KEY", "")
BING_SEARCH_ENDPOINT = os.getenv("BING_SEARCH_ENDPOINT", "https://api.bing.microsoft.com/v7.0/search")
SERPAPI_KEY          = os.getenv("SERPAPI_KEY", "")

app = Flask(__name__)

# ───────────────────── OpenAI client ───────────────────
from openai import OpenAI
def _llm():
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

def fetch_transcript_text(video_id: str, limit_chars: int = 30000) -> str:
    try:
        trs = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            t = trs.find_transcript(["en", "en-US"]).fetch()
        except Exception:
            t = YouTubeTranscriptApi.get_transcript(video_id)
        buf = " ".join(seg.get("text","") for seg in t if seg.get("text","").strip())
        return buf[:limit_chars]
    except Exception:
        return ""

# ─────────────── Frame extraction (yt-dlp + ffmpeg) ───────────────
def extract_frames(youtube_url: str, case_id: str, fps: float = 2.0, max_frames: int = 16) -> List[str]:
    """
    Downloads the video to a temp file (yt-dlp) and extracts PNG frames with ffmpeg.
    Saves into OUT_DIR/frames/<case_id>/frame_001.png ...
    Returns a list of absolute file paths to frames (capped by max_frames).
    """
    frames_dir = os.path.join(OUT_DIR, "frames", case_id)
    if os.path.isdir(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)

    tmpdir = tempfile.mkdtemp(prefix="grab_")
    try:
        # 1) download best mp4
        video_path = os.path.join(tmpdir, "video.mp4")
        cmd_dl = [
            "yt-dlp",
            "-f", "mp4/bv*+ba/b",   # prefer mp4
            "--no-warnings",
            "--quiet",
            "-o", video_path,
            youtube_url
        ]
        subprocess.run(cmd_dl, check=True)

        # 2) extract frames at fps (capped)
        # We do two passes: first extract all at fps; then trim to max_frames by skipping
        raw_pattern = os.path.join(frames_dir, "raw_%06d.png")
        cmd_ff = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", video_path,
            "-vf", f"fps={fps}",
            raw_pattern
        ]
        subprocess.run(cmd_ff, check=True)

        # 3) keep at most max_frames evenly spaced
        raws = sorted(glob.glob(os.path.join(frames_dir, "raw_*.png")))
        if not raws:
            return []
        if len(raws) <= max_frames:
            # rename to frame_001.png numbering
            out_files = []
            for i, p in enumerate(raws, start=1):
                newp = os.path.join(frames_dir, f"frame_{i:03d}.png")
                os.rename(p, newp)
                out_files.append(newp)
            return out_files
        # pick evenly spaced indices
        idxs = [int(round(i*(len(raws)-1)/(max_frames-1))) for i in range(max_frames)]
        picked = []
        for j, k in enumerate(idxs, start=1):
            src = raws[k]
            dst = os.path.join(frames_dir, f"frame_{j:03d}.png")
            shutil.copy2(src, dst)
            picked.append(dst)
        # cleanup raw
        for p in raws:
            os.remove(p)
        return picked
    except Exception:
        return []
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def frame_urls_for_case(case_id: str) -> List[str]:
    """
    Returns Flask URLs for the saved frames so GPT-4o can fetch them.
    """
    rels = sorted(glob.glob(os.path.join(OUT_DIR, "frames", case_id, "frame_*.png")))
    urls = []
    for p in rels:
        fname = os.path.basename(p)
        urls.append(url_for("serve_frame", case_id=case_id, filename=fname, _external=True))
    return urls

# Serve frames
@app.get("/frames/<case_id>/<path:filename>")
def serve_frame(case_id: str, filename: str):
    path = os.path.join(OUT_DIR, "frames", safe_token(case_id))
    return send_from_directory(path, filename)

# ─────────── (Optional) trade-press context (light) ───────────
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
    dedup, seen = [], set()
    for it in results:
        u = it.get("url","")
        if not u or u in seen: continue
        if _host_ok(u):
            dedup.append(it); seen.add(u)
    return dedup[:limit]

def enrich_from_trades_for_prompt(title: str) -> Dict[str, List[str]]:
    queries = [
        f"{title} Super Bowl ad credits",
        f"{title} director",
        f"{title} agency",
        f"{title} voiceover",
        f"{title} adweek",
        f"{title} adage",
        f"{title} shootonline",
    ]
    pages = []
    for q in queries:
        for r in web_search(q, limit=3):
            u = r.get("url","")
            if _host_ok(u):
                pages.append((u, http_get_readable(u)))
    snips, cites = [], []
    for u, t in pages:
        if not t: continue
        # extract short interesting chunks
        for m in re.finditer(r"([^\n\r]{60,240})", t):
            s = m.group(1).strip()
            if any(k in s.lower() for k in ["director","voice","agency","super bowl","spot","commercial"]):
                snips.append(s[:240]); cites.append(u)
                if len(snips) >= 6: break
        if len(snips) >= 6: break
    # dedupe cites
    uniq = []
    for u in cites:
        if u not in uniq: uniq.append(u)
    return {"snippets": snips[:6], "citations": uniq[:6]}

# ───────────── Concrete action enforcement helpers ─────────────
VAGUE_PAT = re.compile(
    r"\b(people|family|someone|person|group|crowd|audience|they|friends|consumers|various settings|reacts?|celebrates?)\b",
    re.I
)
MUST_HAVE_VERB = re.compile(r"\b(dunks|rams|spits|howls|upends|flips|dives|crashes|throws|shatters|smashes|screams|yells|jumps|runs|pours|stacks|tears)\b", re.I)

def drop_vague(items: List[Dict]) -> List[Dict]:
    cleaned = []
    for it in items:
        desc = (it or {}).get("description","").strip()
        prov = (it or {}).get("provenance",[])
        if not desc or "source_verified_visuals" not in prov:
            continue
        if VAGUE_PAT.search(desc) and not MUST_HAVE_VERB.search(desc):
            continue
        cleaned.append({"description": desc, "provenance": ["source_verified_visuals"]})
    return cleaned

# ───────────── Prompt & LLM JSON builder ─────────────
SOURCE_PRIORITY_PROMPT = """
You are a frame-accurate ad analyst. Output ONLY JSON.

STRICT EVIDENCE:
- You are given multiple video frames (images). Only describe on-screen actions that those frames show.
- Dialog must be literal substrings of the transcript text provided.
- If an action is not visible in the frames, do not claim it.

STYLE:
- Use concrete subjects + strong verbs + objects (e.g., “woman crashes through window”, “dog howls”, “man upends coffee table”).
- Ban generic phrases like “family reacts” or “people celebrate”.

OUTPUT JSON (exact shape):
{
  "meta": { "title": "string", "channel": "string", "url": "string" },
  "big_idea": "string",
  "beat_map": [
    {
      "label": "string",
      "time_start": "MM:SS",
      "time_end": "MM:SS",
      "vo_or_dialog": [ "verbatim lines from transcript only" ],
      "visual": "short, concrete on-screen action if visible; else empty",
      "provenance": ["source_verified_visuals" | "transcript_audio" | "trade_press" | "inferred_low"]
    }
  ],
  "dialogs": [
    { "time": "MM:SS", "line": "verbatim from transcript" }
  ],
  "visuals_montage_sourced": [
    { "description": "short, concrete on-screen action", "provenance": ["source_verified_visuals"] }
  ],
  "sources": [ "urls actually referenced (optional)" ]
}

VALIDATION:
- Every item in visuals_montage_sourced must be visible in at least one provided frame; use “source_verified_visuals”.
- Omit anything uncertain. No wardrobe, names, counts, or props unless clearly visible.
- JSON only. No Markdown.
""".strip()

def vision_payload(frames: List[str], title: str, channel: str, url: str, transcript: str,
                   trade_snips: List[str], trade_urls: List[str]) -> List[dict]:
    parts: List[dict] = []
    for u in frames:
        parts.append({"type":"image_url","image_url":{"url":u}})
    text = [
        f"Title: {title}",
        f"Channel: {channel}",
        f"URL: {url}",
        "",
    ]
    if trade_snips:
        text.append("Trade-press snippets (for factual support only; do not invent visuals):")
        for s in trade_snips:
            text.append(f"• {s}")
        if trade_urls:
            text.append("Links:")
            for lu in trade_urls:
                text.append(f"- {lu}")
        text.append("")
    text.append("Transcript (verbatim):")
    text.append(transcript)
    parts.append({"type":"text","text":"\n".join(text)})
    return parts

def gpt_json(system_prompt: str, user_payload: List[dict]) -> dict:
    resp = _llm().chat.completions.create(
        model=OPENAI_MODEL,
        response_format={"type":"json_object"},
        messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_payload}],
        temperature=0.25,
        max_tokens=2200,
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        return json.loads(raw)
    except Exception:
        start, end = raw.find("{"), raw.rfind("}")
        return json.loads(raw[start:end+1]) if start>=0 and end>=0 else {}

# ───────────── Main builder ─────────────
def build_case_json(youtube_url: str, provided_transcript: Optional[str]) -> dict:
    vid = video_id_from_url(youtube_url)
    meta = fetch_basic_metadata(vid)
    title = meta.get("title") or "Untitled Spot"
    channel = meta.get("author") or "Unknown Channel"
    transcript = (provided_transcript or "").strip()
    if not transcript:
        transcript = fetch_transcript_text(vid)

    case_id = safe_token(f"{title}_{vid}")[:120]

    # 1) Extract frames (2fps, <=16 stills), serve as URLs
    _ = extract_frames(youtube_url, case_id, fps=2.0, max_frames=16)
    frame_urls = frame_urls_for_case(case_id)

    # 2) Optional lightweight trade press (small snippets)
    trade = enrich_from_trades_for_prompt(title)
    trade_snips = trade.get("snippets", [])
    trade_urls  = trade.get("citations", [])

    # 3) First pass JSON
    payload = vision_payload(frame_urls, title, channel, youtube_url, transcript, trade_snips, trade_urls)
    data = gpt_json(SOURCE_PRIORITY_PROMPT, payload)

    # 4) Post-validate & concrete enforcement for visuals_montage_sourced
    data.setdefault("meta", {}).update({"title": title, "channel": channel, "url": youtube_url})
    data.setdefault("visuals_montage_sourced", [])
    concrete = drop_vague(data["visuals_montage_sourced"])

    # If too vague or empty but frames exist, run one rewrite pass with stricter instruction
    if len(concrete) < 6 and len(frame_urls) > 0:
        tighten = """
Return ONLY JSON with the same keys. Your 'visuals_montage_sourced' is too vague.
Rewrite it to list 8–14 concrete on-screen actions that are visible in the provided frames.
Each item must include a specific subject + strong verb + object (e.g., “woman crashes through window”, “dog howls”, “man upends coffee table”).
Do NOT use generic words like “people/family/friends react”.
"""
        resp2 = _llm().chat.completions.create(
            model=OPENAI_MODEL,
            response_format={"type":"json_object"},
            messages=[
                {"role":"system","content":SOURCE_PRIORITY_PROMPT},
                {"role":"user","content":payload},
                {"role":"user","content":tighten}
            ],
            temperature=0.25,
            max_tokens=2200,
        )
        raw2 = resp2.choices[0].message.content or "{}"
        try:
            cand = json.loads(raw2)
            concrete = drop_vague((cand or {}).get("visuals_montage_sourced", []))
            if concrete:
                data["visuals_montage_sourced"] = concrete
        except Exception:
            pass
    else:
        data["visuals_montage_sourced"] = concrete

    # Ensure sources present (merge trade urls if any)
    if "sources" not in data or not isinstance(data["sources"], list):
        data["sources"] = []
    for u in trade_urls:
        if u not in data["sources"]:
            data["sources"].append(u)

    data["id"] = case_id
    return data

# ─────────────────── HTML (UI) ───────────────────
INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>YouTube → JSON (frames)</title>
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
    <h2>YouTube → Structured JSON (with frame analysis)</h2>
    <p class="muted">Paste a YouTube URL and (optionally) a transcript. We’ll sample frames, extract concrete on-screen actions, and return JSON. Choose .txt or .pdf.</p>
    <form method="post" action="/generate">
      <label>YouTube URL</label>
      <input name="url" type="text" placeholder="https://www.youtube.com/watch?v=..." required />
      <label style="margin-top:12px">Transcript (optional)</label>
      <textarea name="transcript" placeholder="Paste full transcript if you have it."></textarea>
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
  <title>Done</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; color:#111; margin: 24px; }
    .card { max-width: 860px; margin: 0 auto; border: 1px solid #e5e7eb; border-radius: 14px; padding: 20px; box-shadow: 0 6px 20px rgba(0,0,0,.05); text-align: center; }
    .btn { background:#111827; color:#fff; border:none; padding: 10px 14px; border-radius: 10px; cursor:pointer; text-decoration: none; }
    code { background:#f3f4f6; padding:2px 6px; border-radius:6px; }
  </style>
</head>
<body>
  <div class="card">
    <h2>Success!</h2>
    <p>Your file should auto-download. If not, click here:</p>
    <p><a href="{{ file_url }}" class="btn" download>Download <code>{{ file_name }}</code></a></p>
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
    pre { padding: 14px; border: 1px solid #e5e7eb; border-radius: 10px; background: #f9fafb; white-space: pre-wrap; word-break: break-word; }
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

# ───────────── Writers ─────────────
def write_json_file(data: dict, fmt: str) -> Tuple[str, str]:
    file_id = data.get("id") or safe_token("case_study")
    pretty = json.dumps(data, ensure_ascii=False, indent=2)
    if fmt == "pdf":
        from weasyprint import HTML as WEASY_HTML
        html_doc = PDF_WRAPPER.render(
            title=file_id,
            heading=data.get("meta",{}).get("title", file_id),
            url=data.get("meta",{}).get("url",""),
            json_text=html.escape(pretty),
        )
        outp = os.path.join(OUT_DIR, f"{file_id}.pdf")
        WEASY_HTML(string=html_doc, base_url=".").write_pdf(outp)
        return outp, f"{file_id}.pdf"
    else:
        outp = os.path.join(OUT_DIR, f"{file_id}.txt")
        with open(outp, "w", encoding="utf-8") as f:
            f.write(pretty)
        return outp, f"{file_id}.txt"

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
    url = (request.form.get("url") or "").strip()
    transcript_text = (request.form.get("transcript") or "").strip()
    fmt = (request.form.get("format") or "txt").strip().lower()
    if fmt not in ("txt","pdf"): fmt = "txt"
    try:
        data = build_case_json(url, provided_transcript=transcript_text or None)
        abs_path, file_name = write_json_file(data, fmt)
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
        return f"<pre>Error generating JSON:\n{e}\nSee /out/last_error.txt</pre>", 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=True)
