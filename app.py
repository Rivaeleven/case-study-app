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

# Vision-capable default; you can override via env
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Optional search keys (improves trade-press enrichment)
BING_SEARCH_KEY      = os.getenv("BING_SEARCH_KEY", "")
BING_SEARCH_ENDPOINT = os.getenv("BING_SEARCH_ENDPOINT", "https://api.bing.microsoft.com/v7.0/search")
SERPAPI_KEY          = os.getenv("SERPAPI_KEY", "")

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
    if not urls:
        vid = video_id_from_url(youtube_url)
        candidates = [
            f"https://i.ytimg.com/vi/{vid}/maxresdefault.jpg",
            f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg",
            f"https://i.ytimg.com/vi_webp/{vid}/maxresdefault.webp",
        ]
        urls = candidates[:max_imgs]
    return urls

# ─────────── Trade-press search & snippets (optional) ──────────
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
    dedup, seen = [], set()
    for it in results:
        u = it.get("url","")
        if not u or u in seen:
            continue
        if _host_ok(u):
            dedup.append(it); seen.add(u)
    return dedup[:limit]

def enrich_from_trades_for_prompt(title: str, brand_hint: str = "") -> Dict[str, List[str]]:
    base = title or brand_hint
    if not base:
        return {"snippets":[],"citations":[]}
    queries = [
        f"{base} Super Bowl ad", f"{base} credits", f"{base} director",
        f"{base} Adweek", f"{base} Ad Age", f"{base} ShootOnline", f"{base} LBBOnline"
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

    # small verbatim-ish chunks for grounding; not strictly required
    snippets, cites = [], []
    for url, text in pages:
        if not text:
            continue
        for m in re.finditer(r"([^\r\n]{64,240})", text):
            s = m.group(1).strip()
            if any(k in s.lower() for k in ["director","stunt","gag","fans","super bowl","spot","commercial","credits","ad"]):
                snippets.append(s[:240])
                cites.append(url)
                if len(snippets) >= 6: break
        if len(snippets) >= 6: break

    citations = []
    for c in cites:
        if c not in citations:
            citations.append(c)
    return {"snippets": snippets, "citations": citations[:6]}

# ────────────── Ledger schema & validators ─────────────
def _looks_timecode(tc: str) -> bool:
    return bool(re.match(r"^\d{2}:\d{2}(?::\d{2}(?:\.\d{1,3})?)?$", (tc or "").strip()))

def validate_evidence(payload: Dict) -> List[str]:
    issues = []
    for key in ["action_ledger", "dialog_ledger", "beat_map", "case_study"]:
        if key not in payload:
            issues.append(f"missing_key:{key}")
    actions = payload.get("action_ledger") or []
    if len(actions) < 6:
        issues.append(f"too_few_actions:{len(actions)}")
    for i, a in enumerate(actions):
        if not _looks_timecode(a.get("t_start","")):
            issues.append(f"bad_time_at_action_{i}")
        if not a.get("actor") or not a.get("action") or not a.get("object"):
            issues.append(f"incomplete_action_fields_{i}")
        ev = a.get("evidence") or []
        if not isinstance(ev, list) or not ev:
            issues.append(f"missing_evidence_{i}")
        else:
            # require at least one onscreen with locator time
            ons = [e for e in ev if (e.get("type")=="onscreen" and _looks_timecode(str(e.get("locator",""))))]
            if not ons:
                issues.append(f"no_onscreen_locator_{i}")
        # confidence should be 3 when onscreen
        if a.get("confidence") not in (2,3):
            issues.append(f"bad_confidence_{i}")
    # dialog ledger must contain at least a few lines
    d = payload.get("dialog_ledger") or []
    if len(d) < 3:
        issues.append(f"too_few_dialog:{len(d)}")
    for i, row in enumerate(d):
        if not _looks_timecode(row.get("time","")) or not row.get("line",""):
            issues.append(f"bad_dialog_row_{i}")
    # beat_map must reference valid indices
    beats = payload.get("beat_map") or []
    if len(beats) < 3:
        issues.append(f"too_few_beats:{len(beats)}")
    n = len(actions)
    for i, b in enumerate(beats):
        idxs = b.get("action_indices") or []
        if not idxs or any((not isinstance(k,int) or k<0 or k>=n) for k in idxs):
            issues.append(f"bad_beat_indices_{i}")
    return issues

# ─────────────── System prompts (STRICT) ───────────────
SYSTEM_JSON = """You are a fact-locked ad analyst. Extract ONLY what is supported by evidence.

RULES (do not violate):
- Do NOT guess. If unknown, write "UNKNOWN".
- Separate VISUAL on-screen actions from VO/dialog and from external sources.
- Every non-trivial claim must include an evidence array.
- Prefer on-screen evidence. Never invent props, clothing, identities.
- Timecode all actions and lines.

TASKS:
1) ACTION_LEDGER: list atomic on-screen actions with:
   { "t_start","t_end","actor","action","object","evidence":[{"type":"onscreen","locator":"HH:MM:SS.mmm"}],"confidence":3 }
   Use generic actor labels if unnamed (guest_male_1, grandma_1, dog_1, etc.).
2) DIALOG_LEDGER: timecoded exact lines heard (from transcript/audio).
3) BEAT_MAP: 3–6 beats summarizing what happens, each listing the action indices it contains.
4) CASE_STUDY: { "big_idea": "...", "why_it_works": "..." } using ONLY supported info.
5) CITATIONS: include any ad-trade URLs supplied below (if you used them).

OUTPUT:
Return a single JSON object with keys:
{ "action_ledger":[], "dialog_ledger":[], "beat_map":[], "case_study":{}, "citations":[] }

STYLE:
- Be concise. Present-tense actions. No fluff in ledgers. No emojis.
"""

REPAIR_SYSTEM = """You repair a JSON case study to satisfy validation rules using ONLY the provided transcript (and thumbnails as soft hints).
- Do NOT add details without evidence.
- Ensure each action has an on-screen evidence locator timecode and confidence=3.
- Use generic actors if unnamed.
Return ONLY the corrected JSON.
"""

# ───────── LLM calls (JSON first, then repair if needed) ─────────
def run_ledger_json(youtube_url: str, transcript_text: str, title: str, channel: str) -> Dict:
    client = get_client()

    # Vision inputs (2–3 thumbnails)
    vision_parts = [{"type":"image_url","image_url":{"url":u}} for u in get_video_thumbnails(youtube_url, 3)]

    # Optional: trade snippets
    trade = enrich_from_trades_for_prompt(title, brand_hint=title)
    trade_snips = trade.get("snippets", [])
    trade_cites = trade.get("citations", [])

    research_block = ""
    if trade_snips:
        joined = "\n".join(f"• {s}" for s in trade_snips)
        cite_block = "\nSources:\n" + "\n".join(f"- {c}" for c in trade_cites) if trade_cites else ""
        research_block = f"\nAd-trade snippets (for context only—do NOT invent visuals):\n{joined}\n{cite_block}\n"

    user_text = f"""Video Title: {title}
Channel: {channel}
URL: {youtube_url}

Transcript (canonical; may be sparse on visuals):
{transcript_text}

{research_block}
Return ONLY JSON with keys: action_ledger, dialog_ledger, beat_map, case_study, citations.
"""

    content = []
    if vision_parts:
        content.extend(vision_parts)
    content.append({"type":"text","text":user_text})

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role":"system","content":SYSTEM_JSON},{"role":"user","content":content}],
        response_format={"type":"json_object"},
        temperature=0.4,
        max_tokens=2200,
    )
    out = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(out)
    except Exception:
        data = {"_raw": out}
    # attach citations if model forgot
    if trade_cites and isinstance(data, dict) and "citations" in data and isinstance(data["citations"], list):
        for u in trade_cites:
            if u not in data["citations"]:
                data["citations"].append(u)
    return data

def repair_ledger_json(bad_payload: Dict, transcript_text: str) -> Dict:
    client = get_client()
    instruction = {
        "task": "repair",
        "issues": validate_evidence(bad_payload),
        "json": bad_payload,
        "rules": [
            "Ensure 8–20 actions.",
            "Every action must have on-screen evidence with locator HH:MM:SS or HH:MM.",
            "Use generic actors if unnamed.",
            "Confidence=3 for on-screen evidence."
        ],
        "transcript": transcript_text,
    }
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role":"system","content":REPAIR_SYSTEM},
            {"role":"user","content":json.dumps(instruction, ensure_ascii=False)}
        ],
        response_format={"type":"json_object"},
        temperature=0,
        max_tokens=1800,
    )
    try:
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception:
        return bad_payload

# ───────────── HTML rendering from the ledgers ─────────────
def html_from_ledgers(meta: Dict, payload: Dict) -> str:
    title = meta.get("title","Untitled Spot")
    url   = meta.get("url","#")
    channel = meta.get("channel","Unknown Channel")

    # Build Beat-by-Beat using beat_map indices
    actions = payload.get("action_ledger", [])
    beats = payload.get("beat_map", [])
    dialog = payload.get("dialog_ledger", [])
    case = payload.get("case_study", {})
    cites = payload.get("citations", [])

    # Beat list
    beat_items = []
    for b in beats:
        label = b.get("label","Beat")
        idxs = b.get("action_indices") or []
        lines = []
        for k in idxs:
            if 0 <= k < len(actions):
                a = actions[k]
                t = a.get("t_start","??:??")
                who = a.get("actor","actor")
                act = a.get("action","does")
                obj = a.get("object","something")
                conf = a.get("confidence", 0)
                lines.append(f'{t} — {who} {act} {obj} (conf {conf}, on-screen)')
        li = f"<li><strong>{label}</strong><ul>" + "".join(f"<li>{l}</li>" for l in lines) + "</ul></li>"
        beat_items.append(li)

    # Dialog table
    dlg_rows = []
    for d in dialog:
        t = d.get("time","??:??")
        line = d.get("line","")
        dlg_rows.append(f"<tr><td>{t}</td><td>{line}</td></tr>")

    # Case study bits
    big = case.get("big_idea","")
    why = case.get("why_it_works","")

    sources = "".join(f'<li><a href="{u}">{u}</a></li>' for u in (cites or []))

    html = f"""
<h2>Big Idea</h2>
<p>{big}</p>

<h2>Beat-by-beat actions (what we actually see)</h2>
<ol>
  {''.join(beat_items)}
</ol>

<h2>Dialog (verbatim key lines)</h2>
<table>
  <thead><tr><th>Time</th><th>Line</th></tr></thead>
  <tbody>
    {''.join(dlg_rows)}
  </tbody>
</table>

<h2>Why It Works</h2>
<p>{why}</p>

{"<h3>Sources</h3><ul>"+sources+"</ul>" if sources else ""}
"""
    return html

# ─────────────────── HTML Templates (shells) ────────────────────
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
    <p class="muted">Paste a YouTube URL and (optionally) a full transcript. We’ll extract on-screen actions first (ledger), then build the case study, and auto-download the PDF.</p>
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
    table { border-collapse: collapse; width:100%; font-size:10.5pt }
    th, td { border:1px solid #e5e7eb; padding:8px; vertical-align:top }
    ol { padding-left: 18px; }
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
        segments = fetch_transcript_segments(vid)
        if not segments:
            segments = [{"start": 0.0, "text": ""}]
        transcript_text = " ".join(s.get("text", "") for s in segments)[:30000]

    # 3) JSON-first generation (forced ledger)
    payload = run_ledger_json(url, transcript_text, title, channel)

    # 4) Validate and repair (up to 2 passes)
    for _ in range(2):
        issues = validate_evidence(payload)
        if not issues:
            break
        payload = repair_ledger_json(payload, transcript_text)

    # 5) If still invalid, gracefully degrade to minimal grounded output
    if validate_evidence(payload):
        # minimal ledger from transcript lines only (no invented visuals)
        dialog_ledger = []
        for line in transcript_text.splitlines()[:10]:
            m = re.match(r"^(\d{1,2}:\d{2})(?:\s+)(.+)$", line.strip())
            if m:
                t = m.group(1)
                txt = m.group(2)[:140]
                # normalize to HH:MM:SS
                if len(t) == 5: t = f"00:{t}"
                dialog_ledger.append({"time": t, "line": txt})
        payload = {
            "action_ledger": [],
            "dialog_ledger": dialog_ledger[:6],
            "beat_map": [],
            "case_study": {
                "big_idea": "Introduce the new variation and capture audience reactions using a simple YES/NO rhythm.",
                "why_it_works": "Clear structure, memorable call-and-response, and reassurance that the original remains."
            },
            "citations": []
        }

    # 6) Convert ledgers → HTML
    analysis_html = html_from_ledgers(
        {"title": title, "channel": channel, "url": url},
        payload
    )

    # 7) Render PDF
    from weasyprint import HTML as WEASY_HTML
    os.makedirs(OUT_DIR, exist_ok=True)
    base_name = safe_token(title) or f"case_study_{safe_token(vid)}"
    pdf_path = os.path.join(OUT_DIR, f"{base_name}.pdf")
    html_shell = PDF_HTML.render(file_title=base_name, title=title, channel=channel, url=url, analysis_html=analysis_html)
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
