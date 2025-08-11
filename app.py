import os, re, json
from urllib.parse import urlparse, parse_qs, quote_plus
from typing import Optional, Dict, List, Tuple

from flask import Flask, request, render_template_string, send_from_directory, abort
from youtube_transcript_api import YouTubeTranscriptApi
from jinja2 import Template
from pydantic import BaseModel, Field
import requests

# ───────────────────────── ENV ─────────────────────────
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-5")  # default to GPT‑5
OUT_DIR         = os.getenv("OUT_DIR", "out")
REPAIR_PASSES   = int(os.getenv("REPAIR_PASSES", "2"))     # try to repair twice
OPENAI_FALLBACK = os.getenv("OPENAI_FALLBACK", "1") == "1" # fallback to gpt-4o once on error
os.makedirs(OUT_DIR, exist_ok=True)

# ───────────────────── OpenAI (lazy) ───────────────────
from openai import OpenAI
def get_client():
    # Lazy init so the app boots even if the key is missing; errors occur at call time instead.
    return OpenAI()  # reads OPENAI_API_KEY from env

def _create_with_fallback(**create_kwargs):
    """Call OpenAI; if GPT‑5 errors out and fallback is enabled, retry once on gpt-4o."""
    client = get_client()
    try:
        return client.chat.completions.create(**create_kwargs)
    except Exception as e:
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

# ───────────── Simple external credit scraping ─────────
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

def scrape_ispot_by_title(title: str) -> Dict[str, str]:
    """Try iSpot.tv: search by title → first ad page → regex credit extraction."""
    try:
        q = quote_plus(title or "")
        if not q:
            return {}
        search_url = f"{JINA}www.ispot.tv/search?q={q}"
        rs = requests.get(search_url, timeout=15)
        if not rs.ok:
            return {}
        ad_paths = re.findall(r"/ad/\w+/[a-z0-9\-]+", rs.text, flags=re.IGNORECASE)
        if not ad_paths:
            return {}
        ad_url = f"{JINA}www.ispot.tv{ad_paths[0]}"
        rp = requests.get(ad_url, timeout=15)
        if not rp.ok:
            return {}
        page = rp.text
        out = _extract_credit_fields(page)
        if "product" not in out:
            m = re.search(r"(?:Advertiser)\s*:\s*([^\n\r;|]+)", page, flags=re.IGNORECASE)
            if m: out["product"] = m.group(1).strip()
        return out
    except Exception:
        return {}

def scrape_shots_by_title(title: str) -> Dict[str, str]:
    """Try shots.net search page for credit-like lines."""
    try:
        q = quote_plus(title or "")
        if not q:
            return {}
        url = f"{JINA}www.shots.net/search?q={q}"
        r = requests.get(url, timeout=15)
        if not r.ok:
            return {}
        return _extract_credit_fields(r.text)
    except Exception:
        return {}

def gather_auto_hints(title: str, video_id: str) -> Dict[str, str]:
    """Aggregate credits from YouTube description + iSpot + shots."""
    combined: Dict[str, str] = {}
    desc = fetch_youtube_description(video_id)
    combined.update({k: v for k, v in _extract_credit_fields(desc).items() if v})

    ispot = scrape_ispot_by_title(title or "")
    for k, v in ispot.items():
        if v and k not in combined:
            combined[k] = v

    shots = scrape_shots_by_title(title or "")
    for k, v in shots.items():
        if v and k not in combined:
            combined[k] = v

    return combined

def format_auto_hints(credits: Dict[str, str]) -> str:
    parts = []
    if credits.get("agency"): parts.append(f"Agency: {credits['agency']}")
    if credits.get("director"): parts.append(f"Director: {credits['director']}")
    if credits.get("product"): parts.append(f"Product: {credits['product']}")
    if credits.get("campaign"): parts.append(f"Campaign: {credits['campaign']}")
    return "; ".join(parts)

# ────────────────────── Naming model ───────────────────
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
You will receive a YouTube URL, title, channel, a timecoded transcript, and optional user 'hints'.
Prefer the 'hints' for proper nouns and specific credits, but if hints are absent or appear inconsistent, leave fields as 'Unknown'.
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
- Anchor timestamps to the timecoded transcript.
- 'line' must be a verbatim substring of either the transcript OR the hints; if not possible, keep it short and append ' [inferred]'.
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

# ───────────────────── LLM Helpers ─────────────────────
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
        return {"_error": f"llm_json failed: {e}"}

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
        except Exception as e:
            return {"_error": f"llm_json_structured failed: {e}"}

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
    """Loose verbatim check (ignore case/punctuation/extra whitespace)."""
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
        if not _looks_like_time(sc.get("start")):
            errs.append(f"bad_scene_time_at_{i}")
    for i, row in enumerate(script):
        if not _looks_like_time(row.get("time")):
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
    <p class="muted">Paste a YouTube URL. Optionally add naming overrides and <strong>Hints</strong> (credits, key lines, beats, supers). We’ll return a PDF, save a JSON sidecar, and link to it for you to check.</p>
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
    <p>The case study PDF has been generated. You can download it below.</p>
    <p><a href="{{ pdf_url }}" class="btn" download>Download PDF: {{ pdf_filename }}</a></p>
    <p class="muted">To check the AI's output for accuracy, you can also view the raw JSON sidecar.</p>
    <p><a href="{{ json_url }}" class="btn">View Raw JSON</a></p>
    <p style="margin-top:20px"><a href="/">Generate another video case study</a></p>
  </div>
</body>
</html>
"""

# ─────────────────── Builder pipeline ──────────────────
def _fill_from_title_if_possible(naming: Naming, title: str) -> Naming:
    """Light heuristic: try to fill product/commercial from oEmbed title if Unknown."""
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

def build_case_study(url: str, overrides: Optional[Dict[str, str]] = None) -> Tuple[str, str]:
    """
    Main pipeline to build the case study.
    Returns: (path_to_pdf, path_to_json)
    """
    # Extract and fetch metadata
    vid = video_id_from_url(url)
    meta = fetch_basic_metadata(vid)

    # Transcript (guard for no captions)
    segments = fetch_transcript_segments(vid)
    if not segments:
        segments = [{"start": 0.0, "text": ""}]
    timecoded = "\n".join(f"{_format_time(s['start'])}  {s['text']}" for s in segments[:220])
    transcript_text = " ".join(s["text"] for s in segments)[:22000]

    # Auto-scrape credits before LLM
    auto_credits = gather_auto_hints(meta.get("title", ""), vid)
    auto_hints = format_auto_hints(auto_credits)

    # Hints (user-provided) — merged with auto hints
    user_hints = (overrides or {}).get("_hints", "")
    merged_hints = "; ".join([p for p in [auto_hints, user_hints] if p]).strip()

    # Prompt blocks
    user_block = ANALYSIS_USER_TEMPLATE.render(
        url=url, title=meta.get("title",""), author=meta.get("author",""),
        timecoded=timecoded, transcript=transcript_text
    )
    json_user_block = user_block + ("\n\nHINTS:\n" + merged_hints if merged_hints else "")

    # 1) Naming (cautious) + overrides + tiny title heuristic
    raw_naming = llm_json(NAMING_SYSTEM, user_block + ("\n\nHINTS:\n" + merged_hints if merged_hints else "")) or {}
    overrides = overrides or {}
    raw_naming.update({k:v for k,v in overrides.items() if v and k != "_hints"})
    naming = Naming(
        agency=raw_naming.get("agency","Unknown"),
        product=raw_naming.get("product","Unknown"),
        campaign=raw_naming.get("campaign","Unknown"),
        commercial=raw_naming.get("commercial", meta.get("title","Unknown")),
        director=raw_naming.get("director","Unknown"),
    )
    naming = _fill_from_title_if_possible(naming, meta.get("title",""))

    # 2) Structured JSON (scenes + script + brand)
    json_payload = llm_json_structured(STRUCTURED_JSON_SPEC, json_user_block)
    # Ensure required keys exist
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

    # 3) Validation + repair passes
    for _ in range(REPAIR_PASSES):
        issues = validate_json(json_payload, transcript_text, merged_hints)
        if not issues:
            break
        json_payload = repair_json_with_model(json_payload, transcript_text, merged_hints)

    # 4) Analysis strictly from JSON + transcript
    analysis_html = llm_html_from_json(ANALYSIS_FROM_JSON, json_payload, transcript_text)

    # Save sidecar JSON
    json_filename = naming.filename().replace(".pdf", ".json")
    json_path = os.path.join(OUT_DIR, json_filename)
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # 5) Render PDF (lazy import)
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
    )
    pdf_path = os.path.join(OUT_DIR, naming.filename())
    WEASY_HTML(string=html).write_pdf(pdf_path)
    return pdf_path, json_path

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
    try:
        pdf_path, json_path = build_case_study(url, overrides=overrides)
        pdf_filename = os.path.basename(pdf_path)
        json_filename = os.path.basename(json_path)
        return render_template_string(
            SUCCESS_HTML,
            pdf_url=f"/out/{pdf_filename}",
            pdf_filename=pdf_filename,
            json_url=f"/out/{json_filename}"
        )
    except Exception as e:
        # Write a debug file so you can inspect errors on Render
        try:
            with open(os.path.join(OUT_DIR, "last_error.txt"), "w", encoding="utf-8") as f:
                f.write(str(e))
        except Exception:
            pass
        return f"<pre>Error generating case study:\n{e}\nCheck Logs and /out/last_error.txt for details.</pre>", 400

if __name__ == "__main__":
    # Local dev
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=True)


