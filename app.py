import os, re, json
from urllib.parse import urlparse, parse_qs
from typing import Optional, Dict, List, Tuple

from flask import Flask, request, render_template_string, send_from_directory, abort
from youtube_transcript_api import YouTubeTranscriptApi
from jinja2 import Template
from pydantic import BaseModel, Field
import requests

# ───────────────────────── ENV ─────────────────────────
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o")   # good default for detail
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
    return OpenAI()  # reads OPENAI_API_KEY

def _create_with_fallback(**create_kwargs):
    client = get_client()
    try:
        return client.chat.completions.create(**create_kwargs)
    except Exception:
        if OPENAI_FALLBACK:
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
    # ad trades / credits
    "adage.com", "adweek.com", "campaignlive.com", "campaignlive.co.uk",
    "shots.net", "lbbonline.com", "thedrum.com", "musebycl.io",
    "shootonline.com", "adforum.com", "adsoftheworld.com", "ispot.tv",
    # brand newsroom / press
    "thehersheycompany.com", "businesswire.com", "prnewswire.com",
]

def is_whitelisted(url: str) -> bool:
    try:
        host = (urlparse(url).hostname or "").lower()
        return any(host.endswith(d) or d in host for d in PUBLISHER_WHITELIST)
    except Exception:
        return False

def http_get_readable(url: str, timeout=15):
    """Try Jina Reader proxy first for clean text, then direct."""
    try:
        prox = f"https://r.jina.ai/{url}"
        r = requests.get(prox, timeout=timeout)
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

def web_search(query: str, limit: int = 8) -> List[Dict[str, str]]:
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

    dedup, seen = [], set()
    for it in results:
        u = it.get("url","")
        if not u or u in seen:
            continue
        seen.add(u)
        if is_whitelisted(u):
            dedup.append(it)
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

# ───────────── Credits extraction helpers ──────────────
VOICEOVER_PAT = r"(?:Voice[- ]?over|VO|Narration|Narrator)\s*[:\-]\s*([^\n\r;|,]+)"
DIRECTOR_PAT  = r"(?:Director|Directed by)\s*[:\-]\s*([^\n\r;|,]+)"
AGENCY_PAT    = r"(?:Agency|Creative Agency)\s*[:\-]\s*([^\n\r;|,]+)"

def extract_credit(text: str, pattern: str) -> List[str]:
    try:
        hits = re.findall(pattern, text or "", flags=re.IGNORECASE)
        return [h.strip() for h in hits if h.strip()]
    except Exception:
        return []

def enrich_from_trades(title: str, brand_hint: str = "") -> Dict[str, List[str]]:
    """
    Search ad trades/brand press with focused queries; extract credits.
    Returns: { 'voiceover': [...], 'director': [...], 'agency': [...], 'sources': [...] }
    """
    if not title and not brand_hint:
        return {}

    base = title or brand_hint
    queries = [
        f"site:adweek.com {base}",
        f"site:adage.com {base}",
        f"site:lbbonline.com {base}",
        f"site:shots.net {base}",
        f"site:adforum.com {base}",
        f"site:adsoftheworld.com {base}",
        f"site:ispot.tv {base}",
        f"site:thehersheycompany.com {brand_hint or base}",
        f"{base} Super Bowl voiceover",
        f"{base} Super Bowl director",
        f"{brand_hint} ad agency credits" if brand_hint else base,
    ]

    seen_urls, pages = set(), []
    for q in queries:
        for res in web_search(q, limit=6):
            u = res.get("url", "")
            if not u or u in seen_urls:
                continue
            if not is_whitelisted(u):
                continue
            seen_urls.add(u)
            pages.append((u, http_get_readable(u)))

    bag = {"voiceover": {}, "director": {}, "agency": {}}
    for url, text in pages:
        if not text:
            continue
        for who in extract_credit(text, VOICEOVER_PAT):
            bag["voiceover"].setdefault(who, []).append(url)
        for who in extract_credit(text, DIRECTOR_PAT):
            bag["director"].setdefault(who, []).append(url)
        for who in extract_credit(text, AGENCY_PAT):
            bag["agency"].setdefault(who, []).append(url)

    def pick_best(d: Dict[str, List[str]]) -> Tuple[str, List[str]]:
        if not d: return "Unknown", []
        return max(d.items(), key=lambda kv: len(kv[1]))

    voice, voice_srcs = pick_best(bag["voiceover"])
    direc, direc_srcs = pick_best(bag["director"])
    agcy,  agcy_srcs  = pick_best(bag["agency"])

    return {
        "voiceover": [voice] if voice != "Unknown" else [],
        "director": [direc] if direc != "Unknown" else [],
        "agency":  [agcy]  if agcy  != "Unknown" else [],
        "sources": list({*voice_srcs, *direc_srcs, *agcy_srcs})
    }

# ─────────────────────── Prompts ───────────────────────
NAMING_SYSTEM = """
Extract (or infer cautiously) the work's naming fields from the provided text.
We will give you a YouTube URL, title and channel, plus transcript.
Return ONLY compact JSON with keys: agency, product, campaign, commercial, director.
If uncertain, use 'Unknown'. Prefer the video title for 'commercial' when unclear.
""".strip()

DETAILED_SCENE_SPEC = """
You are building a director-level breakdown of a TV commercial from a transcript preview (timecoded snippet) + full plain transcript.
Your output will be parsed as JSON. Be SPECIFIC and CONCRETE. No generic phrases like "people react"—spell out *what happens* on screen.

Return ONLY this STRICT JSON object:

{
  "video": {"url": string, "title": string, "channel": string},
  "naming": {"agency": string, "product": string, "campaign": string, "commercial": string, "director": string},
  "scenes": [
    {
      "start": "mm:ss",
      "end": "mm:ss" | null,
      "visual": string,
      "audio": string,
      "on_screen_text": [string],
      "purpose": string
    }
  ],
  "script": [
    {
      "time": "mm:ss",
      "speaker": string | null,
      "line": string,
      "directors_notes": string,
      "strategy_note": string
    }
  ],
  "brand": {
    "product": string,
    "distinctive_assets": [string],
    "claims": [string],
    "ctas": [string]
  }
}

Rules and tone:
- Aim for 10–18 scenes in a ~30s spot (merge micro-beats if needed).
- Anchor times to transcript when possible; if estimating, add " [inferred]" to the field you estimated.
- "visual" must describe *specific* actions, props, and reactions (e.g., "man dunks his face into a crockpot of chili").
- "audio" should include VO/SFX/music cues; use short verbatim where possible.
- "line" MUST be a verbatim substring of the transcript when possible; otherwise keep it brief and append " [inferred]".
- Use short, tight sentences. No buzzwords.
""".strip()

# ★ NEW: Chaos Montage expansion step
CHAOS_MONTAGE_SPEC = """
You are given:
- The plain transcript for a commercial (canonical dialogue).
- The current scene list (already extracted).

Task:
Identify every QUICK-CUT gag, physical bit, background reaction, visual joke, stunt, prop-gag, spit-take, crash, fall, explosion, or micro-beat that likely appears in a fast 'chaos montage' sequence. These are often 0.3–1.0s shots layered around lines like "No!" / "Yes!" / "Wait!" or similar emotional pivots.

Return ONLY:
{
  "scenes_augmented": [
    {
      "start": "mm:ss",
      "end": "mm:ss" | null,
      "visual": "extremely concrete physical action (who/what, props, motion, environment). If timing is estimated, add ' [inferred]'.",
      "audio": "VO/SFX/music, short verbatim if possible; append ' [inferred]' if not certain.",
      "on_screen_text": [],
      "purpose": "why this micro-beat exists (e.g., heighten panic, punctuate joke, add absurdity, product beauty, etc.)"
    }
  ]
}

Rules:
- Prefer many short micro-beats (0.3–1.0s implied). DO NOT write generic "people react". Spell out actions (e.g., "grandma spits coffee", "man dives through drywall", "dog howls", "table flips", "guy dunks face in chili").
- Keep times aligned to nearest sensible second from transcript beats. If unknown: choose a plausible nearby time and add ' [inferred]'.
- Do not duplicate existing scenes; add new ones that increase visual specificity.
- Keep language tight, concrete, and filmic (blocking, props, motion).
""".strip()

ANALYSIS_FROM_JSON = """
You will receive a JSON (scenes + script) and the transcript text.
Write VALID HTML for these sections ONLY, using facts from JSON/transcript:
<h2>What Makes It Compelling & Unique</h2>
<h2>Creative Strategy Applied</h2>
<h2>Campaign Objectives, Execution & Reach</h2>
<h2>Performance & Audience Impact</h2>
<h2>Why It’s Award-Worthy</h2>
<h2>Core Insight (The 'Big Idea')</h2>
Mark any deduction as [inferred]. Keep it tight and specific.
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
# --- LLM Helper Function ---
def analyze_transcript(transcript_text: str) -> str:
    from openai import OpenAI
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are an award-winning advertising critic.
When expanding transcripts into scene-by-scene breakdowns:

• Do not just repeat dialogue.  
• Vividly describe the VISUAL ACTIONS on screen — sets, props, crowd reactions, comedic chaos, product shots.  
• Expand sparse beats into full cinematic description, especially absurd or over-the-top reactions typical of Super Bowl spots.  
• Use ad-trade language (e.g. “physical freak-out montage,” “grandma does a spit take,” “director Harold Einstein leans into absurdist escalation”).  
• Always provide:  
   – Timecoded scene description  
   – What we see (visuals)  
   – What we hear (dialogue, VO, music, sfx)  
   – Purpose/strategy of the beat (brand intent)  

After the scene breakdown, include sections:
- Annotated Script (verbatim + critic notes)
- What Makes It Compelling & Unique
- Creative Strategy Applied
- Campaign Objectives, Execution & Reach
- Performance & Audience Impact
- Why It’s Award-Worthy
- Core Insight (The Big Idea)
"""
            },
            {
                "role": "user",
                "content": f"Transcript:\n{transcript_text}\n\nNow expand into a full ad breakdown."
            }
        ],
        temperature=0.7,
        max_tokens=2000
    )

    return response.choices[0].message["content"]


# ─────────────────── Validation / Repair ─────────────
def _is_substring_loosely(s: str, blob: str) -> bool:
    s_norm = re.sub(r"[^\w\s]", "", (s or "").strip().lower())
    blob_norm = re.sub(r"[^\w\s]", "", (blob or "").strip().lower())
    return s_norm and (s_norm in blob_norm)

def _looks_like_time(ts: str) -> bool:
    return bool(re.match(r"^\d{2}:\d{2}$", (ts or "").strip()))

def validate_json(payload: Dict, transcript_blob: str) -> List[str]:
    errs = []
    scenes = payload.get("scenes") or []
    script = payload.get("script") or []
    if len(scenes) < 8:
        errs.append(f"too_few_scenes:{len(scenes)}")
    if len(script) < 8:
        errs.append(f"too_few_script_lines:{len(script)}")
    for i, sc in enumerate(scenes):
        if not _looks_like_time(sc.get("start", "")):
            errs.append(f"bad_scene_time_at_{i}")
    for i, row in enumerate(script):
        if not _looks_like_time(row.get("time", "")):
            errs.append(f"bad_script_time_at_{i}")
        line = (row or {}).get("line", "")
        if line and not _is_substring_loosely(line, transcript_blob):
            errs.append(f"non_verbatim_script_line_at_{i}")
    return errs

def repair_json_with_model(bad_payload: Dict, transcript_blob: str) -> Dict:
    instruction = {
        "task": "repair",
        "issues": validate_json(bad_payload, transcript_blob),
        "rules": [
            "Ensure 10–18 scenes and 8–16 script lines.",
            "Every 'line' must be a substring of transcript; otherwise append ' [inferred]' and keep it short.",
            "Preserve timestamps and supers where possible.",
            "Make 'visual' descriptions concrete and specific (physical actions, props, reactions)."
        ],
        "transcript": transcript_blob,
        "json": bad_payload,
    }
    try:
        resp = _create_with_fallback(
            model=OPENAI_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role":"system","content":"You repair JSON to satisfy validation rules using only the transcript."},
                {"role":"user","content":json.dumps(instruction, ensure_ascii=False)}
            ],
        )
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception:
        return bad_payload

# ─────────────────── Builder pipeline ──────────────────
class Naming(BaseModel):
    agency: str = Field("Unknown")
    product: str = Field("Unknown")
    campaign: str = Field("Unknown")
    commercial: str = Field("Unknown")
    director: str = Field("Unknown")
    voiceover: str = Field("Unknown")
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

def chaos_augment_scenes(current_scenes: List[Dict], transcript_text: str) -> List[Dict]:
    """Ask the model to add micro-beat chaos montage gags, then merge."""
    try:
        user_blob = json.dumps({"scenes": current_scenes, "transcript": transcript_text}, ensure_ascii=False)
        resp = _create_with_fallback(
            model=OPENAI_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": CHAOS_MONTAGE_SPEC},
                {"role": "user", "content": user_blob},
            ],
        )
        payload = json.loads(resp.choices[0].message.content or "{}")
        aug = payload.get("scenes_augmented") or []
        # De-dup: avoid identical visual+start pairs
        seen_keys = { (sc.get("start",""), sc.get("visual","").strip().lower()) for sc in current_scenes }
        merged = list(current_scenes)
        for sc in aug:
            key = (sc.get("start",""), (sc.get("visual","") or "").strip().lower())
            if key not in seen_keys:
                merged.append(sc)
                seen_keys.add(key)
        return merged
    except Exception:
        return current_scenes

def build_case_study(url: str, provided_transcript: Optional[str] = None) -> str:
    """
    Build the PDF (with richer, director-level scene detail + chaos montage) and return its file path.
    """
    # Extract and fetch metadata
    vid = video_id_from_url(url)
    meta = fetch_basic_metadata(vid)

    # Transcript (prefer user-provided)
    if provided_transcript:
        transcript_text = provided_transcript[:24000]
        preview_lines = transcript_text.splitlines()[:24]
        timecoded = "\n".join(preview_lines)
    else:
        segments = fetch_transcript_segments(vid)
        if not segments:
            segments = [{"start": 0.0, "text": ""}]
        timecoded = "\n".join(f"{_format_time(s['start'])}  {s['text']}" for s in segments[:260])
        transcript_text = " ".join(s["text"] for s in segments)[:24000]

    # LLM user block
    user_block = ANALYSIS_USER_TEMPLATE.render(
        url=url, title=meta.get("title",""), author=meta.get("author",""),
        timecoded=timecoded, transcript=transcript_text
    )

    # Naming via LLM
    raw_naming = llm_json(NAMING_SYSTEM, user_block) or {}
    naming = Naming(
        agency=raw_naming.get("agency","Unknown"),
        product=raw_naming.get("product","Unknown"),
        campaign=raw_naming.get("campaign","Unknown"),
        commercial=raw_naming.get("commercial", meta.get("title","Unknown")),
        director=raw_naming.get("director","Unknown"),
        voiceover="Unknown",
    )
    naming = _fill_from_title_if_possible(naming, meta.get("title",""))

    # Enrich with ad-trade research (voiceover/director/agency)
    brand_hint = naming.product if naming.product != "Unknown" else meta.get("title", "")
    trade_hits = enrich_from_trades(meta.get("title","") or brand_hint, brand_hint)
    if trade_hits.get("voiceover"):
        naming.voiceover = trade_hits["voiceover"][0]
    if naming.director == "Unknown" and trade_hits.get("director"):
        naming.director = trade_hits["director"][0]
    if naming.agency == "Unknown" and trade_hits.get("agency"):
        naming.agency = trade_hits["agency"][0]

    # Detailed scene/script JSON
    detailed_payload = llm_json_structured(DETAILED_SCENE_SPEC, user_block)
    # Ensure required keys exist
    detailed_payload.setdefault("video", {"url": url, "title": meta.get("title",""), "channel": meta.get("author","")})
    detailed_payload.setdefault("naming", {
        "agency": naming.agency, "product": naming.product, "campaign": naming.campaign,
        "commercial": naming.commercial, "director": naming.director
    })
    detailed_payload.setdefault("scenes", [])
    detailed_payload.setdefault("script", [])
    detailed_payload.setdefault("brand", {
        "product": naming.product, "distinctive_assets": [], "claims": [], "ctas": []
    })

    # ★ Chaos Montage augmentation (adds micro-beats to scenes)
    detailed_payload["scenes"] = chaos_augment_scenes(detailed_payload["scenes"], transcript_text)

    # Validation & repair to enforce density + verbatim lines
    for _ in range(REPAIR_PASSES):
        issues = validate_json(detailed_payload, transcript_text)
        if not issues:
            break
        detailed_payload = repair_json_with_model(detailed_payload, transcript_text)

    # Analysis HTML (strategy sections)
    analysis_html = llm_html_from_json(ANALYSIS_FROM_JSON, detailed_payload, transcript_text)

    # Render PDF
    from weasyprint import HTML as WEASY_HTML
    heading = f"{naming.agency} – {naming.product} – {naming.campaign} – {naming.commercial} – {naming.director}"
    html = PDF_HTML.render(
        file_title=naming.filename().replace(".pdf",""),
        heading=heading,
        naming=naming,
        scenes=detailed_payload["scenes"],
        script=detailed_payload["script"],
        analysis_html=analysis_html,
        url=url,
    )
    pdf_path = os.path.join(OUT_DIR, naming.filename())
    WEASY_HTML(string=html, base_url=".").write_pdf(pdf_path)
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
    <p class="muted">Paste a YouTube URL and (optionally) a transcript. You’ll get a director-level, scene-by-scene PDF with specific visuals, VO/SFX, and purpose per beat.</p>
    <form method="post" action="/generate">
      <label>YouTube URL</label>
      <input name="url" type="text" placeholder="https://www.youtube.com/watch?v=..." required />
      <label style="margin-top:12px">Transcript (optional)</label>
      <textarea name="transcript" placeholder="Paste the full transcript here if you have it. If empty, we’ll try to fetch captions automatically."></textarea>
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
  <title>Case Study Ready</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; color:#111; margin: 24px; }
    .card { max-width: 860px; margin: 0 auto; border: 1px solid #e5e7eb; border-radius: 14px; padding: 20px; box-shadow: 0 6px 20px rgba(0,0,0,.05); text-align: center; }
    .btn { background:#111827; color:#fff; border:none; padding: 10px 14px; border-radius: 10px; cursor:pointer; text-decoration: none; }
    .muted { color:#6b7280; font-size: 13px; }
  </style>
  <script>
    window.addEventListener('DOMContentLoaded', () => {
      const a = document.getElementById('dl');
      if (a) a.click();
    });
  </script>
</head>
<body>
  <div class="card">
    <h2>Success!</h2>
    <p>Your case study PDF is ready. It should begin downloading automatically.</p>
    <p><a id="dl" href="{{ pdf_url }}" class="btn" download>Download PDF: {{ pdf_filename }}</a></p>
    <p style="margin-top:20px"><a href="/">Generate another</a></p>
  </div>
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
    <strong>Director:</strong> {{ naming.director }} &nbsp;|&nbsp;
    <strong>Voiceover:</strong> {{ naming.voiceover }}
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
        pdf_path = build_case_study(url, provided_transcript=(transcript_text or None))
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
