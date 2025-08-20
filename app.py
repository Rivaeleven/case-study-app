import os, re, requests, json
from typing import List, Dict, Tuple
from fastapi import FastAPI
from pydantic import BaseModel, Field
from jinja2 import Template

# ─────────────────────────────
# Web search & readable content
# ─────────────────────────────

PUBLISHER_WHITELIST = [
    "adage.com", "adweek.com", "campaignlive.com", "campaignlive.co.uk",
    "shots.net", "lbbonline.com", "thedrum.com", "musebycl.io", "ispot.tv",
    "shootonline.com", "adforum.com", "adsoftheworld.com",
    "prnewswire.com", "businesswire.com"
]

def web_search(query: str, limit: int = 5):
    """Use Bing or SerpAPI depending on which key is set."""
    if os.getenv("BING_SEARCH_KEY"):
        key = os.getenv("BING_SEARCH_KEY")
        url = f"https://api.bing.microsoft.com/v7.0/search?q={query}"
        resp = requests.get(url, headers={"Ocp-Apim-Subscription-Key": key}, timeout=10)
        data = resp.json()
        results = []
        for w in data.get("webPages", {}).get("value", []):
            results.append({"title": w["name"], "url": w["url"], "snippet": w["snippet"]})
        return results[:limit]
    elif os.getenv("SERPAPI_KEY"):
        key = os.getenv("SERPAPI_KEY")
        url = f"https://serpapi.com/search.json?q={query}&engine=google&api_key={key}"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        results = []
        for w in data.get("organic_results", []):
            results.append({"title": w["title"], "url": w["link"], "snippet": w.get("snippet", "")})
        return results[:limit]
    return []

def http_get_readable(url: str) -> str:
    """Proxy through Jina.ai to strip boilerplate and return clean text."""
    try:
        prox = "https://r.jina.ai/" + url
        r = requests.get(prox, timeout=10)
        if r.status_code == 200:
            return r.text
    except Exception:
        pass
    return ""

# ─────────────────────────────
# Credit extraction patterns
# ─────────────────────────────

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
    """Search ad trades for credits (voiceover, director, agency)."""
    if not title and not brand_hint:
        return {}

    queries = []
    base = title or brand_hint
    queries += [
        f"{base} voiceover", f"{base} VO", f"{base} narrator",
        f"{base} director", f"{base} credits", f"{base} ad agency"
    ]
    if brand_hint:
        queries += [
            f"{brand_hint} Super Bowl ad voiceover",
            f"{brand_hint} Super Bowl ad director",
            f"{brand_hint} Super Bowl ad credits"
        ]

    seen, pages = set(), []
    for q in queries:
        for res in web_search(q, limit=6):
            u = res.get("url", "")
            if not u or u in seen:
                continue
            if not any(pub in u for pub in PUBLISHER_WHITELIST):
                continue
            seen.add(u)
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
        if not d:
            return "Unknown", []
        best = max(d.items(), key=lambda kv: len(kv[1]))
        return best[0], best[1]

    voice, voice_srcs = pick_best(bag["voiceover"])
    direc, direc_srcs = pick_best(bag["director"])
    agcy, agcy_srcs = pick_best(bag["agency"])
    return {
        "voiceover": [voice] if voice != "Unknown" else [],
        "director": [direc] if direc != "Unknown" else [],
        "agency": [agcy] if agcy != "Unknown" else [],
        "sources": list({*voice_srcs, *direc_srcs, *agcy_srcs})
    }

# ─────────────────────────────
# Data models
# ─────────────────────────────

def safe_token(x: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\-]+", "_", x.strip())

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

# ─────────────────────────────
# HTML template for PDF
# ─────────────────────────────

PDF_HTML = Template("""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{{ naming.commercial }}</title>
  <style>
    body { font-family: Helvetica, sans-serif; margin: 2em; }
    h1 { color: #111; }
    h2 { color: #333; margin-top: 1.4em; }
    .meta { margin-bottom: 1em; font-size: 0.9em; color: #555; }
  </style>
</head>
<body>
  <h1>{{ naming.product }} – {{ naming.commercial }}</h1>
  <div class="meta">
    <strong>Agency:</strong> {{ naming.agency }} &nbsp;|&nbsp;
    <strong>Director:</strong> {{ naming.director }} &nbsp;|&nbsp;
    <strong>Voiceover:</strong> {{ naming.voiceover }}<br>
    <strong>Campaign:</strong> {{ naming.campaign }}
  </div>

  <h2>Scene-by-Scene Breakdown</h2>
  <pre>{{ scenes }}</pre>

  <h2>Annotated Script</h2>
  <pre>{{ script }}</pre>
</body>
</html>
""")

# ─────────────────────────────
# Main app
# ─────────────────────────────

app = FastAPI()

@app.get("/")
def home():
    return {"status": "ok"}

@app.get("/analyze/{video_id}")
def analyze(video_id: str):
    """
    Main entry: fetch metadata + transcript + enrich with ad trades.
    """
    # Mock metadata fetch (replace with YouTube API if you want)
    meta = {"title": "Unknown", "channel": "Unknown"}
    naming = Naming()

    # Enrich with ad trades
    brand_hint = naming.product if naming.product != "Unknown" else meta.get("title", "")
    hits = enrich_from_trades(meta.get("title", ""), brand_hint)
    if hits.get("voiceover"):
        naming.voiceover = hits["voiceover"][0]
    if naming.director == "Unknown" and hits.get("director"):
        naming.director = hits["director"][0]
    if naming.agency == "Unknown" and hits.get("agency"):
        naming.agency = hits["agency"][0]

    # Fake transcript for now
    transcript = "[Transcript goes here]"
    scenes = "[Scene breakdown goes here]"
    script = "[Annotated script goes here]"

    html = PDF_HTML.render(naming=naming, scenes=scenes, script=script)
    return {"naming": naming.dict(), "html": html, "sources": hits.get("sources", [])}
