"""
Gong Call Insights Extractor
Fetches calls from Gong, uses Claude AI to extract use cases, jobs to be done,
and pain points, then attributes them to Tonic Textual or Fabricate.

Credentials are entered via the Settings page in the browser and stored
in a local credentials file (never in code).
"""

import os
import json
import sqlite3
import requests
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from functools import wraps
from authlib.integrations.flask_client import OAuth
import anthropic

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-in-prod")

# Fix for running behind Railway's HTTPS proxy — ensures redirect URIs use https://
from werkzeug.middleware.proxy_fix import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# ── Google OAuth ───────────────────────────────────────────────────────────────
oauth = OAuth(app)
google = oauth.register(
    name="google",
    client_id=os.environ.get("GOOGLE_CLIENT_ID"),
    client_secret=os.environ.get("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

ALLOWED_DOMAIN = "tonic.ai"

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("authenticated"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
# Use /tmp for database on cloud servers (Railway, etc.) which may have read-only filesystems
IS_CLOUD = bool(os.environ.get("RAILWAY_ENVIRONMENT") or os.environ.get("VERCEL"))

# Prefer a persistent data directory (Railway Volume mounted at /data),
# falling back to /tmp (ephemeral) if no volume is attached yet.
if IS_CLOUD:
    _data_dir = Path(os.environ.get("DATA_DIR", "/data"))
    if not _data_dir.exists():
        _data_dir = Path("/tmp")
    DB_PATH = _data_dir / "insights.db"
    SETTINGS_PATH = _data_dir / "settings.json"
else:
    DB_PATH = BASE_DIR / "insights.db"
    SETTINGS_PATH = BASE_DIR / "settings.json"

# ── Default keyword lists for Tonic product attribution ───────────────────────
DEFAULT_TEXTUAL_KEYWORDS = [
    "textual", "unstructured data", "text data", "nlp", "natural language",
    "documents", "pdf", "free text", "text fields", "free-form", "notes",
    "clinical notes", "medical records", "emails", "chat logs", "transcripts",
    "text anonymization", "text masking", "redaction", "pii in text",
    "sensitive text", "llm training data", "ai training",
    "text tokenization", "named entity", "ner", "de-identification",
    "de-identify", "masking", "mask", "anonymize", "rtf",
    "scanned documents", "e-fax", "image-based",
]

DEFAULT_STRUCTURAL_KEYWORDS = [
    "structural", "production-like", "test data", "provisioned", "provision",
    "shift-left", "shift left", "testing environment", "test environment",
    "compliant test data", "mirrors production", "production copy",
    "production data", "staging data", "staging environment", "dev environment",
    "database copy", "data refresh", "release cycles", "accelerate testing",
    "subsetting", "subset", "data subsetting", "referential integrity",
    "structured data", "relational", "database masking", "sql", "tables",
    "schemas", "postgres", "mysql", "snowflake", "redshift", "mongodb",
    "ci/cd", "pipeline", "shape of production", "complexity of production",
]

DEFAULT_FABRICATE_KEYWORDS = [
    "fabricate", "synthetic data", "generate data", "generated data",
    "fake data", "realistic data", "test data generation", "data generation",
    "build and test", "train without real data", "no real data",
    "synthetic database", "data agent", "on demand data", "scalable data",
    "realistic fake", "dummy data", "mock data", "simulate data",
]


# ── Settings helpers ───────────────────────────────────────────────────────────
CREDENTIALS_PATH = Path("/tmp/credentials.json") if IS_CLOUD else BASE_DIR / "credentials.json"


def load_settings():
    """Load all settings from settings.json and credentials from credentials.json."""
    settings = {
        "gong_base_url": "https://api.gong.io",
        "textual_keywords": DEFAULT_TEXTUAL_KEYWORDS,
        "structural_keywords": DEFAULT_STRUCTURAL_KEYWORDS,
        "fabricate_keywords": DEFAULT_FABRICATE_KEYWORDS,
        "lookback_days": 30,
        "gong_access_key": "",
        "gong_access_secret": "",
        "anthropic_api_key": "",
    }

    if SETTINGS_PATH.exists():
        with open(SETTINGS_PATH) as f:
            saved = json.load(f)
        settings.update(saved)

    # Load credentials: environment variables take priority (used in production/Vercel),
    # falling back to local credentials.json (used when running on your own computer)
    env_gong_key = os.environ.get("GONG_ACCESS_KEY", "")
    env_gong_secret = os.environ.get("GONG_ACCESS_SECRET", "")
    env_anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if env_gong_key or env_gong_secret or env_anthropic_key:
        # Running on Vercel or another server — use environment variables
        settings["gong_access_key"] = env_gong_key
        settings["gong_access_secret"] = env_gong_secret
        settings["anthropic_api_key"] = env_anthropic_key
    elif CREDENTIALS_PATH.exists():
        # Running locally — use credentials.json saved via Settings page
        with open(CREDENTIALS_PATH) as f:
            creds = json.load(f)
        settings["gong_access_key"] = creds.get("gong_access_key", "")
        settings["gong_access_secret"] = creds.get("gong_access_secret", "")
        settings["anthropic_api_key"] = creds.get("anthropic_api_key", "")

    return settings


def save_settings(data):
    """Save non-sensitive settings to settings.json, credentials to credentials.json."""
    cred_keys = ["gong_access_key", "gong_access_secret", "anthropic_api_key"]
    config_keys = ["textual_keywords", "structural_keywords", "fabricate_keywords", "lookback_days", "gong_base_url"]

    # Save credentials separately
    if any(k in data for k in cred_keys):
        existing_creds = {}
        if CREDENTIALS_PATH.exists():
            with open(CREDENTIALS_PATH) as f:
                existing_creds = json.load(f)
        for k in cred_keys:
            if k in data and data[k]:  # only update if non-empty
                existing_creds[k] = data[k]
        with open(CREDENTIALS_PATH, "w") as f:
            json.dump(existing_creds, f, indent=2)

    # Save config
    existing_config = {}
    if SETTINGS_PATH.exists():
        with open(SETTINGS_PATH) as f:
            existing_config = json.load(f)
    for k in config_keys:
        if k in data:
            existing_config[k] = data[k]
    with open(SETTINGS_PATH, "w") as f:
        json.dump(existing_config, f, indent=2)


# ── Database setup ─────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS calls (
            id TEXT PRIMARY KEY,
            title TEXT,
            started TEXT,
            duration INTEGER,
            parties TEXT,
            analyzed INTEGER DEFAULT 0,
            analyzed_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            call_id TEXT NOT NULL,
            call_title TEXT,
            call_date TEXT,
            insight_type TEXT NOT NULL,
            content TEXT NOT NULL,
            product TEXT NOT NULL,
            confidence TEXT,
            raw_quote TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (call_id) REFERENCES calls(id)
        )
    """)
    conn.commit()
    conn.close()


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ── Gong API helpers ───────────────────────────────────────────────────────────
def gong_headers(settings):
    import base64
    key = settings["gong_access_key"]
    secret = settings["gong_access_secret"]
    token = base64.b64encode(f"{key}:{secret}".encode()).decode()
    return {
        "Authorization": f"Basic {token}",
        "Content-Type": "application/json",
    }


def fetch_gong_calls(settings, from_date=None, to_date=None):
    base_url = settings.get("gong_base_url", "https://api.gong.io")
    headers = gong_headers(settings)

    if not from_date:
        days = int(settings.get("lookback_days", 30))
        from_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%dT00:00:00Z")
    if not to_date:
        to_date = datetime.utcnow().strftime("%Y-%m-%dT23:59:59Z")

    payload = {
        "filter": {
            "fromDateTime": from_date,
            "toDateTime": to_date,
        }
    }

    resp = requests.post(
        f"{base_url}/v2/calls/extensive",
        headers=headers,
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("calls", [])


def fetch_call_transcript(settings, call_id):
    base_url = settings.get("gong_base_url", "https://api.gong.io")
    headers = gong_headers(settings)

    # Gong's transcript endpoint is a POST with call IDs in the body
    resp = requests.post(
        f"{base_url}/v2/calls/transcript",
        headers=headers,
        json={"filter": {"callIds": [call_id]}},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    transcript_parts = []
    for call in data.get("callTranscripts", []):
        for segment in call.get("transcript", []):
            speaker = segment.get("speakerName", "Unknown")
            for sentence in segment.get("sentences", []):
                text = sentence.get("text", "").strip()
                if text:
                    transcript_parts.append(f"{speaker}: {text}")

    return "\n".join(transcript_parts)


# ── Claude AI extraction ───────────────────────────────────────────────────────
EXTRACTION_PROMPT = """You are an expert product analyst for Tonic.ai, a company that makes data privacy and synthetic data tools.

Tonic has THREE products:

- **Tonic Textual**: Masks, redacts, or de-identifies sensitive information in UNSTRUCTURED data — free-form text, clinical notes, PDFs, emails, scanned documents, RTF files, and chat logs. The data already exists and Textual makes it safe by removing or replacing PII. Also used for creating AI/LLM training datasets from real text data.

- **Tonic Structural**: Takes REAL production data (structured/relational databases) and provisions a compliant, production-like copy for dev and test environments. It mirrors the shape and complexity of production data so engineers can test against realistic data. Used for shift-left testing, accelerating release cycles, subsetting databases, and maintaining referential integrity. The key: it starts with real production data and creates a safe, realistic copy.

- **Tonic Fabricate**: An AI agent that GENERATES completely NEW synthetic data from scratch — no real data needed. Used when a customer wants to create realistic fake data for any domain without starting from real production data. Fabricate creates data that never existed.

CRITICAL DISTINCTIONS — READ CAREFULLY:

1. TEXTUAL vs STRUCTURAL vs FABRICATE comes down to:
   - What TYPE of data? Unstructured text/documents → Textual. Structured databases/tables → Structural or Fabricate.
   - What ACTION? Mask/redact/de-identify existing data → Textual (even if it's in a database). Provision a production-like copy from real data → Structural. Generate brand new data from scratch → Fabricate.

2. THE MOST IMPORTANT RULE: If the words "de-identify", "mask", "redact", or "anonymize" appear, it is ALWAYS Textual — regardless of whether the data is in a database, Snowflake, or any structured system.

3. STRUCTURAL vs FABRICATE:
   - Structural = customer has real production data and wants a safe, production-like copy for testing
   - Fabricate = customer wants to generate new data that doesn't exist yet, with no real production data as input

CONCRETE EXAMPLES:
- "We need to de-identify patient records in Snowflake" → Textual
- "We need to mask free-text notes in our EHR" → Textual
- "We need production-like test data that mirrors our database" → Structural
- "We need to provision compliant copies of our production DB for each dev" → Structural
- "We need to accelerate our release cycles with better test data" → Structural
- "We need synthetic patient data generated from scratch for our dev environment" → Fabricate
- "We need fake test data with no connection to real customers" → Fabricate

Analyze the following sales call transcript and extract:
1. **Use Cases** - specific ways the customer wants to use a Tonic product
2. **Jobs to be Done** - underlying goals or outcomes the customer is trying to achieve
3. **Pain Points** - problems, frustrations, or challenges the customer faces

For each item you extract:
- Classify it as: use_case, job_to_be_done, or pain_point
- Attribute it to: Textual, Structural, Fabricate, or General (if unclear)
- Provide a confidence level: high, medium, or low
- Include a short relevant quote from the transcript (max 50 words)

Return your response as a JSON array with this exact structure:
[
  {
    "type": "use_case" | "job_to_be_done" | "pain_point",
    "content": "concise description of the insight (1-2 sentences)",
    "product": "Textual" | "Fabricate" | "General",
    "confidence": "high" | "medium" | "low",
    "quote": "short relevant quote from transcript"
  }
]

If no relevant insights are found, return an empty array [].
Only return the JSON array, no other text.

TRANSCRIPT:
{transcript}
"""


def extract_insights_with_claude(transcript, settings):
    client = anthropic.Anthropic(api_key=settings["anthropic_api_key"])

    max_chars = 80_000
    if len(transcript) > max_chars:
        transcript = transcript[:max_chars] + "\n\n[Transcript truncated for length]"

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": EXTRACTION_PROMPT.replace("{transcript}", transcript),
            }
        ],
    )

    raw = message.content[0].text.strip()

    try:
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        insights = json.loads(raw)
    except json.JSONDecodeError:
        insights = []

    return insights


def apply_keyword_attribution(insights, settings):
    textual_kw = [k.lower() for k in settings.get("textual_keywords", DEFAULT_TEXTUAL_KEYWORDS)]
    structural_kw = [k.lower() for k in settings.get("structural_keywords", DEFAULT_STRUCTURAL_KEYWORDS)]
    fabricate_kw = [k.lower() for k in settings.get("fabricate_keywords", DEFAULT_FABRICATE_KEYWORDS)]

    for insight in insights:
        if insight.get("product") != "General":
            continue

        content_lower = (insight.get("content", "") + " " + insight.get("quote", "")).lower()
        textual_hits = sum(1 for kw in textual_kw if kw in content_lower)
        structural_hits = sum(1 for kw in structural_kw if kw in content_lower)
        fabricate_hits = sum(1 for kw in fabricate_kw if kw in content_lower)

        best = max(textual_hits, structural_hits, fabricate_hits)
        if best == 0:
            continue
        if textual_hits == best:
            insight["product"] = "Textual"
            insight["confidence"] = "medium"
        elif structural_hits == best:
            insight["product"] = "Structural"
            insight["confidence"] = "medium"
        else:
            insight["product"] = "Fabricate"
            insight["confidence"] = "medium"

    return insights


# ── Flask routes ───────────────────────────────────────────────────────────────
# Initialize database on startup regardless of how the app is launched
init_db()


@app.route("/login")
def login():
    return render_template("login.html")


@app.route("/auth/google")
def auth_google():
    # Use hardcoded redirect URI to avoid http/https mismatch behind Railway proxy
    base = os.environ.get("APP_BASE_URL", "").rstrip("/")
    if base:
        redirect_uri = f"{base}/auth/callback"
    else:
        redirect_uri = url_for("auth_callback", _external=True, _scheme="https")
    return google.authorize_redirect(redirect_uri)


@app.route("/auth/callback")
def auth_callback():
    token = google.authorize_access_token()
    user_info = token.get("userinfo") or google.userinfo()
    email = (user_info.get("email") or "").lower()

    if email.endswith(f"@{ALLOWED_DOMAIN}"):
        session["authenticated"] = True
        session["user_email"] = email
        session["user_name"] = user_info.get("name", email)
        return redirect(url_for("index"))
    else:
        session.clear()
        return render_template("unauthorized.html", email=email)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    return render_template("index.html")


@app.route("/api/settings", methods=["GET"])
@login_required
def get_settings():
    settings = load_settings()
    # Mask credentials — show a hint but never the full value
    def mask(val):
        if not val:
            return ""
        if len(val) <= 8:
            return "••••••••"
        return val[:4] + "••••••••" + val[-4:]

    return jsonify({
        "gong_access_key": mask(settings.get("gong_access_key", "")),
        "gong_access_secret": mask(settings.get("gong_access_secret", "")),
        "anthropic_api_key": mask(settings.get("anthropic_api_key", "")),
        "gong_access_key_set": bool(settings.get("gong_access_key")),
        "gong_access_secret_set": bool(settings.get("gong_access_secret")),
        "anthropic_api_key_set": bool(settings.get("anthropic_api_key")),
        "gong_base_url": settings.get("gong_base_url", "https://api.gong.io"),
        "textual_keywords": settings.get("textual_keywords", DEFAULT_TEXTUAL_KEYWORDS),
        "structural_keywords": settings.get("structural_keywords", DEFAULT_STRUCTURAL_KEYWORDS),
        "fabricate_keywords": settings.get("fabricate_keywords", DEFAULT_FABRICATE_KEYWORDS),
        "lookback_days": settings.get("lookback_days", 30),
    })


@app.route("/api/settings", methods=["POST"])
@login_required
def update_settings():
    data = request.json
    # Only accept non-sensitive settings via this route
    save_settings(data)
    return jsonify({"status": "ok"})


@app.route("/api/calls", methods=["GET"])
@login_required
def get_calls():
    settings = load_settings()
    if not settings.get("gong_access_key") or not settings.get("gong_access_secret"):
        return jsonify({"error": "Gong API credentials not found. Please add GONG_ACCESS_KEY and GONG_ACCESS_SECRET to your .env file."}), 400

    from_date = request.args.get("from_date")
    to_date = request.args.get("to_date")

    try:
        calls = fetch_gong_calls(settings, from_date, to_date)
    except requests.HTTPError as e:
        return jsonify({"error": f"Gong API error: {e.response.status_code} {e.response.text}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    db = get_db()
    for call in calls:
        call_id = call.get("metaData", {}).get("id", "")
        if not call_id:
            continue
        title = call.get("metaData", {}).get("title", "Untitled Call")
        started = call.get("metaData", {}).get("started", "")
        duration = call.get("metaData", {}).get("duration", 0)
        parties = json.dumps([
            p.get("name", p.get("emailAddress", "Unknown"))
            for p in call.get("parties", [])
        ])
        db.execute("""
            INSERT OR IGNORE INTO calls (id, title, started, duration, parties)
            VALUES (?, ?, ?, ?, ?)
        """, (call_id, title, started, duration, parties))
    db.commit()

    result = []
    for call in calls:
        call_id = call.get("metaData", {}).get("id", "")
        row = db.execute("SELECT analyzed, analyzed_at FROM calls WHERE id=?", (call_id,)).fetchone()
        analyzed = bool(row["analyzed"]) if row else False
        result.append({
            "id": call_id,
            "title": call.get("metaData", {}).get("title", "Untitled Call"),
            "started": call.get("metaData", {}).get("started", ""),
            "duration": call.get("metaData", {}).get("duration", 0),
            "parties": [p.get("name", p.get("emailAddress", "")) for p in call.get("parties", [])],
            "analyzed": analyzed,
            "analyzed_at": row["analyzed_at"] if row else None,
        })

    db.close()
    return jsonify({"calls": result, "total": len(result)})


@app.route("/api/analyze/<call_id>", methods=["POST"])
@login_required
def analyze_call(call_id):
    settings = load_settings()
    missing = []
    if not settings.get("gong_access_key"):
        missing.append("GONG_ACCESS_KEY")
    if not settings.get("anthropic_api_key"):
        missing.append("ANTHROPIC_API_KEY")
    if missing:
        return jsonify({"error": f"Missing in .env file: {', '.join(missing)}"}), 400

    db = get_db()
    call_row = db.execute("SELECT * FROM calls WHERE id=?", (call_id,)).fetchone()

    try:
        transcript = fetch_call_transcript(settings, call_id)
        if not transcript.strip():
            return jsonify({"error": "No transcript available for this call"}), 404

        insights = extract_insights_with_claude(transcript, settings)
        insights = apply_keyword_attribution(insights, settings)

        call_title = call_row["title"] if call_row else call_id
        call_date = call_row["started"] if call_row else ""

        db.execute("DELETE FROM insights WHERE call_id=?", (call_id,))

        for ins in insights:
            db.execute("""
                INSERT INTO insights (call_id, call_title, call_date, insight_type, content, product, confidence, raw_quote)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                call_id, call_title, call_date,
                ins.get("type", "unknown"),
                ins.get("content", ""),
                ins.get("product", "General"),
                ins.get("confidence", "medium"),
                ins.get("quote", ""),
            ))

        db.execute(
            "UPDATE calls SET analyzed=1, analyzed_at=? WHERE id=?",
            (datetime.utcnow().isoformat(), call_id)
        )
        db.commit()
        db.close()

        return jsonify({
            "status": "ok",
            "call_id": call_id,
            "insights_extracted": len(insights),
            "insights": insights,
        })

    except Exception as e:
        db.close()
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/insights", methods=["GET"])
@login_required
def get_insights():
    product = request.args.get("product")
    insight_type = request.args.get("type")
    call_id = request.args.get("call_id")

    db = get_db()
    query = "SELECT * FROM insights WHERE 1=1"
    params = []
    if product:
        query += " AND product=?"
        params.append(product)
    if insight_type:
        query += " AND insight_type=?"
        params.append(insight_type)
    if call_id:
        query += " AND call_id=?"
        params.append(call_id)
    query += " ORDER BY created_at DESC"

    rows = db.execute(query, params).fetchall()
    result = [dict(row) for row in rows]
    db.close()
    return jsonify({"insights": result, "total": len(result)})


@app.route("/api/insights/summary", methods=["GET"])
@login_required
def insights_summary():
    db = get_db()
    total = db.execute("SELECT COUNT(*) FROM insights").fetchone()[0]
    by_product = db.execute("SELECT product, COUNT(*) as count FROM insights GROUP BY product").fetchall()
    by_type = db.execute("SELECT insight_type, COUNT(*) as count FROM insights GROUP BY insight_type").fetchall()
    calls_analyzed = db.execute("SELECT COUNT(*) FROM calls WHERE analyzed=1").fetchone()[0]
    recent = db.execute("SELECT * FROM insights ORDER BY created_at DESC LIMIT 10").fetchall()
    db.close()
    return jsonify({
        "total_insights": total,
        "calls_analyzed": calls_analyzed,
        "by_product": [dict(r) for r in by_product],
        "by_type": [dict(r) for r in by_type],
        "recent_insights": [dict(r) for r in recent],
    })


@app.route("/api/insights/ranked-use-cases", methods=["GET"])
@login_required
def ranked_use_cases():
    """
    Returns two ranked lists:
    1. Top Textual use cases ranked by frequency across calls
    2. Use cases from calls that mention both Textual and Fabricate
    """
    db = get_db()

    # ── Textual use cases ranked by frequency ─────────────────────────────────
    textual_rows = db.execute("""
        SELECT content, COUNT(DISTINCT call_id) as call_count, COUNT(*) as total_count,
               GROUP_CONCAT(DISTINCT call_title) as calls
        FROM insights
        WHERE product = 'Textual' AND insight_type = 'use_case'
        GROUP BY content
        ORDER BY call_count DESC, total_count DESC
        LIMIT 25
    """).fetchall()

    textual_ranked = [dict(r) for r in textual_rows]

    # ── Cross-product: calls with both Textual and Fabricate insights ──────────
    cross_calls = db.execute("""
        SELECT DISTINCT call_id FROM insights WHERE product = 'Textual'
        INTERSECT
        SELECT DISTINCT call_id FROM insights WHERE product = 'Fabricate'
    """).fetchall()
    cross_call_ids = [r["call_id"] for r in cross_calls]

    cross_use_cases = []
    if cross_call_ids:
        placeholders = ",".join("?" * len(cross_call_ids))
        cross_rows = db.execute(f"""
            SELECT content, COUNT(DISTINCT call_id) as call_count,
                   GROUP_CONCAT(DISTINCT call_title) as calls,
                   product
            FROM insights
            WHERE call_id IN ({placeholders})
              AND insight_type = 'use_case'
              AND product IN ('Textual', 'Fabricate')
            GROUP BY content
            ORDER BY call_count DESC
            LIMIT 25
        """, cross_call_ids).fetchall()
        cross_use_cases = [dict(r) for r in cross_rows]

    db.close()
    return jsonify({
        "textual_ranked": textual_ranked,
        "cross_product": cross_use_cases,
        "cross_product_call_count": len(cross_call_ids),
    })


@app.route("/api/insights/<int:insight_id>", methods=["DELETE"])
@login_required
def delete_insight(insight_id):
    db = get_db()
    db.execute("DELETE FROM insights WHERE id=?", (insight_id,))
    db.commit()
    db.close()
    return jsonify({"status": "ok"})


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()

    # Warn if credentials are missing
    settings = load_settings()
    missing = []
    if not settings["gong_access_key"]:
        missing.append("Gong Access Key")
    if not settings["gong_access_secret"]:
        missing.append("Gong Access Secret")
    if not settings["anthropic_api_key"]:
        missing.append("Anthropic API Key")
    if missing:
        print(f"⚠️  Missing credentials: {', '.join(missing)}")
        print("   Go to http://localhost:5000 → Settings to add them.")
    else:
        print("✅ All credentials configured")

    print("🚀 Gong Insights Extractor running at http://localhost:5000")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
