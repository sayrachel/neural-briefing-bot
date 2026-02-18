#!/usr/bin/env python3
"""
AI News Telegram Bot
- Sends daily AI news digests at 9am PT
- Supports /start, /stop, /summary commands
"""

import os
import re
import json
import hashlib
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path

import feedparser
import google.generativeai as genai
import psycopg2
import requests
from flask import Flask, request

# Configuration
RSS_FEEDS = [
    ("TechCrunch", "https://techcrunch.com/category/artificial-intelligence/feed/"),
    ("The Verge", "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml"),
    ("MIT Tech Review", "https://www.technologyreview.com/topic/artificial-intelligence/feed"),
    ("Ars Technica", "https://feeds.arstechnica.com/arstechnica/features"),  # Features feed, filter by AI keywords
]

HOURS_LOOKBACK = 24  # Look back 24 hours for daily digest
SIMILARITY_THRESHOLD = 0.7
DIVERSITY_THRESHOLD = 0.4  # Stricter threshold for final article selection (catches same-story coverage)
CACHE_FILE = Path(__file__).parent / "summary_cache.json"
CACHE_TTL = 3600  # 1 hour in seconds
PROCESSED_UPDATES_FILE = Path(__file__).parent / "processed_updates.json"
MAX_PROCESSED_UPDATES = 100  # Keep last 100 update IDs

# In-memory set for fast duplicate detection (primary)
_processed_updates: set[int] = set()

# Quality-based ranking configuration
MIN_ARTICLES = 3
MAX_ARTICLES = 4
MIN_QUALITY_SCORE = 1.3  # Minimum score to include an article (filters out marginal stories)

# Source reputation weights (higher = more credible/in-depth)
SOURCE_WEIGHTS = {
    "MIT Tech Review": 1.5,    # Deep, research-focused
    "Ars Technica": 1.3,       # Technical depth
    "The Verge": 1.0,          # Solid general coverage
    "TechCrunch": 0.9,         # Sometimes clickbaity
}

# Keywords that indicate high-importance articles
HIGH_IMPORTANCE_KEYWORDS = [
    # Major business events
    "breakthrough", "announces", "launches", "acquisition", "merge", "merger",
    "funding", "billion", "million", "regulation", "lawsuit", "antitrust",
    "open source", "safety", "partnership", "research", "paper", "study",
    "infrastructure", "data center", "chip", "semiconductor",
    # Major AI products/companies
    "GPT", "Claude", "Gemini", "OpenAI", "Anthropic", "DeepMind", "Meta AI",
    # Influential figures
    "Sam Altman", "Dario Amodei", "Daniela Amodei", "Demis Hassabis",
    "Yann LeCun", "Fei-Fei Li", "Jensen Huang", "Satya Nadella",
    "Sundar Pichai", "Elon Musk", "Ilya Sutskever", "Andrej Karpathy",
]

# Keywords that indicate an article is about AI/ML (used to filter non-AI content)
AI_RELEVANCE_KEYWORDS = [
    # Core AI/ML terms
    "artificial intelligence", "machine learning", "deep learning", "neural network",
    "large language model", "llm", "generative ai", "gen ai",
    "natural language processing", "nlp", "computer vision",
    "reinforcement learning", "transformer", "diffusion model",
    # AI products and models
    "chatgpt", "gpt-4", "gpt-5", "gpt", "claude", "gemini", "copilot",
    "midjourney", "dall-e", "stable diffusion", "sora", "llama",
    "mistral", "deepseek", "grok",
    # AI companies (when mentioned, article is likely AI-related)
    "openai", "anthropic", "deepmind", "hugging face", "cohere",
    "stability ai", "inflection", "character.ai", "perplexity",
    # AI concepts
    "chatbot", "ai model", "ai agent", "ai safety", "ai regulation",
    "ai chip", "ai training", "ai inference", "foundation model",
    "multimodal", "text-to-image", "text-to-video", "speech recognition",
    "ai-powered", "ai-generated", "machine intelligence",
    "robot", "robotics", "autonomous",
]

# Keywords that indicate lower-value articles
LOW_VALUE_KEYWORDS = [
    # Existing
    "rumor", "might", "could", "speculation", "opinion",
    "podcast", "vergecast", "review",
    # Personal/blog style
    "vibe", "vibes", "feel good", "hot take", "rant",
    "coffee break", "unpopular opinion",
    # Minor incidents
    "brief outage", "short outage", "minute outage", "minutes down",
    "quickly restored", "back online",
]


def get_env_var(name: str) -> str:
    """Get required environment variable or raise error."""
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def get_db_connection():
    """Get a database connection."""
    database_url = get_env_var("DATABASE_URL")
    # Render uses postgres:// but psycopg2 requires postgresql://
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
    return psycopg2.connect(database_url)


def init_db():
    """Initialize the database schema."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        chat_id TEXT PRIMARY KEY,
                        state TEXT NOT NULL DEFAULT 'subscribed',
                        subscribed_at TIMESTAMPTZ DEFAULT NOW(),
                        last_digest_date TEXT
                    )
                """)
                cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS subscribed_at TIMESTAMPTZ DEFAULT NOW()")
                cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS last_digest_date TEXT")
            conn.commit()
        print("Database initialized")
    except Exception as e:
        print(f"Database init error: {e}")


def get_user(chat_id: str) -> dict | None:
    """Get a user by chat_id."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT chat_id, state, subscribed_at, last_digest_date FROM users WHERE chat_id = %s", (chat_id,))
            row = cur.fetchone()
            if row:
                return {"chat_id": row[0], "state": row[1], "subscribed_at": row[2], "last_digest_date": row[3]}
            return None


def upsert_user(chat_id: str, state: str = "subscribed") -> None:
    """Insert or update a user, preserving existing data."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO users (chat_id, state, subscribed_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (chat_id) DO UPDATE SET state = %s
            """, (chat_id, state, state))
        conn.commit()


def delete_user(chat_id: str) -> None:
    """Delete a user."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM users WHERE chat_id = %s", (chat_id,))
        conn.commit()


def get_users_needing_digest(today: str) -> list[str]:
    """Get chat_ids of subscribed users who haven't received today's digest."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT chat_id FROM users WHERE state = 'subscribed' AND (last_digest_date IS NULL OR last_digest_date != %s)",
                (today,)
            )
            return [row[0] for row in cur.fetchall()]


def update_last_digest(chat_id: str, date: str) -> None:
    """Update the last digest date for a user."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE users SET last_digest_date = %s WHERE chat_id = %s", (date, chat_id))
        conn.commit()


def get_articles_hash(articles: list[dict]) -> str:
    """Generate a hash of article titles to use as cache key."""
    titles = sorted([a.get("title", "") for a in articles])
    return hashlib.md5("".join(titles).encode()).hexdigest()


def get_cached_summary(articles: list[dict]) -> str | None:
    """Return cached summary if valid, else None."""
    if not CACHE_FILE.exists():
        return None

    try:
        cache = json.loads(CACHE_FILE.read_text())
        cached_time = cache.get("timestamp", 0)
        cached_hash = cache.get("articles_hash", "")
        cached_summary = cache.get("summary", "")

        # Check if cache is still valid (within TTL and same articles)
        current_hash = get_articles_hash(articles)
        if (datetime.now(timezone.utc).timestamp() - cached_time < CACHE_TTL
                and cached_hash == current_hash
                and cached_summary):
            print("Using cached summary")
            return cached_summary
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Cache read error: {e}")

    return None


def save_summary_cache(articles: list[dict], summary: str) -> None:
    """Save summary to cache with timestamp and article hash."""
    cache = {
        "timestamp": datetime.now(timezone.utc).timestamp(),
        "articles_hash": get_articles_hash(articles),
        "summary": summary,
    }
    try:
        CACHE_FILE.write_text(json.dumps(cache, indent=2))
        print("Summary cached")
    except Exception as e:
        print(f"Cache write error: {e}")


def is_update_processed(update_id: int) -> bool:
    """Check if an update ID has already been processed."""
    # Check in-memory set first (fast path)
    if update_id in _processed_updates:
        return True
    # Fall back to file check
    if not PROCESSED_UPDATES_FILE.exists():
        return False
    try:
        processed = json.loads(PROCESSED_UPDATES_FILE.read_text())
        return update_id in processed.get("ids", [])
    except (json.JSONDecodeError, KeyError):
        return False


def mark_update_processed(update_id: int) -> None:
    """Mark an update ID as processed."""
    _processed_updates.add(update_id)

    # Trim in-memory set if too large
    if len(_processed_updates) > MAX_PROCESSED_UPDATES * 2:
        excess = len(_processed_updates) - MAX_PROCESSED_UPDATES
        for item in list(_processed_updates)[:excess]:
            _processed_updates.discard(item)

    # Persist to file (backup)
    try:
        ids = []
        if PROCESSED_UPDATES_FILE.exists():
            ids = json.loads(PROCESSED_UPDATES_FILE.read_text()).get("ids", [])
        ids.append(update_id)
        PROCESSED_UPDATES_FILE.write_text(json.dumps({"ids": ids[-MAX_PROCESSED_UPDATES:]}))
    except Exception as e:
        print(f"Error persisting update to file: {e}")


def send_telegram_message(token: str, chat_id: str, message: str, parse_mode: str = "HTML") -> bool:
    """Send message to Telegram."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Failed to send message to {chat_id}: {e}")
        return False


def get_telegram_updates(token: str, offset: int = None) -> list:
    """Get new messages from Telegram."""
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    params = {"timeout": 5}
    if offset:
        params["offset"] = offset

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json().get("result", [])
    except Exception as e:
        print(f"Failed to get updates: {e}")
        return []


def handle_command(token: str, chat_id: str, text: str) -> None:
    """Handle a single command from a user."""
    if text == "/start":
        upsert_user(chat_id, "subscribed")
        send_telegram_message(
            token, chat_id,
            "Welcome to the Neural Briefing Bot! I'll send you a daily summary of the top AI news at 9am PT daily.\n\n"
            "Commands:\n"
            "/summary - Generate summary now\n"
            "/stop - Unsubscribe"
        )

    elif text == "/stop":
        delete_user(chat_id)
        send_telegram_message(
            token, chat_id,
            "You've been unsubscribed. Send /start to subscribe again."
        )

    elif text == "/summary":
        send_telegram_message(token, chat_id, "Generating your summary...")
        try:
            gemini_api_key = get_env_var("GEMINI_API_KEY")
            articles = fetch_recent_articles()
            if articles:
                articles = rank_and_filter_articles(articles)
            if articles:
                summaries = get_cached_summary(articles)
                if not summaries:
                    summaries = summarize_with_gemini(articles, gemini_api_key)
                    save_summary_cache(articles, summaries)
                message = format_telegram_message(articles, summaries)
                send_telegram_message(token, chat_id, message)
            else:
                send_telegram_message(token, chat_id, "No recent AI news found.")
        except Exception as e:
            print(f"Error generating summary: {e}")
            send_telegram_message(token, chat_id, "Sorry, couldn't generate summary right now.")

    else:
        user = get_user(chat_id)
        if user and user.get("state") == "subscribed":
            send_telegram_message(
                token, chat_id,
                "You're subscribed to receive AI news daily at <b>9am PT</b>.\n\n"
                "Commands:\n"
                "/summary - Generate summary now\n"
                "/stop - Unsubscribe"
            )
        else:
            send_telegram_message(
                token, chat_id,
                "Send /start to subscribe to daily AI news digests."
            )


def handle_messages(token: str) -> None:
    """Process incoming Telegram messages (polling mode)."""
    updates = get_telegram_updates(token)

    for update in updates:
        if "message" not in update:
            continue
        message = update["message"]
        chat_id = str(message["chat"]["id"])
        text = message.get("text", "").strip()
        handle_command(token, chat_id, text)

    if updates:
        last_update_id = updates[-1]["update_id"]
        get_telegram_updates(token, offset=last_update_id + 1)


def fetch_recent_articles(hours: int = HOURS_LOOKBACK) -> list[dict]:
    """Fetch articles from all RSS feeds published within the last N hours."""
    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
    articles = []

    for source_name, feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                pub_date = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    pub_date = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                    pub_date = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)

                if not pub_date or pub_date < cutoff_time:
                    continue

                articles.append({
                    "title": entry.get("title", "No title"),
                    "link": entry.get("link", ""),
                    "source": source_name,
                    "published": pub_date,
                    "summary": entry.get("summary", "")[:500],
                })
        except Exception as e:
            print(f"Error fetching {source_name}: {e}")

    return articles


def score_article(article: dict) -> float:
    """Calculate a quality score based on source reputation and keyword signals."""
    source = article.get("source", "")
    score = SOURCE_WEIGHTS.get(source, 1.0)

    text = (article.get("title", "") + " " + article.get("summary", "") + " " + article.get("link", "")).lower()

    # Boost for high-importance keywords (each +0.2, capped at +1.0)
    keyword_boost = sum(0.2 for kw in HIGH_IMPORTANCE_KEYWORDS if kw.lower() in text)
    score += min(keyword_boost, 1.0)

    # Penalty for low-value keywords (-0.3 each)
    score -= sum(0.3 for kw in LOW_VALUE_KEYWORDS if kw.lower() in text)

    # Penalty for first-person articles (opinion/blog posts)
    title_lower = article.get("title", "").lower()
    if title_lower.startswith("i ") or " i " in title_lower[:30]:
        score -= 0.5

    return max(score, 0)


def is_ai_relevant(article: dict) -> bool:
    """Check if an article is relevant to AI/ML topics."""
    text = (article.get("title", "") + " " + article.get("summary", "")).lower()
    if any(kw in text for kw in AI_RELEVANCE_KEYWORDS):
        return True
    # Check for "AI" as a standalone uppercase word in the original text
    original_text = article.get("title", "") + " " + article.get("summary", "")
    return bool(re.search(r'\bAI\b', original_text))


def deduplicate_articles(articles: list[dict], threshold: float = SIMILARITY_THRESHOLD) -> list[dict]:
    """Remove duplicate articles based on title similarity."""
    unique = []
    for article in articles:
        if not any(SequenceMatcher(None, article["title"].lower(), e["title"].lower()).ratio() > threshold for e in unique):
            unique.append(article)
    return unique


def rank_and_filter_articles(articles: list[dict]) -> list[dict]:
    """Rank articles by quality score and return top high-value articles."""
    if not articles:
        return []

    # Filter to AI-relevant articles only
    articles = [a for a in articles if is_ai_relevant(a)]
    if not articles:
        return []

    # Deduplicate, score, and sort
    articles = deduplicate_articles(articles, SIMILARITY_THRESHOLD)
    for article in articles:
        article["_score"] = score_article(article)
    articles.sort(key=lambda x: x["_score"], reverse=True)

    # Filter by quality, ensuring minimum count
    quality_articles = [a for a in articles if a["_score"] >= MIN_QUALITY_SCORE]
    if len(quality_articles) < MIN_ARTICLES:
        quality_articles = articles[:MIN_ARTICLES]

    # Stricter diversity check, then cap
    final_articles = deduplicate_articles(quality_articles, DIVERSITY_THRESHOLD)[:MAX_ARTICLES]

    for article in final_articles:
        article.pop("_score", None)

    return final_articles


def summarize_with_gemini(articles: list[dict], api_key: str) -> str:
    """Use Gemini to create insightful news summaries."""
    if not articles:
        return ""

    genai.configure(api_key=api_key)

    # Use gemini-2.5-flash (better free tier quota than 2.0-flash-lite)
    model = genai.GenerativeModel("gemini-2.5-flash")

    # Use all provided articles (already filtered to 3-10 by rank_and_filter_articles)
    articles_text = "\n\n".join([
        f"Title: {a['title']}\nSource: {a['source']}\nSummary: {a['summary']}"
        for a in articles
    ])

    article_count = len(articles)
    prompt = f"""You are writing a daily AI news digest in the style of Chamath Palihapitiya's "What I Read This Week."

For each article, write a summary that covers what matters. Use these elements as needed (not every article needs all of them):

- State the concrete news: who did what. Include company/product names.
- Explain any technical concept in simple terms, using an analogy if helpful.
- Why it matters: what problem does this solve, or what does it change?
- The bottom line: is it cheaper, faster, more powerful? Who benefits?

IMPORTANT guidelines on length and format:
- Match summary length to the substance of the story. A major breakthrough deserves more detail; a straightforward product launch might need only 2-3 sentences. Do NOT pad summaries to hit a word count.
- For longer summaries, break them into short paragraphs (2-3 sentences each) for readability. Don't write a wall of text.
- Vary the length naturally. Not every summary should be the same size.
- Capture the full scope of the article, not just one detail. If the article covers multiple examples, companies, or developments, reflect that breadth rather than fixating on a single one.
- Do NOT end with a generic concluding sentence that just restates the point already made. If the takeaway is clear, stop.

Example of the style to match:
"DeepSeek recently published a new AI architecture paper called Manifold-Constrained Hyper-Connections. The paper focuses on improving how information moves inside large AI models.

For the last decade, all AI models have used a single, narrow 'express lane' to pass information between their internal layers. DeepSeek's new paper is a blueprint for turning that single lane into a multi-lane 'superhighway'. The result is an AI that is significantly more powerful but costs almost nothing extra to build or run."

Articles:
{articles_text}

Write exactly {article_count} summaries. Separate each with "---" on its own line. No labels or headers - just the summary text."""

    fallback = "\n---\n".join([a['title'] for a in articles])

    # Retry logic - try twice before falling back
    for attempt in range(2):
        try:
            response = model.generate_content(prompt)

            # Validate response - accessing .text can raise if blocked
            try:
                response_text = response.text
            except (ValueError, AttributeError) as e:
                print(f"Could not get response text (attempt {attempt + 1}): {e}")
                if attempt == 0:
                    continue
                return fallback

            if not response_text:
                print(f"Gemini returned empty response (attempt {attempt + 1})")
                if attempt == 0:
                    continue
                return fallback

            summary_text = response_text.strip()

            # Check if response is too short (likely an error)
            if len(summary_text) < 100:
                print(f"Gemini response too short ({len(summary_text)} chars): {summary_text[:100]}")
                if attempt == 0:
                    continue
                return fallback

            # Validate we got approximately the right number of summaries
            summary_count = summary_text.count("---") + 1
            if summary_count < article_count:
                print(f"Warning: Expected {article_count} summaries, got {summary_count}")

            return summary_text

        except Exception as e:
            print(f"Gemini API error (attempt {attempt + 1}): {e}")
            # Handle rate limiting - don't retry, just fall back immediately
            if "429" in str(e) or "quota" in str(e).lower():
                print("Rate limited, falling back to titles")
                return fallback
            if attempt == 0:
                print("Retrying...")
                continue
            print(f"Falling back to titles only for {len(articles)} articles")
            return fallback

    return fallback


def format_telegram_message(articles: list[dict], summaries: str) -> str:
    """Format the final Telegram message with title, summary, and source."""
    today = datetime.now().strftime("%B %d, %Y")
    summary_blocks = [s.strip() for s in summaries.split("---") if s.strip()]

    message_parts = [f"<b>Daily Neural Briefing</b>\n{today}\n"]
    for i, article in enumerate(articles):
        takeaway = summary_blocks[i] if i < len(summary_blocks) else ""
        message_parts.append(
            f"<b>{article['title']}</b> - <a href=\"{article['link']}\">{article['source']}</a>\n"
            f"{takeaway}\n"
        )
    return "\n".join(message_parts)


def send_digests(token: str, gemini_key: str) -> None:
    """Send digests to all subscribed users (at most once per day per user)."""
    # Pacific Time (UTC-8); off by 1 hr during DST, acceptable since cron is hourly
    pt = timezone(timedelta(hours=-8))
    now_pt = datetime.now(pt)
    today = now_pt.strftime("%Y-%m-%d")

    # Only send digests during the 9am PT hour
    if now_pt.hour != 9:
        print(f"Skipping digest - current PT hour is {now_pt.hour}, waiting for 9am")
        return

    recipients = get_users_needing_digest(today)

    if not recipients:
        print("No users need digest (all already received today or none subscribed)")
        return

    print(f"Sending digest to {len(recipients)} users...")

    # Fetch and prepare news
    articles = fetch_recent_articles()
    print(f"Found {len(articles)} articles")

    if not articles:
        print("No articles found, skipping digest")
        return

    articles = rank_and_filter_articles(articles)
    print(f"After ranking: {len(articles)} quality articles")

    if not articles:
        print("No AI-relevant articles found after filtering, skipping digest")
        return

    # Check cache first to avoid burning Gemini quota
    summaries = get_cached_summary(articles)
    if not summaries:
        summaries = summarize_with_gemini(articles, gemini_key)
        save_summary_cache(articles, summaries)

    message = format_telegram_message(articles, summaries)

    # Send to all recipients; only mark date on success
    for chat_id in recipients:
        try:
            send_telegram_message(token, chat_id, message)
            update_last_digest(chat_id, today)
            print(f"Sent digest to {chat_id}")
        except Exception as e:
            print(f"Failed to send digest to {chat_id}: {e}")


def main():
    """Main bot execution - runs continuously."""
    import time

    print(f"AI News Bot starting - {datetime.now(timezone.utc).isoformat()}")

    telegram_token = get_env_var("TELEGRAM_TOKEN")
    gemini_api_key = get_env_var("GEMINI_API_KEY")

    last_check_hour = None

    while True:
        try:
            # Check for new messages every loop (responds immediately)
            handle_messages(telegram_token)

            # Check for scheduled digests once per hour
            current_hour = datetime.now(timezone.utc).hour
            if current_hour != last_check_hour:
                print(f"Checking for scheduled digests... (hour {current_hour})")
                send_digests(telegram_token, gemini_api_key)
                last_check_hour = current_hour

            # Wait 5 seconds before checking again
            time.sleep(5)

        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(10)  # Wait a bit longer on error


app = Flask(__name__)
init_db()


def process_webhook_update(update: dict) -> None:
    """Process a single update from Telegram webhook."""
    if "message" not in update:
        return
    telegram_token = get_env_var("TELEGRAM_TOKEN")
    message = update["message"]
    chat_id = str(message["chat"]["id"])
    text = message.get("text", "").strip()
    handle_command(telegram_token, chat_id, text)


@app.route("/webhook", methods=["POST"])
def webhook():
    """Handle incoming Telegram webhook updates."""
    try:
        update = request.get_json()
        if update:
            update_id = update.get("update_id")
            if update_id and is_update_processed(update_id):
                print(f"Skipping already processed update: {update_id}")
                return "OK", 200
            if update_id:
                mark_update_processed(update_id)
            process_webhook_update(update)
    except Exception as e:
        print(f"Webhook error: {e}")
    return "OK", 200


@app.route("/setup-webhook", methods=["GET", "POST"])
def setup_webhook_endpoint():
    """Trigger webhook setup via browser."""
    try:
        result = setup_webhook()
        return f"Webhook setup: {'success' if result else 'failed (check WEBHOOK_URL)'}", 200
    except Exception as e:
        return f"Error: {e}", 500


@app.route("/cron/digest", methods=["GET", "POST"])
def cron_digest():
    """Endpoint for scheduled digest sending (called by external cron service)."""
    try:
        telegram_token = get_env_var("TELEGRAM_TOKEN")
        gemini_api_key = get_env_var("GEMINI_API_KEY")
        send_digests(telegram_token, gemini_api_key)
        return "Digest check complete", 200
    except Exception as e:
        print(f"Cron digest error: {e}")
        return f"Error: {e}", 500


@app.route("/migrate-users", methods=["GET", "POST"])
def migrate_users_endpoint():
    """One-time migration from users.json to database."""
    users_file = Path(__file__).parent / "users.json"
    if not users_file.exists():
        return "users.json not found", 404
    try:
        users = json.loads(users_file.read_text())
        count = 0
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                for chat_id, data in users.items():
                    cur.execute("""
                        INSERT INTO users (chat_id, state, subscribed_at, last_digest_date)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (chat_id) DO NOTHING
                    """, (chat_id, data.get("state", "subscribed"),
                          data.get("subscribed_at"), data.get("last_digest_date")))
                    count += 1
            conn.commit()
        return f"Migrated {count} user(s) from users.json to database", 200
    except Exception as e:
        return f"Error: {e}", 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return "OK", 200


@app.route("/", methods=["GET"])
def index():
    """Root endpoint."""
    return "AI News Bot is running!", 200


def setup_webhook():
    """Set up Telegram webhook (run once after deployment)."""
    telegram_token = get_env_var("TELEGRAM_TOKEN")
    webhook_url = os.environ.get("WEBHOOK_URL")

    if not webhook_url:
        print("WEBHOOK_URL not set, skipping webhook setup")
        return False

    url = f"https://api.telegram.org/bot{telegram_token}/setWebhook"
    response = requests.post(url, json={"url": f"{webhook_url}/webhook"})
    print(f"Webhook setup response: {response.json()}")
    return response.ok


if __name__ == "__main__":
    import sys

    command = sys.argv[1] if len(sys.argv) > 1 else None
    if command == "setup-webhook":
        setup_webhook()
    elif command == "polling":
        main()
    else:
        port = int(os.environ.get("PORT", 5000))
        app.run(host="0.0.0.0", port=port)
