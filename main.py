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
import requests

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
USERS_FILE = Path(__file__).parent / "users.json"
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


def load_users() -> dict:
    """Load user preferences from JSON file."""
    if USERS_FILE.exists():
        try:
            return json.loads(USERS_FILE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_users(users: dict) -> None:
    """Save user preferences to JSON file."""
    USERS_FILE.write_text(json.dumps(users, indent=2))




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
    global _processed_updates

    # Add to in-memory set immediately (fast, atomic)
    _processed_updates.add(update_id)

    # Trim in-memory set if too large
    if len(_processed_updates) > MAX_PROCESSED_UPDATES:
        _processed_updates = set(list(_processed_updates)[-MAX_PROCESSED_UPDATES:])

    # Also persist to file (backup)
    try:
        if PROCESSED_UPDATES_FILE.exists():
            processed = json.loads(PROCESSED_UPDATES_FILE.read_text())
            ids = processed.get("ids", [])
        else:
            ids = []

        ids.append(update_id)
        ids = ids[-MAX_PROCESSED_UPDATES:]

        PROCESSED_UPDATES_FILE.write_text(json.dumps({"ids": ids}))
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


def handle_messages(token: str) -> None:
    """Process incoming Telegram messages."""
    users = load_users()
    updates = get_telegram_updates(token)

    for update in updates:
        if "message" not in update:
            continue

        message = update["message"]
        chat_id = str(message["chat"]["id"])
        text = message.get("text", "").strip()

        user = users.get(chat_id, {"state": "new"})

        if text == "/start":
            users[chat_id] = {"state": "subscribed"}
            send_telegram_message(
                token, chat_id,
                "Welcome to the Neural Briefing Bot! I'll send you a daily summary of the top AI news at 9am PT daily.\n\n"
                "Commands:\n"
                "/summary - Generate summary now\n"
                "/stop - Unsubscribe"
            )

        elif text == "/stop":
            if chat_id in users:
                del users[chat_id]
            send_telegram_message(
                token, chat_id,
                "You've been unsubscribed. Send /start to subscribe again."
            )

        elif text == "/summary":
            # Send summary now
            send_telegram_message(token, chat_id, "Generating your summary...")
            try:
                gemini_api_key = get_env_var("GEMINI_API_KEY")
                articles = fetch_recent_articles()
                if articles:
                    articles = rank_and_filter_articles(articles)
                    # Check cache first
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

        elif user.get("state") == "subscribed":
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

    # Mark updates as read
    if updates:
        last_update_id = updates[-1]["update_id"]
        get_telegram_updates(token, offset=last_update_id + 1)

    save_users(users)


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
    """
    Calculate a quality score for an article based on:
    - Source reputation
    - Presence of high-importance keywords
    - Absence of low-value keywords

    Returns a score where higher = more important.
    """
    # Base score from source reputation (default 1.0 for unknown sources)
    source = article.get("source", "")
    score = SOURCE_WEIGHTS.get(source, 1.0)

    # Combine title, summary, AND link for keyword matching
    text = (article.get("title", "") + " " + article.get("summary", "") + " " + article.get("link", "")).lower()

    # Boost for high-importance keywords (each keyword adds 0.2)
    keyword_boost = 0
    for keyword in HIGH_IMPORTANCE_KEYWORDS:
        if keyword.lower() in text:
            keyword_boost += 0.2
    # Cap keyword boost at 1.0 to prevent runaway scores
    score += min(keyword_boost, 1.0)

    # Penalty for low-value keywords (each reduces score by 0.3)
    for keyword in LOW_VALUE_KEYWORDS:
        if keyword.lower() in text:
            score -= 0.3

    # Penalty for first-person articles (opinion/blog posts)
    title_lower = article.get("title", "").lower()
    if title_lower.startswith("i ") or " i " in title_lower[:30]:
        score -= 0.5

    return max(score, 0)  # Don't go negative


def is_ai_relevant(article: dict) -> bool:
    """Check if an article is relevant to AI/ML topics.

    Filters out non-AI articles that slip through from general feeds
    (e.g., Ars Technica features feed).
    """
    text = (article.get("title", "") + " " + article.get("summary", "")).lower()

    for keyword in AI_RELEVANCE_KEYWORDS:
        if keyword in text:
            return True

    # Check for "AI" as a standalone uppercase word in the original text
    original_text = article.get("title", "") + " " + article.get("summary", "")
    if re.search(r'\bAI\b', original_text):
        return True

    return False


def deduplicate_articles(articles: list[dict], threshold: float = SIMILARITY_THRESHOLD) -> list[dict]:
    """Remove duplicate articles based on title similarity."""
    if not articles:
        return []

    unique = []
    for article in articles:
        is_duplicate = False
        for existing in unique:
            similarity = SequenceMatcher(
                None,
                article["title"].lower(),
                existing["title"].lower()
            ).ratio()
            if similarity > threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique.append(article)

    return unique


def rank_and_filter_articles(articles: list[dict]) -> list[dict]:
    """
    Rank articles by quality score and filter to return only high-value articles.

    Returns 3-10 articles based on quality thresholds, ensuring diversity.
    """
    if not articles:
        return []

    # Filter out non-AI articles (e.g., crypto, cybersecurity from general feeds)
    articles = [a for a in articles if is_ai_relevant(a)]

    if not articles:
        return []

    # First pass: basic deduplication with standard threshold
    articles = deduplicate_articles(articles, SIMILARITY_THRESHOLD)

    # Score all articles
    for article in articles:
        article["_score"] = score_article(article)

    # Sort by score (highest first), not by recency
    articles.sort(key=lambda x: x["_score"], reverse=True)

    # Filter by minimum quality threshold
    quality_articles = [a for a in articles if a["_score"] >= MIN_QUALITY_SCORE]

    # If we don't have enough quality articles, take top articles anyway
    if len(quality_articles) < MIN_ARTICLES:
        quality_articles = articles[:MIN_ARTICLES]

    # Apply stricter diversity check to avoid similar topics in final selection
    diverse_articles = deduplicate_articles(quality_articles, DIVERSITY_THRESHOLD)

    # Cap at maximum
    final_articles = diverse_articles[:MAX_ARTICLES]

    # Clean up internal score field before returning
    for article in final_articles:
        if "_score" in article:
            del article["_score"]

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

    # Split summaries by --- separator
    summary_blocks = [s.strip() for s in summaries.split("---") if s.strip()]

    message_parts = [f"<b>Daily Neural Briefing</b>\n{today}\n"]

    # Process all articles - title + summary + hyperlinked source
    for i, article in enumerate(articles):
        if i < len(summary_blocks):
            takeaway = summary_blocks[i]
        else:
            takeaway = ""

        message_parts.append(
            f"<b>{article['title']}</b>\n"
            f"{takeaway}\n"
            f"<a href=\"{article['link']}\">{article['source']}</a>\n"
        )

    return "\n".join(message_parts)


def send_digests(token: str, gemini_key: str) -> None:
    """Send digests to all subscribed users (at most once per day per user)."""
    users = load_users()

    # Pacific Time (UTC-8); off by 1 hr during DST, acceptable since cron is hourly
    pt = timezone(timedelta(hours=-8))
    today = datetime.now(pt).strftime("%Y-%m-%d")

    recipients = [chat_id for chat_id, data in users.items()
                  if data.get("state") == "subscribed"
                  and data.get("last_digest_date") != today]

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
            users[chat_id]["last_digest_date"] = today
            print(f"Sent digest to {chat_id}")
        except Exception as e:
            print(f"Failed to send digest to {chat_id}: {e}")

    save_users(users)


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


# ============ Flask Web App for Webhooks ============

from flask import Flask, request

app = Flask(__name__)



def process_webhook_update(update: dict) -> None:
    """Process a single update from Telegram webhook."""
    telegram_token = get_env_var("TELEGRAM_TOKEN")
    users = load_users()

    if "message" not in update:
        return

    message = update["message"]
    chat_id = str(message["chat"]["id"])
    text = message.get("text", "").strip()

    user = users.get(chat_id, {"state": "new"})

    if text == "/start":
        users[chat_id] = {"state": "subscribed"}
        send_telegram_message(
            telegram_token, chat_id,
            "Welcome to the Neural Briefing Bot! I'll send you a daily summary of the top AI news at 9am PT daily.\n\n"
            "Commands:\n"
            "/summary - Generate summary now\n"
            "/stop - Unsubscribe"
        )

    elif text == "/stop":
        if chat_id in users:
            del users[chat_id]
        send_telegram_message(
            telegram_token, chat_id,
            "You've been unsubscribed. Send /start to subscribe again."
        )

    elif text == "/summary":
        # Send summary now
        send_telegram_message(telegram_token, chat_id, "Generating your summary...")
        try:
            gemini_api_key = get_env_var("GEMINI_API_KEY")
            articles = fetch_recent_articles()
            if articles:
                articles = rank_and_filter_articles(articles)
                # Check cache first
                summaries = get_cached_summary(articles)
                if not summaries:
                    summaries = summarize_with_gemini(articles, gemini_api_key)
                    save_summary_cache(articles, summaries)
                message = format_telegram_message(articles, summaries)
                send_telegram_message(telegram_token, chat_id, message)
            else:
                send_telegram_message(telegram_token, chat_id, "No recent AI news found.")
        except Exception as e:
            print(f"Error generating summary: {e}")
            send_telegram_message(telegram_token, chat_id, "Sorry, couldn't generate summary right now.")

    elif user.get("state") == "subscribed":
        send_telegram_message(
            telegram_token, chat_id,
            "You're subscribed to receive AI news daily at <b>9am PT</b>.\n\n"
            "Commands:\n"
            "/summary - Generate summary now\n"
            "/stop - Unsubscribe"
        )

    else:
        send_telegram_message(
            telegram_token, chat_id,
            "Send /start to subscribe to daily AI news digests."
        )

    save_users(users)


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

    if len(sys.argv) > 1 and sys.argv[1] == "setup-webhook":
        # Run: python main.py setup-webhook
        setup_webhook()
    elif len(sys.argv) > 1 and sys.argv[1] == "polling":
        # Run: python main.py polling (for local testing)
        main()
    else:
        # Default: run Flask app (for production with gunicorn)
        port = int(os.environ.get("PORT", 5000))
        app.run(host="0.0.0.0", port=port)
