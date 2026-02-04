#!/usr/bin/env python3
"""
AI News Telegram Bot
- Interactive: responds to /start, asks for preferred digest time
- Stores user preferences
- Sends personalized daily digests
"""

import os
import re
import json
import hashlib
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from zoneinfo import ZoneInfo

import feedparser
import google.generativeai as genai
import requests

# Configuration
RSS_FEEDS = [
    ("TechCrunch", "https://techcrunch.com/category/artificial-intelligence/feed/"),
    ("The Verge", "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml"),
    ("VentureBeat", "https://venturebeat.com/category/ai/feed/"),
    ("MIT Tech Review", "https://www.technologyreview.com/topic/artificial-intelligence/feed"),
    ("Ars Technica", "https://feeds.arstechnica.com/arstechnica/technology-lab"),
    ("Wired", "https://www.wired.com/feed/tag/ai/latest/rss"),
]

HOURS_LOOKBACK = 24  # Look back 24 hours for daily digest
SIMILARITY_THRESHOLD = 0.7
USERS_FILE = Path(__file__).parent / "users.json"

# Time mappings for natural language
TIME_MAPPINGS = {
    "morning": "09:00",
    "early morning": "07:00",
    "late morning": "11:00",
    "noon": "12:00",
    "afternoon": "14:00",
    "evening": "18:00",
    "night": "21:00",
    "midnight": "00:00",
}

# Common timezone abbreviations to IANA timezone names
TIMEZONE_MAPPINGS = {
    # US timezones
    "est": "America/New_York",
    "edt": "America/New_York",
    "eastern": "America/New_York",
    "cst": "America/Chicago",
    "cdt": "America/Chicago",
    "central": "America/Chicago",
    "mst": "America/Denver",
    "mdt": "America/Denver",
    "mountain": "America/Denver",
    "pst": "America/Los_Angeles",
    "pdt": "America/Los_Angeles",
    "pacific": "America/Los_Angeles",
    "hst": "Pacific/Honolulu",
    "akst": "America/Anchorage",
    "akdt": "America/Anchorage",
    # Europe
    "gmt": "Europe/London",
    "bst": "Europe/London",
    "utc": "UTC",
    "cet": "Europe/Paris",
    "cest": "Europe/Paris",
    "eet": "Europe/Helsinki",
    "eest": "Europe/Helsinki",
    # Asia
    "ist": "Asia/Kolkata",
    "jst": "Asia/Tokyo",
    "kst": "Asia/Seoul",
    "cst china": "Asia/Shanghai",
    "sgt": "Asia/Singapore",
    "hkt": "Asia/Hong_Kong",
    # Australia
    "aest": "Australia/Sydney",
    "aedt": "Australia/Sydney",
    "acst": "Australia/Adelaide",
    "awst": "Australia/Perth",
    # Other
    "nzst": "Pacific/Auckland",
    "nzdt": "Pacific/Auckland",
}


def parse_timezone(text: str) -> tuple[str, str]:
    """
    Parse user's timezone from input.
    Returns (iana_timezone, friendly_name) or (None, None) if unparseable.
    """
    text = text.lower().strip()

    # Check abbreviation mappings
    for abbrev, tz_name in TIMEZONE_MAPPINGS.items():
        if abbrev in text:
            try:
                tz = ZoneInfo(tz_name)
                return tz_name, abbrev.upper()
            except Exception:
                pass

    # Try as IANA timezone directly (e.g., "America/New_York")
    # Also try common formats like "US/Pacific"
    candidates = [
        text,
        text.replace(" ", "_"),
        text.title().replace(" ", "_"),
        f"America/{text.title().replace(' ', '_')}",
        f"Europe/{text.title().replace(' ', '_')}",
        f"Asia/{text.title().replace(' ', '_')}",
    ]

    for candidate in candidates:
        try:
            tz = ZoneInfo(candidate)
            return candidate, candidate
        except Exception:
            pass

    return None, None


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


def parse_time_preference(text: str) -> tuple[str, str, str, str]:
    """
    Parse user's time preference and timezone from natural language.
    Returns (time_24h, friendly_time, iana_timezone, friendly_tz) or (None, None, None, None) if unparseable.
    Defaults to America/Los_Angeles (PST) if no timezone specified.
    """
    text_lower = text.lower().strip()

    # Parse timezone from input (default to PST)
    iana_tz, friendly_tz = parse_timezone(text_lower)
    if not iana_tz:
        iana_tz = "America/Los_Angeles"
        friendly_tz = "PST"

    # Check for natural language times
    for phrase, time_24h in TIME_MAPPINGS.items():
        if phrase in text_lower:
            hour = int(time_24h.split(":")[0])
            friendly = format_friendly_time(hour)
            return time_24h, friendly, iana_tz, friendly_tz

    # Check for specific times like "9am", "14:00", "9:30 pm"
    # Pattern for "9am", "9 am", "9:00am", "9:30 pm"
    pattern = r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?'
    match = re.search(pattern, text_lower)

    if match:
        hour = int(match.group(1))
        minutes = match.group(2) or "00"
        period = match.group(3)

        # Convert to 24-hour format
        if period:
            if period == "pm" and hour != 12:
                hour += 12
            elif period == "am" and hour == 12:
                hour = 0

        if 0 <= hour <= 23:
            time_24h = f"{hour:02d}:{minutes}"
            friendly = format_friendly_time(hour, int(minutes))
            return time_24h, friendly, iana_tz, friendly_tz

    return None, None, None, None


def format_friendly_time(hour: int, minutes: int = 0) -> str:
    """Format hour as friendly time string."""
    period = "am" if hour < 12 else "pm"
    display_hour = hour if hour <= 12 else hour - 12
    if display_hour == 0:
        display_hour = 12
    if minutes:
        return f"{display_hour}:{minutes:02d}{period}"
    return f"{display_hour}{period}"


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
            # New user or restart
            send_telegram_message(
                token, chat_id,
                "Welcome to the AI News Bot!\n\n"
                "I'll send you a daily digest of the latest AI news from:\n"
                "- TechCrunch\n"
                "- The Verge\n"
                "- VentureBeat\n"
                "- MIT Technology Review\n"
                "- Ars Technica\n"
                "- Wired\n\n"
                "What time would you like to receive your daily digest? "
                "Please also include your timezone, otherwise I'll use Pacific Time (PST)."
            )
            users[chat_id] = {"state": "awaiting_time"}

        elif text == "/stop":
            if chat_id in users:
                del users[chat_id]
            send_telegram_message(
                token, chat_id,
                "You've been unsubscribed. Send /start to subscribe again."
            )

        elif text == "/time":
            # Change time preference
            send_telegram_message(
                token, chat_id,
                "What time would you like to receive your daily digest? "
                "Please also include your timezone, otherwise I'll use Pacific Time (PST)."
            )
            users[chat_id] = {**user, "state": "awaiting_time"}

        elif user.get("state") == "awaiting_time":
            time_24h, friendly_time, iana_tz, friendly_tz = parse_time_preference(text)

            if time_24h:
                users[chat_id] = {
                    "state": "subscribed",
                    "time": time_24h,
                    "timezone": iana_tz,
                    "subscribed_at": datetime.now(timezone.utc).isoformat(),
                }
                send_telegram_message(
                    token, chat_id,
                    f"You'll receive your daily AI news digest at <b>{friendly_time}</b> ({friendly_tz}).\n\n"
                    "Commands:\n"
                    "/time - Change delivery time\n"
                    "/stop - Unsubscribe"
                )
            else:
                send_telegram_message(
                    token, chat_id,
                    "I didn't understand that time. Please try again.\n"
                    "Examples: \"9am\", \"9am PST\", \"morning\", \"14:00 EST\", \"6:30pm\""
                )

        elif user.get("state") == "subscribed":
            # Already subscribed, remind them of commands
            time_24h = user.get("time", "09:00")
            user_tz = user.get("timezone", "America/Los_Angeles")
            hour = int(time_24h.split(":")[0])
            friendly = format_friendly_time(hour)
            # Get friendly timezone name
            tz_friendly = next(
                (abbr.upper() for abbr, tz in TIMEZONE_MAPPINGS.items() if tz == user_tz),
                user_tz
            )
            send_telegram_message(
                token, chat_id,
                f"You're subscribed to receive AI news at <b>{friendly}</b> ({tz_friendly}).\n\n"
                "Commands:\n"
                "/time - Change delivery time\n"
                "/stop - Unsubscribe"
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


def deduplicate_articles(articles: list[dict]) -> list[dict]:
    """Remove duplicate articles based on title similarity."""
    if not articles:
        return []

    articles.sort(key=lambda x: x["published"], reverse=True)

    unique = []
    for article in articles:
        is_duplicate = False
        for existing in unique:
            similarity = SequenceMatcher(
                None,
                article["title"].lower(),
                existing["title"].lower()
            ).ratio()
            if similarity > SIMILARITY_THRESHOLD:
                is_duplicate = True
                break
        if not is_duplicate:
            unique.append(article)

    return unique


def summarize_with_gemini(articles: list[dict], api_key: str) -> str:
    """Use Gemini to create a concise news briefing."""
    if not articles:
        return ""

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    articles_text = "\n\n".join([
        f"Title: {a['title']}\nSource: {a['source']}\nSummary: {a['summary']}"
        for a in articles[:10]
    ])

    prompt = f"""You are an AI news curator. Create a brief, engaging summary of these AI news stories.

For each story, write a single compelling sentence (max 15 words) that captures the key point.

Articles:
{articles_text}

Format your response as a simple list with one line per story. Just the summary sentences, nothing else.
Keep it concise and newsworthy. Maximum 8 stories."""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini API error: {e}")
        return "\n".join([f"• {a['title']}" for a in articles[:8]])


def format_telegram_message(articles: list[dict], summaries: str) -> str:
    """Format the final Telegram message."""
    today = datetime.now().strftime("%b %d, %Y")
    summary_lines = [s.strip() for s in summaries.split("\n") if s.strip()]

    message_parts = ["<b>AI News Digest</b>\n"]

    for i, article in enumerate(articles[:8]):
        if i < len(summary_lines):
            summary = summary_lines[i].lstrip("•-123456789. ")
        else:
            summary = article["title"]

        message_parts.append(
            f"• {summary}\n"
            f"  <a href=\"{article['link']}\">{article['source']}</a>\n"
        )

    message_parts.append(f"\n{today} | {len(articles[:8])} stories")

    return "\n".join(message_parts)


def send_digests(token: str, gemini_key: str) -> None:
    """Send digests to users whose scheduled time matches current hour in their timezone."""
    users = load_users()
    now_utc = datetime.now(timezone.utc)

    # Find users who should receive digest now
    recipients = []
    for chat_id, data in users.items():
        if data.get("state") != "subscribed":
            continue

        user_time = data.get("time", "09:00")
        user_tz_name = data.get("timezone", "America/Los_Angeles")

        try:
            user_tz = ZoneInfo(user_tz_name)
            # Get current time in user's timezone
            now_local = now_utc.astimezone(user_tz)
            current_hour_local = now_local.strftime("%H")

            # Check if user's scheduled hour matches current hour in their timezone
            user_hour = user_time.split(":")[0]
            if user_hour == current_hour_local:
                recipients.append(chat_id)
        except Exception as e:
            print(f"Error processing timezone for {chat_id}: {e}")
            continue

    if not recipients:
        print(f"No recipients scheduled for current hour")
        return

    print(f"Sending digest to {len(recipients)} users...")

    # Fetch and prepare news
    articles = fetch_recent_articles()
    print(f"Found {len(articles)} articles")

    if not articles:
        print("No articles found, skipping digest")
        return

    articles = deduplicate_articles(articles)
    print(f"After dedup: {len(articles)} unique articles")

    summaries = summarize_with_gemini(articles, gemini_key)
    message = format_telegram_message(articles, summaries)

    # Send to all recipients
    for chat_id in recipients:
        send_telegram_message(token, chat_id, message)
        print(f"Sent digest to {chat_id}")


def main():
    """Main bot execution - runs continuously."""
    import time

    print(f"AI News Bot starting - {datetime.now(timezone.utc).isoformat()}")

    telegram_token = get_env_var("TELEGRAM_TOKEN")
    gemini_api_key = get_env_var("GEMINI_API_KEY")

    last_digest_hour = None

    while True:
        try:
            # Check for new messages every loop (responds immediately)
            handle_messages(telegram_token)

            # Check for scheduled digests once per hour
            current_hour = datetime.now(timezone.utc).hour
            if current_hour != last_digest_hour:
                print(f"Checking for scheduled digests... (hour {current_hour})")
                send_digests(telegram_token, gemini_api_key)
                last_digest_hour = current_hour

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
        send_telegram_message(
            telegram_token, chat_id,
            "Welcome to the AI News Bot!\n\n"
            "I'll send you a daily digest of the latest AI news from:\n"
            "- TechCrunch\n"
            "- The Verge\n"
            "- VentureBeat\n"
            "- MIT Technology Review\n"
            "- Ars Technica\n"
            "- Wired\n\n"
            "What time would you like to receive your daily digest? "
            "Please also include your timezone, otherwise I'll use Pacific Time (PST)."
        )
        users[chat_id] = {"state": "awaiting_time"}

    elif text == "/stop":
        if chat_id in users:
            del users[chat_id]
        send_telegram_message(
            telegram_token, chat_id,
            "You've been unsubscribed. Send /start to subscribe again."
        )

    elif text == "/time":
        send_telegram_message(
            telegram_token, chat_id,
            "What time would you like to receive your daily digest? "
            "Please also include your timezone, otherwise I'll use Pacific Time (PST)."
        )
        users[chat_id] = {**user, "state": "awaiting_time"}

    elif user.get("state") == "awaiting_time":
        time_24h, friendly_time, iana_tz, friendly_tz = parse_time_preference(text)

        if time_24h:
            users[chat_id] = {
                "state": "subscribed",
                "time": time_24h,
                "timezone": iana_tz,
                "subscribed_at": datetime.now(timezone.utc).isoformat(),
            }
            send_telegram_message(
                telegram_token, chat_id,
                f"You'll receive your daily AI news digest at <b>{friendly_time}</b> ({friendly_tz}).\n\n"
                "Commands:\n"
                "/time - Change delivery time\n"
                "/stop - Unsubscribe"
            )
        else:
            send_telegram_message(
                telegram_token, chat_id,
                "I didn't understand that time. Please try again.\n"
                "Examples: \"9am\", \"9am PST\", \"morning\", \"14:00 EST\", \"6:30pm\""
            )

    elif user.get("state") == "subscribed":
        time_24h = user.get("time", "09:00")
        user_tz = user.get("timezone", "America/Los_Angeles")
        hour = int(time_24h.split(":")[0])
        friendly = format_friendly_time(hour)
        tz_friendly = next(
            (abbr.upper() for abbr, tz in TIMEZONE_MAPPINGS.items() if tz == user_tz),
            user_tz
        )
        send_telegram_message(
            telegram_token, chat_id,
            f"You're subscribed to receive AI news at <b>{friendly}</b> ({tz_friendly}).\n\n"
            "Commands:\n"
            "/time - Change delivery time\n"
            "/stop - Unsubscribe"
        )

    save_users(users)


@app.route("/webhook", methods=["POST"])
def webhook():
    """Handle incoming Telegram webhook updates."""
    try:
        update = request.get_json()
        if update:
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
        return "Digests sent", 200
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
