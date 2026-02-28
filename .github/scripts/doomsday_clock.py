#!/usr/bin/env python3
"""
AI Doomsday Clock — Weekly update script.

Gathers AI news + Reddit sentiment, sends briefing to 4 AI models via
OpenRouter, aggregates their scores, and generates a static HTML dashboard.
"""

import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from html import escape
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TEMPLATE_PATH = REPO_ROOT / ".github" / "scripts" / "clock_template.html"
PROMPT_PATH = REPO_ROOT / "docs" / "ai-doomsday-clock-prompt.md"
CLOCK_DIR = REPO_ROOT / "clock"
INDEX_HTML = REPO_ROOT / "index.html"
MANIFEST_PATH = CLOCK_DIR / "manifest.json"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = [
    ("Claude Opus 4.6", "anthropic/claude-opus-4.6"),
    ("GPT-5.1", "openai/gpt-5.1"),
    ("Gemini 3 Pro", "google/gemini-3-pro-preview"),
    ("Grok 4", "x-ai/grok-4"),
]

REDDIT_SUBS = [
    "singularity",
    "artificial",
    "MachineLearning",
    "technology",
    "ArtificialIntelligence",
]

NEWS_QUERIES = [
    "artificial intelligence",
    "AI safety risk",
    "AI layoffs jobs",
    "AI regulation",
    "AI autonomous agents",
]


# ---------------------------------------------------------------------------
# Phase A: Research
# ---------------------------------------------------------------------------


def fetch_google_news_rss(query: str, num_results: int = 10) -> list[dict]:
    """Fetch AI news headlines from Google News RSS."""
    url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en&gl=US&ceid=US:en"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; DoomsdayClock/1.0)"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        items = []
        for item in root.findall(".//item")[:num_results]:
            title = item.findtext("title", "")
            link = item.findtext("link", "")
            pub_date = item.findtext("pubDate", "")
            source = item.findtext("source", "")
            items.append(
                {"title": title, "link": link, "date": pub_date, "source": source}
            )
        return items
    except Exception as e:
        print(f"  Warning: Google News fetch failed for '{query}': {e}")
        return []


def fetch_reddit_posts(subreddit: str, limit: int = 10) -> list[dict]:
    """Fetch top posts from a subreddit (past week, public JSON API)."""
    url = f"https://www.reddit.com/r/{subreddit}/top.json?t=week&limit={limit}"
    headers = {"User-Agent": "python:ai-doomsday-clock:v1.0 (by /u/bobbai_dev)"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        posts = []
        for child in data.get("data", {}).get("children", []):
            d = child.get("data", {})
            posts.append(
                {
                    "title": d.get("title", ""),
                    "score": d.get("score", 0),
                    "num_comments": d.get("num_comments", 0),
                    "subreddit": subreddit,
                    "url": f"https://reddit.com{d.get('permalink', '')}",
                }
            )
        return posts
    except Exception as e:
        print(f"  Warning: Reddit fetch failed for r/{subreddit}: {e}")
        return []


def get_previous_reading() -> dict | None:
    """Load the previous week's reading from manifest (excluding current week)."""
    if not MANIFEST_PATH.exists():
        return None
    current_week = get_week_id()
    try:
        manifest = json.loads(MANIFEST_PATH.read_text())
        readings = manifest.get("readings", [])
        # Filter out the current week — we don't want to compare against ourselves
        past_readings = [r for r in readings if r.get("week_id") != current_week]
        if past_readings:
            return past_readings[-1]
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def gather_briefing() -> tuple[str, list[dict]]:
    """Gather news + sentiment + previous week context into a briefing."""
    print("Phase A: Gathering research...")

    print("  Fetching news headlines...")
    all_news = []
    for query in NEWS_QUERIES:
        articles = fetch_google_news_rss(query)
        all_news.extend(articles)
        print(f"    '{query}': {len(articles)} articles")

    seen = set()
    unique_news = []
    for article in all_news:
        if article["title"] not in seen:
            seen.add(article["title"])
            unique_news.append(article)

    print("  Fetching Reddit sentiment...")
    all_reddit = []
    for sub in REDDIT_SUBS:
        posts = fetch_reddit_posts(sub)
        all_reddit.extend(posts)
        print(f"    r/{sub}: {len(posts)} posts")

    all_reddit.sort(key=lambda x: x["score"], reverse=True)

    briefing = "# WEEKLY AI BRIEFING\n\n"
    briefing += f"**Period:** {(datetime.now() - timedelta(days=7)).strftime('%B %d')} — {datetime.now().strftime('%B %d, %Y')}\n\n"

    briefing += "## AI News Headlines (Past 7 Days)\n\n"
    for article in unique_news[:30]:
        source = f" — {article['source']}" if article["source"] else ""
        briefing += f"- {article['title']}{source}\n"

    briefing += "\n## Reddit Sentiment (Top Posts This Week)\n\n"
    for post in all_reddit[:25]:
        briefing += f"- [{post['subreddit']}] (score: {post['score']}, {post['num_comments']} comments) {post['title']}\n"

    # Append previous week's reading for context
    prev = get_previous_reading()
    if prev:
        briefing += "\n## Previous Week's Reading\n\n"
        briefing += f"- **Week:** {prev.get('week_id', 'unknown')}\n"
        briefing += f"- **Clock time:** {prev.get('clock_time', 'unknown')}\n"
        briefing += f"- **Seconds to midnight:** {prev.get('total_seconds_to_midnight', 'unknown')}\n"
        briefing += f"- **Weighted score:** {prev.get('weighted_score', 'unknown')}\n"
        consensus = prev.get("consensus_summary", "")
        if consensus:
            briefing += f"- **Consensus summary:** {consensus}\n"
        print(f"  Previous week context included: {prev.get('week_id', '?')} ({prev.get('clock_time', '?')})")
    else:
        briefing += "\n## Previous Week's Reading\n\nNo previous reading available. This is the first week.\n"
        print("  No previous week data (first reading)")

    print(f"  Briefing compiled: {len(unique_news)} news articles, {len(all_reddit)} Reddit posts")
    return briefing, unique_news


# ---------------------------------------------------------------------------
# Phase B: AI Calls via OpenRouter
# ---------------------------------------------------------------------------


def call_model(model_name: str, model_id: str, prompt: str, briefing: str) -> dict | None:
    """Call a single model via OpenRouter and parse the JSON response."""
    api_key = os.environ.get("OPEN_ROUTER")
    if not api_key:
        print("  Error: OPEN_ROUTER not set")
        return None

    print(f"  Calling {model_name} ({model_id})...")

    try:
        resp = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://bobbai.dev",
                "X-Title": "AI Doomsday Clock",
            },
            json={
                "model": model_id,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": briefing},
                ],
                "max_tokens": 4000,
                "temperature": 0.7,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        content = data["choices"][0]["message"]["content"]

        # Try multiple patterns: fenced json, fenced without label, raw JSON object
        json_match = re.search(r"```json\s*\n?(.*?)\n?\s*```", content, re.DOTALL)
        if not json_match:
            json_match = re.search(r"```\s*\n?(\{.*?\"clock_time\".*?\})\s*\n?```", content, re.DOTALL)
        if not json_match:
            # Find JSON block containing clock_time — greedy to capture nested {} like scores
            json_match = re.search(r'(\{\s*"clock_time".*"verdict"\s*:\s*"[^"]*"\s*\})', content, re.DOTALL)
        if not json_match:
            # Most permissive: from clock_time to last } in content
            json_match = re.search(r'(\{\s*"clock_time"[\s\S]*\})', content)

        if not json_match:
            print(f"  Warning: No JSON block found in {model_name} response")
            print(f"  Response preview: {content[-500:]}")
            return None

        try:
            result = json.loads(json_match.group(1))
        except json.JSONDecodeError:
            # Try cleaning: remove trailing commas, fix common issues
            json_str = json_match.group(1)
            json_str = re.sub(r',\s*}', '}', json_str)  # trailing commas
            json_str = re.sub(r',\s*]', ']', json_str)
            try:
                result = json.loads(json_str)
            except json.JSONDecodeError as e2:
                print(f"  Error parsing {model_name} JSON even after cleanup: {e2}")
                print(f"  JSON snippet: {json_str[:300]}")
                return None
        result["_provider"] = model_name
        result["_full_response"] = content
        print(f"  {model_name}: {result.get('clock_time', '?')} (score: {result.get('weighted_score', '?')})")
        return result

    except requests.exceptions.HTTPError as e:
        print(f"  Error calling {model_name}: HTTP {e.response.status_code} — {e.response.text[:200]}")
        return None
    except json.JSONDecodeError as e:
        print(f"  Error parsing {model_name} JSON: {e}")
        return None
    except Exception as e:
        print(f"  Error calling {model_name}: {e}")
        return None


def query_all_models(prompt: str, briefing: str) -> list[dict]:
    """Query all 4 models and return successful results."""
    print("\nPhase B: Querying AI models...")
    results = []
    for name, model_id in MODELS:
        result = call_model(name, model_id, prompt, briefing)
        if result:
            results.append(result)
    print(f"  {len(results)}/{len(MODELS)} models responded successfully")
    return results


# ---------------------------------------------------------------------------
# Phase C: Aggregate
# ---------------------------------------------------------------------------


def score_to_seconds(score: float) -> int:
    """Convert weighted score (1-10) to seconds to midnight (600-1)."""
    return max(1, round(600 - (score - 1) * (599 / 9)))


def seconds_to_clock_time(total_seconds: int) -> str:
    """Convert seconds-to-midnight into a clock time string like 11:58:18."""
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    clock_minutes = 59 - minutes
    clock_seconds = 60 - seconds if seconds > 0 else 0
    if seconds == 0:
        clock_minutes += 1
    return f"11:{clock_minutes:02d}:{clock_seconds:02d}"


def save_to_manifest(aggregated: dict, week_id: str) -> None:
    """Save the current reading to manifest for future comparisons."""
    manifest = {"readings": []}
    if MANIFEST_PATH.exists():
        try:
            manifest = json.loads(MANIFEST_PATH.read_text())
        except json.JSONDecodeError:
            pass

    # Don't duplicate if same week
    manifest["readings"] = [
        r for r in manifest.get("readings", []) if r.get("week_id") != week_id
    ]

    manifest["readings"].append({
        "week_id": week_id,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "clock_time": aggregated["clock_time"],
        "total_seconds_to_midnight": aggregated["total_seconds_to_midnight"],
        "weighted_score": aggregated["weighted_score"],
        "consensus_summary": re.sub(r'<[^>]+>', '', aggregated["consensus_summary"]),
    })

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def generate_consensus(results: list[dict], news_articles: list[dict], clock_time: str, avg_score: float) -> str:
    """Ask a model to synthesize a consensus summary with news links from all reports."""
    print("  Generating consensus summary...")

    # Build context from all model verdicts + news URLs
    model_summaries = ""
    for r in results:
        name = r.get("_provider", "Unknown")
        model_summaries += f"\n### {name} (score: {r.get('weighted_score', '?')})\n"
        model_summaries += f"{r.get('verdict', r.get('summary', 'No summary.'))}\n"

    news_links = ""
    for article in news_articles[:20]:
        if article.get("link"):
            news_links += f"- [{article['title']}]({article['link']})\n"

    prompt = (
        "You are writing a 2-3 sentence consensus summary for the AI Doomsday Clock page. "
        f"The consensus clock reads {clock_time} with a weighted score of {avg_score:.2f}.\n\n"
        "Below are the individual model verdicts and a list of news articles with URLs.\n\n"
        "Write a concise, punchy 2-3 sentence summary that:\n"
        "1. References the most significant events from this week\n"
        "2. Embeds hyperlinks to relevant news articles using HTML <a> tags "
        "(e.g., <a href=\"URL\" target=\"_blank\">Company Name</a>)\n"
        "3. Explains why the clock moved\n\n"
        "Return ONLY the HTML paragraph text — no wrapping tags, no markdown, no explanation.\n\n"
        f"## Model Verdicts\n{model_summaries}\n\n"
        f"## News Articles with URLs\n{news_links}"
    )

    api_key = os.environ.get("OPEN_ROUTER")
    if not api_key:
        print("  Warning: No API key, using fallback consensus")
        closest = min(results, key=lambda r: abs(r.get("weighted_score", 0) - avg_score))
        return escape(closest.get("summary", "No summary available."))

    try:
        resp = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://bobbai.dev",
                "X-Title": "AI Doomsday Clock",
            },
            json={
                "model": "anthropic/claude-opus-4.6",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
                "temperature": 0.5,
            },
            timeout=60,
        )
        resp.raise_for_status()
        consensus = resp.json()["choices"][0]["message"]["content"].strip()
        print(f"  Consensus generated ({len(consensus)} chars)")
        return consensus
    except Exception as e:
        print(f"  Warning: Consensus generation failed: {e}")
        closest = min(results, key=lambda r: abs(r.get("weighted_score", 0) - avg_score))
        return escape(closest.get("summary", "No summary available."))


def aggregate_results(results: list[dict], news_articles: list[dict]) -> dict:
    """Average scores across all models to get final reading."""
    print("\nPhase C: Aggregating results...")

    if not results:
        print("  FATAL: No successful model responses. Cannot generate clock.")
        sys.exit(1)

    # Average weighted score
    scores = [r["weighted_score"] for r in results if "weighted_score" in r]
    avg_score = sum(scores) / len(scores)

    # Average individual vector scores
    vectors = [
        "capability_acceleration",
        "autonomy_agency",
        "economic_displacement",
        "military_surveillance",
        "governance_safety",
        "public_sentiment",
    ]
    avg_vectors = {}
    for v in vectors:
        v_scores = [r["scores"][v] for r in results if "scores" in r and v in r["scores"]]
        avg_vectors[v] = round(sum(v_scores) / len(v_scores), 1) if v_scores else 5.0

    total_seconds_to_midnight = score_to_seconds(avg_score)
    minutes = total_seconds_to_midnight // 60
    seconds = total_seconds_to_midnight % 60
    clock_time = seconds_to_clock_time(total_seconds_to_midnight)

    # SVG rotation angles
    minutes_decimal = 60 - total_seconds_to_midnight / 60
    minute_angle = (minutes_decimal / 60) * 360
    hour_angle = ((11 + minutes_decimal / 60) / 12) * 360

    # Generate consensus with news links via extra model call
    consensus_summary = generate_consensus(results, news_articles, clock_time, avg_score)

    # Comparison with previous week
    prev = get_previous_reading()
    if prev:
        prev_seconds = prev["total_seconds_to_midnight"]
        diff = prev_seconds - total_seconds_to_midnight
        if diff > 0:
            comparison = f'<span class="closer">{abs(diff)} seconds closer</span><br>to midnight vs last week'
        elif diff < 0:
            comparison = f'<span class="further">{abs(diff)} seconds further</span><br>from midnight vs last week'
        else:
            comparison = "No change from last week"
    else:
        comparison = "First reading"

    aggregated = {
        "weighted_score": round(avg_score, 2),
        "total_seconds_to_midnight": total_seconds_to_midnight,
        "minutes": minutes,
        "seconds": seconds,
        "clock_time": clock_time,
        "minute_hand_rotation": round(minute_angle, 1),
        "hour_hand_rotation": round(hour_angle, 1),
        "scores": avg_vectors,
        "consensus_summary": consensus_summary,
        "comparison": comparison,
        "providers": results,
    }

    print(f"  Final reading: {clock_time} — {minutes} min {seconds} sec to midnight")
    print(f"  Weighted score: {avg_score:.2f}")
    return aggregated


# ---------------------------------------------------------------------------
# Phase D: Generate HTML
# ---------------------------------------------------------------------------


def inline_markdown(text: str) -> str:
    """Convert inline markdown (bold) to HTML within a line."""
    # **text** -> <strong>text</strong>
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # *text* -> <em>text</em> (but not inside <strong>)
    text = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'<em>\1</em>', text)
    return text


def markdown_to_html(text: str) -> str:
    """Convert a model's markdown response to HTML for inline report display."""
    html_parts = []
    lines = text.split("\n")
    in_table = False
    in_json = False
    table_rows = []
    is_header_row = True

    for line in lines:
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            if in_table:
                html_parts.append(flush_table(table_rows))
                table_rows = []
                in_table = False
                is_header_row = True
            continue

        # Skip code fences
        if stripped.startswith("```"):
            in_json = not in_json
            continue

        # Skip everything inside fenced code blocks
        if in_json:
            continue

        # Skip JSON lines (unfenced): lines that look like JSON key/value pairs
        if stripped in ("{", "}") or stripped.startswith('"') and '":' in stripped:
            continue

        # Skip horizontal rules
        if stripped == "---":
            continue

        # Table rows
        if stripped.startswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            # Skip separator rows (---|---|---)
            if all(c.replace("-", "").replace(":", "") == "" for c in cells):
                continue
            in_table = True
            # Limit to 3 columns (Vector, Weight, Score) — drop justification
            if len(cells) > 3:
                cells = cells[:3]
            table_rows.append({"cells": cells, "is_header": is_header_row})
            is_header_row = False
            continue

        if in_table:
            html_parts.append(flush_table(table_rows))
            table_rows = []
            in_table = False
            is_header_row = True

        # Headers
        if stripped.startswith("# ") and not stripped.startswith("## "):
            # Skip top-level title (the report title)
            continue
        elif stripped.startswith("## "):
            html_parts.append(f'<h2>{escape(stripped[3:])}</h2>')
        elif stripped.startswith("### "):
            html_parts.append(f'<h3>{escape(stripped[4:])}</h3>')
        elif stripped.startswith("#### "):
            html_parts.append(f'<h4>{escape(stripped[5:])}</h4>')
        # Bullet points
        elif stripped.startswith("- ") or stripped.startswith("* "):
            content = inline_markdown(escape(stripped[2:].lstrip()))
            html_parts.append(f'<p class="signal-item">{content}</p>')
        # Regular paragraphs — convert inline bold/italic
        else:
            content = inline_markdown(escape(stripped))
            html_parts.append(f'<p>{content}</p>')

    if in_table:
        html_parts.append(flush_table(table_rows))

    return "\n            ".join(html_parts)


def flush_table(rows: list[dict]) -> str:
    """Convert collected table rows to an HTML table."""
    if not rows:
        return ""
    html = '<table class="matrix-table">\n'
    for row in rows:
        tag = "th" if row["is_header"] else "td"
        cells_html = ""
        for i, cell in enumerate(row["cells"]):
            # Last column in data rows gets score-cell class
            cls = ""
            if tag == "td" and i == len(row["cells"]) - 1:
                try:
                    float(cell)
                    cls = ' class="score-cell"'
                except ValueError:
                    pass
            cells_html += f"<{tag}{cls}>{inline_markdown(escape(cell))}</{tag}>"
        html += f"                <tr>{cells_html}</tr>\n"
    html += "            </table>"
    return html


def build_provider_cards(results: list[dict]) -> str:
    """Build HTML for provider cards linking to inline report views."""
    cards = []
    for r in results:
        name = escape(r.get("_provider", "Unknown"))
        slug = name.lower().replace(" ", "-").replace(".", "")
        time = escape(r.get("clock_time", "??:??:??"))
        score = r.get("weighted_score", "?")
        summary = escape(r.get("summary", "No summary available."))
        total_sec = r.get("total_seconds_to_midnight", 0)
        mins = total_sec // 60 if total_sec else "?"
        secs = total_sec % 60 if total_sec else "?"

        cards.append(
            f'            <a href="#{slug}-report" class="provider-card">\n'
            f'                <div class="provider-header">\n'
            f'                    <span class="provider-name">{name}</span>\n'
            f'                    <span class="provider-time">{time}</span>\n'
            f'                </div>\n'
            f'                <span class="provider-meta">{mins} min {secs} sec to midnight &middot; score {score}</span>\n'
            f'                <p class="provider-summary">{summary}</p>\n'
            f'                <span class="provider-report">read full report &rarr;</span>\n'
            f'            </a>'
        )
    return "\n".join(cards)


def build_report_views(results: list[dict], week_id: str) -> str:
    """Build inline report view divs for each provider (CSS :target toggled)."""
    views = []
    for r in results:
        name = escape(r.get("_provider", "Unknown"))
        slug = name.lower().replace(" ", "-").replace(".", "")
        clock_time = escape(r.get("clock_time", "??:??:??"))
        score = r.get("weighted_score", "?")
        total_sec = r.get("total_seconds_to_midnight", 0)
        mins = total_sec // 60 if total_sec else "?"
        secs = total_sec % 60 if total_sec else "?"

        model_id = ""
        for m_name, m_id in MODELS:
            if m_name == r.get("_provider"):
                model_id = m_id
                break

        full_response = r.get("_full_response", "No report available.")
        report_html = markdown_to_html(full_response)

        views.append(
            f'    <!-- Report: {name} -->\n'
            f'    <div id="{slug}-report" class="report-view">\n'
            f'        <a href="#" class="back-to-cards">&larr; back to overview</a>\n'
            f'        <div class="report-header">\n'
            f'            <p class="report-provider">{name}</p>\n'
            f'            <p class="report-model">{escape(model_id)}</p>\n'
            f'            <div class="report-time">{clock_time}</div>\n'
            f'            <p class="report-meta">Score: {score} &middot; {mins} min {secs} sec to midnight &middot; {week_id}</p>\n'
            f'        </div>\n'
            f'        <div class="report-content">\n'
            f'            {report_html}\n'
            f'        </div>\n'
            f'    </div>\n'
        )
    return "\n".join(views)


def get_week_id() -> str:
    """Get ISO week identifier like 2026-W09."""
    now = datetime.now()
    return f"{now.year}-W{now.isocalendar()[1]:02d}"


def get_prev_week_file() -> Path | None:
    """Find the most recent existing archive file."""
    if not CLOCK_DIR.exists():
        return None
    files = sorted(CLOCK_DIR.glob("????-W??.html"), reverse=True)
    return files[0] if files else None


def generate_html(aggregated: dict) -> None:
    """Generate clock HTML pages from template."""
    print("\nPhase D: Generating HTML...")

    template = TEMPLATE_PATH.read_text()
    week_id = get_week_id()
    date_str = datetime.now().strftime("%B %d, %Y")

    provider_cards = build_provider_cards(aggregated["providers"])
    report_views = build_report_views(aggregated["providers"], week_id)

    # Previous/next week links
    prev_file = get_prev_week_file()
    if prev_file and prev_file.stem != week_id:
        prev_link = f'<a href="/clock/{prev_file.name}">&larr; {prev_file.stem}</a>'
    else:
        prev_link = '<span class="nav-placeholder">&nbsp;</span>'
    next_link = '<span class="nav-placeholder">&nbsp;</span>'

    # Replace placeholders
    html = template
    replacements = {
        "{{CLOCK_TIME}}": aggregated["clock_time"],
        "{{MINUTES}}": str(aggregated["minutes"]),
        "{{SECONDS}}": str(aggregated["seconds"]),
        "{{MINUTE_HAND_ROTATION}}": str(aggregated["minute_hand_rotation"]),
        "{{HOUR_HAND_ROTATION}}": str(aggregated["hour_hand_rotation"]),
        "{{CONSENSUS_SUMMARY}}": aggregated["consensus_summary"],
        "{{CONSENSUS_PLAIN}}": escape(re.sub(r'<[^>]+>', '', aggregated["consensus_summary"])),
        "{{COMPARISON}}": aggregated["comparison"],
        "{{PROVIDER_CARDS}}": provider_cards,
        "{{REPORT_VIEWS}}": report_views,
        "{{DATE}}": date_str,
        "{{WEEK_ID}}": week_id,
        "{{PREV_LINK}}": prev_link,
        "{{NEXT_LINK}}": next_link,
        "{{MODEL_NAMES}}": ", ".join(name for name, _ in MODELS),
    }
    for placeholder, value in replacements.items():
        html = html.replace(placeholder, value)

    CLOCK_DIR.mkdir(exist_ok=True)

    # Latest
    index_path = CLOCK_DIR / "index.html"
    index_path.write_text(html)
    print(f"  Written: {index_path}")

    # Archive
    archive_path = CLOCK_DIR / f"{week_id}.html"
    archive_path.write_text(html)
    print(f"  Written: {archive_path}")

    # Update previous week's next link (replace last nav-placeholder)
    if prev_file and prev_file.stem != week_id:
        prev_html = prev_file.read_text()
        placeholder = '<span class="nav-placeholder">&nbsp;</span>'
        next_link_html = f'<a href="/clock/{week_id}.html">{week_id} &rarr;</a>'
        idx = prev_html.rfind(placeholder)
        if idx != -1:
            prev_html = prev_html[:idx] + next_link_html + prev_html[idx + len(placeholder):]
            prev_file.write_text(prev_html)
            print(f"  Updated next link in: {prev_file}")

    # Save to manifest for future comparisons
    save_to_manifest(aggregated, week_id)

    # Update main page clock
    update_main_page(aggregated)


def update_main_page(aggregated: dict) -> None:
    """Update the clock SVG and time label on the main index.html."""
    if not INDEX_HTML.exists():
        print("  Warning: index.html not found, skipping main page update")
        return

    html = INDEX_HTML.read_text()

    html = re.sub(
        r'(x2="18" y2="5"[^>]*transform="rotate\()(\d+)(, 18, 18\)")',
        rf"\g<1>{round(aggregated['minute_hand_rotation'])}\g<3>",
        html,
    )

    html = re.sub(
        r'(x2="18" y2="8"[^>]*transform="rotate\()(\d+)(, 18, 18\)")',
        rf"\g<1>{round(aggregated['hour_hand_rotation'])}\g<3>",
        html,
    )

    html = re.sub(
        r"(<span>)11:\d{2}:\d{2}(</span>\s*</span>\s*</(?:div|a)>)",
        rf"\g<1>{aggregated['clock_time']}\g<2>",
        html,
    )

    INDEX_HTML.write_text(html)
    print(f"  Updated main page clock: {aggregated['clock_time']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("AI DOOMSDAY CLOCK — Weekly Update")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if not PROMPT_PATH.exists():
        print(f"FATAL: Prompt file not found: {PROMPT_PATH}")
        sys.exit(1)
    prompt = PROMPT_PATH.read_text()

    briefing, news_articles = gather_briefing()
    results = query_all_models(prompt, briefing)
    aggregated = aggregate_results(results, news_articles)
    generate_html(aggregated)

    print("\n" + "=" * 60)
    print(f"Done. Clock set to {aggregated['clock_time']}.")
    print(f"{aggregated['minutes']} min {aggregated['seconds']} sec to midnight.")
    print("=" * 60)


if __name__ == "__main__":
    main()
