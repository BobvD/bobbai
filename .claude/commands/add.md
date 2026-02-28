---
description: Add a new item (article, podcast, video, tool) to the bobbai feed
---

# Add Item to bobbai

Add a new item to the feed in `index.html`. The user provides a URL (and optionally a date and category).

## Steps

1. **Determine category** from context or ask: `article`, `podcast`, `video`, `tool`
2. **Determine date** — use provided date, or today's date formatted as `DD Mon` (e.g. `07 Feb`)
3. **Get title** — fetch the URL via oembed (for YouTube) or WebFetch to get the page title. Clean it up to max 50 characters. Use `—` (em dash) to separate title from source when needed.
4. **Check doomsday indicators** per AI_RULES.md:
   - AI existential risk / accelerating capabilities → add class `clock-tick`
   - AI job cuts / layoffs → add class `clock-tick skull`
5. **Insert the list item** in chronological order (newest first) inside `<ul>` in `.content`:

```html
<li data-category="CATEGORY" class="OPTIONAL_CLOCK_CLASSES">
    <span class="date">DD Mon</span>
    <span class="type">[CATEGORY]</span>
    <a href="URL" target="_blank" rel="noopener noreferrer">TITLE</a>
</li>
```

6. **For tools only** — also add a tool card in `.tools-grid` with an SVG icon and description.

## Input format

The user will say something like:
- `/add https://example.com/article` — infer category, use today's date
- `/add https://youtube.com/watch?v=xyz feb 7` — video, specific date
- `/add https://example.com article, 15 mar` — explicit category and date

## Rules

- Single file: all changes go in `index.html`
- No JavaScript
- Max 50 char titles
- Only 3 colors: `#1a1512`, `#d7ccc8`, `#ffb3c1`
- Insert in chronological position (compare dates, newest near top)

## After adding

7. **Commit and push** — after inserting the item, automatically:
   - `git add index.html`
   - Commit with message: `Add TITLE`
   - `git push origin main`
