# AI Doomsday Clock — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a subtle SVG doomsday clock to the site header showing 11:58 (2 minutes to midnight).

**Architecture:** A `.doomsday-clock` container with an inline SVG clock face is added after the `.github` element inside `.header`. CSS handles opacity, hover label reveal, and responsive behavior.

**Tech Stack:** HTML, CSS, inline SVG. No JavaScript.

---

### Task 1: Add the SVG clock to index.html

**Files:**
- Modify: `index.html:23-28` (after the `.github` paragraph, before closing `</div>`)

**Step 1: Add the doomsday clock HTML**

Insert after the closing `</p>` of `.github` (line 28), before the closing `</div>` of `.header` (line 29):

```html
<div class="doomsday-clock" title="ai doomsday clock">
    <svg viewBox="0 0 36 36" width="36" height="36" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
        <circle cx="18" cy="18" r="16" fill="none" stroke="currentColor" stroke-width="1.5"/>
        <line x1="18" y1="18" x2="18" y2="5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" transform="rotate(348, 18, 18)"/>
        <line x1="18" y1="18" x2="18" y2="8" stroke="currentColor" stroke-width="2" stroke-linecap="round" transform="rotate(330, 18, 18)"/>
    </svg>
    <span class="doomsday-label">11:58</span>
</div>
```

SVG geometry notes:
- Circle: centered at (18,18), radius 16, thin stroke
- Minute hand: vertical line from center to near top, rotated 348° (= 58 minutes × 6°/min)
- Hour hand: shorter vertical line from center, rotated 330° (= 11 hours × 30°/hr = 330°, close to but not at 12)

**Step 2: Commit**

```bash
git add index.html
git commit -m "feat: add AI doomsday clock SVG to header"
```

---

### Task 2: Style the doomsday clock in style.css

**Files:**
- Modify: `style.css` (add after `.github svg` block, ~line 58)

**Step 1: Add the CSS**

Insert after the `.github svg` rule (line 58):

```css
.doomsday-clock {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--text);
    opacity: 0.35;
    transition: opacity var(--transition);
    flex-shrink: 0;
    line-height: 1;
}

.doomsday-clock:hover {
    opacity: 0.7;
}

.doomsday-label {
    font-size: 0.7em;
    opacity: 0;
    transition: opacity var(--transition);
    white-space: nowrap;
}

.doomsday-clock:hover .doomsday-label {
    opacity: 1;
}
```

**Step 2: Commit**

```bash
git add style.css
git commit -m "style: add subtle doomsday clock styling with hover reveal"
```

---

### Task 3: Verify visually and adjust

**Step 1:** Open `index.html` in a browser and verify:
- Clock appears small and subtle in the header, after the GitHub link
- Default opacity is low (~0.35) — reads as a quiet decorative element
- On hover, the clock brightens and "11:58" label fades in
- On mobile widths (≤600px), the header wraps naturally with the clock

**Step 2:** Adjust opacity/size values if needed, then final commit.
