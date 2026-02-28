# AI Doomsday Clock — Design

## Summary

A subtle, minimal SVG clock face placed in the header next to the GitHub link. Shows the time at 11:58 (2 minutes to midnight), evoking the Bulletin of Atomic Scientists' Doomsday Clock but themed around AI.

## Visual Specification

- **Size:** ~36px inline SVG
- **Shape:** Thin circle outline + two clock hands (hour near 12, minute at 58)
- **Color:** Cream (`#d7ccc8`) at reduced opacity (~0.4 default)
- **Hover:** Opacity increases to ~0.7, a small label fades in: `11:58 — ai doomsday clock`
- **Label:** Tiny monospace text, low opacity, appears via CSS transition

## Layout

- Placed in the existing header flex container, after the GitHub link
- Wraps naturally on mobile with existing responsive behavior
- Vertically centered with the GitHub link text

## Constraints

- No JavaScript — static SVG + CSS hover transition
- Only uses the 3 project colors (cream strokes at reduced opacity)
- All code in index.html and style.css
- Hand positions are hardcoded SVG `transform: rotate()` values

## Interaction

- Default: quiet, low-opacity decorative element
- Hover: opacity rises, label appears via `transition: opacity var(--transition)`
