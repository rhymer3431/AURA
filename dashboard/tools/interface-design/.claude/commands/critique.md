---
name: interface-design:critique
description: Codex workflow reference for critiquing a UI build for craft and correcting defaulted decisions.
---

# Critique

Use this workflow after a first pass ships the structure but before presenting the result as finished.

## The Gap

There is distance between correct and crafted. Correct means the layout holds, the grid aligns, and the colors do not clash. Crafted means the decisions carry intent down to the last detail.

This workflow is for closing that gap.

## Review Lenses

### Composition

- Does the layout have rhythm, or is every zone equally dense?
- Are proportions declaring what matters?
- Is there a focal point tied to the user's primary task?

### Craft

- Is spacing deliberate, or only technically consistent?
- Does typography create hierarchy beyond size alone?
- Do surfaces establish hierarchy without harsh borders or loud shadows?
- Do interactive elements respond with believable hover, active, and focus states?

### Content

- Do visible strings tell one coherent story?
- Does the interface feel like it belongs to one real product context?

### Structure

- Is the layout built cleanly, or held together with workarounds?
- Are there negative margins, brittle `calc()` values, or unnecessary absolute positioning?

## Process

1. Open the UI you just built or reviewed
2. Walk through composition, craft, content, and structure
3. Identify every place where the design defaulted instead of being intentionally decided
4. Rebuild those parts from the underlying decision, not from a cosmetic patch
5. Present the improved result rather than narrating the critique process
