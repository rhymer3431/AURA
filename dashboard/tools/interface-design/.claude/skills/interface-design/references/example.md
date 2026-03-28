# Craft In Action

This example shows how the approved AURA Pipeline Overview target should translate into concrete decisions. Learn the reasoning, not the literal numbers.

---

## The Board Mindset

Before looking at any example, internalize this: the operator should understand the page in one glance.

The page should read in big beats:

1. Where am I in the app?
2. What is the robot doing right now?
3. What is the main live surface I should watch?
4. Which support modules tell me whether the system is healthy?

If the answer hides behind whisper-light shells or chat-style controls, the hierarchy is wrong.

---

## Example: Pipeline Overview

### The Navigation Decision

**Why a dedicated rail, not a disappearing sidebar?**

The operator needs stable orientation. A distinct left rail with section labels and a highlighted active row grounds the dashboard immediately. It should feel like an app region, not like content floating on a blank canvas.

### The KPI Decision

**Why pastel tiles instead of neutral stat cards?**

The top row carries the fastest telemetry read. Each tile belongs to a telemetry family, so the family color should appear at the tile level, not only in a tiny badge. The large number does the first read. The small delta does the second.

### The Main Workbench Decision

**Why one dominant center panel?**

The live robot view is the main job surface. Treat it as the hero workstation:

- large rounded shell
- white inner stage
- concise header controls
- compact inference footer

If the live view is just one card among many, the page loses its operational center.

### The Right Column Decision

**Why stacked support modules instead of more KPI cards?**

The right column is secondary but still essential. Process composition, sensor readiness, and health modules support the main workbench. Stacking them in one column keeps the main live view dominant while preserving quick glanceability.

### The Surface Decision

**Why fog shell plus white module?**

This creates readable grouping without going dark or flat:

- warm canvas for the page
- fog shell for major sections
- white modules for detailed content

The result is brighter and more legible than a monochrome workspace shell.

---

## Example: Control Treatment

### Live Badge Decision

**Why should live state be visibly red?**

Live state is not decorative. It is an operational condition. Let it stand out as a small but clear coral/red pill near the panel title.

### Sensor State Decision

**Why white buttons with green state dots?**

They read like compact instrument modules. The white body keeps them readable; the green dot carries readiness without turning the whole module green.

### Toggle Decision

**Why keep overlay toggles secondary?**

Controls like overlay toggles matter, but they should not compete with the camera stage. Let them sit in the header row as secondary utilities.

---

## Adapt To Context

Your product may need:

- denser right-column modules
- larger or smaller KPI rows
- alternate telemetry families
- different media surfaces besides a camera frame

The principle stays the same: the operator should understand the board in a glance, and the main workstation should remain visually dominant.

---

## The Craft Check

Apply these checks to your work:

1. Can you identify the navigation rail, KPI ribbon, main workbench, and right support column immediately?
2. Do the KPI tiles read faster than the lower modules?
3. Is the live-view panel clearly the main working surface?
4. Does the page feel like an operations board rather than a workspace shell or chat product?

If any answer is "no", fix the hierarchy before presenting.
