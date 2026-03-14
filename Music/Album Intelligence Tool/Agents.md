# AGENTS.md

## Project Overview
This repository contains experimental tools for analyzing music projects and albums.

The primary project is the **Album Intelligence Tool**, which analyzes audio files to extract musical structure, cohesion, and creative insights.

The goal is to provide musicians and producers with **useful musical intelligence**, not generic genre descriptions.

Good outputs should feel like insights from a **producer or musicologist**, not a generic audio feature report.

---

## Core Principles

### Evidence-Based Analysis
All musical conclusions should be based on measurable signals from the audio.

Avoid generic descriptions unless they are supported by data.

Examples of useful evidence:
- tempo distribution
- tonal center detection
- dynamic range
- rhythmic density
- spectral balance
- motif repetition
- section boundaries

---

### Avoid Generic Language
Avoid repeating vague phrases such as:
- "bright-edged timbre"
- "cinematic feel"
- "through-composed rather than loop-driven"

Instead, explain **what in the audio leads to that interpretation**.

Example improvement:

Bad:
"bright-edged timbre"

Better:
"high-frequency energy concentrated between 4–8 kHz suggesting bright guitar or cymbal presence"

---

## Desired Features

The analysis system should prioritize identifying:

### Song DNA
Each track should attempt to identify:

- groove identity
- melodic identity
- arrangement identity
- energy arc
- distinctive traits

Example structure:

Song DNA
- Groove: steady fast pulse with consistent kick placement
- Melody: descending minor contour
- Arrangement: even intensity across sections
- Identity: tension driven by rhythmic motion rather than harmony

---

### Album-Level Intelligence

The album report should include:

- tempo center
- tonal center clustering
- dynamic range comparison
- cohesion score
- outlier detection
- sequencing suggestions

The report should help answer:

- Do these songs feel like the same album?
- What traits unify them?
- What contrasts might improve the project?

---

### Creative Direction

Suggestions should be actionable.

Examples:

Good suggestions:
- introduce a slower tempo track for contrast
- push dynamic range further in one song
- emphasize recurring melodic motifs

Avoid vague suggestions such as:
"make it more unique".

---

## Technical Constraints

Keep the notebook **Colab-friendly**.

Preferred libraries:

- librosa
- numpy
- scipy
- sklearn
- matplotlib

Avoid fragile dependencies unless clearly optional.

---

## Code Quality Expectations

Changes should:

- keep the notebook runnable
- avoid breaking existing analysis
- add comments explaining new logic
- organize complex logic into helper functions

When possible, refactor repeated logic into reusable functions.

---

## Commit Strategy

When making changes:

Use logical commits such as:

feat: add motif detection heuristics  
feat: improve arrangement analysis  
feat: upgrade album cohesion scoring  

Avoid committing large unrelated changes in a single commit.

---

## Pull Request Expectations

Pull requests should:

- summarize the change
- explain the motivation
- mention any limitations

If analysis is heuristic or approximate, clearly state that.

---

## Long-Term Vision

The tool should evolve toward a system that can detect:

- recurring motifs
- groove fingerprints
- arrangement patterns
- album identity

The goal is **deep musical insight**, not just audio statistics.