---
marp: true
theme: gaia
paginate: true
backgroundColor: #0b1020
color: #e5eefb
size: 16:9
style: |
  section {
    font-family: 'Segoe UI', sans-serif;
    background: linear-gradient(135deg, #0b1020 0%, #101b2d 100%);
    color: #e5eefb;
  }
  h1, h2, h3 {
    color: #f7d77a;
  }
  strong { color: #8fe3ff; }
  a { color: #8fe3ff; }
  code { font-size: 0.8em; }
  .small { font-size: 0.8em; }
  .lead {
    background: linear-gradient(135deg, #172a46 0%, #0f172a 100%);
  }
  .accent { color: #8fe3ff; }
---

<!-- _class: lead -->

# NarrativeGraph

## Detecting coordinated narratives in social media

### A fusion of content similarity + network amplification

<p class="small">Pallav Rathod • Narrative Intelligence Demo</p>

---

## The problem

Social media narratives are hard to detect because many real campaigns look like ordinary activity:

- breaking-news bursts create near-duplicate text
- viral celebrity posts create high graph activity
- organic communities often resemble coordinated behavior

> A single signal is not enough.

We need a method that is skeptical of one-layer false positives and robust to organic look-alikes.

---

## Why existing detectors fail

### Content-only approaches
- catch “look-alike” posts from a news event
- miss coordinated campaigns that use varied wording

### Graph-only approaches
- catch viral propagation patterns
- miss coordinated messaging that is textually disguised

### NarrativeGraph
- combines both layers
- only flags pairs/communities when both signals agree within the same time window

---

## Core intuition

For an account pair (i, j):

- c = content-sync count
- a = co-amplification count

We transform both to bounded values:

```text
c' = 1 - exp(-c / kappa_content)
a' = 1 - exp(-a / kappa_amp)
w = sqrt(c' * a')
```

### Why this matters
- if either layer is zero, fused score is zero
- content-only artifacts and graph-only artifacts are strongly suppressed
- the method asks for agreement, not just correlation

---

## Pipeline overview

```text
Posts / public social data
        ↓
Validate schema
        ↓
Embed text + cluster topics
        ↓
Compute content-sync graph
        ↓
Compute co-amplification graph
        ↓
Fuse both into one edge score
        ↓
Detect communities + rank suspicious accounts
        ↓
Semantic retrieval + narrative agent
```

---

## Data sources and workflow

### Synthetic benchmark
- organic communities
- coordinated campaigns with decoy patterns
- explicit labels for evaluation

### Live public data
- Bluesky search ingestion
- anonymized identities by default
- public-only data collection

### Outputs
- ranked suspicious accounts
- coordinated communities
- topic summaries
- narrative retrieval results

---

## What the model actually computes

### 1) Content-sync edges
Pairs of users posting semantically similar text inside a short time window.

### 2) Co-amplification edges
Pairs of users engaging with the same target or post within the same window.

### 3) Fused edge score
A score that requires both layers to agree.

### 4) User-level coordination scores
Each user receives:
- `score_content`
- `score_amp`
- `score_fused`

This creates a robust ranking of suspicious actors.

---

## Community detection

The algorithm builds a graph of fused edges and detects communities with Louvain clustering.

Key outputs:

- `coord_comm`: coordinated community assignment
- `social_comm`: ordinary social interaction communities
- `mismatch`: topic mismatch between user behavior and social neighborhood

This makes it possible to distinguish:
- coordinated communities
- organic discussion clusters
- decoy bursts or viral outliers

---

## Why the graph-aware retrieval matters

Semantic retrieval alone retrieves posts based on text similarity.

NarrativeGraph adds a prior from suspicious authors:

```text
score(q, p) = cos(e_q, e_p) + λ * s_hat(author(p))
```

This means that:
- strong narrative claims from suspicious users get boosted
- the retrieval becomes network-aware
- investigators can surface likely narrative drivers faster

---

## Agent + analyst workflow

The project includes a local LangGraph-based agent that can answer questions like:

> “Who is pushing the claim that energy bills will triple?”

### Tools available to the agent
- semantic search over posts
- community summaries
- account-level reports
- top suspicious accounts

### Result
The system grounds answers in evidence: post IDs, user IDs, community membership, and scores.

---

## Dashboard experience

The Streamlit app includes tabs for:

- Overview
- Network graph
- Narrative map
- Suspicious accounts
- Ask the agent

### Investigators can explore
- post counts and account counts
- fused edges and community size
- semantic topic layout over time
- suspicious account rankings
- natural-language narrative questions

---

## Evaluation strategy

The project evaluates the detector against synthetic benchmarks using metrics such as:

- Average Precision (AP)
- ROC AUC
- Precision@K
- Retrieval Recall@K

### Important result
The fused approach is designed to outperform content-only and graph-only baselines when both false-positive patterns are present at once.

---

## Real-world use cases

### Narrative monitoring
- detect coordinated campaigns around policy or health stories
- identify suspicious amplification clusters

### Trust & safety
- surface network behavior that looks synthetic or manipulated
- prioritize manual review on the highest-risk accounts

### Research and investigations
- compare organic vs coordinated communities
- explain why a topic is emerging and who is driving it

---

## Project status and strengths

### What this repo already does
- synthetic data generation
- live Bluesky ingestion
- embedding and clustering
- fusion and community detection
- graph-aware retrieval
- interactive dashboard
- narrative Q&A agent

### Design strengths
- evidence-first
- robust against one-layer false positives
- modular architecture
- easy to tune with configuration thresholds

---

## Key takeaway

NarrativeGraph is not a claim of “truth”; it is a tool for surfacing suspicious coordination patterns and narrative drivers.

> It is built around a simple principle:
> if the content layer and the network layer agree in the same window, the signal is more credible.

That is the core idea that makes the system useful.

---

## Closing

### NarrativeGraph helps answer:
- what narratives are emerging?
- which communities are acting in concert?
- which accounts are pushing the message?
- how do the social structure and language align?

### In one sentence
NarrativeGraph detects coordinated narratives by combining semantic similarity and social amplification, then turning the result into an evidence-backed investigation workflow.

---

## Q&A

Questions? 

Thank you.
