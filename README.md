# NarrativeGraph

NarrativeGraph is a Python-based coordination-detection pipeline for social media narratives. It looks for suspicious account pairs and communities where two signals agree at the same time:

- content similarity: users post near-duplicate text in a short time window
- co-amplification: users repeatedly engage with the same targets (reply/quote/repost) in the same time window

The project is designed to reduce the false positives that arise when a single signal is misleading:

- a breaking-news spike can look like near-duplicate content without being coordinated
- a viral celebrity post can trigger a high graph signal without being coordinated

NarrativeGraph only keeps a pair or community if the content layer and the amplification layer both fire within the same window, then adds graph-aware retrieval and an LLM-explained narrative analysis layer.

## Core idea

For each account pair (i, j), the pipeline computes:

- content-sync count c
- co-amplification count a
- transformed values

  c' = 1 - exp(-c / kappa_content)
  a' = 1 - exp(-a / kappa_amp)
  w = sqrt(c' * a')

This fused score is zero if either signal is zero, which prevents a single-layer artifact from dominating the result.

The retrieval layer also adds a graph-aware prior:

score(q, p) = cos(e_q, e_p) + lambda * s_hat(author(p))

where the author prior is derived from coordination scores, so posts from suspicious accounts are boosted during semantic retrieval.

## What the project contains

This repository includes:

- a synthetic data generator for a coordinated misinformation-style campaign
- a live Bluesky ingestion pipeline
- embedding + topic projection + community detection
- user-level coordination scoring
- dashboard and narrative analysis tooling
- an optional agent that can answer natural-language questions about the detected narrative

## Architecture overview

The workflow is implemented in several modules under `src/ng/`:

- `config.py` — pipeline configuration and thresholds
- `schema.py` — dataframe validation and standard data schema
- `ingest_synthetic.py` — synthetic social graph generation with organic posts and coordinated campaigns
- `ingest_bluesky.py` — public Bluesky search ingestion with anonymized handles and repost handling
- `embed.py` — embedding generation and dimension reduction helpers
- `topics.py` — clustering and keyword extraction for narrative topics
- `graphs.py` — content-sync, interaction, and co-amplification graph construction
- `coordination.py` — fusion, scoring, community detection, mismatch metrics
- `retrieval.py` — dense semantic retrieval with graph-aware reranking
- `agent.py` — LangGraph-based research agent over suspicious accounts and posts
- `evaluate.py` — detection and retrieval metrics
- `pipeline.py` — main end-to-end orchestration and artifact persistence

## Repository layout

```text
narrativegraph/
├── app/
│   └── dashboard.py              # Streamlit dashboard for exploring posts, graph, topics, and suspect accounts
├── data/
│   └── processed/                # generated analysis artifacts such as artifacts.pkl
├── scripts/
│   └── run_pipeline.py           # CLI to generate synthetic data or pull live Bluesky results
├── src/
│   └── ng/
│       ├── __init__.py
│       ├── agent.py              # narrative QA agent via LangGraph
│       ├── config.py             # tunable thresholds
│       ├── coordination.py       # fusion and community logic
│       ├── embed.py              # embeddings
│       ├── evaluate.py           # evaluation metrics
│       ├── graphs.py             # graph construction
│       ├── ingest_bluesky.py     # public Bluesky ingestion
│       ├── ingest_synthetic.py   # synthetic social data
│       ├── pipeline.py           # end-to-end run + save/load
│       ├── retrieval.py          # semantic search with graph-aware rerank
│       ├── schema.py             # data validation schema
│       └── topics.py             # topic clustering and keyword extraction
├── tests/
│   └── test_pipeline.py          # verifies fused score beats single-layer baselines
├── Makefile
├── pyproject.toml
├── README.md
├── requirements.txt
└── scripts/run_pipeline.py
```

## Data model

The pipeline expects a flat post-level dataframe with the following schema (standardized by `schema.validate`):

- `post_id`: unique post identifier
- `user`: posting account identifier
- `text`: message text
- `created_at`: timestamp in UTC
- `kind`: one of `post`, `reply`, `quote`, `repost`
- `target_user`: the target account for replies/quotes/reposts
- `target_post_id`: target post for engagement links
- `label`: optional ground-truth label for evaluation
- `campaign`: campaign identifier for synthetic data evaluation

The synthetic generator creates organic behavior mixed with decoys and coordinated campaigns, while the Bluesky ingest creates a real-world-like social stream from public queries.

## Installation

This project is set up as a Python package under `src/`.

### Option 1: using the repo helper

```bash
python -m venv .venv
source .venv/bin/activate
make install
```

### Option 2: manual install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The package currently declares Python 3.10+ in `pyproject.toml`.

## Quick start

### Run the synthetic pipeline

```bash
python scripts/run_pipeline.py --source synthetic
```

This generates a synthetic dataset, runs the detection pipeline, and saves the artifacts to `data/processed/artifacts.pkl`.

### Launch the dashboard

```bash
make demo
```

This runs the pipeline on synthetic data and starts a Streamlit dashboard at:

```bash
streamlit run app/dashboard.py
```

### Run tests

```bash
make test
```

The current test checks that the fused method outperforms content-only and graph-only baselines on synthetic data.

## Live Bluesky usage

The repository also supports live data collection from public Bluesky feeds.

### 1. Set environment variables

Create a `.env` file in the project root with values like:

```bash
BSKY_HANDLE=your-handle.example.com
BSKY_APP_PASSWORD=your-app-password
ANTHROPIC_API_KEY=optional-if-you-want-the-agent
NG_LLM_PROVIDER=anthropic    # optional; defaults to anthropic if key is present, otherwise ollama
OLLAMA_HOST=http://localhost:11434   # optional if using ollama
```

The Bluesky fetcher uses public posts only and hashes handles by default for privacy.

### 2. Run the pipeline against Bluesky

```bash
python scripts/run_pipeline.py --source bluesky --queries "energy prices" "ai privacy" --limit 1000
```

A few important notes:

- Bluesky does not expose repost timestamps through its API, so repost-based amplification is approximated using replies and quotes in the interactive config.
- The default amplification kinds are `("reply", "quote", "repost")`, but the CLI switches to `("reply", "quote")` for live data to stay faithful to the API constraints.
- Search results and thresholds may need tuning for a specific topic or account set.

## Running the pipeline programmatically

```python
from ng.config import Config
from ng.pipeline import run, save
from ng.ingest_synthetic import generate

posts, _ = generate(n_users=400, n_campaign=(25, 20))
cfg = Config()
art = run(posts, cfg)
print(art["users"].head())
print(art["communities"])
save(art, "data/processed/artifacts.pkl")
```

The `run()` function returns a dictionary with the most important outputs:

- `cfg`: configuration object
- `posts`: validated post dataframe
- `txt`: non-repost text entries used for similarity analysis
- `emb`: embedding matrix for the text corpus
- `edges`: fused user-to-user edge dataframe
- `users`: user-level coordination scores and labels
- `communities`: detected coordinated communities
- `keywords`: top topic keywords per detected community/topic

## Graph construction details

The pipeline builds an interaction graph and a content graph separately, then fuses them.

### Content sync graph

`content_sync_edges` compares post embeddings within a time bucket and creates an edge between users when they post similar text in a short time window.

This catches synchronized copy-paste or near-duplicate campaigns, but it can also create false positives when a general topic goes viral.

### Co-amplification graph

`coamplification_edges` finds users who engage with the same target within the same time window. This helps identify coordinated reply/quote/repost behavior, even when messages are not textually similar.

### Fused graph

The fused score is computed in `coordination.py`:

- both layers are transformed with a saturating exponential function
- the final edge weight is the geometric mean
- the result is zero if either layer is absent

This is the main mechanism used to reduce false positives.

## Community detection and scoring

The pipeline adds user-level scores:

- `score_content`: content-only coordination score
- `score_amp`: amplification-only coordination score
- `score_fused`: combined score
- `coord_comm`: community assignment for coordinated communities
- `social_comm`: organic social-community assignment
- `mismatch`: Jensen-Shannon divergence between an account's topic mix and its social community's average topic mix

The `detect_communities` logic uses a graph of fused edges and Louvain community detection. Only communities above the configured size threshold are reported as coordinated communities.

## Retrieval and narrative analysis

The retrieval layer uses a dense vector index over posts. It supports:

- semantic search over the message corpus
- optional graph-aware reranking based on author coordination priors
- recall metrics against campaign-related post sets

This is implemented in `retrieval.py` and supports analysis tasks such as:

- "which accounts are most associated with a narrative?"
- "what posts are most central to the claim?"
- "how does graph-aware ranking improve retrieval of coordinated content?"

The optional agent in `agent.py` wraps the retrieval and community tools in a local LLM workflow so that a user can ask natural-language questions like:

> Who is pushing the claim that energy bills will triple?

The agent uses tools for:

- semantic post search
- community summaries
- account reports
- top suspicious accounts

The agent is intentionally designed to ground results in tool outputs and avoid unsupported claims.

## Dashboard

The Streamlit app in `app/dashboard.py` offers a rich interactive interface with tabs for:

- Overview
- Network visualization
- Narrative map
- Suspicious accounts
- Ask the agent

It shows:

- number of posts and accounts
- detected communities
- community attributes
- edge threshold tuning
- content/amp/fused graph views
- semantic map of posts
- temporal narrative volume
- agent-based Q&A over the network

## Configuration

The primary tuning parameters live in `src/ng/config.py`:

```python
Config(
    embed_model="sentence-transformers/all-MiniLM-L6-v2",
    window_sec=1800,
    sim_threshold=0.70,
    kappa_content=2.0,
    kappa_amp=2.0,
    edge_threshold=0.35,
    min_comm_size=4,
    n_topics=8,
    max_group=300,
    amp_kinds=("reply", "quote", "repost"),
    seed=7,
    data_dir="data/processed",
)
```

Key notes:

- `window_sec` determines the time window in which content and amplification are considered synchronous
- `sim_threshold` controls the cosine similarity threshold for content duplication
- `kappa_content` and `kappa_amp` scale the saturation behavior of the fused score
- `edge_threshold` controls which edges are retained in the community graph
- `min_comm_size` filters extremely small communities

## Evaluation

The evaluation utilities in `src/ng/evaluate.py` compute metrics such as:

- average precision
- ROC AUC
- precision at k
- retrieval recall@k

This allows the project to compare the fused model against content-only and graph-only baselines on synthetic/known ground-truth data.

## Important caveats

This project is designed as a research and detection aid, not as a proof of real-world coordination.

- scores are evidence, not proof
- public-data ingestion uses anonymized identities by default
- social behavior can look coordinated even when it is organic
- real-world tuning will likely require dataset-specific thresholds
- package internals such as LangGraph, atproto, and pyvis may evolve between versions

## Development notes

- `pytest` is used for automated checks.
- the package is installed with `pip install -e .` so imports resolve from `src/`
- artifacts are stored in `data/processed/`
- the generated synthetic data intentionally includes both coordinated campaigns and decoy patterns to stress-test the detector

## Summary

NarrativeGraph is a practical, end-to-end framework for surfacing coordinated narratives by combining:

- near-duplicate content detection
- synchronized amplification patterns
- community discovery in the fused interaction graph
- graph-aware retrieval
- explanatory agent tooling

It is especially useful when you want to distinguish orchestrated narrative campaigns from ordinary social-media activity such as news bursts or viral content.
