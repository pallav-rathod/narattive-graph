from dataclasses import dataclass


@dataclass
class Config:
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    window_sec: int = 1800         # temporal window for "synchronized"
    sim_threshold: float = 0.70    # cosine threshold for near-duplicate content
    kappa_content: float = 2.0     # saturation scale for content-sync counts
    kappa_amp: float = 2.0         # saturation scale for co-amplification counts
    edge_threshold: float = 0.35   # fused weight needed to keep an edge
    min_comm_size: int = 4
    n_topics: int = 8
    max_group: int = 300           # cap per amplified post (keeps pairs bounded)
    amp_kinds: tuple = ("reply", "quote", "repost")
    seed: int = 7
    data_dir: str = "data/processed"
