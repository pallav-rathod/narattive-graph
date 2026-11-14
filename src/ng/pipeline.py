import os
import pickle
import pandas as pd
from .config import Config
from .schema import validate
from .embed import Embedder
from .topics import cluster_topics, topic_keywords, project_2d
from .graphs import content_sync_edges, coamplification_edges
from .coordination import (fuse, account_scores, detect_communities, community_report,
                           social_communities, topic_mismatch)


def run(posts, cfg: Config = Config()):
    posts = validate(posts).sort_values("created_at").reset_index(drop=True)
    txt = posts[(posts.kind != "repost") & (posts.text.str.len() > 0)].reset_index(drop=True)

    embedder = Embedder(cfg.embed_model)
    emb = embedder.fit_transform(txt.text.tolist())
    k = min(cfg.n_topics, len(txt))
    txt["topic"] = cluster_topics(emb, k, cfg.seed)
    xy = project_2d(emb, cfg.seed)
    txt["x"], txt["y"] = xy[:, 0], xy[:, 1]
    keywords = topic_keywords(txt.text, txt.topic)

    Wc = content_sync_edges(txt, emb, cfg.window_sec, cfg.sim_threshold)
    Wa = coamplification_edges(posts[posts.kind.isin(cfg.amp_kinds)], cfg.window_sec, cfg.max_group)
    edges = fuse(Wc, Wa, cfg.kappa_content, cfg.kappa_amp)

    users = pd.DataFrame(index=pd.Index(posts.user.unique(), name="user"))
    for col, name in [("content", "score_content"), ("amp", "score_amp"), ("fused", "score_fused")]:
        users[name] = account_scores(edges, col).reindex(users.index).fillna(0.0)
    cmap, comms = detect_communities(edges, cfg.edge_threshold, cfg.min_comm_size, cfg.seed)
    users["coord_comm"] = pd.Series(cmap).reindex(users.index).fillna(-1).astype(int)
    smap = social_communities(posts, cfg.seed)
    users["social_comm"] = pd.Series(smap).reindex(users.index)
    users["mismatch"] = topic_mismatch(txt, smap, k)
    users["n_posts"] = posts.groupby("user").size()
    users["label"] = posts.groupby("user").label.max()
    users = users.reset_index()
    txt["coord_comm"] = txt.user.map(cmap).fillna(-1).astype(int)

    return dict(cfg=cfg, posts=posts, txt=txt, emb=emb, edges=edges, users=users,
                communities=community_report(edges, comms, cfg.edge_threshold), keywords=keywords)


def save(art, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(art, f)


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)
