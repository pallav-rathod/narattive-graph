import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import jensenshannon


def fuse(Wc, Wa, kc, ka) -> pd.DataFrame:
    rows = []
    for k in set(Wc) | set(Wa):
        c = 1 - np.exp(-Wc.get(k, 0) / kc)
        a = 1 - np.exp(-Wa.get(k, 0) / ka)
        rows.append((k[0], k[1], c, a, float(np.sqrt(c * a))))
    return pd.DataFrame(rows, columns=["u", "v", "content", "amp", "fused"])


def account_scores(edges, col):
    if edges.empty:
        return pd.Series(dtype=float)
    s = pd.concat([edges.groupby("u")[col].sum(), edges.groupby("v")[col].sum()])
    return s.groupby(level=0).sum()


def detect_communities(edges, thr, min_size, seed):
    g = nx.Graph()
    for r in edges[edges.fused >= thr].itertuples():
        g.add_edge(r.u, r.v, weight=r.fused)
    if g.number_of_nodes() == 0:
        return {}, []
    comms = [c for c in nx.community.louvain_communities(g, weight="weight", seed=seed) if len(c) >= min_size]
    comms.sort(key=len, reverse=True)
    return {u: i for i, c in enumerate(comms) for u in c}, comms


def community_report(edges, comms, thr):
    e = edges[edges.fused >= thr]
    rows = []
    for i, c in enumerate(comms):
        sub = e[e.u.isin(c) & e.v.isin(c)]
        n = len(c)
        rows.append(dict(coord_comm=i, size=n,
                         density=len(sub) / (n * (n - 1) / 2) if n > 1 else 0.0,
                         mean_content=sub.content.mean(), mean_amp=sub.amp.mean(),
                         mean_fused=sub.fused.mean(), members=sorted(c)))
    return pd.DataFrame(rows, columns=["coord_comm", "size", "density", "mean_content",
                                       "mean_amp", "mean_fused", "members"])


def social_communities(posts, seed):
    """Organic social structure: Louvain on the undirected interaction graph."""
    from .graphs import interaction_graph
    g = interaction_graph(posts).to_undirected()
    comms = nx.community.louvain_communities(g, weight="weight", seed=seed)
    return {u: i for i, c in enumerate(comms) for u in c}


def topic_mismatch(txt, social_comm, k):
    """JS divergence between a user's topic mix (what they say) and the average topic mix
    of their social community (who they talk to). High => behavior doesn't match company."""
    ct = pd.crosstab(txt.user, txt.topic).reindex(columns=range(k), fill_value=0)
    P = ct.div(ct.sum(axis=1), axis=0)
    comm = pd.Series(social_comm).reindex(P.index)
    C = P.groupby(comm).mean()
    out = {}
    for u, row in P.iterrows():
        c = comm[u]
        out[u] = np.nan if pd.isna(c) else float(jensenshannon(row.values, C.loc[c].values, base=2))
    return pd.Series(out)
