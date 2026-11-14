from collections import defaultdict
import numpy as np
import networkx as nx
from .schema import to_sec


def interaction_graph(posts) -> nx.DiGraph:
    ev = posts[posts.target_user.notna() & (posts.user != posts.target_user)]
    g = nx.DiGraph()
    g.add_nodes_from(posts.user.unique())
    for (u, v), w in ev.groupby(["user", "target_user"]).size().items():
        g.add_edge(u, v, weight=int(w))
    return g


def content_sync_edges(txt, emb, window, thr, chunk=2048):
    """Pairs of different users posting near-duplicate content within `window` seconds.
    Time-bucketing => only compare bucket b with b and b+1 (no O(N^2) over all posts)."""
    t, users = to_sec(txt.created_at), txt.user.to_numpy()
    buckets = defaultdict(list)
    for i, b in enumerate((t // window).astype(int)):
        buckets[b].append(i)
    W = defaultdict(float)
    for b, ids in buckets.items():
        A = np.array(ids)
        for B, same in ((A, True), (np.array(buckets.get(b + 1, []), dtype=int), False)):
            if len(B) == 0:
                continue
            for s in range(0, len(A), chunk):
                Ac = A[s:s + chunk]
                S = emb[Ac] @ emb[B].T
                dt = np.abs(t[Ac][:, None] - t[B][None, :])
                ii, jj = np.where((S >= thr) & (dt <= window))
                for a, bj in zip(ii, jj):
                    p, q = Ac[a], B[bj]
                    if (same and p >= q) or users[p] == users[q]:
                        continue
                    W[tuple(sorted((users[p], users[q])))] += 1
    return W


def coamplification_edges(ev, window, max_group):
    """Pairs of users that reply/quote/repost the SAME target within `window` seconds."""
    W = defaultdict(float)
    for _, g in ev[ev.target_post_id.notna()].groupby("target_post_id"):
        if len(g) > max_group:
            g = g.sample(max_group, random_state=0)
        g = g.sort_values("created_at")
        u, t = g.user.to_numpy(), to_sec(g.created_at)
        for i in range(len(g)):
            j = i + 1
            while j < len(g) and t[j] - t[i] <= window:
                if u[i] != u[j]:
                    W[tuple(sorted((u[i], u[j])))] += 1
                j += 1
    return W
