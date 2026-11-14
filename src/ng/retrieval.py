import numpy as np


class PostIndex:
    """Dense retrieval over posts with optional graph-aware rerank (author coordination prior)."""

    def __init__(self, txt, emb, embedder, user_prior=None):
        self.posts = txt.reset_index(drop=True)
        self.emb, self.embedder = emb, embedder
        self.prior = (self.posts.user.map(user_prior).fillna(0).to_numpy()
                      if user_prior is not None else np.zeros(len(self.posts)))
        try:
            import faiss
            self.ix = faiss.IndexFlatIP(emb.shape[1])
            self.ix.add(emb)
        except Exception:
            self.ix = None

    def search(self, query, k=10, lam=0.0, pool=100):
        qv = self.embedder.encode([query])
        pool = min(pool, len(self.posts))
        if self.ix is not None:
            s, i = self.ix.search(qv, pool)
            s, i = s[0], i[0]
        else:
            sc = self.emb @ qv[0]
            i = np.argsort(-sc)[:pool]
            s = sc[i]
        s2 = s + lam * self.prior[i]
        o = np.argsort(-s2)[:k]
        out = self.posts.iloc[i[o]].copy()
        out["score"] = s2[o]
        return out


def recall_at_k(index, queries, ks=(5, 10, 20), lam=0.0):
    """queries: {query_text: set(relevant post_ids)}"""
    res = {}
    for k in ks:
        vals = []
        for q, rel in queries.items():
            got = set(index.search(q, k=k, lam=lam).post_id)
            vals.append(len(got & rel) / max(1, min(k, len(rel))))
        res[f"recall@{k}"] = float(np.mean(vals))
    return res
