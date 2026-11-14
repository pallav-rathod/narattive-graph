import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from .retrieval import recall_at_k


def detection_report(users):
    y = users["label"].to_numpy()
    k = int(y.sum())
    rows = []
    for name, col in [("content-only", "score_content"), ("graph-only", "score_amp"), ("fused", "score_fused")]:
        s = users[col].to_numpy()
        top = s.argsort()[::-1][:k]
        rows.append(dict(method=name, AP=average_precision_score(y, s), AUC=roc_auc_score(y, s),
                         precision_at_k=float(y[top].mean())))
    return pd.DataFrame(rows).round(3)


def retrieval_report(index, txt, ks=(5, 10, 20), lams=(0.0, 0.5, 1.0)):
    """Queries = campaign message text; relevant = posts belonging to that campaign."""
    queries = {}
    for k, g in txt[txt.campaign >= 0].groupby("campaign"):
        queries[g.text.iloc[0]] = set(g.post_id)
    return pd.DataFrame([dict(lam=l, **recall_at_k(index, queries, ks, l)) for l in lams]).round(3)
