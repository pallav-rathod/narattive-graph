import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def cluster_topics(emb, k, seed):
    return KMeans(n_clusters=k, n_init="auto", random_state=seed).fit_predict(emb)


def topic_keywords(texts, labels, n=6):
    df = pd.DataFrame({"t": list(texts), "l": list(labels)})
    docs = df.groupby("l").t.apply(" ".join)
    v = TfidfVectorizer(stop_words="english", max_features=20000)
    X = v.fit_transform(docs)
    vocab = np.array(v.get_feature_names_out())
    return {int(l): ", ".join(vocab[np.argsort(-X[i].toarray()[0])[:n]]) for i, l in enumerate(docs.index)}

def project_2d(emb, seed):
    if os.getenv("NG_PROJECTION", "umap").lower() != "pca":
        try:
            import umap
            return umap.UMAP(n_components=2, metric="cosine", random_state=seed,
                             compatibility_layout=True).fit_transform(emb)
        except TypeError:
            import umap
            return umap.UMAP(n_components=2, metric="cosine", random_state=seed).fit_transform(emb)
        except Exception:
            pass
    from sklearn.decomposition import PCA
    return PCA(2, random_state=seed).fit_transform(emb)
