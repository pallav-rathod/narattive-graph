import numpy as np


def _norm(Z):
    return (Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)).astype("float32")


class Embedder:
    """Sentence-transformers if available, else TF-IDF+SVD (works offline)."""

    def __init__(self, model_name):
        self.kind, self.m = "st", None
        try:
            from sentence_transformers import SentenceTransformer
            self.m = SentenceTransformer(model_name)
        except Exception:
            self.kind = "tfidf"

    def _enc_st(self, texts):
        return self.m.encode(texts, batch_size=128, normalize_embeddings=True,
                             show_progress_bar=False).astype("float32")

    def fit_transform(self, texts):
        if self.kind == "st":
            return self._enc_st(texts)
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        self.vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
        X = self.vec.fit_transform(texts)
        self.svd = TruncatedSVD(max(2, min(128, X.shape[1] - 1, X.shape[0] - 1)), random_state=0)
        return _norm(self.svd.fit_transform(X))

    def prepare(self, texts):
        """Call at load time so TF-IDF mode can embed queries (no-op for ST)."""
        if self.kind == "tfidf":
            self.fit_transform(texts)

    def encode(self, texts):
        if self.kind == "st":
            return self._enc_st(texts)
        return _norm(self.svd.transform(self.vec.transform(texts)))
