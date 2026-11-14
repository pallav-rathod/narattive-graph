import pandas as pd

COLUMNS = ["post_id", "user", "text", "created_at", "kind",
           "target_user", "target_post_id", "label", "campaign"]


def to_sec(s: pd.Series):
    return (s - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()


def validate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    defaults = {"label": 0, "campaign": -1}
    for c in COLUMNS:
        if c not in df:
            df[c] = defaults.get(c, None)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["text"] = df["text"].fillna("")
    return df[COLUMNS]
