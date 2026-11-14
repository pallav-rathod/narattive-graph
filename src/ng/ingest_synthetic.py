import numpy as np
import pandas as pd
from .schema import validate

TOPICS = {
    "energy": ["Energy bills are out of control and nobody in government is listening",
               "Switching to solar paid for itself in four years, highly recommend",
               "The grid needs serious investment before next winter",
               "Heat pumps are finally affordable for ordinary families"],
    "transit": ["The new tram line cut my commute in half",
                "Bus routes in the east side desperately need more frequency",
                "Cycling lanes are the cheapest way to fix downtown traffic",
                "Rail delays this week were completely unacceptable"],
    "health": ["Waiting times at the local clinic have doubled this year",
               "Great to see the vaccination drive reaching rural areas",
               "Mental health services need far more funding",
               "Nurses deserve a real pay rise, plain and simple"],
    "ai": ["Open source language models are catching up faster than expected",
           "Using an AI coding assistant saved me hours on a boring migration",
           "We need clear rules for how AI is used in hiring",
           "Benchmarks for AI agents are still pretty unreliable"],
    "sports": ["What a comeback in the second half, unbelievable game",
               "The transfer window has been wild this season",
               "Marathon training week six done, legs are toast",
               "That last-minute goal will be replayed for years"],
    "climate": ["Reforestation projects only work with local community buy-in",
                "Record temperatures again this summer, we cannot ignore this",
                "Coastal towns are already planning for higher seas",
                "Carbon markets need much stricter verification"],
}
NEWS = ["BREAKING: government announces review of the energy price cap",
        "Breaking news - energy price cap under review, government confirms",
        "Government confirms the energy price cap will be reviewed, details to follow",
        "Just in: energy price cap review announced by the government"]
CAMP = [["Government is secretly planning to triple your energy bills, share before they delete this",
         "They are hiding the real energy price hike from you, wake up and share",
         "Leaked: energy bills set to triple next month, nobody is telling you"],
        ["Your AI assistant is secretly recording everything you say, wake up people",
         "Big tech AI chatbots are spying on your private chats, share this now",
         "Leaked: AI assistants log every conversation and sell it, spread the word"]]
TAGS = ["#wakeup", "#share", "#truth", "🔥", "🚨", "#exposed", ""]


def generate(n_users=400, n_comm=5, n_campaign=(25, 20), seed=7):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2025-03-01", tz="UTC")
    topics = list(TOPICS)
    rows = []

    def add(user, text, ts, kind="post", tu=None, tp=None, label=0, camp=-1):
        pid = f"p{len(rows)}"
        rows.append(dict(post_id=pid, user=user, text=text.strip(), created_at=ts, kind=kind,
                         target_user=tu, target_post_id=tp, label=label, campaign=camp))
        return pid

    def rand_ts():
        return t0 + pd.Timedelta(seconds=float(rng.uniform(0, 3 * 86400)))

    def jitter(sec):
        return pd.Timedelta(seconds=float(rng.uniform(0, sec)))

    users = [f"u{i}" for i in range(n_users)]
    roles = {u: "organic" for u in users}
    comm = {u: i % n_comm for i, u in enumerate(users)}
    by_comm = {c: [] for c in range(n_comm)}

    # organic posts, topic-homophilous communities
    for u in users:
        for _ in range(rng.poisson(12)):
            tp = topics[comm[u] % len(topics)] if rng.random() < .7 else topics[rng.integers(len(topics))]
            ts = rand_ts()
            pid = add(u, str(rng.choice(TOPICS[tp])) + " " + str(rng.choice(TAGS)), ts)
            by_comm[comm[u]].append((pid, u, ts))
    # organic replies, mostly within community, long lags (not synchronized)
    for u in users:
        for _ in range(rng.poisson(6)):
            c = comm[u] if rng.random() < .85 else int(rng.integers(n_comm))
            pid, tu, ts = by_comm[c][rng.integers(len(by_comm[c]))]
            if tu == u:
                continue
            tp = topics[comm[u] % len(topics)]
            add(u, str(rng.choice(TOPICS[tp])), ts + pd.Timedelta(seconds=float(rng.exponential(3600))),
                "reply", tu, pid)

    # coordinated campaigns: near-duplicate text + shared amplification, same minutes
    for k, size in enumerate(n_campaign):
        members = [f"c{k}_{i}" for i in range(size)]
        seeds = [f"seed{k}_{j}" for j in range(3)]
        for m in members:
            roles[m] = "coordinated"
            for _ in range(5):  # organic-looking cover activity
                add(m, str(rng.choice(TOPICS[topics[rng.integers(len(topics))]])), rand_ts(), label=1)
        for s in seeds:
            roles[s] = "seed"
        for b in range(4):
            T = t0 + pd.Timedelta(seconds=float(rng.uniform(3600, 3 * 86400 - 3600)))
            sp = [(add(s, f"Thoughts on today's news ({b})", T - pd.Timedelta(seconds=120)), s) for s in seeds]
            for m in members:
                if rng.random() < .8:
                    msg = CAMP[k][rng.integers(len(CAMP[k]))] + " " + str(rng.choice(TAGS))
                    add(m, msg, T + jitter(240), label=1, camp=k)
                    for pid, s in sp:
                        add(m, "", T + jitter(300), "repost", s, pid, 1, k)

    # decoy 1: organic news burst (content-only false positive)
    others = list(users)
    rng.shuffle(others)
    T = t0 + pd.Timedelta(days=1.5)
    for u in others[:40]:
        roles[u] = "news_decoy"
        add(u, NEWS[rng.integers(len(NEWS))], T + jitter(240))
    # decoy 2: viral celebrity post (graph-only false positive)
    T2 = t0 + pd.Timedelta(days=2.2)
    roles["celeb0"] = "celeb"
    cp = add("celeb0", "Huge announcement coming tomorrow!", T2 - pd.Timedelta(seconds=60))
    for u in others[40:100]:
        roles[u] = "viral_decoy"
        add(u, "", T2 + jitter(240), "repost", "celeb0", cp)

    posts = validate(pd.DataFrame(rows)).sort_values("created_at").reset_index(drop=True)
    udf = pd.DataFrame({"user": list(roles), "role": list(roles.values())})
    udf["label"] = (udf.role == "coordinated").astype(int)
    return posts, udf
