import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
from ng.config import Config
from ng.pipeline import run, save

load_dotenv()
ap = argparse.ArgumentParser()
ap.add_argument("--source", choices=["synthetic", "bluesky"], default="synthetic")
ap.add_argument("--queries", nargs="*", default=["energy prices", "ai privacy"])
ap.add_argument("--limit", type=int, default=500)
a = ap.parse_args()

cfg = Config()
if a.source == "synthetic":
    from ng.ingest_synthetic import generate
    posts, _ = generate()
else:
    from ng.ingest_bluesky import fetch
    posts = fetch(a.queries, a.limit)
    cfg.amp_kinds = ("reply", "quote")   # repost times are not exposed by the API
art = run(posts, cfg)
save(art, f"{cfg.data_dir}/artifacts.pkl")
print(f"posts={len(art['posts'])} users={len(art['users'])} coordinated communities={len(art['communities'])}")
if art["users"]["label"].sum() > 0:
    from ng.evaluate import detection_report
    print(detection_report(art["users"]))
