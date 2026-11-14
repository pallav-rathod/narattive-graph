from ng.ingest_synthetic import generate
from ng.pipeline import run
from ng.evaluate import detection_report


def test_fused_beats_single_layer():
    posts, _ = generate(n_users=200, n_campaign=(15, 12))
    rep = detection_report(run(posts)["users"]).set_index("method")
    assert rep.loc["fused", "AP"] >= rep.loc["content-only", "AP"]
    assert rep.loc["fused", "AP"] >= rep.loc["graph-only", "AP"]
