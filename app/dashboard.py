import os
os.environ.setdefault("NG_PROJECTION", "pca")   # only if you applied Fix 3 from earlier
import sys
import tempfile
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from pyvis.network import Network
from ng.config import Config
from ng.pipeline import run, load
from ng.embed import Embedder
from ng.retrieval import PostIndex
from ng.evaluate import detection_report, retrieval_report

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


@st.cache_resource
def _run_lock():
    return threading.Lock()


def run_locked(posts, cfg):
    with _run_lock():
        return run(posts, cfg)


def safe(df):
    """Make a DataFrame safe for Streamlit's Arrow serialization."""
    d = df.copy().reset_index(drop=True)
    for c in d.columns:
        s = d[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            d[c] = s.astype(str)
        elif s.dtype == object or str(s.dtype).startswith("str"):
            d[c] = s.map(lambda v: v if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v))
            d[c] = d[c].astype(str)
    return d

st.set_page_config(page_title="NarrativeGraph", layout="wide")
st.title("NarrativeGraph: what they say × who they talk to")

PATH = "data/processed/artifacts.pkl"

with st.sidebar:
    st.header("Data")
    mode = st.radio("Source", ["Precomputed / synthetic", "Live Bluesky"])
    if mode == "Live Bluesky":
        qs = st.text_input("Search terms (comma separated)", "energy prices, ai privacy")
        lim = st.slider("Posts per term", 100, 2000, 500, 100)
        if st.button("Fetch & analyze"):
            if not (os.getenv("BSKY_HANDLE") and os.getenv("BSKY_APP_PASSWORD")):
                st.error("Set BSKY_HANDLE and BSKY_APP_PASSWORD in .env, then restart Streamlit.")
            else:
                from ng.ingest_bluesky import fetch
                try:
                    with st.spinner("Fetching public posts and running pipeline..."):
                        cfg = Config(amp_kinds=("reply", "quote"))
                        st.session_state["art"] = run_locked(fetch([q.strip() for q in qs.split(",")], lim), cfg)
                except Exception as e:
                    st.error(f"Fetch failed: {e}")
    elif st.button("Generate synthetic data"):
        from ng.ingest_synthetic import generate
        with st.spinner("Running pipeline..."):
            st.session_state["art"] = run_locked(generate()[0], Config())

art = st.session_state.get("art") or (load(PATH) if os.path.exists(PATH) else None)
if art is None:
    st.info("Run `python scripts/run_pipeline.py` or use the sidebar.")
    st.stop()

txt, users, edges, comms = art["txt"], art["users"], art["edges"], art["communities"]
cfg = art["cfg"]
thr = st.sidebar.slider("Edge threshold (fused)", 0.1, 0.9, float(cfg.edge_threshold), 0.05)


@st.cache_resource
def get_index(_art):
    emb = Embedder(_art["cfg"].embed_model)
    emb.prepare(_art["txt"].text.tolist())
    prior = _art["users"].set_index("user").score_fused
    prior = prior / (prior.max() + 1e-9)
    return PostIndex(_art["txt"], _art["emb"], emb, prior)


def network_html(layer="fused", max_nodes=300):
    e = edges[edges[layer] >= thr]
    deg = e.groupby("u")[layer].sum().add(e.groupby("v")[layer].sum(), fill_value=0)
    keep = set(deg.nlargest(max_nodes).index)
    e = e[e.u.isin(keep) & e.v.isin(keep)]
    U = users.set_index("user")
    pal = px.colors.qualitative.Bold
    net = Network(height="620px", width="100%", bgcolor="#0e1117", font_color="white", cdn_resources="in_line")
    for n in set(e.u) | set(e.v):
        c = int(U.loc[n, "coord_comm"])
        net.add_node(str(n), label=str(n)[:8], color="#888" if c < 0 else pal[c % len(pal)],
                     size=float(5 + 3 * U.loc[n, "score_fused"] ** .5),
                     title=f"comm {c} | fused {U.loc[n, 'score_fused']:.2f}")
    for r in e.itertuples():
        net.add_edge(str(r.u), str(r.v), value=float(getattr(r, layer)))
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
        path = f.name
    net.write_html(path, notebook=False)
    return open(path).read(), len(e)


t1, t2, t3, t4, t5 = st.tabs(["Overview", "Network", "Narrative map", "Suspicious accounts", "Ask"])

with t1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Posts", len(art["posts"]))
    c2.metric("Accounts", len(users))
    c3.metric("Coordinated communities", len(comms))
    c4.metric("Accounts in them", int((users.coord_comm >= 0).sum()))
    st.subheader("Communities (both layers must agree)")
    st.dataframe(comms.drop(columns="members"), use_container_width=True)
    if users["label"].sum() > 0:
        st.subheader("Ground-truth check: content-only vs graph-only vs fused")
        st.dataframe(detection_report(users), use_container_width=True)
        st.subheader("Retrieval recall@k: plain vs graph-aware (λ)")
        st.dataframe(retrieval_report(get_index(art), txt), use_container_width=True)
    st.subheader("Topics")
    st.table(pd.Series(art["keywords"], name="keywords"))

with t2:
    layer = st.radio("Edge layer", ["fused", "content", "amp"], horizontal=True)
    st.write(f"{len(edges)} candidate pairs. "
             f"Pairs with content > 0: {(edges.content > 0).sum()}, "
             f"amp > 0: {(edges.amp > 0).sum()}, "
             f"both: {((edges.content > 0) & (edges.amp > 0)).sum()}")
    html, n_e = network_html(layer)
    if n_e == 0:
        st.warning("No edges at this threshold for this layer. For live data this often means "
                   "no account pairs showed BOTH near-duplicate text and shared amplification "
                   "in the same window. Try the content or amp layer, or loosen the config.")
    else:
        components.html(html, height=640, scrolling=False)
    st.caption("Fused edges exist only where content-sync AND co-amplification both fire.")

with t3:
    sel = st.selectbox("Highlight community", ["all"] + sorted(txt.coord_comm.unique().tolist()))
    d = txt.copy()
    d["group"] = d.coord_comm.map(lambda c: "organic" if c < 0 else f"coord {c}")
    if sel != "all":
        d = d[d.coord_comm == sel]
    a, b = st.columns(2)
    a.plotly_chart(px.scatter(d, x="x", y="y", color="group", hover_data=["user", "text"],
                              title="Semantic map of posts"), use_container_width=True)
    b.plotly_chart(px.histogram(d, x="created_at", color="group", nbins=120,
                                title="Narrative volume over time"), use_container_width=True)

with t4:
    st.caption("Mismatch = JS divergence between what an account says and what its social community says.")
    st.dataframe(users.sort_values("score_fused", ascending=False).head(50), use_container_width=True)

with t5:
    q = st.text_input("Ask about the network", "Who is pushing the claim that energy bills will triple?")
    idx = get_index(art)
    if st.button("Ask the agent"):
        from ng.agent import build_agent, ask
        try:
            with st.spinner("Agent is working (a local model can take a minute)..."):
                st.markdown(ask(build_agent(art, idx), q))
        except Exception as e:
            st.error(f"Agent failed: {e}")
    lam = st.slider("Graph-aware weight λ", 0.0, 1.0, 0.5, 0.1)
    l, r = st.columns(2)
    l.write("**Content only**")
    l.dataframe(idx.search(q, 10, 0.0)[["user", "text", "coord_comm", "score"]])
    r.write("**Content + network prior**")
    r.dataframe(idx.search(q, 10, lam)[["user", "text", "coord_comm", "score"]])
