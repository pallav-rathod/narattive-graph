import json
import os
import re
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

SYSTEM = ("You are a network-analysis assistant. Use the tools to answer questions about narratives, "
          "communities and suspicious accounts. Ground every claim in tool output; cite post_ids and "
          "account ids. Coordination scores are evidence, not proof: say so, and never speculate "
          "about real identities.")


def make_llm():
    """NG_LLM_PROVIDER=ollama (default if no Anthropic key) or anthropic."""
    provider = os.getenv("NG_LLM_PROVIDER") or ("anthropic" if os.getenv("ANTHROPIC_API_KEY") else "ollama")
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=os.getenv("NG_OLLAMA_MODEL", "qwen3:4b"),
                          base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
                          temperature=0)
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=os.getenv("NG_ANTHROPIC_MODEL", "claude-sonnet-5-5"))


def build_agent(art, index):
    users = art["users"].set_index("user")
    comms = art["communities"].set_index("coord_comm")
    txt = art["txt"]

    @tool
    def search_posts(query: str, k: int = 8, graph_aware: bool = True) -> str:
        """Semantic search over posts. graph_aware=True boosts posts by accounts with high coordination scores."""
        r = index.search(query, k=k, lam=0.5 if graph_aware else 0.0)
        return r[["post_id", "user", "created_at", "text", "coord_comm", "score"]].to_json(orient="records", date_format="iso")

    @tool
    def community_report(community_id: int) -> str:
        """Stats and top messages for a coordinated community."""
        if community_id not in comms.index:
            return "unknown community"
        row = comms.loc[community_id]
        sample = txt[txt.coord_comm == community_id].text.value_counts().head(3).to_dict()
        return json.dumps(dict(size=int(row["size"]), density=float(row.density), mean_content=float(row.mean_content),
                               mean_amp=float(row.mean_amp), top_messages=sample, members=row.members[:15]))

    @tool
    def account_report(user: str) -> str:
        """Coordination scores, communities and topic mismatch for one account."""
        if user not in users.index:
            return "unknown account"
        r = users.loc[user]
        recent = txt[txt.user == user].text.head(3).tolist()
        return json.dumps(dict(r.drop("label", errors="ignore").to_dict(), sample_posts=recent), default=str)

    @tool
    def top_suspicious(n: int = 10) -> str:
        """Accounts with the highest fused (content AND network) coordination score."""
        return users.nlargest(n, "score_fused")[["score_fused", "score_content", "score_amp", "coord_comm"]].to_json()

    return create_react_agent(make_llm(), [search_posts, community_report, account_report, top_suspicious],
                              prompt=SYSTEM)


def ask(agent, question):
    out = agent.invoke({"messages": [("user", question)]})
    text = out["messages"][-1].content
    if isinstance(text, list):  # Anthropic can return content blocks
        text = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in text)
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()