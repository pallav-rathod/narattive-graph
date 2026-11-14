import hashlib
import os
import pandas as pd
from .schema import validate


def _anon(x, on=True):
    return hashlib.blake2b(x.encode(), digest_size=5).hexdigest() if (on and x) else x


def _did(uri):  # at://did:plc:xxx/app.bsky.feed.post/rkey
    return uri.split("/")[2] if uri else None


def fetch(queries, limit_per_query=500, anonymize=True, min_reposts=5, handle=None, password=None):
    """Public Bluesky posts only. Handles are hashed by default (privacy).
    Repost timestamps are not exposed by the API, so reposts get the target's time;
    keep them out of amp_kinds for real data (see Config.amp_kinds)."""
    from atproto import Client
    c = Client()
    c.login(handle or os.environ["BSKY_HANDLE"], password or os.environ["BSKY_APP_PASSWORD"])
    rows, seen = [], set()
    for q in queries:
        cursor, n = None, 0
        while n < limit_per_query:
            r = c.app.bsky.feed.search_posts(params={"q": q, "limit": 100, "cursor": cursor, "sort": "latest"})
            if not r.posts:
                break
            for p in r.posts:
                if p.uri in seen:
                    continue
                seen.add(p.uri)
                rec = p.record
                kind, target = "post", None
                if getattr(rec, "reply", None):
                    kind, target = "reply", rec.reply.parent.uri
                emb = getattr(rec, "embed", None)
                if emb is not None and getattr(emb, "record", None) is not None:
                    r0 = emb.record
                    uri = getattr(r0, "uri", None) or getattr(getattr(r0, "record", None), "uri", None)
                    if uri:
                        kind, target = "quote", uri
                created = pd.to_datetime(getattr(rec, "created_at", None) or p.indexed_at, utc=True)
                rows.append(dict(post_id=p.uri, user=_anon(p.author.did, anonymize), text=rec.text or "",
                                 created_at=created, kind=kind,
                                 target_user=_anon(_did(target), anonymize), target_post_id=target))
                if (p.repost_count or 0) >= min_reposts:
                    try:
                        rr = c.get_reposted_by(p.uri, limit=100)
                        for a in rr.reposted_by:
                            rows.append(dict(post_id=f"rp:{p.uri}:{a.did}", user=_anon(a.did, anonymize), text="",
                                             created_at=created, kind="repost",
                                             target_user=_anon(p.author.did, anonymize), target_post_id=p.uri))
                    except Exception:
                        pass
                n += 1
            cursor = r.cursor
            if not cursor:
                break
    return validate(pd.DataFrame(rows))
