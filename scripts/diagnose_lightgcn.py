"""Is LightGCN undertrained? Train at several epoch counts on the SAME split and
measure the mf channel directly (retrieval recall + mf-as-scorer hit@10).
Run: python -m scripts.diagnose_lightgcn
"""
import asyncio
import random

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector

from app.core.config import settings
from app.ml import config
from app.ml.lightgcn import train_lightgcn
from app.ml.reco import build_context_from_conn

SAMPLE = 3000
LIKE = 7.0
EXPERIMENTS = [
    ("epochs=30 (current)", dict(epochs=30, neg_k=1)),
    ("epochs=150", dict(epochs=150, neg_k=1)),
    ("epochs=400", dict(epochs=400, neg_k=1)),
    ("epochs=400 neg_k=4", dict(epochs=400, neg_k=4)),
]


def dsn():
    return settings.database_url.replace("+asyncpg", "")


def train_variant(pu, pi, nU, nI, epochs, neg_k):
    """Wrapper that optionally averages BPR over neg_k negatives per positive."""
    import torch
    import torch.nn.functional as F
    torch.manual_seed(42)
    n = nU + nI
    e = len(pu)
    u_nodes = torch.as_tensor(pu, dtype=torch.long)
    i_nodes = torch.as_tensor(pi, dtype=torch.long) + nU
    src = torch.cat([u_nodes, i_nodes]); dst = torch.cat([i_nodes, u_nodes])
    deg = torch.zeros(n); deg.scatter_add_(0, src, torch.ones(2 * e))
    dis = deg.pow(-0.5); dis[torch.isinf(dis)] = 0.0
    vals = dis[src] * dis[dst]
    adj = torch.sparse_coo_tensor(torch.stack([src, dst]), vals, (n, n)).coalesce()
    E0 = torch.nn.Parameter(torch.empty(n, config.LIGHTGCN_DIM)); torch.nn.init.normal_(E0, std=0.1)
    opt = torch.optim.Adam([E0], lr=config.LIGHTGCN_LR)
    put = torch.as_tensor(pu, dtype=torch.long); pit = torch.as_tensor(pi, dtype=torch.long) + nU
    L = config.LIGHTGCN_LAYERS

    def prop():
        ek = E0; out = E0
        for _ in range(L):
            ek = torch.sparse.mm(adj, ek); out = out + ek
        return out / (L + 1)

    for _ in range(epochs):
        out = prop(); u = out[put]; ip = out[pit]
        loss = 0.0
        for _ in range(neg_k):
            neg = torch.randint(0, nI, (e,)) + nU
            loss = loss - F.logsigmoid((u * ip).sum(1) - (u * out[neg]).sum(1)).mean()
        loss = loss / neg_k + config.LIGHTGCN_REG * (u.pow(2).sum(1).mean() + ip.pow(2).sum(1).mean())
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        fin = prop().numpy().astype(np.float32)
    return fin[:nU], fin[nU:], float(loss)


async def main():
    conn = await asyncpg.connect(dsn())
    await register_vector(conn)
    ctx = await build_context_from_conn(conn)

    rows = await conn.fetch(
        "SELECT user_id, movie_id, preference_score, review_date FROM interactions "
        "WHERE preference_score IS NOT NULL ORDER BY user_id, review_date NULLS FIRST")
    per_user = {}
    for r in rows:
        per_user.setdefault(r["user_id"], []).append(
            (r["movie_id"], float(r["preference_score"]), r["review_date"]))
    train_items, test_items = {}, {}
    for uid, items in per_user.items():
        if len(items) >= 2:
            k = max(1, round(0.2 * len(items)))
            train_items[uid], test_items[uid] = items[:-k], items[-k:]
        else:
            train_items[uid] = items

    # build LightGCN training edges from train positives
    user_map, item_map, pu, pi = {}, {}, [], []
    for uid, items in train_items.items():
        for mid, pref, _ in items:
            if pref < config.CF_POS_THRESHOLD:
                continue
            pu.append(user_map.setdefault(uid, len(user_map)))
            pi.append(item_map.setdefault(mid, len(item_map)))
    pu, pi = np.array(pu), np.array(pi)
    print(f"edges={len(pu)} users={len(user_map)} items={len(item_map)}")

    # eval pool
    uids, seen, rel = [], {}, {}
    for uid, test in test_items.items():
        r = {m for m, p, _ in test if p >= LIKE and m in ctx.idx}
        if not r:
            continue
        uids.append(uid); rel[uid] = r
        seen[uid] = [ctx.idx[m] for m, _, _ in train_items.get(uid, []) if m in ctx.idx]
    random.Random(42).shuffle(uids); uids = uids[:SAMPLE]

    import time
    print(f"\n{'variant':22s} {'sec':>6} {'recall@300':>11} {'recall@1000':>12} {'scorer hit@10|r':>16}")
    for name, kw in EXPERIMENTS:
        t = time.time()
        ue, ie, loss = train_variant(pu, pi, len(user_map), len(item_map), **kw)
        sec = time.time() - t
        # item matrix in ctx order
        n = len(ctx.ids)
        mf_item = np.zeros((n, config.LIGHTGCN_DIM), dtype=np.float32)
        mask = np.zeros(n, bool)
        for mid, row in item_map.items():
            j = ctx.idx.get(mid)
            if j is not None:
                mf_item[j] = ie[row]; mask[j] = True
        umf = {uid: ue[row] for uid, row in user_map.items()}
        r300 = r1000 = hit = retr = 0
        for uid in uids:
            uv = umf.get(uid)
            if uv is None:
                continue
            s = np.where(mask, mf_item @ uv, -1e9)
            if seen[uid]:
                s[np.array(seen[uid])] = -1e9
            top = np.argpartition(s, -1000)[-1000:]
            top = top[np.argsort(s[top])[::-1]]
            recs = [int(ctx.ids[j]) for j in top]
            retr += 1
            if rel[uid] & set(recs[:300]): r300 += 1
            if rel[uid] & set(recs[:1000]): r1000 += 1
            if rel[uid] & set(recs[:10]): hit += 1
        print(f"{name:22s} {sec:6.1f} {r300/retr:11.4f} {r1000/retr:12.4f} {hit/retr:16.4f}")
    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
