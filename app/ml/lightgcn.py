"""LightGCN collaborative filtering (replaces BPR matrix factorisation).

Propagates user/item embeddings over the normalized user-item bipartite graph
(no feature transforms, no nonlinearity — LightGCN), trained with BPR loss.
Negatives never include a user's own rated items. Output embeddings are stored in
the same artifact shape as the MF model (user_factors / item_factors / item_bias=0)
so the existing CollaborativeModel runtime + hybrid scorer work unchanged.

Full-batch BPR per epoch (one propagation + all positives) keeps it CPU/VPS-feasible
at this scale (~82k users, ~31k items, ~1.1M edges).
"""
from __future__ import annotations

import numpy as np

from app.ml import config


def _device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_lightgcn(
    pos_u: np.ndarray,        # user row indices (0..nU-1)
    pos_i: np.ndarray,        # item row indices (0..nI-1)
    n_users: int,
    n_items: int,
    *,
    dim: int = config.LIGHTGCN_DIM,
    layers: int = config.LIGHTGCN_LAYERS,
    epochs: int = config.LIGHTGCN_EPOCHS,
    lr: float = config.LIGHTGCN_LR,
    reg: float = config.LIGHTGCN_REG,
    seed: int = 42,
):
    """Returns (user_emb (nU,dim), item_emb (nI,dim), losses)."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    dev = _device()
    n = n_users + n_items
    e = len(pos_u)

    # --- normalized symmetric adjacency  Â = D^-1/2 A D^-1/2 -------------- #
    u_nodes = torch.as_tensor(pos_u, dtype=torch.long)
    i_nodes = torch.as_tensor(pos_i, dtype=torch.long) + n_users
    src = torch.cat([u_nodes, i_nodes])
    dst = torch.cat([i_nodes, u_nodes])
    deg = torch.zeros(n, dtype=torch.float32)
    deg.scatter_add_(0, src, torch.ones(2 * e))
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    vals = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
    adj = torch.sparse_coo_tensor(
        torch.stack([src, dst]), vals, (n, n)).coalesce().to(dev)

    E0 = torch.nn.Parameter(torch.empty(n, dim, device=dev))
    torch.nn.init.normal_(E0, std=0.1)
    opt = torch.optim.Adam([E0], lr=lr)

    pu = torch.as_tensor(pos_u, dtype=torch.long, device=dev)
    pi = torch.as_tensor(pos_i, dtype=torch.long, device=dev) + n_users
    losses: list[float] = []

    def propagate():
        e_k = E0
        out = E0
        for _ in range(layers):
            e_k = torch.sparse.mm(adj, e_k)
            out = out + e_k
        return out / (layers + 1)

    for _ in range(epochs):
        out = propagate()
        u = out[pu]
        ip = out[pi]
        neg = torch.randint(0, n_items, (e,), device=dev) + n_users
        ineg = out[neg]
        pos_s = (u * ip).sum(1)
        neg_s = (u * ineg).sum(1)
        loss = -F.logsigmoid(pos_s - neg_s).mean()
        loss = loss + reg * (u.pow(2).sum(1).mean()
                             + ip.pow(2).sum(1).mean()
                             + ineg.pow(2).sum(1).mean())
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    with torch.no_grad():
        final = propagate().cpu().numpy().astype(np.float32)
    return final[:n_users], final[n_users:], losses
