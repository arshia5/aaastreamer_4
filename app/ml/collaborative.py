"""Collaborative filtering via PyTorch BPR matrix factorisation.

Replaces LightFM's collaborative core (metadata is handled separately by the
content embeddings in the hybrid scorer). Trains on positive interactions
(preference >= threshold), weighted by preference. Supports incremental
``partial_fit_user`` for real-time updates between full retrains.

Artifact (joblib dict): user_factors, item_factors, item_bias, user_map
(db_user_id -> row), item_map (db_movie_id -> row), meta.
"""
from __future__ import annotations

import os
import threading
import time

import joblib
import numpy as np

from app.ml import config


def _device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_bpr(
    u_idx: np.ndarray,
    i_idx: np.ndarray,
    weights: np.ndarray,
    n_users: int,
    n_items: int,
    *,
    dim: int = config.CF_DIM,
    epochs: int = config.CF_EPOCHS,
    lr: float = config.CF_LR,
    reg: float = config.CF_REG,
    batch: int = config.CF_BATCH,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """Train BPR-MF. Returns (user_factors, item_factors, item_bias, losses)."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    dev = _device()
    U = torch.nn.Embedding(n_users, dim).to(dev)
    I = torch.nn.Embedding(n_items, dim).to(dev)
    B = torch.nn.Embedding(n_items, 1).to(dev)
    torch.nn.init.normal_(U.weight, std=0.05)
    torch.nn.init.normal_(I.weight, std=0.05)
    torch.nn.init.zeros_(B.weight)
    opt = torch.optim.Adam([*U.parameters(), *I.parameters(), *B.parameters()], lr=lr)

    u_t = torch.as_tensor(u_idx, dtype=torch.long, device=dev)
    i_t = torch.as_tensor(i_idx, dtype=torch.long, device=dev)
    w_t = torch.as_tensor(weights, dtype=torch.float32, device=dev)
    n = len(u_idx)
    losses: list[float] = []
    for _ in range(epochs):
        perm = torch.randperm(n, device=dev)
        ep_loss = 0.0
        for start in range(0, n, batch):
            b = perm[start:start + batch]
            u, ip, w = u_t[b], i_t[b], w_t[b]
            ineg = torch.randint(0, n_items, (len(b),), device=dev)
            ux, ipx, inx = U(u), I(ip), I(ineg)
            x = (ux * ipx).sum(1) + B(ip).squeeze(1) - (ux * inx).sum(1) - B(ineg).squeeze(1)
            loss = -(w * F.logsigmoid(x)).mean()
            loss = loss + reg * (ux.pow(2).sum(1).mean()
                                 + ipx.pow(2).sum(1).mean()
                                 + inx.pow(2).sum(1).mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(b)
        losses.append(ep_loss / n)
    return (
        U.weight.detach().cpu().numpy().astype(np.float32),
        I.weight.detach().cpu().numpy().astype(np.float32),
        B.weight.detach().cpu().numpy().astype(np.float32).reshape(-1),
        losses,
    )


def save_artifact(
    user_factors, item_factors, item_bias, user_map, item_map, meta: dict
) -> tuple[str, str]:
    """Persist a model. Returns (version_name, artifact_path)."""
    config.COLLAB_DIR.mkdir(parents=True, exist_ok=True)
    version = "mf_" + time.strftime("%Y%m%d_%H%M%S")
    path = config.COLLAB_DIR / f"{version}.joblib"
    tmp = path.with_suffix(".joblib.tmp")
    # Write to a temp file then atomically rename so a partially-written
    # artifact is never visible to a concurrent loader (real-time/serving).
    joblib.dump(
        {
            "user_factors": user_factors,
            "item_factors": item_factors,
            "item_bias": item_bias,
            "user_map": user_map,
            "item_map": item_map,
            "meta": meta,
        },
        tmp,
    )
    os.replace(tmp, path)
    return version, str(path)


class CollaborativeModel:
    """Runtime wrapper around trained factors with incremental updates."""

    def __init__(self, data: dict, path: str | None = None):
        self.user_factors = np.asarray(data["user_factors"], dtype=np.float32)
        self.item_factors = np.asarray(data["item_factors"], dtype=np.float32)
        self.item_bias = np.asarray(data["item_bias"], dtype=np.float32)
        self.user_map: dict[int, int] = {int(k): int(v) for k, v in data["user_map"].items()}
        self.item_map: dict[int, int] = {int(k): int(v) for k, v in data["item_map"].items()}
        # row index -> db movie id
        self.item_ids = np.empty(len(self.item_map), dtype=np.int64)
        for mid, idx in self.item_map.items():
            self.item_ids[idx] = mid
        self.meta = data.get("meta", {})
        self.path = path
        self._lock = threading.Lock()

    @classmethod
    def load(cls, path: str) -> "CollaborativeModel":
        path = config.resolve_artifact_path(path)
        return cls(joblib.load(path), path=path)

    def has_user(self, user_id: int) -> bool:
        return user_id in self.user_map

    def has_item(self, movie_id: int) -> bool:
        return movie_id in self.item_map

    def scores_for_user(self, user_id: int) -> np.ndarray | None:
        """Raw collaborative score per item row (aligned to self.item_ids)."""
        idx = self.user_map.get(user_id)
        if idx is None:
            return None
        return self.item_factors @ self.user_factors[idx] + self.item_bias

    def partial_fit_user(
        self,
        user_id: int,
        positive_movie_ids: list[tuple[int, float]],
        epochs: int = config.CF_PARTIAL_EPOCHS,
        lr: float = config.CF_PARTIAL_LR,
        blend: float = config.CF_PARTIAL_BLEND,
    ) -> bool:
        """Gently nudge one existing user's factor toward their (new) positives.

        Item factors stay frozen (the collaborative space is never distorted —
        only this user's own position moves). The update is intentionally small:
        a low learning rate, then a blend back toward the original nightly factor
        (`new = blend*updated + (1-blend)*nightly`) so a single odd review can't
        re-learn the user. Returns False if the user is unknown."""
        uidx = self.user_map.get(user_id)
        if uidx is None:
            return False
        pos = [(self.item_map[m], w) for m, w in positive_movie_ids if m in self.item_map]
        if not pos:
            return False
        import torch
        import torch.nn.functional as F

        with self._lock:
            original = self.user_factors[uidx].copy()
            u = torch.tensor(original, requires_grad=True)
            I = torch.tensor(self.item_factors)
            B = torch.tensor(self.item_bias)
            pos_idx = torch.tensor([p[0] for p in pos], dtype=torch.long)
            pos_w = torch.tensor([p[1] for p in pos], dtype=torch.float32)
            opt = torch.optim.Adam([u], lr=lr)
            n_items = I.shape[0]
            for _ in range(epochs):
                neg_idx = torch.randint(0, n_items, (len(pos),))
                x = (I[pos_idx] @ u) + B[pos_idx] - (I[neg_idx] @ u) - B[neg_idx]
                loss = -(pos_w * F.logsigmoid(x)).mean() + config.CF_REG * u.pow(2).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
            updated = u.detach().numpy().astype(np.float32)
            self.user_factors[uidx] = (blend * updated + (1.0 - blend) * original)
        return True
