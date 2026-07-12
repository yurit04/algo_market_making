"""
Tile-coding function approximator and the linear combination of tile codings (LCTC).

Implements the value-function approximator of Spooner et al. (2018, §4.3, Eq. 7). Three
independent tile codings — over the *agent* state, the *market* state, and the *full*
state — each estimate the action-value; the overall estimate is a fixed linear
combination:

    q(s, a) = sum_i lambda_i * q_i(s, a),   sum_i lambda_i = 1.

Each coding uses hashed tile coding: ``num_tilings`` overlapping, offset tilings of the
(scaled) feature subvector, with the action folded in as an extra coordinate so a single
weight vector represents all actions. Tiles are hashed **statelessly** with a
deterministic FNV-1a hash into ``[0, size)``: unlike Sutton's insertion-order IHT this is
reproducible across processes (so only the weight vectors need saving) and never
"saturates" — it degrades gracefully via hash collisions instead of hitting a hard cap.
Learning uses replacing eligibility traces stored sparsely (only non-zero traces kept).
"""

from __future__ import annotations

import math

import numpy as np

# 64-bit FNV-1a — deterministic (unlike Python's per-process-salted ``hash``), so tile
# indices reproduce across processes without persisting a tile->index table.
_FNV_OFFSET = 1469598103934665603
_FNV_PRIME = 1099511628211
_MASK64 = (1 << 64) - 1


def _fnv_mix(h: int, x: int) -> int:
    return ((h ^ (x & _MASK64)) * _FNV_PRIME) & _MASK64


_OFFSET_U64 = np.uint64(_FNV_OFFSET)
_PRIME_U64 = np.uint64(_FNV_PRIME)


def _fnv_mix_np(h: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Vectorised FNV-1a mix over uint64 arrays (numpy wraps unsigned overflow mod 2^64)."""
    with np.errstate(over="ignore"):
        return (h ^ x) * _PRIME_U64


class IHT:
    """Index Hash Table (Sutton tiles3): maps tile tuples to a bounded index range."""

    def __init__(self, size: int) -> None:
        self.size = size
        self._table: dict[tuple, int] = {}
        self._overfull = 0

    def index(self, obj: tuple) -> int:
        table = self._table
        idx = table.get(obj)
        if idx is not None:
            return idx
        if len(table) >= self.size:
            # Collision fallback once full: hash into the existing range.
            self._overfull += 1
            return hash(obj) % self.size
        idx = len(table)
        table[obj] = idx
        return idx


def tiles(iht: IHT, num_tilings: int, floats: np.ndarray, action: int) -> list[int]:
    """
    Return ``num_tilings`` active tile indices for ``floats`` (already scaled to tile
    units) with ``action`` as an extra integer coordinate.
    """
    qfloats = [int(math.floor(f * num_tilings)) for f in floats]
    result = []
    for tiling in range(num_tilings):
        coords = [tiling]
        b = tiling
        for q in qfloats:
            coords.append((q + b) // num_tilings)
            b += 2 * tiling + 1
        coords.append(action)
        result.append(iht.index(tuple(coords)))
    return result


class _Coding:
    """One tile coding: an IHT, a weight vector, and a sparse eligibility trace."""

    def __init__(self, n_features: int, num_tilings: int, iht_size: int) -> None:
        self.num_tilings = num_tilings
        self.size = iht_size
        self.w = np.zeros(iht_size, dtype=np.float64)
        self.n_features = n_features
        # Eligibility trace stored as persistent parallel arrays (idx, val) plus a
        # slot map, so decay/update are vectorised and no dict<->array conversion
        # happens per step. Below-floor entries are pruned periodically, not each step.
        self._tr_idx = np.empty(0, dtype=np.int64)
        self._tr_val = np.empty(0, dtype=np.float64)
        self._tr_n = 0
        self._tr_cap = 0
        self._tr_slot: dict[int, int] = {}

    # -- eligibility trace (array-backed) ------------------------------------
    def reset_trace(self) -> None:
        self._tr_n = 0
        self._tr_slot.clear()

    def _ensure_cap(self, need: int) -> None:
        if need <= self._tr_cap:
            return
        new_cap = max(64, self._tr_cap * 2, need)
        idx = np.empty(new_cap, dtype=np.int64)
        val = np.empty(new_cap, dtype=np.float64)
        idx[: self._tr_n] = self._tr_idx[: self._tr_n]
        val[: self._tr_n] = self._tr_val[: self._tr_n]
        self._tr_idx, self._tr_val, self._tr_cap = idx, val, new_cap

    def trace_set_active(self, idxs) -> None:
        """Replacing traces: set each active tile's trace to 1 (append if unseen)."""
        slot = self._tr_slot
        if isinstance(idxs, np.ndarray):
            idxs = idxs.tolist()
        for idx in idxs:
            pos = slot.get(idx)
            if pos is not None:
                self._tr_val[pos] = 1.0
            else:
                self._ensure_cap(self._tr_n + 1)
                self._tr_idx[self._tr_n] = idx
                self._tr_val[self._tr_n] = 1.0
                slot[idx] = self._tr_n
                self._tr_n += 1

    def trace_update_decay(self, step: float, decay: float, floor: float, prune: bool) -> None:
        n = self._tr_n
        if n == 0:
            return
        idx = self._tr_idx[:n]
        val = self._tr_val[:n]
        self.w[idx] += step * val
        val *= decay
        if prune:
            mask = val > floor
            m = int(mask.sum())
            if m < n:
                self._tr_idx[:m] = idx[mask]
                self._tr_val[:m] = val[mask]
                self._tr_n = m
                self._tr_slot = {int(k): j for j, k in enumerate(self._tr_idx[:m])}

    def base_coords(self, scaled: np.ndarray) -> np.ndarray:
        """
        Per-tiling partial FNV hash of the (tiling, state-coordinates) as a uint64 array,
        computed once (vectorised) and reused across actions. The action is mixed in later
        by ``active_from_base``. Tiling ``t`` offsets feature ``f`` by ``t + f*(2t+1)``.
        """
        nt = self.num_tilings
        q = np.floor(np.asarray(scaled, dtype=np.float64) * nt).astype(np.int64)  # (F,)
        tilings = np.arange(nt, dtype=np.int64)                                   # (nt,)
        f_idx = np.arange(q.shape[0], dtype=np.int64)                             # (F,)
        b = tilings[:, None] + f_idx[None, :] * (2 * tilings[:, None] + 1)        # (nt, F)
        coords = (q[None, :] + b) // nt                                           # (nt, F)

        h = np.full(nt, _OFFSET_U64, dtype=np.uint64)
        h = _fnv_mix_np(h, tilings.astype(np.uint64))
        for f in range(coords.shape[1]):
            h = _fnv_mix_np(h, coords[:, f].astype(np.uint64))
        return h

    def active_from_base(self, base: np.ndarray, action: int) -> np.ndarray:
        mixed = _fnv_mix_np(base, np.uint64(action))
        return (mixed % np.uint64(self.size)).astype(np.int64)

    def active(self, scaled: np.ndarray, action: int) -> np.ndarray:
        return self.active_from_base(self.base_coords(scaled), action)

    def value(self, scaled: np.ndarray, action: int) -> float:
        return float(self.w[self.active(scaled, action)].sum())


class LCTC:
    """
    Linear combination of three tile codings (agent / market / full state).

    Feature scaling: each feature is mapped from its ``(lo, hi)`` range onto
    ``[0, tiles_per_dim]`` so a unit step is one tile width; tiles() then offsets across
    ``num_tilings``.
    """

    def __init__(
        self,
        feature_names: tuple[str, ...],
        subsets: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]],
        ranges: dict[str, tuple[float, float]],
        lambda_weights: tuple[float, float, float],
        *,
        num_tilings: int = 32,
        tiles_per_dim: int = 8,
        iht_size: int = 1 << 20,
    ) -> None:
        self.feature_names = feature_names
        self.name_to_idx = {n: i for i, n in enumerate(feature_names)}
        self.lambdas = np.asarray(lambda_weights, dtype=np.float64)
        self.tiles_per_dim = tiles_per_dim
        self.num_tilings = num_tilings

        # Per-coding: the indices of its features and the affine scaling to tile units.
        self.subset_idx: list[np.ndarray] = []
        self.scale: list[np.ndarray] = []
        self.offset: list[np.ndarray] = []
        self.codings: list[_Coding] = []
        for subset in subsets:
            idx = np.array([self.name_to_idx[n] for n in subset], dtype=np.int64)
            los = np.array([ranges[n][0] for n in subset], dtype=np.float64)
            his = np.array([ranges[n][1] for n in subset], dtype=np.float64)
            span = np.where(his > los, his - los, 1.0)
            self.subset_idx.append(idx)
            self.scale.append(tiles_per_dim / span)
            self.offset.append(los)
            self.codings.append(_Coding(len(subset), num_tilings, iht_size))

    def _scaled(self, coding: int, state: np.ndarray) -> np.ndarray:
        sub = state[self.subset_idx[coding]]
        return (sub - self.offset[coding]) * self.scale[coding]

    def q(self, state: np.ndarray, action: int) -> float:
        total = 0.0
        for i, coding in enumerate(self.codings):
            total += self.lambdas[i] * coding.value(self._scaled(i, state), action)
        return total

    def q_from_active(self, active: list[list[int]]) -> float:
        """Action-value from precomputed active tiles per coding (avoids recompute)."""
        total = 0.0
        for i, idxs in enumerate(active):
            total += self.lambdas[i] * float(self.codings[i].w[idxs].sum())
        return total

    def q_all(self, state: np.ndarray, n_actions: int) -> np.ndarray:
        out = np.zeros(n_actions, dtype=np.float64)
        for i, coding in enumerate(self.codings):
            base = coding.base_coords(self._scaled(i, state))
            lam = self.lambdas[i]
            w = coding.w
            for a in range(n_actions):
                idxs = coding.active_from_base(base, a)
                out[a] += lam * w[idxs].sum()
        return out

    def active_all(self, state: np.ndarray, action: int) -> list[list[int]]:
        """Active tile indices per coding for (state, action) — used for trace updates."""
        return [self.codings[i].active(self._scaled(i, state), action)
                for i in range(len(self.codings))]

    # -- eligibility traces ---------------------------------------------------
    def reset_traces(self) -> None:
        for c in self.codings:
            c.reset_trace()

    def accumulate_replacing(self, active: list[list[int]]) -> None:
        """Replacing traces: set the active tiles of the current (s, a) to 1."""
        for c, idxs in zip(self.codings, active):
            c.trace_set_active(idxs)

    def update_and_decay(self, alpha: float, delta: float, decay: float, floor: float = 1e-4) -> None:
        """
        Vectorised eligibility-trace step: for every coding apply
        ``w_i += alpha * delta * lambda_i * e_i`` then decay ``e_i *= decay``. Pruning of
        below-floor traces is amortised (every few steps) to avoid rebuilding the slot
        map on every update.
        """
        self._prune_ctr = getattr(self, "_prune_ctr", 0) + 1
        prune = self._prune_ctr % 8 == 0
        for i, c in enumerate(self.codings):
            c.trace_update_decay(alpha * delta * self.lambdas[i], decay, floor, prune)
