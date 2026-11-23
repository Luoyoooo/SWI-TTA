"""Deterministic spatial-block and per-station period-holdout split generators."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple
import numpy as np


def create_spatial_block_split(
    coords: np.ndarray,
    ratio_a: float = 0.30,
    grid_nx: int = 10,
    grid_ny: int = 10,
    seed: int = 0,
    dataset_name: str = "LongBeach",
) -> Dict[str, Any]:
    """Partition stations into disjoint spatial blocks (Tuning/Validation vs. Report)."""
    pts = np.asarray(coords, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"coords must have shape (N, 2); got {pts.shape}")

    n_total = int(pts.shape[0])
    lon, lat = pts[:, 0], pts[:, 1]
    lon_min, lon_max = float(np.min(lon)), float(np.max(lon))
    lat_min, lat_max = float(np.min(lat)), float(np.max(lat))

    lon_edges = np.linspace(lon_min, lon_max + 1e-9, grid_nx + 1)
    lat_edges = np.linspace(lat_min, lat_max + 1e-9, grid_ny + 1)
    ix = np.clip(np.digitize(lon, lon_edges) - 1, 0, grid_nx - 1)
    iy = np.clip(np.digitize(lat, lat_edges) - 1, 0, grid_ny - 1)

    block_map: Dict[str, List[int]] = {}
    for idx in range(n_total):
        b_key = f"{int(ix[idx])}_{int(iy[idx])}"
        block_map.setdefault(b_key, []).append(int(idx))

    block_keys = sorted(block_map.keys())
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(block_keys))
    shuffled_keys = [block_keys[i] for i in perm]

    target_a = int(round(ratio_a * n_total))
    blocks_a: List[str] = []
    blocks_b: List[str] = []
    indices_a: List[int] = []
    indices_b: List[int] = []

    for b_key in shuffled_keys:
        members = block_map[b_key]
        if abs((len(indices_a) + len(members)) - target_a) <= abs(len(indices_a) - target_a):
            blocks_a.append(b_key)
            indices_a.extend(members)
        else:
            blocks_b.append(b_key)
            indices_b.extend(members)

    indices_a.sort()
    indices_b.sort()
    blocks_a.sort()
    blocks_b.sort()

    return {
        "dataset": str(dataset_name),
        "seed": int(seed),
        "grid_nx": int(grid_nx),
        "grid_ny": int(grid_ny),
        "target_ratio_a": float(ratio_a),
        "actual_ratio_a": float(len(indices_a) / n_total),
        "actual_ratio_b": float(len(indices_b) / n_total),
        "num_total": n_total,
        "num_a": len(indices_a),
        "num_b": len(indices_b),
        "num_blocks_total": len(block_keys),
        "num_blocks_a": len(blocks_a),
        "num_blocks_b": len(blocks_b),
        "bounds": {
            "lon_min": lon_min,
            "lon_max": lon_max,
            "lat_min": lat_min,
            "lat_max": lat_max,
        },
        "blocks_a": blocks_a,
        "blocks_b": blocks_b,
        "indices_a": indices_a,
        "indices_b": indices_b,
    }


def save_split_file(split_dict: Dict[str, Any], out_path: str) -> None:
    """Serialize a spatial or period split dictionary to UTF-8 JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(split_dict, f, indent=2, ensure_ascii=True, sort_keys=False)
        f.write("\n")


def load_split_file(path: str) -> Dict[str, Any]:
    """Load a JSON split manifest."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_period_holdout_split(
    mask: np.ndarray,
    holdout_ratio: float = 0.20,
    seed: int = 0,
    keep_endpoints: bool = True,
    min_valid_periods: int = 5,
    dataset_name: str = "unknown",
) -> Dict[str, Any]:
    """Deterministically hold out a fixed fraction of valid periods per station."""
    m = np.asarray(mask)
    if m.ndim != 3 or m.shape[2] != 2:
        raise ValueError(f"mask must have shape (N, P, 2); got {m.shape}")

    n_stations, n_periods, _ = m.shape
    period_valid = (m[:, :, 0] > 0.5) | (m[:, :, 1] > 0.5)
    rng = np.random.RandomState(seed)

    heldout_indices: List[List[int]] = []
    n_valid_total = 0
    n_heldout_total = 0
    stations_with_ho = 0

    for i in range(n_stations):
        valid_idx = np.where(period_valid[i])[0]
        nv = int(len(valid_idx))
        n_valid_total += nv
        if nv < min_valid_periods:
            heldout_indices.append([])
            continue

        candidates = valid_idx[1:-1] if (keep_endpoints and nv > 2) else valid_idx
        n_ho = min(int(round(holdout_ratio * nv)), len(candidates))
        if n_ho <= 0:
            heldout_indices.append([])
            continue

        chosen = np.sort(rng.choice(candidates, size=n_ho, replace=False)).astype(int).tolist()
        heldout_indices.append(chosen)
        n_heldout_total += n_ho
        stations_with_ho += 1

    return {
        "dataset": str(dataset_name),
        "seed": int(seed),
        "holdout_ratio": float(holdout_ratio),
        "keep_endpoints": bool(keep_endpoints),
        "min_valid_periods": int(min_valid_periods),
        "num_stations": int(n_stations),
        "num_periods": int(n_periods),
        "stations_with_holdout": int(stations_with_ho),
        "total_valid_periods": int(n_valid_total),
        "total_heldout_periods": int(n_heldout_total),
        "actual_holdout_fraction": float(n_heldout_total / max(n_valid_total, 1)),
        "heldout_indices": heldout_indices,
    }


def apply_period_holdout_mask(
    obs_mask: np.ndarray,
    heldout_indices: List[List[int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Split ``obs_mask`` into disjoint ``(kept_mask, heldout_obs_mask)`` arrays."""
    m = (np.asarray(obs_mask) > 0.5).astype(np.uint8)
    n_stations, n_periods, _ = m.shape
    if len(heldout_indices) != n_stations:
        raise ValueError(f"heldout_indices length {len(heldout_indices)} != num_stations {n_stations}")

    ho_flag = np.zeros((n_stations, n_periods, 2), dtype=np.uint8)
    for i, idxs in enumerate(heldout_indices):
        if idxs:
            ho_flag[i, idxs, :] = 1

    heldout_obs_mask = (m & ho_flag).astype(np.uint8)
    kept_mask = (m & (1 - ho_flag)).astype(np.uint8)
    return kept_mask, heldout_obs_mask
