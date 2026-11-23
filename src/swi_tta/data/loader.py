"""Unified data loaders for OpenSWI synthetic subsets and LongBeach/CSRM field arrays."""

from __future__ import annotations

import os
from typing import Dict, NamedTuple, Optional, Sequence, Tuple
import numpy as np

SHALLOW_STRUCTURE_TYPES: Tuple[str, ...] = (
    "FlatVelA",
    "FlatFaultA",
    "CurveVelA",
    "CurveFaultA",
    "StyleA",
)

DEEP_BASE_MODELS: Tuple[str, ...] = (
    "US-upper-mantle",
    "Continental_China_Shen2016",
    "USTCLitho1",
    "Central_and_Western_US_Shen2013",
    "CSEM_Australasia",
    "CSEM_South_Atlantic",
    "CSEM_lberia",
    "CSEM_Eastmed",
    "CSEM_North_Atlantic",
    "CSEM_Japan",
    "CSEM_Europe",
    "Alaska",
    "EUCrust",
    "LITHO1",
)


class SWIDatasetBatch(NamedTuple):
    """Unified surface-wave dispersion dataset container.

    Attributes
    ----------
    period : np.ndarray
        Observation periods (s), shape ``(N, P)``, float32.
    phase_vel : np.ndarray
        Phase velocity observations (km/s), shape ``(N, P)``, float32; missing entries set to 0.
    group_vel : np.ndarray
        Group velocity observations (km/s), shape ``(N, P)``, float32; missing entries set to 0.
    mask : np.ndarray
        Binary validity mask, shape ``(N, P, 2)``, uint8 (channel 0 = phase, channel 1 = group).
    vs_profile : Optional[np.ndarray]
        Ground-truth or published reference shear-wave velocity profiles (km/s), shape ``(N, L)``.
    depth : Optional[np.ndarray]
        Depth coordinate grid (km), shape ``(N, L)``.
    structure_type : Optional[np.ndarray]
        Geological structure or regional source-model label per sample, shape ``(N,)``.
    coords : Optional[np.ndarray]
        Station or grid coordinates ``(lon, lat)`` in degrees, shape ``(N, 2)``.
    """

    period: np.ndarray
    phase_vel: np.ndarray
    group_vel: np.ndarray
    mask: np.ndarray
    vs_profile: Optional[np.ndarray]
    depth: Optional[np.ndarray]
    structure_type: Optional[np.ndarray]
    coords: Optional[np.ndarray]

    def as_dict(self) -> Dict[str, Optional[np.ndarray]]:
        return self._asdict()


def _build_mask_and_clean_disp(
    raw_disp: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract period, phase velocity, group velocity, and strict binary mask from ``(N, P, 3)``."""
    arr = np.asarray(raw_disp, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Dispersion array must have shape (N, P, 3); got {arr.shape}")

    raw_t = arr[:, :, 0]
    raw_c = arr[:, :, 1]
    raw_u = arr[:, :, 2]

    valid_t = np.isfinite(raw_t) & (raw_t > 0.0)
    mask_c = valid_t & np.isfinite(raw_c) & (raw_c > 0.0)
    mask_u = valid_t & np.isfinite(raw_u) & (raw_u > 0.0)

    period = np.where(valid_t, raw_t, 0.0).astype(np.float32)
    phase_vel = np.where(mask_c, raw_c, 0.0).astype(np.float32)
    group_vel = np.where(mask_u, raw_u, 0.0).astype(np.float32)
    mask = np.stack([mask_c.astype(np.uint8), mask_u.astype(np.uint8)], axis=-1)
    return period, phase_vel, group_vel, mask


def load_shallow_subset(
    raw_dir: str,
    structure_types: Optional[Sequence[str]] = None,
    max_samples_per_type: Optional[int] = None,
    seed: int = 0,
) -> SWIDatasetBatch:
    """Load OpenSWI shallow synthetic subsets (0.2-10 s base dataset)."""
    base_dir = os.path.join(raw_dir, "OpenSWI-shallow", "0.2-10s-Base")
    types_to_check = list(structure_types) if structure_types is not None else list(SHALLOW_STRUCTURE_TYPES)

    periods_list, phase_list, group_list, mask_list = [], [], [], []
    vs_list, depth_list, st_list = [], [], []

    for st in types_to_check:
        disp_path = os.path.join(base_dir, f"{st}_disp.npz")
        model_path = os.path.join(base_dir, f"{st}_model.npz")
        if not (os.path.exists(disp_path) and os.path.exists(model_path)):
            if structure_types is not None:
                raise FileNotFoundError(f"Missing shallow dataset files: {disp_path} or {model_path}")
            continue

        d_arr = np.load(disp_path)["data"]
        m_arr = np.load(model_path)["data"]
        if d_arr.shape[0] != m_arr.shape[0]:
            raise ValueError(f"{st} count mismatch: disp={d_arr.shape[0]} vs model={m_arr.shape[0]}")

        if max_samples_per_type is not None and d_arr.shape[0] > max_samples_per_type:
            rng = np.random.RandomState(seed)
            idx = np.sort(rng.choice(d_arr.shape[0], size=max_samples_per_type, replace=False))
            d_arr = d_arr[idx]
            m_arr = m_arr[idx]

        p, c, u, m = _build_mask_and_clean_disp(d_arr)
        periods_list.append(p)
        phase_list.append(c)
        group_list.append(u)
        mask_list.append(m)
        depth_list.append(m_arr[:, :, 0].astype(np.float32))
        vs_list.append(m_arr[:, :, 2].astype(np.float32))
        st_list.append(np.full(d_arr.shape[0], st, dtype="<U16"))

    if not periods_list:
        raise FileNotFoundError(f"No shallow synthetic files found under {base_dir}")

    return SWIDatasetBatch(
        period=np.concatenate(periods_list, axis=0),
        phase_vel=np.concatenate(phase_list, axis=0),
        group_vel=np.concatenate(group_list, axis=0),
        mask=np.concatenate(mask_list, axis=0),
        vs_profile=np.concatenate(vs_list, axis=0),
        depth=np.concatenate(depth_list, axis=0),
        structure_type=np.concatenate(st_list, axis=0),
        coords=None,
    )


def load_deep_subset(
    raw_dir: str,
    model_names: Optional[Sequence[str]] = None,
    max_samples_per_model: Optional[int] = None,
    seed: int = 0,
) -> SWIDatasetBatch:
    """Load OpenSWI deep lithospheric synthetic subsets (1-100 s base dataset)."""
    base_dir = os.path.join(raw_dir, "OpenSWI-deep", "1s-100s-Base")
    names_to_check = list(model_names) if model_names is not None else list(DEEP_BASE_MODELS)

    periods_list, phase_list, group_list, mask_list = [], [], [], []
    vs_list, depth_list, st_list, coords_list = [], [], [], []

    for name in names_to_check:
        disp_path = os.path.join(base_dir, f"{name}_disp.npz")
        model_path = os.path.join(base_dir, f"{name}_model.npz")
        loc_path = os.path.join(base_dir, f"{name}_loc.npz")
        if not (os.path.exists(disp_path) and os.path.exists(model_path)):
            if model_names is not None:
                raise FileNotFoundError(f"Missing deep dataset files: {disp_path} or {model_path}")
            continue

        d_arr = np.load(disp_path)["data"]
        m_arr = np.load(model_path)["data"]
        loc_arr = np.load(loc_path)["data"] if os.path.exists(loc_path) else np.zeros((d_arr.shape[0], 2), dtype=np.float32)

        if max_samples_per_model is not None and d_arr.shape[0] > max_samples_per_model:
            rng = np.random.RandomState(seed)
            idx = np.sort(rng.choice(d_arr.shape[0], size=max_samples_per_model, replace=False))
            d_arr = d_arr[idx]
            m_arr = m_arr[idx]
            loc_arr = loc_arr[idx]

        p, c, u, m = _build_mask_and_clean_disp(d_arr)
        periods_list.append(p)
        phase_list.append(c)
        group_list.append(u)
        mask_list.append(m)
        depth_list.append(m_arr[:, :, 0].astype(np.float32))
        vs_list.append(m_arr[:, :, 2].astype(np.float32))
        st_list.append(np.full(d_arr.shape[0], name, dtype="<U36"))
        coords_list.append(loc_arr.astype(np.float32))

    if not periods_list:
        raise FileNotFoundError(f"No deep synthetic files found under {base_dir}")

    return SWIDatasetBatch(
        period=np.concatenate(periods_list, axis=0),
        phase_vel=np.concatenate(phase_list, axis=0),
        group_vel=np.concatenate(group_list, axis=0),
        mask=np.concatenate(mask_list, axis=0),
        vs_profile=np.concatenate(vs_list, axis=0),
        depth=np.concatenate(depth_list, axis=0),
        structure_type=np.concatenate(st_list, axis=0),
        coords=np.concatenate(coords_list, axis=0),
    )


def load_longbeach(raw_dir: str, include_ref_model: bool = True) -> SWIDatasetBatch:
    """Load the LongBeach dense-array field dataset (5,297 stations)."""
    lb_dir = os.path.join(raw_dir, "OpenSWI-real", "LongBeach")
    disp_arr = np.load(os.path.join(lb_dir, "longbeach_disp.npz"))["data"]
    loc_arr = np.load(os.path.join(lb_dir, "longbeach_loc.npz"))["data"].astype(np.float32)

    period, phase_vel, group_vel, mask = _build_mask_and_clean_disp(disp_arr)
    if include_ref_model:
        ref_arr = np.load(os.path.join(lb_dir, "syn_model.npz"))["data"]
        depth = ref_arr[:, :, 0].astype(np.float32)
        vs_ref = ref_arr[:, :, 2].astype(np.float32)
    else:
        depth = None
        vs_ref = None

    return SWIDatasetBatch(
        period=period,
        phase_vel=phase_vel,
        group_vel=group_vel,
        mask=mask,
        vs_profile=vs_ref,
        depth=depth,
        structure_type=None,
        coords=loc_arr,
    )


def load_csrm(raw_dir: str, include_ref_model: bool = True) -> SWIDatasetBatch:
    """Load the China Seismological Reference Model (CSRM) field dataset (12,705 grid points)."""
    csrm_dir = os.path.join(raw_dir, "OpenSWI-real", "CSRM")
    disp_arr = np.load(os.path.join(csrm_dir, "obs_period_phase_group.npz"))["data"]
    loc_arr = np.load(os.path.join(csrm_dir, "obs_depth_vs_loc.npz"))["data"].astype(np.float32)

    period, phase_vel, group_vel, mask = _build_mask_and_clean_disp(disp_arr)
    if include_ref_model:
        ref_arr = np.load(os.path.join(csrm_dir, "obs_depth_vs.npz"))["data"]
        depth = ref_arr[:, :, 0].astype(np.float32)
        vs_ref = ref_arr[:, :, 1].astype(np.float32)
    else:
        depth = None
        vs_ref = None

    return SWIDatasetBatch(
        period=period,
        phase_vel=phase_vel,
        group_vel=group_vel,
        mask=mask,
        vs_profile=vs_ref,
        depth=depth,
        structure_type=None,
        coords=loc_arr,
    )
