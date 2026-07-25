"""IMM model-bank builder (CV, CA, CT±ω), reusing the validated per-mode filters from
imm_synthetic.py so the tracker and the prototype share one source of model math."""
from __future__ import annotations

import numpy as np
from imm_synthetic import CaModelFilter, CtModelFilter, CvModelFilter


def build_model_bank(dt, sigma_pos, q_accel, r, omegas):
    """Return (filters, mode_names). Order is CV, CA, then one CT per turn rate in `omegas`
    (rad/s). `omegas=()` gives a CV+CA bank; the tracker default is (+ω, -ω)."""
    r = np.asarray(r, dtype=float)
    filters = [
        CvModelFilter(dt, sigma_pos, q_accel, r),
        CaModelFilter(dt, sigma_pos, q_accel, r),
    ]
    names = ["CV", "CA"]
    for w in omegas:
        filters.append(CtModelFilter(dt, w, q_accel, r))
        names.append(f"CT{'+' if w >= 0 else ''}{w:g}")
    return filters, names
