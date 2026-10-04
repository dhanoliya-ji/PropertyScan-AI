"""Per-tier error budget. All terms are 1-sigma.

Length-like quantities L get sigma = sqrt(abs_terms^2 + (scale_rel * L)^2), so a tier
with weak metric scale (monocular depth) widens proportionally with size, while LiDAR
widens only with surface-fit noise.

`scale_rel` for video/photos is calibrated against the LiDAR tier on the benchmark
captures (scripts/calibrate.py); the defaults below are the pre-calibration priors.
"""
from __future__ import annotations

import math

from .schema import Measurement

Z95 = 1.96

BUDGETS = {
    # surface: residual systematic per surface position (range bias, residual drift)
    # scale_rel: relative metric-scale error
    "lidar": {"surface": 0.006, "scale_rel": 0.003, "height_surface": 0.004},
    "video": {"surface": 0.02, "scale_rel": 0.03, "height_surface": 0.02},
    "photos": {"surface": 0.04, "scale_rel": 0.07, "height_surface": 0.04},
}


def length(tier: str, value: float, fit_sigmas=(), unit="m", observed=True, extra_rel=0.0) -> Measurement:
    b = BUDGETS[tier]
    abs2 = sum(s * s for s in fit_sigmas) + 2 * b["surface"] ** 2
    rel = math.hypot(b["scale_rel"], extra_rel)
    sigma = math.sqrt(abs2 + (rel * value) ** 2)
    return _m(value, sigma, unit, observed)


def height(tier: str, value: float, fit_sigmas=(), observed=True) -> Measurement:
    b = BUDGETS[tier]
    abs2 = sum(s * s for s in fit_sigmas) + 2 * b["height_surface"] ** 2
    sigma = math.sqrt(abs2 + (b["scale_rel"] * value) ** 2)
    return _m(value, sigma, "m", observed)


def area_from_dims(tier: str, area: float, perimeter: float, fit_sigma_pos: float = 0.0) -> Measurement:
    """Area of a polygon whose edges are each uncertain by sigma_pos and whose scale is
    uncertain by scale_rel: dA ~ perimeter/2 * sigma_pos (edge shifts) + 2*scale_rel*A."""
    b = BUDGETS[tier]
    s_pos = math.hypot(b["surface"], fit_sigma_pos)
    sigma = math.hypot(0.5 * perimeter * s_pos, 2 * b["scale_rel"] * area)
    return _m(area, sigma, "m2", True)


def widen(m: Measurement, factor: float, observed: bool | None = None) -> Measurement:
    return _m(m.value, m.sigma * factor, m.unit, m.observed if observed is None else observed)


def _m(value, sigma, unit, observed) -> Measurement:
    return Measurement(value=round(value, 4), sigma=round(sigma, 4),
                       ci95=(round(value - Z95 * sigma, 4), round(value + Z95 * sigma, 4)),
                       unit=unit, observed=observed)
