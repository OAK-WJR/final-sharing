"""Lidar geometry: bearing tables, scan warping, swept-arc free length."""

import math

import numpy as np

N = 720
ARES = 2.0 * math.pi / N
IDX = np.arange(N)
PHI = -(((IDX + N // 2) % N) - N // 2) * ARES     # CCW-positive bearing
COS_P = np.cos(PHI)
SIN_P = np.sin(PHI)
FRONT_M = np.abs(PHI) < math.radians(22.0)        # wedge window
CONE_M = np.abs(PHI) < math.radians(15.0)         # anti-stall cone

RMAX = 10.0          # m, range clip
S_SKIP = 0.25        # m of arc the swath ignores (apex-graze guard)
ANG_CAP = 2.0        # rad of arc beyond which the plan is stale


def warp(scan_m, dx, dy, dth):
    """Project the capture-frame scan into the predicted frame.

    Empty bins after re-binning stay 0.0 ("no return") — the steering's
    zero-ignoring window average is the occlusion repair.
    """
    vmask = scan_m > 0.05
    px = scan_m[vmask] * COS_P[vmask]
    py = scan_m[vmask] * SIN_P[vmask]
    ct, st = math.cos(-dth), math.sin(-dth)
    ox, oy = px - dx, py - dy
    qx = ct * ox - st * oy
    qy = st * ox + ct * oy
    d = np.hypot(qx, qy)
    idx = np.rint(-np.arctan2(qy, qx) / ARES).astype(np.int64) % N
    out = np.full(N, np.inf)
    np.minimum.at(out, idx, d)
    out[~np.isfinite(out)] = 0.0
    return np.minimum(out, RMAX), qx, qy


def swath(qx, qy, kappa, half_w):
    """Arc length until the planned circular path first sweeps a point."""
    if qx.size == 0:
        return RMAX
    ahead = qx > -0.05
    if not ahead.any():
        return RMAX
    ax, ay = qx[ahead], qy[ahead]
    if abs(kappa) < 1e-3:
        m = (np.abs(ay) < half_w) & (ax > S_SKIP)
        return float(ax[m].min()) if m.any() else RMAX
    r = 1.0 / kappa
    sg = 1.0 if r > 0 else -1.0
    ra = abs(r)
    yy = ay * sg
    e = np.abs(np.hypot(ax, yy - ra) - ra)
    ang = np.arctan2(ax, ra - yy)
    ang = np.where(ang < 0.0, ang + 2.0 * math.pi, ang)
    s = ra * ang
    m = (e < half_w) & (s > S_SKIP) & (ang < ANG_CAP)
    if not m.any():
        return RMAX
    return float(min(s[m].min(), RMAX))
