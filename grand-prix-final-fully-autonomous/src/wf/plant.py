"""Vehicle model: measured maps, lags, bicycle step."""

import math

import numpy as np

L = 0.225
K_US = 0.77          # venue calib 2026-08-01, jointly fit with STE_D
SPD_U = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
SPD_V = np.array([0.0, 1.94, 2.51, 3.0, 4.12, 4.5])
STE_S = np.array([0.0, 0.25, 0.5, 1.0])
STE_D = np.array([0.0, 0.43, 0.55, 0.79])   # venue calib 2026-08-01; the 0.5
                                            # cell failed its fit gate and is
                                            # interpolated between neighbours
SPEED_DELAY = 0.10
STEER_DELAY = 0.184  # venue calib 2026-08-01 (was 0.12)
TAU_UP = 0.373
TAU_DOWN = 0.247
TAU_STEER = 0.10
LIDAR_LATENCY = 0.06

# mirrors rc.drive.set_max_speed(spd_scale): the plant sees u * spd_scale, so
# the twin stays honest while the command stays in the band the driver asked
# for. Set by WfLite.__init__.
SPD_SCALE = 1.0


def speed_map(u):
    return math.copysign(float(np.interp(abs(u) * SPD_SCALE, SPD_U, SPD_V)), u)


def speed_inv(v):
    return math.copysign(
        float(np.interp(abs(v), SPD_V, SPD_U)) / max(SPD_SCALE, 1e-6), v)


def steer_map(s):
    return math.copysign(float(np.interp(abs(s), STE_S, STE_D)), s)


def step(v, delta, x, y, th, u, s_cmd, dt):
    """One tick of the twin. Returns (v, delta, x, y, th, yaw_rate)."""
    v_ss = speed_map(u)
    tau = TAU_UP if abs(v_ss) > abs(v) else TAU_DOWN
    v = v + (v_ss - v) * (1.0 - math.exp(-dt / tau))
    d_ss = steer_map(-s_cmd)             # sim negates the student angle
    delta = delta + (d_ss - delta) * (1.0 - math.exp(-dt / TAU_STEER))
    x += v * math.cos(th) * dt
    y += v * math.sin(th) * dt
    w = v * math.tan(delta) / (L * (1.0 + K_US * v * v))
    th += w * dt
    return v, delta, x, y, th, w
