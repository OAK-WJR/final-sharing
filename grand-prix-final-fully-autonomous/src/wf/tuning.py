"""Control constants and the tunable parameter spec."""

V_MAX = 4.0          # m/s, user cap (plant ceiling is 4.5)
V_CREEP = 1.40       # m/s anti-stall floor while the front cone is open
CREEP_D0 = 0.30      # m of cone clearance where the floor starts
CREEP_SPAN = 0.45    # m over which the floor ramps to V_CREEP
TAU_PRED = 0.15      # s, prediction horizon (ablation: tunes flat)
BLIND_TRUST = 1.8    # m of blind travel at full speed (not a knob on purpose)
BLIND_STOP = 3.0     # m: beyond this the scan is dead, not stale. A mid-run
                     # lidar freeze once left blind at 31 m while the crawl
                     # floor kept the car creeping into the unknown.
T_REACT = 0.15       # s, reaction time inside the brake law
NOSE = 0.26          # m, stop margin ahead of the lidar
BRAKE_W = 0.52       # brake swath width, fraction of half_width (0.62 phantom-
                     # braked: wall-pointing during corrections clipped it)
BRAKE_KAP = 0.40     # fraction of commanded curvature the swath sweeps
SLEW = 0.20          # per-tick steering slew (0.13 lagged turn-in and un-turn)
OPEN_M = 3.0         # m of front clearance that counts as open floor
OPEN_DAMP = 0.35     # steering multiplier there — no gradient to follow in the
                     # open, so damp toward straight instead of circling
U_SLEW = 0.30        # per-tick throttle slew
BRAKE_FF = 1.2       # over-speed feedforward (ablation: load-bearing)
U_MIN = -0.40        # deepest braking throttle
REP_LP = 0.35        # wall-repulsion low-pass blend per tick

WEDGE_DIST = 0.35    # m; wedge recovery is load-bearing (5/8 -> 3/8 finishes)
WEDGE_HOLD = 0.5     # s blocked before reversing
REV_T = 0.34         # s of reverse
REV_U = -0.32        # reverse throttle
RECOV_COOL = 1.30    # s after a reverse ...
RECOV_WIN = 70.0     # ... with the steering window capped (no U-turns)

GYRO_W = 0.25        # gyro weight in the yaw complementary filter
GYRO_KB = 0.004      # gyro bias tracker gain per tick
BIAS0 = 0.16         # rad/s, sim-injected mean yaw bias (real /attitude: ~0)
V_BLEND = 0.20       # per-tick blend toward a sane measured wheel speed
V_MEAS_LO = 0.30     # m/s, below this a sensorless encoder reads garbage
V_MEAS_HI = 6.0

PARAM_SPEC = {
    "window_deg":     {"default": 110.0, "min": 60.0,  "max": 120.0},
    "ray_width_deg":  {"default": 25.0,  "min": 8.0,   "max": 40.0},
    "weight_range_m": {"default": 1.50,  "min": 0.40,  "max": 4.00},
    "kp":             {"default": 0.02,  "min": 0.004, "max": 0.06},
    "half_width":     {"default": 0.21,  "min": 0.13,  "max": 0.36},
    "a_brake":        {"default": 13.0,  "min": 2.0,   "max": 20.0},
    "a_lat":          {"default": 21.0,  "min": 3.0,   "max": 32.0},
    "wall_rep":       {"default": 0.0,   "min": 0.0,   "max": 2.5},
    "u_floor":        {"default": 0.0,   "min": 0.0,   "max": 0.9},
    "spd_scale":      {"default": 1.0,   "min": 0.30,  "max": 1.0},
}
