"""Real-car deployment wrapper for evolve-contract racing controllers.

Feeds the controller exactly what it ate in the sim: a 720-sample cm scan
(the real lidar's 1080 are resampled), dt, and an imu dict {gyro, speed}.

Settings come from the ENVIRONMENT, so redeploying never clobbers them:

    python3 race_deploy.py                     # static check, does not drive
    CHECK=0 VMAX=3.0 python3 race_deploy.py    # drive, capped at 3.0 m/s
    CHECK=0 VMAX=3.0 GYRO_SIGN=-1 ...          # if the gyro sign is inverted
    CTRL=wf_distill.py CHECK=0 VMAX=1.5 ...    # a different controller file

CTRL names a controller file next to this script (default wf_lite.py). If a
matching <stem>_params.json exists there it is loaded, otherwise PARAM_SPEC
defaults apply. VMAX caps the speed either through the controller's own
"v_max" knob or, for controllers that fixed it as a module constant (V_MAX),
by clamping that constant — the twin stays consistent either way because it
only ever integrates the commands actually issued.

Static check first (car may stay on the ground, it will not move): verify the
scan directions, that gyro_y shows small noise and goes NEGATIVE when the car
is rotated counter-clockwise, and that enc reads out when the car is pushed.
Then climb the speed ladder VMAX = 1.5 -> 2.0 -> 2.5 -> 3.0 -> 4.0.

Sensor chain on this car (rc.physics's /imu does not exist here):
gyro  <- /attitude        (sensor_msgs/Imu, BEST_EFFORT, ~112 Hz)
speed <- /encoder/speed   (std_msgs/Float32, median-filtered + sanity-gated)

See car/README.md for the telemetry field reference and fault table.
"""

import importlib.util

import numpy as np
import os
import sys

try:                                   # car: racecar_core is already on PYTHONPATH
    import racecar_core
except ImportError:                    # laptop/sim: walk up to racecar-student/
    from pathlib import Path
    for _root in Path(__file__).resolve().parents:
        if (_root / "library").is_dir():
            sys.path.insert(0, str(_root / "library"))
            if (_root / "sim2d").is_dir():
                sys.path.insert(0, str(_root / "sim2d"))
            break
    import racecar_core

rc = racecar_core.create_racecar()

# Settings come from the ENVIRONMENT so redeploying this file never clobbers
# your choices:   CHECK=0 VMAX=3.0 python3 race_deploy.py
CHECK_ONLY = os.environ.get("CHECK", "1") != "0"   # CHECK=0 -> drive
GYRO_SIGN = float(os.environ.get("GYRO_SIGN", "1.0"))
V_MAX_OVERRIDE = float(os.environ.get("VMAX", "1.5"))   # m/s cap

# ------------------------------------------------------------ load controller
import json

_here = os.path.dirname(os.path.abspath(__file__))
_ctrl_file = os.environ.get("CTRL", "wf_lite.py")
_ctrl_path = os.path.join(_here, _ctrl_file)
_stem = os.path.splitext(os.path.basename(_ctrl_file))[0]
_spec = importlib.util.spec_from_file_location(_stem, _ctrl_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_params = {k: v["default"] for k, v in _mod.PARAM_SPEC.items()}
_pfile = os.path.join(_here, os.environ.get("PARAMS", _stem + "_params.json"))
if os.path.exists(_pfile):
    _tuned = json.load(open(_pfile))
    _params.update({k: v for k, v in _tuned.items() if k in _params})
    print(f"race_deploy: loaded tuned params from {os.path.basename(_pfile)}")

if V_MAX_OVERRIDE is not None:
    if "v_max" in _params:
        _params["v_max"] = float(V_MAX_OVERRIDE)
    elif hasattr(_mod, "V_MAX"):
        # controllers like wf_distill fix v_max as a module constant; clamp it
        # (never raise it) so the speed ladder still works
        _mod.V_MAX = min(float(V_MAX_OVERRIDE), float(_mod.V_MAX))
ctrl = _mod.create_controller(_params, seed=0)

# ------------------------------------------------------------ sensor ADAPTER
# Subscriptions piggyback on the racecar's OWN (already-spinning) node.
# This car's live chain (verified): RealSense IMU -> imu_preprocessor
# (body frame, bias removed, yaw scale-corrected) -> /attitude @112 Hz;
# localizer fuses /encoder/speed. rc.physics's /imu topic does NOT exist
# here, and /velocity is TwistWithCovarianceStamped (crashes take_message
# in this process) -- so: gyro from /attitude, speed from /encoder/speed.
import time as _time

SPEED = {"hist": []}
GYRO = {"z": None, "t": 0.0}


def _enc_cb(m):
    h = SPEED["hist"]
    h.append(float(m.data))
    if len(h) > 5:
        del h[0]
try:
    from rclpy.qos import qos_profile_sensor_data   # BEST_EFFORT: matches the
    from sensor_msgs.msg import Imu                 # sensor publishers' QoS
    from std_msgs.msg import Float32
    rc.physics.node.create_subscription(
        Float32, "/encoder/speed", _enc_cb, qos_profile_sensor_data)
    # re-provisioned car 2026-08-01: /attitude is gone, raw IMU publishes on
    # /imu/primary (same sensor_msgs/Imu, REP-103; bias NOT pre-removed, the
    # controller's own bias tracker handles that). GYRO_TOPIC env overrides.
    _gt = os.environ.get("GYRO_TOPIC", "/imu/primary")
    rc.physics.node.create_subscription(
        Imu, _gt,
        lambda m: GYRO.update(z=float(m.angular_velocity.z),
                              t=_time.monotonic()), qos_profile_sensor_data)
except Exception:
    pass


def make_imu():
    g = None
    if GYRO["z"] is not None and _time.monotonic() - GYRO["t"] < 0.5:
        # /attitude follows REP-103: angular_velocity.z = +yaw rate (CCW+).
        # The controller expects the sim convention gyro[1] = -yaw_ccw(+bias),
        # hence the negation. Upstream already removed the bias; the
        # controller's own bias tracker just converges to ~0 (harmless).
        g = [0.0, GYRO_SIGN * (-GYRO["z"]), 0.0]
    imu = {"gyro": g, "accel": None}
    # sensorless-ESC encoder: garbage spikes and stuck zeros at low RPM.
    # median-of-5 + sanity window; outside it the controller's model twin
    # is more trustworthy than the sensor, so omit the key entirely.
    h = SPEED["hist"]
    if len(h) >= 3:
        vm = sorted(h)[len(h) // 2]
        if 0.30 < vm < 4.4:
            imu["speed"] = vm
    return imu


_t_print = [0.0]
_frames = [0]


def start():
    rc.drive.stop()
    # spd_scale mirrors the controller's internal model (wf_lite SPD_SCALE):
    # the command stays in the driver's high band, the ESC gets u * spd_scale.
    rc.drive.set_max_speed(float(_params.get("spd_scale", 1.0)))
    if hasattr(ctrl, "reset"):
        ctrl.reset()
    mode = "STATIC CHECK (no driving)" if CHECK_ONLY else \
        f"RACE  v_max={_params.get('v_max', getattr(_mod, 'V_MAX', '?'))} m/s"
    print(f"race_deploy: {_mod.ALGO_FAMILY}  [{mode}]  "
          f"(CHECK={'1' if CHECK_ONLY else '0'} VMAX={V_MAX_OVERRIDE} "
          f"GYRO_SIGN={GYRO_SIGN:+.0f})")


_RS = {}


def to720(scan):
    """Resample any full-revolution scan to the controller's 720-bin contract
    (real RPLidar with angle_compensate emits ~1080/rev; sim emits 720)."""
    if scan is None:
        return None
    n = len(scan)
    if n == 720:
        return scan
    idx = _RS.get(n)
    if idx is None:
        idx = (np.round(np.arange(720) * (n / 720.0)).astype(np.int64)) % n
        _RS[n] = idx
    return np.asarray(scan, dtype=np.float32)[idx]


def update():
    dt = rc.get_delta_time()
    scan = to720(rc.lidar.get_samples())
    imu = make_imu()

    if CHECK_ONLY:
        rc.drive.stop()
        _t_print[0] += dt
        _frames[0] += 1
        if _t_print[0] >= 1.0:
            fps = _frames[0] / _t_print[0]
            _t_print[0] = 0.0
            _frames[0] = 0
            g = imu.get("gyro")
            gtxt = "%+.2f" % g[1] if g else "N/A"
            if scan is None:
                print(f"[{fps:4.0f}fps] scan=None (lidar not publishing?) | "
                      f"gyro_y={gtxt} | enc={imu.get('speed', 'N/A')}")
            elif len(scan) != 720:
                print(f"[{fps:4.0f}fps] scan len={len(scan)} (expected 720!) | "
                      f"gyro_y={gtxt} | enc={imu.get('speed', 'N/A')}")
            else:
                nz = float((np.asarray(scan) > 0).mean()) * 100
                print(f"[{fps:4.0f}fps] front={scan[0]:6.0f} right={scan[180]:6.0f} "
                      f"back={scan[360]:6.0f} left={scan[540]:6.0f}cm "
                      f"valid={nz:3.0f}% | gyro_y={gtxt} | "
                      f"enc={imu.get('speed', 'N/A')}")
        return

    if scan is None or len(scan) != 720:
        rc.drive.stop()
        return
    speed, angle = ctrl.update(scan, dt, imu)
    rc.drive.set_speed_angle(max(min(speed, 1.0), -1.0),
                             max(min(angle, 1.0), -1.0))
    _t_print[0] += dt
    if _t_print[0] >= 1.0:
        _t_print[0] = 0.0
        # controller-agnostic telemetry: r180z carries .ms and a dbg dict,
        # wf_lite carries .v (m/s) and .blind — read all of it defensively
        d = getattr(ctrl, "dbg", {})
        v_twin = getattr(ctrl, "v", 0.0) * getattr(ctrl, "ms", 1.0)
        blind = getattr(ctrl, "blind", None)
        extra = f" blind={blind:4.2f}m" if isinstance(blind, float) else (
            f" free={d.get('free', 0):4.2f} k1={d.get('k1', 0):+5.2f}" if d else "")
        # enc: trusted value; gated garbage shows as raw!x.xx; dead topic as N/A
        if "speed" in imu:
            etxt = f"{imu['speed']:.2f}"
        elif SPEED["hist"]:
            h = sorted(SPEED["hist"])
            etxt = f"raw!{h[len(h)//2]:.2f}"
        else:
            etxt = "N/A"
        print(f"v={v_twin:4.2f}{extra} enc={etxt} "
              f"cmd=({speed:+.2f},{angle:+.2f})")


rc.set_start_update(start, update)
rc.go()
