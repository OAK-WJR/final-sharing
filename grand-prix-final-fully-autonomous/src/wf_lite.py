"""wf_lite — lidar-only racer: model twin + scan carry-forward + ets steering.

Commands land 0.10-0.18 s late and the lidar refreshes at 7 Hz, so what the car
sees and what its next command acts on are different worlds. Three ideas fix it:

  1. Simulate your own car. A replica (delay buffers -> measured maps -> lags ->
     bicycle with understeer) always knows the pose the next command executes in.
  2. Carry the old scan forward. A repeated frame is byte-identical, so the last
     real cloud is warped into the predicted pose; steering never sees a stale
     frame.
  3. Brake for the arc you chose. v = min(stop-within-measured-free-length,
     lateral-accel cap), decayed once blind dead-reckoning exceeds trust.

Ablation-trimmed from wf_distill: wedge recovery, brake feedforward, the blind
governor and the anti-stall creep floor each cost finishes when removed.

Layout:  wf/plant.py   vehicle model      wf/scan.py   lidar geometry
         wf/tuning.py  constants + spec
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wf import plant, scan
from wf.tuning import (BIAS0, BLIND_STOP, BLIND_TRUST, BRAKE_FF, BRAKE_KAP,
                       BRAKE_W, CREEP_D0, CREEP_SPAN, GYRO_KB, GYRO_W, NOSE,
                       OPEN_DAMP, OPEN_M, PARAM_SPEC, RECOV_COOL, RECOV_WIN,
                       REP_LP, REV_T, REV_U, SLEW, T_REACT, TAU_PRED, U_MIN,
                       U_SLEW, V_BLEND, V_CREEP, V_MAX, V_MEAS_HI, V_MEAS_LO,
                       WEDGE_DIST, WEDGE_HOLD)

N = scan.N
RMAX = scan.RMAX
ALGO_FAMILY = "wf_lite"

__all__ = ["PARAM_SPEC", "ALGO_FAMILY", "create_controller", "WfLite", "V_MAX"]


def create_controller(params, seed):
    return WfLite(params)


class WfLite:
    def __init__(self, p):
        self.p = p
        plant.SPD_SCALE = float(p.get("spd_scale", 1.0))
        self.reset()

    def reset(self):
        self.uh = [0.0]
        self.ah = [0.0]
        self.v = 0.0
        self.delta = 0.0
        self.x = self.y = self.th = 0.0
        self.pose_hist = [(0.0, 0.0, 0.0)] * 16
        self.bias = BIAS0
        self.prev_raw = None
        self.scan_m = None
        self.cap_pose = (0.0, 0.0, 0.0)
        self.blind = 0.0
        self.prev_angle = 0.0
        self.rep_f = 0.0
        self.prev_u = 0.0
        self.wedge_t = 0.0
        self.rev_t = 0.0
        self.rev_dir = 1.0
        self.cool = 0.0

    @staticmethod
    def _hist(h, back):
        i = len(h) - 1 - back
        return h[i] if i >= 0 else h[0]

    def _trim(self):
        if len(self.uh) > 40:
            del self.uh[:20]
            del self.ah[:20]

    # ------------------------------------------------------------- main tick
    def update(self, scan_cm, dt, imu):
        p = self.p
        if dt <= 0.0:
            dt = 1.0 / 60.0
        ks = max(1, int(round(plant.SPEED_DELAY / dt)))
        ka = max(1, int(round(plant.STEER_DELAY / dt)))
        kl = max(1, int(round(plant.LIDAR_LATENCY / dt)))

        self._advance_twin(dt, ks, ka, imu)

        rr = self._carry_forward(scan_cm, dt, kl)
        if rr is None:                       # no usable scan: creep straight
            self.uh.append(0.25)
            self.ah.append(0.0)
            self._trim()
            return 0.25, 0.0
        rr, qx, qy, v_f = rr

        angle = self._steer(p, rr, qx, qy, dt)
        u = self._throttle(p, angle, qx, qy, rr, v_f)

        angle, u = self._wedge(float(np.clip(angle, -1, 1)),
                               float(np.clip(u, -1, 1)), dt)
        self.uh.append(u)
        self.ah.append(angle)
        self._trim()
        self.prev_angle, self.prev_u = angle, u
        return u, angle

    # ----------------------------------------------------------- plant twin
    def _advance_twin(self, dt, ks, ka, imu):
        """Step the replica to now, then let live sensors correct it.

        A measured omega bypasses the steering/wheelbase/understeer model and a
        measured v bypasses the throttle map — the two model-error axes the
        vehicle-perturbation sweep found DNFs on. Absent or insane readings
        fall through to the pure model, so lidar stays the only required sensor.
        """
        self.v, self.delta, self.x, self.y, self.th, w_mod = plant.step(
            self.v, self.delta, self.x, self.y, self.th,
            self._hist(self.uh, ks), self._hist(self.ah, ka), dt)

        try:
            gy = float(imu["gyro"][1])       # sim convention: -yaw_ccw + bias
            self.bias += GYRO_KB * (gy - (self.bias - w_mod))
            w_fus = (1.0 - GYRO_W) * w_mod + GYRO_W * (self.bias - gy)
            self.th += (w_fus - w_mod) * dt
        except Exception:
            pass
        try:
            vm = float(imu["speed"])
            if V_MEAS_LO < vm < V_MEAS_HI:
                self.v += V_BLEND * (vm - self.v)
        except Exception:
            pass

        self.pose_hist.append((self.x, self.y, self.th))
        if len(self.pose_hist) > 16:
            del self.pose_hist[0]

    # --------------------------------------------------------- carry forward
    def _carry_forward(self, scan_cm, dt, kl):
        """Detect a repeated frame, then warp the last real cloud into the pose
        the next command will execute in. Returns (rr, qx, qy, v_pred) or None.
        """
        raw = np.asarray(scan_cm, dtype=np.float64)
        fresh = self.prev_raw is None or not np.array_equal(raw, self.prev_raw)
        if fresh:
            self.prev_raw = raw.copy()
            m = raw * 0.01
            self.scan_m = np.where((m > 0.05) & (m < RMAX + 2.0), m, 0.0)
            self.cap_pose = self.pose_hist[max(0, len(self.pose_hist) - 1 - kl)]
            self.blind = 0.0
        else:
            self.blind += abs(self.v) * dt

        if self.scan_m is None or not np.any(self.scan_m > 0.05):
            return None

        ks = max(1, int(round(plant.SPEED_DELAY / dt)))
        ka = max(1, int(round(plant.STEER_DELAY / dt)))
        v_f, d_f, x_f, y_f, th_f = self.v, self.delta, self.x, self.y, self.th
        for j in range(int(TAU_PRED / dt)):
            v_f, d_f, x_f, y_f, th_f, _ = plant.step(
                v_f, d_f, x_f, y_f, th_f,
                self._hist(self.uh, max(0, ks - 1 - j)),
                self._hist(self.ah, max(0, ka - 1 - j)), dt)

        cx, cy, cth = self.cap_pose
        ct0, st0 = math.cos(cth), math.sin(cth)
        gx, gy = x_f - cx, y_f - cy
        rr, qx, qy = scan.warp(self.scan_m,
                               ct0 * gx + st0 * gy,
                               -st0 * gx + ct0 * gy,
                               th_f - cth)
        return rr, qx, qy, v_f

    # ------------------------------------------------------- ets steering
    def _steer(self, p, rr, qx, qy, dt):
        """Windowed openness per side, blended by the openness difference."""
        win_deg = float(p["window_deg"])
        if self.cool > 0.0:
            self.cool -= dt
            win_deg = min(win_deg, RECOV_WIN)   # no U-turn hunting after a reverse

        w = max(int(float(p["ray_width_deg"]) * N / 360), 2)
        vals = np.where(rr > 0.05, rr, 0.0)
        mask = (vals > 0).astype(float)
        k = np.ones(w)
        s = np.convolve(np.r_[vals, vals[:w]], k, "valid")[:N]
        c = np.convolve(np.r_[mask, mask[:w]], k, "valid")[:N]
        avg = np.roll(np.where(c > 0, s / np.maximum(c, 1), 0.0), w // 2)

        win = int(win_deg * N / 360)
        right = avg[:win]
        left = avg[N - win:]
        ri = int(right.argmax())
        li = int(left.argmax())
        shift = float(np.clip((float(right[ri]) - float(left[li]))
                              / float(p["weight_range_m"]), -0.5, 0.5))
        blend = (ri * 360.0 / N * (0.5 + shift)
                 - (win - li) * 360.0 / N * (0.5 - shift)) / 2
        angle = float(np.clip(blend * float(p["kp"]), -1.0, 1.0))

        angle = self._wall_repel(p, angle, qx, qy)

        _fr = rr[scan.FRONT_M]
        _fr = _fr[_fr > 0.05]
        if _fr.size and float(np.median(_fr)) > OPEN_M:
            angle *= OPEN_DAMP                  # open floor: go find a wall

        return self.prev_angle + max(-SLEW, min(SLEW, angle - self.prev_angle))

    def _wall_repel(self, p, angle, qx, qy):
        """1/d centring pull off the deep outside sweeps openness-following
        loves in wide sections. wall_rep=0 disables it entirely.
        """
        wrep = float(p.get("wall_rep", 0.0))
        if wrep <= 0.0:
            return angle
        beside = (qx > -0.30) & (qx < 0.85)
        by = qy[beside]
        pl = by[by > 0.0]
        pr = by[by < 0.0]
        dl = float(pl.min()) if pl.size else 2.0
        dr = float(-pr.max()) if pr.size else 2.0
        hwd0 = float(p["half_width"])
        # student convention: angle + = RIGHT, so a close LEFT wall pushes +
        rep = wrep * hwd0 * (1.0 / max(dl, 0.5 * hwd0)
                             - 1.0 / max(dr, 0.5 * hwd0))
        # width gate: centring only helps in wide sections. In a narrow corridor
        # the nearest wall at turn-in is the apex wall, so ungated repulsion
        # fights the ets turn — late, twisty entries.
        gate = min(max((dl + dr - 4.0 * hwd0) / (2.0 * hwd0), 0.0), 1.0)
        # low-pass: dl/dr are box minima, single-point noise otherwise lands
        # straight on the servo
        self.rep_f += REP_LP * (gate * rep - self.rep_f)
        return float(np.clip(angle + self.rep_f, -1.0, 1.0))

    # -------------------------------------------------------- physics speed
    def _throttle(self, p, angle, qx, qy, rr, v_f):
        kappa = math.tan(plant.steer_map(-angle)) / (
            plant.L * (1.0 + plant.K_US * v_f * v_f))
        d_free = max(scan.swath(qx, qy, BRAKE_KAP * kappa,
                                BRAKE_W * float(p["half_width"])) - NOSE, 0.0)
        ab = float(p["a_brake"])
        v_brake = -ab * T_REACT + math.sqrt((ab * T_REACT) ** 2 + 2.0 * ab * d_free)
        v_lat = math.sqrt(float(p["a_lat"]) / max(abs(kappa), 1e-3))
        v_tgt = min(V_MAX, v_brake, v_lat)

        # anti-stall: at v = 0 the bicycle model cannot yaw at all
        cone = rr[scan.CONE_M]
        cone = cone[cone > 0.05]
        d_cone = float(np.percentile(cone, 25)) if cone.size else RMAX
        floor = V_CREEP * min(max((d_cone - CREEP_D0) / CREEP_SPAN, 0.0), 1.0)
        v_tgt = max(v_tgt, min(floor, V_MAX))

        if self.blind > BLIND_TRUST:
            # reference the ACHIEVABLE top speed, not V_MAX: under spd_scale a
            # 4.0-based cap sat above the whole reachable band and never bound,
            # and blind corners crashed.
            v_tgt = min(v_tgt,
                        max(0.9, plant.speed_map(1.0) * BLIND_TRUST / self.blind))
        if self.blind > BLIND_STOP:             # sensor is dead, not stale
            v_tgt = 0.0

        over = self.v - v_tgt
        u = (plant.speed_inv(v_tgt - BRAKE_FF * over) if over > 0.35
             else plant.speed_inv(v_tgt))
        u = max(U_MIN, min(1.0, u))
        u = self._floor(p, u, d_free, v_f)
        return self.prev_u + max(-U_SLEW, min(U_SLEW, u - self.prev_u))

    def _floor(self, p, u, d_free, v_f):
        """Race-day throttle floor. Deliberately overrides the brake law —
        steering must carry the corners — but yields to the blind-stop
        governor, the wedge cooldown, and an arc that says a wall arrives
        within ~0.8 s (capped at 0.45 so the car keeps rolling and re-floors
        the moment the arc clears).
        """
        uf = float(p.get("u_floor", 0.0))
        if uf <= 0.0 or self.blind > BLIND_STOP or self.cool > 0.0:
            return u
        u_lo = 0.45 if (d_free < v_f * 0.8 or self.blind > BLIND_TRUST) else uf
        return max(u, u_lo)

    # ------------------------------------------------------ wedge recovery
    def _wedge(self, angle, u, dt):
        if self.rev_t > 0.0:
            self.rev_t -= dt
            return self.rev_dir, REV_U
        fr = self.scan_m[scan.FRONT_M]
        fr = fr[fr > 0.05]
        front = float(np.percentile(fr, 20)) if fr.size >= 4 else RMAX
        if front < WEDGE_DIST:
            self.wedge_t += dt
        elif front > 1.6 * WEDGE_DIST:
            self.wedge_t = 0.0
        if self.wedge_t > WEDGE_HOLD:
            self.wedge_t = 0.0
            self.rev_t = REV_T
            self.cool = RECOV_COOL
            self.rev_dir = -1.0 if self.prev_angle >= 0.0 else 1.0
            return self.rev_dir, REV_U
        return angle, u
