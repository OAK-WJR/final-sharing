# Grand Prix — final algorithm

```
run.sh                       launcher
src/race_deploy.py           entry point: sensors -> controller -> drive
src/wf_lite.py               controller: the tick, steering, speed law, wedge
src/wf/plant.py              vehicle model (measured maps, lags, bicycle step)
src/wf/scan.py               lidar geometry, scan warping, swept-arc length
src/wf/tuning.py             constants + PARAM_SPEC
config/wf_lite_finalloop_params.json    raced parameters
```

## Run

```bash
bash run.sh              # race, VMAX=4.0
VMAX=2.0 bash run.sh     # speed ladder on a new track
CHECK=1 bash run.sh      # static sensor check, never drives
bash run.sh -s           # simulator
```

START launches, BACK reclaims manual.

Equivalent long form, if you'd rather type it out on the car:

```bash
cd src
PARAMS=../config/wf_lite_finalloop_params.json CTRL=wf_lite.py \
    CHECK=0 VMAX=4.0 python3 race_deploy.py
```

## How it works

The car's commands land 0.10-0.12 s late and the lidar only refreshes at 7 Hz,
so what the car sees and what its next command acts on are different worlds.
Four layers fix that:

**Plant twin.** A replica of the car runs in software: delay buffers, measured
throttle/steer maps, first-order lags, bicycle model with understeer. It always
knows the pose the next command will execute in. Gyro and encoder correct it
when they're alive; lidar is the only required sensor.

**Scan carry-forward.** A repeated lidar frame is byte-identical, so it is
detected and the last real point cloud is rigid-body warped into the predicted
pose. Steering never looks at a stale frame. Blind travel over 1.8 m decays
speed, over 3.0 m stops the car — that's a dead sensor, not a stale one.

**Steering.** Windowed openness per bearing; take the most open bearing on each
side, blend them by their openness difference, multiply by `kp`. Plus a 1/d
wall repulsion that keeps the car off the outside sweeps.

**Speed.** Sweep the planned arc through the point cloud to get a free length,
then take the smaller of "can stop inside it" (`a_brake`) and "lateral accel
stays under `a_lat`". An anti-stall floor keeps ~1.4 m/s while the front cone is
open, because a stopped bicycle model cannot yaw at all.

Wedge recovery reverses for 0.34 s if the front stays blocked for 0.5 s, then
caps the steering window for 1.3 s so it doesn't U-turn out of the recovery.

## Parameters

```
window_deg      85.93   steering search window each side
ray_width_deg   28.65   smoothing width of the openness average
weight_range_m   2.99   how much a left/right difference steers (bigger = straighter line)
kp              0.032   openness blend -> steering
half_width      0.343   braking swath half-width
a_brake         12.0    how late to brake
a_lat           16.0    how fast to corner
wall_rep         0.4    centring pull
```

Three constants in `wf_lite.py` matter as much as the JSON:

- `SLEW = 0.20` — steering rate limit. It caps turn-in *and* un-turn, so too
  low reads as "turns too much" (it's really straightening out too slowly).
- `OPEN_M = 3.0`, `OPEN_DAMP = 0.35` — open floor has no openness gradient, so
  every heading looks equal and the car circles at the creep speed. Damping
  toward straight makes it cross and re-acquire a wall. Real corners have a
  near wall ahead, so they're unaffected.

Tuning map: slow turn-in → `SLEW`; wrong steering amount → `kp`; line too wavy →
`weight_range_m`; hits walls in corners → `a_lat`; slow on straights →
`a_brake`; hugs walls → `wall_rep`; circles in place → `OPEN_DAMP`.

`spd_scale` and `u_floor` exist in `PARAM_SPEC` but are absent from the raced
JSON, so they default to full speed and no throttle floor.

The camera is never touched: `racecar_core` subscribes lazily and then decodes
every frame at 30 Hz inside the control loop's executor, and the twin steers
badly on a jittery loop.

## Deploying to the car

The car's directory is flat, so copy the two source files and the JSON into it:

```bash
scp -o ProxyJump=jetson@10.1.14.5 src/*.py config/*.json \
    racecar@10.42.0.1:~/jupyter_ws/team1/labs/racing/
```

Then on the car the long form applies with `PARAMS=wf_lite_finalloop_params.json`
(same directory, no `../config`).
