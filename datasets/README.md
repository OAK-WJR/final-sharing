# Datasets — RACECAR sign detection & car (conga line) detection

All annotations are YOLO format (`class x_center y_center width height`,
normalized 0-1), one `.txt` per image, empty file = negative image.
Each folder has its own `data.yaml`.

## racecar-real/ — 318 real photos, 1 class (`racecar`)

Real photos of a RACECAR Neo on classroom floors, extracted from handheld
videos at 3 fps (7 videos, several rooms/lighting conditions). Includes
partial views, motion blur, cars with covers/decorations and lit LED strips,
plus 13 true-negative frames. Distances range from a few meters down to
close-ups.

Annotation provenance: boxes were produced by a YOLO26n model trained on
synthetic-only data (`ML_models/racecar_v1.pt`, mAP50 0.898 on held-out real
frames), then every frame was manually reviewed; 10 frames with missed or
uncertain annotations were removed rather than kept. Treat as high-quality
pseudo-labels, not hand-drawn ground truth.

## racecar-synthetic/ — 1000 synthetic images, 1 class (`racecar`)

Random subset of a domain-randomized synthetic set: the car was 3D-scanned
(GLB), rendered from randomized viewpoints/lighting (elevation biased to a
car-mounted camera height), pasted onto varied backgrounds with random
sticker/LED-glow/color augmentations, then degraded (motion blur, noise,
compression). ~8% negatives, up to 3 cars per image. 640x480.

## signs-synthetic/ — 1300 synthetic images, 9 classes

The validation split of a printed-sign synthetic dataset. Classes:

```
0 DO_NOT_ENTER      3 GO_AROUND        6 ONE_WAY_(right)
1 DO_NOT_GO_AROUND  4 NOT_YIELD        7 STOP
2 FAKE_STOP         5 ONE_WAY_(left)   8 YIELD
```

Sign artwork was composited with print degradation, perspective warp,
occlusion, and camera degradation onto varied backgrounds. Fully synthetic —
no real photos of the printed signs were available. A model trained on the
full 13k-image version of this set reached mAP50 0.987 (synthetic val) and
worked on-car on real printed signs, so the domain gap is small but nonzero.

## Caveats for use as a student validation set

- `racecar-real` is the only real-image folder; prefer it for scoring.
- The synthetic folders are best used as extra training data or as a
  sanity-check benchmark, since models trained on similar synthetic
  pipelines will score optimistically on them.
- ONE_WAY left/right are mirror classes: do NOT use horizontal-flip
  augmentation when training on signs, and inverted signs are labeled by
  arrow direction, not text orientation.
