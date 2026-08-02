#!/usr/bin/env python3
"""Slice a 3D field scan (binary PLY point cloud) into 2D top-down maps.

Outputs into final-sharing/2d_map/:
  occupancy_map.png   black = obstacle, white = floor, gray = unknown
  topdown_color.png   RGB orthographic top-down view
  map.pgm + map.yaml  ROS map_server format
  slice_preview.png   several candidate height bands side by side

Usage:
  uv run --with numpy,pillow slice_2d_map.py <field.ply> [--res 0.01]
      [--slice-min 0.05] [--slice-max 0.50]
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def line_structs(length):
    h = np.ones((1, length), bool)
    v = np.ones((length, 1), bool)
    d = np.eye(length, dtype=bool)
    return [h, v, d, d[::-1]]


def read_ply(path):
    with open(path, "rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            chunk = f.readline()
            if not chunk:
                sys.exit("bad PLY: no end_header")
            header += chunk
        n_vertex = 0
        for line in header.decode("ascii", "replace").splitlines():
            if line.startswith("element vertex"):
                n_vertex = int(line.split()[-1])
        dtype = np.dtype(
            [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
             ("r", "u1"), ("g", "u1"), ("b", "u1")]
        )
        data = np.fromfile(f, dtype=dtype, count=n_vertex)
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1)
    rgb = np.stack([data["r"], data["g"], data["b"]], axis=1)
    keep = np.isfinite(xyz).all(axis=1)
    return xyz[keep], rgb[keep]


def find_up_axis(xyz):
    """Gravity axis = the axis whose histogram has the sharpest peak (the floor
    is the biggest planar cluster in the scan)."""
    best_axis, best_score, best_floor = 2, -1.0, 0.0
    for axis in range(3):
        v = xyz[:, axis]
        lo, hi = np.percentile(v, [0.5, 99.5])
        hist, edges = np.histogram(v, bins=200, range=(lo, hi))
        score = hist.max() / max(hist.mean(), 1)
        if score > best_score:
            i = int(hist.argmax())
            best_axis, best_score = axis, score
            best_floor = 0.5 * (edges[i] + edges[i + 1])
    return best_axis, best_floor


def local_ground(u, v, z, u0, v0, u1, v1, cell=0.25):
    """Per-cell ground height (robust local floor), median-filtered to reject
    outliers, holes filled by dilation. Handles scan drift / sloped floors that
    break a single global floor plane. Returns height-above-ground per point."""
    nc = int(np.ceil((u1 - u0) / cell)) + 1
    nr = int(np.ceil((v1 - v0) / cell)) + 1
    ci = np.clip(((u - u0) / cell).astype(np.int64), 0, nc - 1)
    cj = np.clip(((v - v0) / cell).astype(np.int64), 0, nr - 1)
    flat = cj * nc + ci
    ground = np.full(nr * nc, np.inf, np.float32)
    np.minimum.at(ground, flat, z.astype(np.float32))
    g = ground.reshape(nr, nc)
    g[~np.isfinite(g)] = np.nan

    # 3x3 median filter; clamp cells deviating > 15 cm from neighborhood median
    pad = np.pad(g, 1, constant_values=np.nan)
    stack = np.stack([pad[dr:dr + nr, dc:dc + nc]
                      for dr in range(3) for dc in range(3)])
    med = np.nanmedian(stack, axis=0)
    bad = np.isfinite(g) & np.isfinite(med) & (np.abs(g - med) > 0.15)
    g[bad] = med[bad]

    # fill holes by repeated nan-aware 3x3 mean dilation
    for _ in range(6):
        if not np.isnan(g).any():
            break
        pad = np.pad(g, 1, constant_values=np.nan)
        stack = np.stack([pad[dr:dr + nr, dc:dc + nc]
                          for dr in range(3) for dc in range(3)])
        fill = np.nanmean(stack, axis=0)
        hole = np.isnan(g)
        g[hole] = fill[hole]

    return z - g.ravel()[flat]


def rasterize(u, v, grid_shape, u0, v0, res):
    """Map planar coords to integer pixel indices (row = flipped v)."""
    col = ((u - u0) / res).astype(np.int64)
    row = grid_shape[0] - 1 - ((v - v0) / res).astype(np.int64)
    ok = (col >= 0) & (col < grid_shape[1]) & (row >= 0) & (row < grid_shape[0])
    return row[ok], col[ok], ok


def dilate(m, it=1):
    H, W = m.shape
    for _ in range(it):
        p = np.pad(m, 1)
        out = np.zeros_like(m)
        for dr in range(3):
            for dc in range(3):
                out |= p[dr:dr + H, dc:dc + W]
        m = out
    return m


def erode(m, it=1):
    return ~dilate(~m, it)


def box_sum(m, k=3):
    """Sum of mask over a k x k window (k odd)."""
    H, W = m.shape
    r = k // 2
    p = np.pad(m.astype(np.uint16), r)
    s = np.zeros((H, W), np.uint16)
    for dr in range(k):
        for dc in range(k):
            s += p[dr:dr + H, dc:dc + W]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply")
    ap.add_argument("--res", type=float, default=0.01, help="m per pixel")
    ap.add_argument("--slice-min", type=float, default=0.05,
                    help="obstacle band lower bound above floor (m)")
    ap.add_argument("--slice-max", type=float, default=0.50,
                    help="obstacle band upper bound above floor (m)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "2d_map"))
    args = ap.parse_args()

    xyz, rgb = read_ply(args.ply)
    print(f"loaded {len(xyz):,} points")

    up, floor = find_up_axis(xyz)
    plane = [a for a in range(3) if a != up]
    h = xyz[:, up] - floor
    print(f"up axis = {'xyz'[up]}, floor at {'xyz'[up]}={floor:.3f} m")

    u_all, v_all = xyz[:, plane[0]], xyz[:, plane[1]]
    u0, u1 = np.percentile(u_all, [0.2, 99.8])
    v0, v1 = np.percentile(v_all, [0.2, 99.8])
    h = local_ground(u_all, v_all, xyz[:, up], u0, v0, u1, v1)
    print("using local ground estimation (25 cm cells)")
    W = int(np.ceil((u1 - u0) / args.res)) + 1
    H = int(np.ceil((v1 - v0) / args.res)) + 1
    print(f"grid {W} x {H} px @ {args.res*100:.0f} cm/px  "
          f"({(u1-u0):.1f} x {(v1-v0):.1f} m)")

    os.makedirs(args.out, exist_ok=True)

    # --- occupancy grid ---
    # per-pixel aggregates: scanned coverage, max height, in-band point count
    r, c, ok = rasterize(u_all, v_all, (H, W), u0, v0, args.res)
    scanned = np.zeros((H, W), bool)
    scanned[r, c] = True
    maxh = np.full((H, W), -np.inf, np.float32)
    np.maximum.at(maxh, (r, c), h[ok].astype(np.float32))

    band = (h >= args.slice_min) & (h <= args.slice_max)
    rb, cb, _ = rasterize(u_all[band], v_all[band], (H, W), u0, v0, args.res)
    cnt = np.zeros((H, W), np.uint16)
    np.add.at(cnt, (rb, cb), 1)

    # 2 cm aggregation pools sparse wall evidence (thin corridor walls)
    res2 = args.res * 2
    W2 = int(np.ceil((u1 - u0) / res2)) + 1
    H2 = int(np.ceil((v1 - v0) / res2)) + 1
    rb2, cb2, _ = rasterize(u_all[band], v_all[band], (H2, W2), u0, v0, res2)
    cnt2 = np.zeros((H2, W2), np.uint16)
    np.add.at(cnt2, (rb2, cb2), 1)
    r2, c2, ok2 = rasterize(u_all, v_all, (H2, W2), u0, v0, res2)
    maxh2 = np.full((H2, W2), -np.inf, np.float32)
    np.maximum.at(maxh2, (r2, c2), h[ok2].astype(np.float32))
    cand2 = (cnt2 >= 2) & (maxh2 >= 0.18)
    cand2_up = np.kron(cand2, np.ones((2, 2), bool))[:H, :W]

    # a wall pixel has vertical structure and linear neighborhood support;
    # isolated registration-noise pixels have neither
    cand = ((cnt >= 1) & (maxh >= 0.18)) | cand2_up
    support = box_sum(cand, 5)
    obst = cand & (support >= 4)

    # stitch dashed walls along their direction: closing with line elements
    # (walls are long lines; noise blobs are not collinear and stay isolated)
    stitched = np.zeros_like(obst)
    for s in line_structs(11):
        stitched |= ndimage.binary_closing(obst, structure=s)
    obst = ndimage.binary_closing(stitched, structure=disk(2))
    lab, n = ndimage.label(obst)
    sizes = np.bincount(lab.ravel())
    obst &= (sizes > 80)[lab]
    obst = ndimage.binary_dilation(obst, structure=disk(1))

    free = scanned & ~obst
    free = ndimage.binary_closing(free, structure=disk(3)) & ~obst

    # fill unknown holes fully enclosed by scanned area (occlusion shadows)
    unknown = ~(free | obst)
    lab_u, _ = ndimage.label(unknown)
    border = np.zeros_like(unknown)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    outside = np.unique(lab_u[border & unknown])
    hole = unknown & ~np.isin(lab_u, outside)
    free |= hole & ~obst

    occ = np.full((H, W), 205, np.uint8)         # unknown gray
    occ[free] = 254                              # free white
    occ[obst] = 0                                # occupied black
    Image.fromarray(occ).save(os.path.join(args.out, "occupancy_map.png"))
    print(f"obstacle pixels: {int(obst.sum()):,}  free pixels: {int(free.sum()):,}")

    # ROS map_server pair
    with open(os.path.join(args.out, "map.pgm"), "wb") as f:
        f.write(f"P5\n{W} {H}\n255\n".encode())
        f.write(occ.tobytes())
    with open(os.path.join(args.out, "map.yaml"), "w") as f:
        f.write(
            f"image: map.pgm\nresolution: {args.res}\n"
            f"origin: [{u0:.4f}, {v0:.4f}, 0.0]\n"
            "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n"
        )

    # --- height-layer composite: obstacle pixels colored by their height ---
    hbins = [(0.10, 0.30, (255, 160, 0)),    # low: orange
             (0.30, 0.60, (220, 40, 40)),    # mid: red
             (0.60, 1.00, (150, 40, 200)),   # tall: purple
             (1.00, 2.50, (30, 60, 220))]    # very tall: blue
    maxh_f = ndimage.grey_dilation(np.nan_to_num(maxh, neginf=-1), size=5)
    layers = np.full((H, W, 3), 205, np.uint8)
    layers[free] = 255
    for lo_, hi_, col in hbins:
        m = obst & (maxh_f >= lo_) & (maxh_f < hi_)
        layers[m] = col
    img = Image.fromarray(layers)
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=60)
    except TypeError:
        font = ImageFont.load_default()
    y0 = 40
    for lo_, hi_, col in hbins:
        d.rectangle([40, y0, 120, y0 + 60], fill=col)
        d.text((140, y0 + 5), f"{lo_:.1f} - {hi_:.1f} m", fill=(0, 0, 0), font=font)
        y0 += 90
    img.save(os.path.join(args.out, "height_layers.png"))

    # --- per-band raw slices, labeled comparison grid ---
    grid_bands = [(0.05, 0.15), (0.15, 0.30), (0.30, 0.60),
                  (0.60, 1.00), (1.00, 1.60), (1.60, 2.60)]
    tiles = []
    for lo_, hi_ in grid_bands:
        m = (h >= lo_) & (h < hi_)
        rb_, cb_, _ = rasterize(u_all[m], v_all[m], (H, W), u0, v0, args.res)
        sl = np.zeros((H, W), bool)
        sl[rb_, cb_] = True
        sl &= box_sum(sl, 5) >= 3                # light despeckle only
        tile = np.full((H, W), 255, np.uint8)
        tile[sl] = 0
        tile[~scanned & ~sl] = 230
        t = Image.fromarray(tile).reduce(3)
        dt = ImageDraw.Draw(t)
        dt.rectangle([0, 0, t.width, 70], fill=(200,))
        dt.text((20, 8), f"{lo_:.2f} - {hi_:.2f} m", fill=0, font=font)
        tiles.append(t)
    tw, th = tiles[0].size
    grid = Image.new("L", (tw * 3 + 20, th * 2 + 10), 255)
    for i, t in enumerate(tiles):
        grid.paste(t, ((i % 3) * (tw + 10), (i // 3) * (th + 5)))
    grid.save(os.path.join(args.out, "height_slices_grid.png"))

    # --- colored orthographic top-down (highest point wins per pixel) ---
    color = np.zeros((H, W, 3), np.uint8)
    zbuf = np.full((H, W), -np.inf, np.float32)
    order = np.argsort(h)                         # low first, high overwrites
    r, c, ok = rasterize(u_all[order], v_all[order], (H, W), u0, v0, args.res)
    sel = order[ok]
    color[r, c] = rgb[sel]
    zbuf[r, c] = h[sel]
    Image.fromarray(color).save(os.path.join(args.out, "topdown_color.png"))

    # --- preview of candidate slice bands ---
    bands = [(0.02, 0.15), (0.05, 0.30), (0.05, 0.50), (0.10, 1.00)]
    scale = max(1, max(H, W) // 600)
    tiles = []
    for lo, hi in bands:
        img = np.full((H, W), 255, np.uint8)
        m = (h >= lo) & (h <= hi)
        r, c, _ = rasterize(u_all[m], v_all[m], (H, W), u0, v0, args.res)
        img[r, c] = 0
        tiles.append(np.array(Image.fromarray(img).reduce(scale)))
    th, tw = tiles[0].shape
    strip = np.full((th + 20, (tw + 10) * len(tiles)), 255, np.uint8)
    for i, t in enumerate(tiles):
        strip[20:, i * (tw + 10):i * (tw + 10) + tw] = t
    prev = Image.fromarray(strip)
    prev.save(os.path.join(args.out, "slice_preview.png"))
    print("bands in preview (left to right):",
          ", ".join(f"[{lo}-{hi} m]" for lo, hi in bands))
    print(f"outputs in {args.out}")


if __name__ == "__main__":
    main()
