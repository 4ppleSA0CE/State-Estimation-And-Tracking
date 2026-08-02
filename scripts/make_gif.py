#!/usr/bin/env python3
"""Render a recorded pipeline run to an animated GIF for the README.

Generated from the recorded npz, not screen-captured, so it is reproducible: rerun the script and
you get the same GIF. Left panel is the BEV world view (ego truth vs estimate, target truth, live
tracks); right panel is ego position error against time, which is what makes the coupling legible.

Ego estimate and truth are paired by TIMESTAMP, never by index: the filter publishes the state for
t_k only when the IMU sample at t_{k+1} arrives, so the estimate array is one sample shorter and an
index pairing silently shifts the whole trajectory.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

TRACK_COLORS = plt.get_cmap("tab10")


def pair_by_stamp(est_t_ns: np.ndarray, truth_t_ns: np.ndarray) -> np.ndarray:
    """Index into truth for each estimate stamp. Raises if any stamp has no exact match."""
    idx = np.searchsorted(truth_t_ns, est_t_ns)
    idx = np.clip(idx, 0, len(truth_t_ns) - 1)
    if not np.array_equal(truth_t_ns[idx], est_t_ns):
        bad = int((truth_t_ns[idx] != est_t_ns).sum())
        raise ValueError(f"{bad} estimate stamps have no exact truth match")
    return idx


def render(npz_path: Path, out_path: Path, fps: int = 10, stride: int = 1) -> Path:
    d = np.load(npz_path, allow_pickle=True)
    est, truth = d["ego_est"], d["ego_truth"]
    tidx = pair_by_stamp(d["ego_est_t_ns"], d["t_ns"])
    truth_at_est = truth[tidx]
    err = np.linalg.norm(est - truth_at_est, axis=1)
    est_t = (d["ego_est_t_ns"] - d["t_ns"][0]) / 1e9

    frame_t = d["frame_t"]
    track_pos, track_ids, track_count = d["track_pos_enu"], d["track_ids"], d["track_count"]
    tgt, tgt_vis = d["target_truth_enu"], d["target_visible"]

    frames = range(0, len(frame_t), stride)
    fig, (ax, axe) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.45, 1]})

    # Frame on the ego path plus only the targets that are ever visible. Including invisible
    # targets pads the view with dead space they never occupy on screen.
    pad = 12.0
    xs = np.concatenate([truth[:, 0], tgt[..., 0][tgt_vis]])
    ys = np.concatenate([truth[:, 1], tgt[..., 1][tgt_vis]])
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.set_ylim(ys.min() - pad, ys.max() + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]")
    ax.set_title("World view (ENU)")
    ax.grid(alpha=0.2)

    axe.set_xlim(0, est_t[-1]); axe.set_ylim(0, max(err.max() * 1.15, 0.1))
    axe.set_xlabel("time [s]"); axe.set_ylabel("ego position error [m]")
    axe.set_title("Localization error")
    axe.grid(alpha=0.2)

    # Truth is drawn wider and underneath so it stays visible as a halo; at sub-metre error the
    # two trails overlap almost exactly, and equal widths would hide truth completely.
    truth_line, = ax.plot([], [], "-", color="#e8a33d", lw=3.6, alpha=0.9, label="ego truth",
                          zorder=2, solid_capstyle="round")
    est_line, = ax.plot([], [], "-", color="#3ec6e0", lw=1.5, label="ego ESKF", zorder=3)
    ego_dot, = ax.plot([], [], "o", color="#3ec6e0", ms=7)
    tgt_dots, = ax.plot([], [], "s", color="#888888", ms=6, mfc="none", label="target truth")
    trk_scat = ax.scatter([], [], s=44, marker="x", linewidths=1.8, label="tracks")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    err_line, = axe.plot([], [], "-", color="#d1495b", lw=1.6)
    err_dot, = axe.plot([], [], "o", color="#d1495b", ms=5)
    stamp = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=9,
                    family="monospace")

    def update(k):
        t = frame_t[k]
        upto = np.searchsorted(est_t, t) + 1
        truth_line.set_data(truth_at_est[:upto, 0], truth_at_est[:upto, 1])
        est_line.set_data(est[:upto, 0], est[:upto, 1])
        ego_dot.set_data(est[upto - 1:upto, 0], est[upto - 1:upto, 1])

        vis = tgt_vis[k]
        tgt_dots.set_data(tgt[k, vis, 0], tgt[k, vis, 1])

        n = int(track_count[k])
        if n:
            trk_scat.set_offsets(track_pos[k, :n, :2])
            trk_scat.set_color([TRACK_COLORS(int(i) % 10) for i in track_ids[k, :n]])
        else:
            trk_scat.set_offsets(np.empty((0, 2)))

        err_line.set_data(est_t[:upto], err[:upto])
        err_dot.set_data(est_t[upto - 1:upto], err[upto - 1:upto])
        stamp.set_text(f"t = {t:5.1f} s\nerr = {err[upto - 1]:4.2f} m\ntracks = {n}")
        return truth_line, est_line, ego_dot, tgt_dots, trk_scat, err_line, err_dot, stamp

    fig.tight_layout()
    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", type=Path, default=Path("data/cache/pipeline_baseline.npz"))
    ap.add_argument("--out", type=Path, default=Path("docs/images/demo.gif"))
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args(argv)

    if not args.npz.exists():
        raise SystemExit(f"recording not found: {args.npz} — run the pipeline first")
    out = render(args.npz, args.out, args.fps, args.stride)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
