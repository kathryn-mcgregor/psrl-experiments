"""
Visualize a participant's fruit-collection maze and the path they took.

Usage:
    python visualize.py                                   # all users → data/path-vis/
    python visualize.py data/user_meta_<seed>.json        # display single user
    python visualize.py data/user_meta_<seed>.json --save plot.png
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse

from nicewebrl.utils import read_all_records_sync
import gridworld as gw

SEGMENT_COLORS = [
    "#3b82f6", "#f97316", "#8b5cf6", "#06b6d4",
    "#ec4899", "#84cc16", "#ef4444", "#14b8a6",
    "#f59e0b", "#6366f1",
]
OFFSET_STEP = 0.13

APPLE_COLOR  = "#dc2626"   # red
BANANA_COLOR = "#fbbf24"   # amber-yellow
GRAY_COLOR   = "#d1d5db"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(meta_path: str):
    with open(meta_path) as f:
        meta = json.load(f)

    if "grid" not in meta or "fruit_positions" not in meta:
        print("Note: layout not in metadata, regenerating from seed.")
        episode = gw.init_episode(seed=meta.get("seed"))
        meta.setdefault("grid",            episode["grid"])
        meta.setdefault("fruit_positions", episode["fruit_positions"])
        meta.setdefault("fruit_types",     episode["fruit_types"])
        meta.setdefault("fruit_values",    episode["fruit_values"])
        meta.setdefault("agent_pos",       episode["agent_pos"])

    data_path = meta_path.replace("user_meta_", "user_data_").replace(".json", ".msgpack")
    try:
        records = read_all_records_sync(data_path)
    except FileNotFoundError:
        print(f"Warning: step file not found at {data_path}, path will not be drawn.")
        records = []

    return meta, records


# ---------------------------------------------------------------------------
# Path segmentation  (new segment each time a fruit is collected)
# ---------------------------------------------------------------------------

def build_segments(records: list) -> list[list[tuple]]:
    if not records:
        return []
    segments = []
    current = [tuple(records[0]["prev_pos"])]
    for r in records:
        current.append(tuple(r["new_pos"]))
        if r.get("collected_fruit") is not None:
            segments.append(current)
            current = [tuple(r["new_pos"])]
    if len(current) > 1:
        segments.append(current)
    return segments


def compute_offsets(segments: list) -> dict:
    edge_visits = defaultdict(list)
    for seg_idx, seg in enumerate(segments):
        for step_idx in range(len(seg) - 1):
            p1, p2 = seg[step_idx], seg[step_idx + 1]
            edge = (min(p1, p2), max(p1, p2))
            edge_visits[edge].append((seg_idx, step_idx))

    offsets = {}
    for edge, visits in edge_visits.items():
        n = len(visits)
        for i, key in enumerate(visits):
            offsets[key] = (i - (n - 1) / 2) * OFFSET_STEP
    return offsets


# ---------------------------------------------------------------------------
# Fruit rendering helpers (matplotlib)
# ---------------------------------------------------------------------------

def _draw_apple(ax, gc, gr, color, zorder=3):
    r = 0.30
    circle = plt.Circle((gc, gr), r, color=color, zorder=zorder,
                         linewidth=1.2, edgecolor="#555")
    ax.add_patch(circle)
    # Stem
    ax.plot([gc, gc], [gr - r, gr - r - 0.15],
            color="#15803d", linewidth=1.5, zorder=zorder + 1,
            solid_capstyle="round")


def _draw_banana(ax, gc, gr, color, zorder=3):
    from matplotlib.transforms import Affine2D
    ellipse = Ellipse(
        xy=(gc, gr), width=0.72, height=0.36,
        angle=-30, color=color, zorder=zorder,
        linewidth=1.2, edgecolor="#555"
    )
    ax.add_patch(ellipse)


def _draw_fruit(ax, gc, gr, fruit_type: str, collected: bool, zorder=3):
    color = GRAY_COLOR if collected else (
        APPLE_COLOR if fruit_type == "apple" else BANANA_COLOR
    )
    if fruit_type == "apple":
        _draw_apple(ax, gc, gr, color, zorder)
    else:
        _draw_banana(ax, gc, gr, color, zorder)


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------

def plot(meta: dict, records: list, save_path: str | None = None):
    grid            = np.array(meta["grid"])
    fruit_positions = [tuple(p) for p in meta["fruit_positions"]]
    fruit_types     = meta["fruit_types"]
    fruit_values    = meta.get("fruit_values", {})
    score_history   = meta.get("score_history", [])
    seed            = meta.get("seed", "?")
    total_steps     = meta.get("steps", len(records))
    final_score     = meta.get("score", sum(e["points"] for e in score_history))
    size            = grid.shape[0]

    # Which fruits were collected (from score_history in meta, fallback to records)
    collected_idxs = {e["fruit_idx"] for e in score_history}

    segments = build_segments(records)
    offsets  = compute_offsets(segments)
    flat_path = (
        segments[0][:1] + [pt for seg in segments for pt in seg[1:]]
        if segments else []
    )

    # ------------------------------------------------------------------ #
    # Figure                                                               #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(7, 7))

    # Grid background
    cmap = ListedColormap(["#f5f0e8", "#4a4a4a"])
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, origin="upper", zorder=0)
    for x in range(size + 1):
        ax.axhline(x - 0.5, color="#cccccc", linewidth=0.4, zorder=1)
        ax.axvline(x - 0.5, color="#cccccc", linewidth=0.4, zorder=1)

    # Fruits
    for idx, (fr, fc) in enumerate(fruit_positions):
        collected = idx in collected_idxs
        _draw_fruit(ax, fc, fr, fruit_types[idx], collected, zorder=3)

    # Path segments with per-edge offsets
    for seg_idx, seg in enumerate(segments):
        color = SEGMENT_COLORS[seg_idx % len(SEGMENT_COLORS)]
        for step_idx in range(len(seg) - 1):
            (r1, c1), (r2, c2) = seg[step_idx], seg[step_idx + 1]
            off = offsets.get((seg_idx, step_idx), 0.0)
            dr, dc = r2 - r1, c2 - c1
            pr, pc = (off, 0.0) if dr == 0 else (0.0, off)
            ax.plot(
                [c1 + pc, c2 + pc], [r1 + pr, r2 + pr],
                color=color, linewidth=1.8, zorder=4,
                solid_capstyle="round", solid_joinstyle="round"
            )

    # Start / end markers
    if flat_path:
        arr = np.array(flat_path, dtype=float)
        ax.plot(arr[0, 1],  arr[0, 0],  "o", color="#22c55e", markersize=8,
                zorder=5, markeredgecolor="white", markeredgewidth=1.2)
        ax.plot(arr[-1, 1], arr[-1, 0], "*", color="#dc2626", markersize=12,
                zorder=5, markeredgecolor="white", markeredgewidth=1)

    # Axes
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.tick_params(labelsize=7, length=2)
    ax.set_xlabel("Column", fontsize=9)
    ax.set_ylabel("Row",    fontsize=9)

    apple_pts  = fruit_values.get("apple",  "?")
    banana_pts = fruit_values.get("banana", "?")
    ax.set_title(
        f"Participant {seed}  |  {total_steps} steps  |  Score: {final_score}\n"
        f"🍎 = {apple_pts} pt   🍌 = {banana_pts} pt",
        fontsize=10, pad=10
    )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_all(data_dir: str = "data"):
    vis_dir    = os.path.join(data_dir, "path-vis")
    os.makedirs(vis_dir, exist_ok=True)

    meta_files = sorted(glob.glob(os.path.join(data_dir, "user_meta_*.json")))
    if not meta_files:
        print(f"No user_meta_*.json files found in {data_dir}/")
        return

    created = skipped = 0
    for meta_path in meta_files:
        seed     = os.path.basename(meta_path).replace("user_meta_", "").replace(".json", "")
        out_path = os.path.join(vis_dir, f"path_{seed}.png")

        if os.path.exists(out_path):
            print(f"  skip  {seed}  (already exists)")
            skipped += 1
            continue

        print(f"  plot  {seed}")
        try:
            meta, records = load_data(meta_path)
            plot(meta, records, save_path=out_path)
            created += 1
        except Exception as e:
            print(f"    ERROR: {e}")

    print(f"\nDone — {created} created, {skipped} skipped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize participant fruit-collection maze runs."
    )
    parser.add_argument(
        "meta_file", nargs="?", default=None,
        help="Path to a single user_meta_<seed>.json. "
             "Omit to process all users in data/ and save to data/path-vis/."
    )
    parser.add_argument("--save", metavar="FILE", default=None,
                        help="(single-file mode) save plot here instead of displaying.")
    parser.add_argument("--data-dir", metavar="DIR", default="data",
                        help="Directory containing user data files (default: data).")
    args = parser.parse_args()

    if args.meta_file is None:
        process_all(data_dir=args.data_dir)
    else:
        meta, records = load_data(args.meta_file)
        plot(meta, records, save_path=args.save)


if __name__ == "__main__":
    main()
