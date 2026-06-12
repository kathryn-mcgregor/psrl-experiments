"""
Visualize a participant's maze and the path they took.

Usage:
    python visualize.py data/user_meta_<seed>.json
    python visualize.py data/user_meta_<seed>.json --save path_plot.png
"""

import sys
import json
import argparse
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from nicewebrl.utils import read_all_records_sync
import gridworld as gw


def load_data(meta_path: str):
    with open(meta_path) as f:
        meta = json.load(f)

    # Derive the msgpack path from the meta path
    data_path = meta_path.replace("user_meta_", "user_data_").replace(".json", ".msgpack")
    try:
        records = read_all_records_sync(data_path)
    except FileNotFoundError:
        print(f"Warning: step file not found at {data_path}, path will not be drawn.")
        records = []

    return meta, records


def plot(meta: dict, records: list, save_path: str | None = None):
    # Fall back to regenerating layout from seed if grid wasn't saved
    if "grid" not in meta or "goal_positions" not in meta:
        print("Note: grid not found in metadata, regenerating from seed.")
        episode = gw.init_episode(seed=meta.get("seed"))
        grid = np.array(episode["grid"])
        goal_positions = [tuple(p) for p in episode["goal_positions"]]
        true_goal_idx = episode["true_goal_idx"]
    else:
        grid = np.array(meta["grid"])
        goal_positions = [tuple(p) for p in meta["goal_positions"]]
        true_goal_idx = meta["true_goal_idx"]
    true_goal_pos = tuple(goal_positions[true_goal_idx])
    seed = meta.get("seed", "?")
    total_steps = meta.get("steps", len(records))

    # ------------------------------------------------------------------ #
    # Build path segments — split each time a candidate goal is entered   #
    # ------------------------------------------------------------------ #
    # Each segment is a list of (row, col) points drawn in a single color.
    segments = []
    if records:
        current_segment = [tuple(records[0]["prev_pos"])]
        for r in records:
            current_segment.append(tuple(r["new_pos"]))
            if r.get("visited_goal") is not None:
                segments.append(current_segment)
                current_segment = [tuple(r["new_pos"])]  # next segment starts here
        if len(current_segment) > 1:
            segments.append(current_segment)

    # Flat path still needed for start/end markers
    path = segments[0][:1] + [pt for seg in segments for pt in seg[1:]] if segments else []

    # ------------------------------------------------------------------ #
    # Figure                                                               #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(7, 7))
    size = grid.shape[0]

    # Background: white for empty, dark grey for wall
    cmap = ListedColormap(["#f5f0e8", "#4a4a4a"])
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, origin="upper", zorder=0)

    # Light grid lines
    for x in range(size + 1):
        ax.axhline(x - 0.5, color="#cccccc", linewidth=0.4, zorder=1)
        ax.axvline(x - 0.5, color="#cccccc", linewidth=0.4, zorder=1)

    # Candidate goals
    for idx, (gr, gc) in enumerate(goal_positions):
        is_true = (idx == true_goal_idx)
        color = "#16a34a" if is_true else "#f59e0b"
        circle = plt.Circle(
            (gc, gr), 0.32,
            color=color, zorder=3,
            linewidth=1.2, edgecolor="#555"
        )
        ax.add_patch(circle)

    # Distinct colors for path segments (cycles if more segments than colors)
    SEGMENT_COLORS = [
        "#3b82f6",  # blue
        "#f97316",  # orange
        "#8b5cf6",  # purple
        "#06b6d4",  # cyan
        "#ec4899",  # pink
        "#84cc16",  # lime
        "#ef4444",  # red
        "#14b8a6",  # teal
        "#f59e0b",  # amber
        "#6366f1",  # indigo
    ]

    # ------------------------------------------------------------------
    # Compute perpendicular offsets so overlapping edges show both colors
    # ------------------------------------------------------------------
    OFFSET_STEP = 0.13   # distance between parallel lines sharing an edge

    # Map undirected edge -> list of (seg_idx, step_idx)
    edge_visits = defaultdict(list)
    for seg_idx, seg in enumerate(segments):
        for step_idx in range(len(seg) - 1):
            p1, p2 = seg[step_idx], seg[step_idx + 1]
            edge = (min(p1, p2), max(p1, p2))
            edge_visits[edge].append((seg_idx, step_idx))

    # Assign a perpendicular offset to each (seg_idx, step_idx)
    step_offset = {}
    for edge, visits in edge_visits.items():
        n = len(visits)
        for i, key in enumerate(visits):
            step_offset[key] = (i - (n - 1) / 2) * OFFSET_STEP

    # Draw each step individually with its offset
    if segments:
        for seg_idx, seg in enumerate(segments):
            color = SEGMENT_COLORS[seg_idx % len(SEGMENT_COLORS)]
            for step_idx in range(len(seg) - 1):
                (r1, c1), (r2, c2) = seg[step_idx], seg[step_idx + 1]
                off = step_offset.get((seg_idx, step_idx), 0.0)
                # Perpendicular to direction of travel
                dr, dc = r2 - r1, c2 - c1
                if dr == 0:          # horizontal move → offset vertically
                    pr, pc = off, 0.0
                else:                # vertical move   → offset horizontally
                    pr, pc = 0.0, off
                ax.plot(
                    [c1 + pc, c2 + pc], [r1 + pr, r2 + pr],
                    color=color, linewidth=1.8, zorder=4,
                    solid_capstyle="round", solid_joinstyle="round"
                )

    if path:
        path_arr = np.array(path, dtype=float)
        rows, cols = path_arr[:, 0], path_arr[:, 1]
        # Start marker
        ax.plot(cols[0], rows[0], "o", color="#22c55e", markersize=8,
                zorder=5, markeredgecolor="white", markeredgewidth=1.2)
        # End marker
        ax.plot(cols[-1], rows[-1], "*", color="#dc2626", markersize=12,
                zorder=5, markeredgecolor="white", markeredgewidth=1)

    # ------------------------------------------------------------------ #
    # Axes formatting                                                       #
    # ------------------------------------------------------------------ #
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.tick_params(labelsize=7, length=2)
    ax.set_xlabel("Column", fontsize=9)
    ax.set_ylabel("Row", fontsize=9)
    ax.set_title(
        f"Participant {seed}  |  {total_steps} steps  |  True goal: {true_goal_idx + 1} "
        f"@ {true_goal_pos}",
        fontsize=10, pad=10
    )


    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


def process_all(data_dir: str = "data"):
    """Create visualizations for every user that doesn't already have one."""
    import glob
    import os

    vis_dir = os.path.join(data_dir, "path-vis")
    os.makedirs(vis_dir, exist_ok=True)

    meta_files = sorted(glob.glob(os.path.join(data_dir, "user_meta_*.json")))
    if not meta_files:
        print(f"No user_meta_*.json files found in {data_dir}/")
        return

    created = 0
    skipped = 0
    for meta_path in meta_files:
        seed = os.path.basename(meta_path).replace("user_meta_", "").replace(".json", "")
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


def main():
    parser = argparse.ArgumentParser(description="Visualize participant maze runs.")
    parser.add_argument(
        "meta_file", nargs="?", default=None,
        help="Path to a single user_meta_<seed>.json. "
             "Omit to process all users in data/ and save to data/path-vis/."
    )
    parser.add_argument("--save", metavar="FILE", default=None,
                        help="(single-file mode) save plot to this path instead of displaying.")
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
