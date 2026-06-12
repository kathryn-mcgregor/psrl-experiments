"""
Convert user maze data and run the PSRL agent on the same problem.

The maze is treated as a tabular MDP:
  - States  : free cells (104 states for a 13x13 four-room grid)
  - Actions : up / down / left / right  (4 actions)
  - Transitions : deterministic and known from the grid
  - Rewards : uncertain — one of the 15 candidate goals gives reward 1,
              the rest give 0.  PSRL maintains a Dirichlet posterior over
              the reward distribution at each candidate goal and updates it
              each time a goal is tested (episode boundary).

Each PSRL *episode* corresponds to navigating to one candidate goal:
  - At episode start, sample an MDP from the posterior and run value
    iteration to get a policy.
  - Follow that policy until a candidate goal cell is reached (done=True).
  - Observe reward (1 if true goal, else 0), update posteriors, repeat.

Usage
-----
    python run_psrl.py data/user_meta_<seed>.json
    python run_psrl.py data/user_meta_<seed>.json --gamma 0.99 --seed 0

Output
------
  - Prints a step-by-step comparison of the human and PSRL trajectories.
  - Saves a side-by-side visualisation to data/path-vis/psrl_<seed>.png.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from nicewebrl.utils import read_all_records_sync
import gridworld as gw
from psrl_kathryn import PSRLAgent

# ---------------------------------------------------------------------------
# Action definitions (must match gridworld.py DIRECTIONS order)
# ---------------------------------------------------------------------------
ACTIONS = {
    0: (-1,  0),   # up
    1: ( 1,  0),   # down
    2: ( 0, -1),   # left
    3: ( 0,  1),   # right
}
N_ACTIONS = len(ACTIONS)


# ---------------------------------------------------------------------------
# MDP construction
# ---------------------------------------------------------------------------

def build_state_space(grid):
    """Map free cells ↔ integer state indices."""
    cell_to_state = {}
    state_to_cell = {}
    idx = 0
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == 0:
                cell_to_state[(r, c)] = idx
                state_to_cell[idx] = (r, c)
                idx += 1
    return cell_to_state, state_to_cell


def build_transition_matrix(grid, cell_to_state, state_to_cell):
    """
    Deterministic transition matrix T[S, A, S].

    T[s, a, s'] = 1 if taking action a in state s leads to s', else 0.
    Wall collisions leave the agent in state s.
    """
    S = len(cell_to_state)
    T = np.zeros((S, N_ACTIONS, S), dtype=np.float32)
    rows, cols = len(grid), len(grid[0])

    for s, (r, c) in state_to_cell.items():
        for a, (dr, dc) in ACTIONS.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
                s_next = cell_to_state[(nr, nc)]
            else:
                s_next = s
            T[s, a, s_next] = 1.0

    return T


def build_reward_prior(S, cell_to_state, goal_positions, alpha_goal=1.0,
                       alpha_noncand=1e6):
    """
    Dirichlet prior over reward categories at each state.

    R_vals = [0., 1.].  For candidate goals the prior is uniform (each
    equally likely to carry reward 1).  For all other cells the prior
    strongly favours reward 0.

    Returns R_post of shape (S, N_ACTIONS, 2).
    """
    R_post = np.zeros((S, N_ACTIONS, 2), dtype=np.float32)
    candidate_states = {cell_to_state[tuple(p)] for p in goal_positions}

    for s in range(S):
        if s in candidate_states:
            # Uniform: equal weight on reward 0 and reward 1
            R_post[s, :, 0] = alpha_goal
            R_post[s, :, 1] = alpha_goal
        else:
            # Near-certain zero reward
            R_post[s, :, 0] = alpha_noncand
            R_post[s, :, 1] = 1e-6

    return R_post


# ---------------------------------------------------------------------------
# PSRL simulation
# ---------------------------------------------------------------------------

def run_psrl(meta: dict, gamma: float = 0.99, seed: int = 0,
             max_total_steps: int = 5000):
    """
    Run the PSRL agent on the maze described by *meta*.

    Returns
    -------
    psrl_path : list of (row, col) cells visited
    episodes  : list of dicts, one per episode, with keys
                  'path', 'goal_tested', 'goal_pos', 'reached_true'
    """
    grid = meta["grid"]
    goal_positions = [tuple(p) for p in meta["goal_positions"]]
    true_goal_pos  = tuple(meta["true_goal_pos"])
    start_pos      = tuple(meta["agent_pos"])

    cell_to_state, state_to_cell = build_state_space(grid)
    T      = build_transition_matrix(grid, cell_to_state, state_to_cell)
    R_post = build_reward_prior(len(cell_to_state), cell_to_state, goal_positions)

    S = len(cell_to_state)
    candidate_states = {cell_to_state[p] for p in goal_positions}
    true_goal_state  = cell_to_state[true_goal_pos]

    agent = PSRLAgent(
        num_states=S,
        num_actions=N_ACTIONS,
        gamma=gamma,
        seed=seed,
        R_vals=[0., 1.],
        T_alpha0=0.0,    # transitions are known — placeholder
        R_alpha0=1.0,
    )

    # Override posteriors:
    #   T_post : encode known transitions with high concentration
    #   R_post : our custom prior
    T_post = (T * 1000.0).astype(np.float32) + 1e-6
    agent.agent_state = agent.agent_state._replace(
        T_post=jnp.array(T_post),
        R_post=jnp.array(R_post),
    )

    # Simulate
    obs          = cell_to_state[start_pos]
    psrl_path    = [state_to_cell[obs]]
    episodes     = []
    ep_path      = [state_to_cell[obs]]
    total_steps  = 0

    while total_steps < max_total_steps:
        action   = agent.act(obs)
        next_obs = int(np.argmax(T[obs, action]))

        obs        = next_obs
        cell       = state_to_cell[obs]
        psrl_path.append(cell)
        ep_path.append(cell)
        total_steps += 1

        at_candidate = obs in candidate_states
        reached_true = obs == true_goal_state
        reward       = 1.0 if reached_true else 0.0

        agent.update(
            obs     = cell_to_state[state_to_cell[obs]],   # current state
            action  = action,
            reward  = reward,
            next_obs= obs,
            done    = at_candidate,
        )

        if at_candidate:
            goal_idx = next(
                i for i, p in enumerate(goal_positions) if p == cell
            )
            episodes.append({
                "path":         list(ep_path),
                "goal_tested":  goal_idx,
                "goal_pos":     cell,
                "reached_true": reached_true,
            })
            ep_path = [cell]

            if reached_true:
                break

    return psrl_path, episodes


# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------

def load_user_data(meta_path: str):
    with open(meta_path) as f:
        meta = json.load(f)

    # Fill in fields that may be missing from older saves
    if "grid" not in meta:
        episode = gw.init_episode(seed=meta.get("seed"))
        meta.setdefault("grid",          episode["grid"])
        meta.setdefault("goal_positions", episode["goal_positions"])
        meta.setdefault("true_goal_idx",  episode["true_goal_idx"])
        true_idx = meta["true_goal_idx"]
        meta.setdefault("true_goal_pos",
                        meta["goal_positions"][true_idx])

    data_path = meta_path.replace("user_meta_", "user_data_").replace(".json", ".msgpack")
    try:
        records = read_all_records_sync(data_path)
    except FileNotFoundError:
        print(f"Warning: step file not found at {data_path}")
        records = []

    return meta, records


def human_path_from_records(records, meta):
    """Extract the human's (row, col) path from step records."""
    if not records:
        return []
    path = [tuple(records[0]["prev_pos"])]
    for r in records:
        path.append(tuple(r["new_pos"]))
    return path


def print_comparison(meta, records, psrl_path, episodes):
    human_path = human_path_from_records(records, meta)
    true_goal  = tuple(meta["true_goal_pos"])
    true_idx   = meta["true_goal_idx"]

    print("=" * 60)
    print(f"Seed            : {meta.get('seed')}")
    print(f"True goal       : #{true_idx + 1}  at {true_goal}")
    print()
    print(f"Human  — total steps : {len(human_path) - 1}")

    # Goals the human visited
    visited = meta.get("visited_goals", [])
    goal_pos_list = [tuple(p) for p in meta["goal_positions"]]
    print(f"Human  — goals tested: {[g + 1 for g in visited]}")

    print()
    print(f"PSRL   — total steps : {len(psrl_path) - 1}")
    print(f"PSRL   — goals tested: "
          f"{[ep['goal_tested'] + 1 for ep in episodes]}")
    print(f"PSRL   — episodes    : {len(episodes)}")
    print()
    print("PSRL episode breakdown:")
    for i, ep in enumerate(episodes):
        status = "TRUE GOAL ✓" if ep["reached_true"] else "not true goal"
        print(f"  Episode {i+1}: navigated to goal #{ep['goal_tested']+1} "
              f"at {ep['goal_pos']}  [{status}]  "
              f"({len(ep['path'])-1} steps)")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

SEGMENT_COLORS = [
    "#3b82f6", "#f97316", "#8b5cf6", "#06b6d4",
    "#ec4899", "#84cc16", "#ef4444", "#14b8a6",
]
OFFSET_STEP = 0.13


def _compute_offsets(human_segs, psrl_segs):
    """Assign perpendicular offsets to steps shared between human and PSRL."""
    edge_visits = defaultdict(list)
    for label, segs in [("human", human_segs), ("psrl", psrl_segs)]:
        for seg_idx, seg in enumerate(segs):
            for step_idx in range(len(seg) - 1):
                p1, p2 = seg[step_idx], seg[step_idx + 1]
                edge = (min(p1, p2), max(p1, p2))
                edge_visits[edge].append((label, seg_idx, step_idx))

    offsets = {}
    for edge, visits in edge_visits.items():
        n = len(visits)
        for i, key in enumerate(visits):
            offsets[key] = (i - (n - 1) / 2) * OFFSET_STEP
    return offsets


def _split_into_segments(path, goal_positions):
    """Split a path into segments at candidate goal visits."""
    goal_set = set(goal_positions)
    segments = []
    current = [path[0]]
    for pt in path[1:]:
        current.append(pt)
        if pt in goal_set:
            segments.append(current)
            current = [pt]
    if len(current) > 1:
        segments.append(current)
    return segments if segments else [path]


def _draw_path(ax, segs, offsets_dict, label, colors, lw=1.8, alpha=1.0):
    for seg_idx, seg in enumerate(segs):
        color = colors[seg_idx % len(colors)]
        for step_idx in range(len(seg) - 1):
            (r1, c1), (r2, c2) = seg[step_idx], seg[step_idx + 1]
            off = offsets_dict.get((label, seg_idx, step_idx), 0.0)
            dr, dc = r2 - r1, c2 - c1
            pr, pc = (off, 0.0) if dr == 0 else (0.0, off)
            ax.plot(
                [c1 + pc, c2 + pc], [r1 + pr, r2 + pr],
                color=color, linewidth=lw, alpha=alpha,
                solid_capstyle="round", solid_joinstyle="round", zorder=4,
            )


def plot_comparison(meta, records, psrl_path, episodes, save_path=None):
    grid           = np.array(meta["grid"])
    goal_positions = [tuple(p) for p in meta["goal_positions"]]
    true_goal_pos  = tuple(meta["true_goal_pos"])
    size           = grid.shape[0]

    human_path = human_path_from_records(records, meta)

    human_segs = _split_into_segments(human_path, goal_positions) if human_path else []
    psrl_segs  = _split_into_segments(psrl_path,  goal_positions) if psrl_path  else []
    offsets    = _compute_offsets(human_segs, psrl_segs)

    HUMAN_COLORS = ["#3b82f6","#f97316","#8b5cf6","#06b6d4","#ec4899",
                    "#84cc16","#ef4444","#14b8a6"]
    PSRL_COLORS  = ["#1d4ed8","#c2410c","#7c3aed","#0e7490","#be185d",
                    "#4d7c0f","#b91c1c","#0f766e"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    titles = [
        f"Human  ({len(human_path)-1 if human_path else '?'} steps)",
        f"PSRL   ({len(psrl_path)-1} steps)",
    ]

    for ax, segs, colors, title in zip(
        axes,
        [human_segs, psrl_segs],
        [HUMAN_COLORS, PSRL_COLORS],
        titles,
    ):
        cmap = ListedColormap(["#f5f0e8", "#4a4a4a"])
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, origin="upper", zorder=0)

        for x in range(size + 1):
            ax.axhline(x - 0.5, color="#cccccc", linewidth=0.4, zorder=1)
            ax.axvline(x - 0.5, color="#cccccc", linewidth=0.4, zorder=1)

        # Candidate goals
        for idx, (gr, gc) in enumerate(goal_positions):
            is_true = ((gr, gc) == true_goal_pos)
            circle = plt.Circle(
                (gc, gr), 0.32,
                color="#16a34a" if is_true else "#f59e0b",
                zorder=3, linewidth=1.2, edgecolor="#555",
            )
            ax.add_patch(circle)

        # Path
        label = "human" if colors is HUMAN_COLORS else "psrl"
        _draw_path(ax, segs, offsets, label, colors)

        # Start / end markers
        path = human_path if label == "human" else psrl_path
        if path:
            ax.plot(path[0][1], path[0][0], "o", color="#22c55e",
                    markersize=8, zorder=5, markeredgecolor="white",
                    markeredgewidth=1.2)
            ax.plot(path[-1][1], path[-1][0], "*", color="#dc2626",
                    markersize=12, zorder=5, markeredgecolor="white",
                    markeredgewidth=1)

        ax.set_xlim(-0.5, size - 0.5)
        ax.set_ylim(size - 0.5, -0.5)
        ax.set_xticks(range(size))
        ax.set_yticks(range(size))
        ax.tick_params(labelsize=7, length=2)
        ax.set_title(title, fontsize=11, pad=8)

    seed = meta.get("seed", "?")
    fig.suptitle(
        f"Participant {seed}  |  True goal #{meta['true_goal_idx']+1} "
        f"at {true_goal_pos}",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualisation to {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run PSRL on a participant's maze and compare to human data."
    )
    parser.add_argument("meta_file", help="Path to user_meta_<seed>.json")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor for value iteration (default: 0.99)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for the PSRL agent (default: 0)")
    parser.add_argument("--save", metavar="FILE", default=None,
                        help="Save comparison plot to this path. "
                             "Defaults to data/path-vis/psrl_<seed>.png.")
    args = parser.parse_args()

    meta, records = load_user_data(args.meta_file)

    user_seed = meta.get("seed", "unknown")
    save_path = args.save or os.path.join(
        "data", "path-vis", f"psrl_{user_seed}.png"
    )

    print(f"Running PSRL for participant {user_seed} ...")
    psrl_path, episodes = run_psrl(meta, gamma=args.gamma, seed=args.seed)

    print_comparison(meta, records, psrl_path, episodes)
    plot_comparison(meta, records, psrl_path, episodes, save_path=save_path)


if __name__ == "__main__":
    main()
