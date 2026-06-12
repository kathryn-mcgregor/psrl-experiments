"""
13x13 four-room gridworld with random goal placement.

Layout (mid = 6):
  - Outer border walls
  - Horizontal wall at row 6 with two doorways
  - Vertical wall at col 6 with two doorways
  - 4 rooms: top-left, top-right, bottom-left, bottom-right
"""
import random
import numpy as np

EMPTY = 0
WALL = 1

GRID_SIZE = 13
N_CANDIDATE_GOALS = 15

# Directions: (dr, dc)
DIRECTIONS = {
    "ArrowUp":    (-1,  0),
    "ArrowDown":  ( 1,  0),
    "ArrowLeft":  ( 0, -1),
    "ArrowRight": ( 0,  1),
}


def make_fourroom_grid() -> np.ndarray:
    size = GRID_SIZE
    mid = size // 2  # 6

    grid = np.zeros((size, size), dtype=np.int8)

    # Outer walls
    grid[0, :] = WALL
    grid[-1, :] = WALL
    grid[:, 0] = WALL
    grid[:, -1] = WALL

    # Interior dividing walls
    grid[mid, :] = WALL   # horizontal
    grid[:, mid] = WALL   # vertical

    # Doorways: one per wall segment
    # Horizontal wall, left half  (cols 1-5)  → door at col 2
    grid[mid, 2] = EMPTY
    # Horizontal wall, right half (cols 7-11) → door at col 9
    grid[mid, 9] = EMPTY
    # Vertical wall, top half    (rows 1-5)  → door at row 2
    grid[2, mid] = EMPTY
    # Vertical wall, bottom half (rows 7-11) → door at row 9
    grid[9, mid] = EMPTY

    return grid


def get_free_cells(grid: np.ndarray) -> list[tuple[int, int]]:
    rows, cols = np.where(grid == EMPTY)
    return list(zip(rows.tolist(), cols.tolist()))


def init_episode(seed: int | None = None) -> dict:
    """
    Randomly place agent and 15 candidate goals; choose one as the true goal.

    Returns a plain dict (JSON-serialisable) describing the episode.
    """
    rng = random.Random(seed)
    grid = make_fourroom_grid()
    free = get_free_cells(grid)
    rng.shuffle(free)

    agent_pos = list(free[0])                          # [row, col]
    goal_positions = [list(p) for p in free[1: N_CANDIDATE_GOALS + 1]]
    true_goal_idx = rng.randint(0, N_CANDIDATE_GOALS - 1)

    return {
        "grid": grid.tolist(),
        "agent_pos": agent_pos,
        "goal_positions": goal_positions,
        "true_goal_idx": true_goal_idx,
        "step": 0,
        "done": False,
        "visited_goals": [],   # list of goal indices visited so far
    }


def step(state: dict, action: str) -> tuple[dict, dict]:
    """
    Apply action to state. Returns (new_state, info).

    info keys:
        moved        – whether the agent actually moved
        visited_goal – index of candidate goal just entered (or None)
        reached_true – whether the true goal was just reached
    """
    if state["done"] or action not in DIRECTIONS:
        return state, {"moved": False, "visited_goal": None, "reached_true": False}

    dr, dc = DIRECTIONS[action]
    r, c = state["agent_pos"]
    nr, nc = r + dr, c + dc

    grid = np.array(state["grid"], dtype=np.int8)

    # Boundary / wall check
    if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE) or grid[nr, nc] == WALL:
        return state, {"moved": False, "visited_goal": None, "reached_true": False}

    new_state = dict(state)
    new_state["agent_pos"] = [nr, nc]
    new_state["step"] = state["step"] + 1

    # Check if agent entered a candidate goal cell
    visited_goal = None
    reached_true = False
    for idx, gpos in enumerate(state["goal_positions"]):
        if [nr, nc] == gpos:
            visited_goal = idx
            if idx == state["true_goal_idx"]:
                reached_true = True
                new_state["done"] = True
            if idx not in new_state.get("visited_goals", []):
                new_state["visited_goals"] = list(state["visited_goals"]) + [idx]
            break

    return new_state, {
        "moved": True,
        "visited_goal": visited_goal,
        "reached_true": reached_true,
    }
