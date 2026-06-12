"""
13x13 four-room gridworld with 7 apples and 7 bananas.

One fruit type is worth 5 points and the other 1 point; which is which
is randomly assigned at the start of each episode.

Reward timing:
  - Points are awarded the moment the agent steps ONTO a fruit cell.
  - The fruit is marked for graying-out when the agent steps OFF the cell.
"""

import random
import numpy as np

EMPTY = 0
WALL  = 1

GRID_SIZE  = 13
N_APPLES   = 7
N_BANANAS  = 7
N_FRUITS   = N_APPLES + N_BANANAS

DIRECTIONS = {
    "ArrowUp":    (-1,  0),
    "ArrowDown":  ( 1,  0),
    "ArrowLeft":  ( 0, -1),
    "ArrowRight": ( 0,  1),
}


def make_fourroom_grid() -> np.ndarray:
    size = GRID_SIZE
    mid  = size // 2   # 6
    grid = np.zeros((size, size), dtype=np.int8)

    grid[0, :]  = WALL
    grid[-1, :] = WALL
    grid[:, 0]  = WALL
    grid[:, -1] = WALL

    grid[mid, :] = WALL   # horizontal divider
    grid[:, mid] = WALL   # vertical divider

    # One doorway per wall segment
    grid[mid, 2] = EMPTY   # horizontal left
    grid[mid, 9] = EMPTY   # horizontal right
    grid[2, mid] = EMPTY   # vertical top
    grid[9, mid] = EMPTY   # vertical bottom

    return grid


def get_free_cells(grid: np.ndarray) -> list[tuple[int, int]]:
    rows, cols = np.where(grid == EMPTY)
    return list(zip(rows.tolist(), cols.tolist()))


def init_episode(seed: int | None = None) -> dict:
    """
    Randomly place the agent and 14 fruits (7 apples, 7 bananas).
    Randomly assign which fruit type is worth 5 pts vs 1 pt.
    """
    rng  = random.Random(seed)
    grid = make_fourroom_grid()
    free = get_free_cells(grid)
    rng.shuffle(free)

    agent_pos       = list(free[0])
    fruit_positions = [list(p) for p in free[1: N_FRUITS + 1]]

    # Shuffle which positions get apples vs bananas
    fruit_types = ["apple"] * N_APPLES + ["banana"] * N_BANANAS
    rng.shuffle(fruit_types)

    # Randomly decide which fruit is the high-value one
    if rng.random() < 0.5:
        fruit_values = {"apple": 5, "banana": 1}
    else:
        fruit_values = {"apple": 1, "banana": 5}

    return {
        "grid":             grid.tolist(),
        "agent_pos":        agent_pos,
        "fruit_positions":  fruit_positions,   # list of [row, col]
        "fruit_types":      fruit_types,        # list of "apple"/"banana"
        "fruit_values":     fruit_values,       # {"apple": X, "banana": Y}
        "collected_fruits": [],                 # indices collected so far
        "current_fruit":    None,               # index agent is standing on
        "score":            0,
        "score_history":    [],                 # [{fruit_idx, fruit_type, points, step}]
        "step":             0,
        "done":             False,
    }


def step(state: dict, action: str) -> tuple[dict, dict]:
    """
    Apply *action* to *state*.

    Returns (new_state, info) where info contains:
        moved            – whether the agent actually moved
        collected_fruit  – index of fruit just entered (or None)
        left_fruit       – index of fruit just exited (or None)
    """
    if state["done"] or action not in DIRECTIONS:
        return state, {"moved": False, "collected_fruit": None, "left_fruit": None}

    dr, dc = DIRECTIONS[action]
    r, c   = state["agent_pos"]
    nr, nc = r + dr, c + dc

    grid = np.array(state["grid"], dtype=np.int8)

    if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE) or grid[nr][nc] == WALL:
        return state, {"moved": False, "collected_fruit": None, "left_fruit": None}

    new_state = {**state}
    new_state["agent_pos"] = [nr, nc]
    new_state["step"]      = state["step"] + 1

    # ---- leaving a fruit cell? ----------------------------------------
    left_fruit = None
    if state["current_fruit"] is not None:
        left_fruit = state["current_fruit"]
        new_state["current_fruit"] = None

    # ---- entering a fruit cell? ----------------------------------------
    collected_fruit = None
    collected = set(state["collected_fruits"])

    for idx, fpos in enumerate(state["fruit_positions"]):
        if [nr, nc] == fpos and idx not in collected:
            collected_fruit = idx
            new_state["current_fruit"] = idx

            fruit_type = state["fruit_types"][idx]
            points     = state["fruit_values"][fruit_type]

            new_state["score"]            = state["score"] + points
            new_state["collected_fruits"] = list(state["collected_fruits"]) + [idx]
            new_state["score_history"]    = list(state["score_history"]) + [{
                "fruit_idx":  idx,
                "fruit_type": fruit_type,
                "points":     points,
                "step":       new_state["step"],
            }]
            break

    # ---- all fruits collected? -----------------------------------------
    if len(new_state["collected_fruits"]) == N_FRUITS:
        new_state["done"] = True

    return new_state, {
        "moved":           True,
        "collected_fruit": collected_fruit,
        "left_fruit":      left_fruit,
    }
