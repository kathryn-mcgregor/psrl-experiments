"""
13x13 four-room gridworld for the concept-learning experiment.

Rule: each participant is assigned a hidden rule — either "all of one shape"
      or "all of one color".  Goals matching the rule give reward 1, others 0.

Layout per trial:
  - 4 candidate goals, one placed randomly in each room
  - Each goal has a unique (shape, color) combination within its maze
  - 4 mazes per participant; the rule is constant across all mazes
"""

import random
import numpy as np

EMPTY = 0
WALL  = 1

GRID_SIZE = 13
N_GOALS   = 4
N_MAZES   = 4

SHAPES = ["circle", "square", "triangle"]
COLORS = ["blue", "red", "yellow"]

DIRECTIONS = {
    "ArrowUp":    (-1,  0),
    "ArrowDown":  ( 1,  0),
    "ArrowLeft":  ( 0, -1),
    "ArrowRight": ( 0,  1),
}


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

def make_fourroom_grid() -> np.ndarray:
    size = GRID_SIZE
    mid  = size // 2
    grid = np.zeros((size, size), dtype=np.int8)

    grid[0, :]  = WALL;  grid[-1, :] = WALL
    grid[:, 0]  = WALL;  grid[:, -1] = WALL
    grid[mid, :] = WALL;  grid[:, mid] = WALL

    grid[mid, 2] = EMPTY;  grid[mid, 9] = EMPTY   # horizontal doors
    grid[2, mid] = EMPTY;  grid[9, mid] = EMPTY   # vertical doors

    return grid


def get_free_cells(grid: np.ndarray) -> list[tuple[int, int]]:
    rows, cols = np.where(grid == EMPTY)
    return list(zip(rows.tolist(), cols.tolist()))


def get_room_cells(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return free cells in each of the 4 rooms: [TL, TR, BL, BR]."""
    mid = GRID_SIZE // 2
    rooms = []
    for r_lo, r_hi, c_lo, c_hi in [
        (1, mid,           1, mid),
        (1, mid,           mid + 1, GRID_SIZE - 1),
        (mid + 1, GRID_SIZE - 1, 1, mid),
        (mid + 1, GRID_SIZE - 1, mid + 1, GRID_SIZE - 1),
    ]:
        cells = [
            (r, c) for r in range(r_lo, r_hi) for c in range(c_lo, c_hi)
            if grid[r][c] == EMPTY
        ]
        rooms.append(cells)
    return rooms


# ---------------------------------------------------------------------------
# Episode initialisation
# ---------------------------------------------------------------------------

def _generate_goal_attributes(rng: random.Random, n: int) -> list[tuple[str, str]]:
    """Return n unique (shape, color) pairs chosen at random."""
    all_combos = [(s, c) for s in SHAPES for c in COLORS]
    return rng.sample(all_combos, n)


def _make_maze(rng: random.Random) -> dict:
    grid  = make_fourroom_grid()
    rooms = get_room_cells(grid)

    # One goal per room
    goal_positions = [list(rng.choice(room)) for room in rooms]
    goal_set       = {tuple(p) for p in goal_positions}

    # Agent at a random free cell not on a goal
    free      = [c for c in get_free_cells(grid) if c not in goal_set]
    agent_pos = list(rng.choice(free))

    attrs = _generate_goal_attributes(rng, N_GOALS)
    rng.shuffle(attrs)

    return {
        "grid":          grid.tolist(),
        "agent_pos":     agent_pos,
        "goal_positions": goal_positions,
        "goal_shapes":   [a[0] for a in attrs],
        "goal_colors":   [a[1] for a in attrs],
        "visited_goals": [],
        "current_goal":  None,
        "done":          False,
        "step":          0,
    }


def init_episode(seed: int | None = None) -> dict:
    rng = random.Random(seed)

    rule_type  = rng.choice(["shape", "color"])
    rule_value = rng.choice(SHAPES if rule_type == "shape" else COLORS)

    return {
        "rule_type":       rule_type,
        "rule_value":      rule_value,
        "current_maze_idx": 0,
        "mazes":           [_make_maze(rng) for _ in range(N_MAZES)],
        "log":             [],
        "total_score":     0,
        "done":            False,
    }


# ---------------------------------------------------------------------------
# Step logic
# ---------------------------------------------------------------------------

def step(state: dict, action: str) -> tuple[dict, dict]:
    """
    Apply *action* to the current maze.

    Returns (new_state, info) where info contains:
        moved        – agent actually moved
        visited_goal – index of goal just entered (or None)
        left_goal    – index of goal just exited (or None)
        reward       – reward received this step (0 or 1)
    """
    maze_idx = state["current_maze_idx"]
    maze     = state["mazes"][maze_idx]

    if maze["done"] or action not in DIRECTIONS:
        return state, {"moved": False, "visited_goal": None,
                       "left_goal": None, "reward": 0}

    dr, dc = DIRECTIONS[action]
    r, c   = maze["agent_pos"]
    nr, nc = r + dr, c + dc
    grid   = np.array(maze["grid"], dtype=np.int8)

    if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE) or grid[nr][nc] == WALL:
        return state, {"moved": False, "visited_goal": None,
                       "left_goal": None, "reward": 0}

    new_maze         = dict(maze)
    new_maze["agent_pos"] = [nr, nc]
    new_maze["step"]      = maze["step"] + 1

    # Leaving a goal?
    left_goal = None
    if maze["current_goal"] is not None:
        left_goal = maze["current_goal"]
        new_maze["current_goal"] = None

    # Entering a new goal?
    visited_goal = None
    reward       = 0
    visited      = set(maze["visited_goals"])

    for idx, gpos in enumerate(maze["goal_positions"]):
        if [nr, nc] == gpos and idx not in visited:
            visited_goal = idx
            new_maze["current_goal"]  = idx
            new_maze["visited_goals"] = list(maze["visited_goals"]) + [idx]

            shape = maze["goal_shapes"][idx]
            color = maze["goal_colors"][idx]
            reward = int(
                (state["rule_type"] == "shape" and shape == state["rule_value"]) or
                (state["rule_type"] == "color" and color == state["rule_value"])
            )
            break

    if len(new_maze["visited_goals"]) == N_GOALS:
        new_maze["done"] = True

    new_mazes         = list(state["mazes"])
    new_mazes[maze_idx] = new_maze

    new_log   = list(state["log"])
    new_score = state["total_score"]

    if visited_goal is not None:
        new_score += reward
        new_log.append({
            "maze_idx":  maze_idx,
            "goal_idx":  visited_goal,
            "shape":     maze["goal_shapes"][visited_goal],
            "color":     maze["goal_colors"][visited_goal],
            "reward":    reward,
            "maze_step": new_maze["step"],
        })

    new_state = {
        **state,
        "mazes":       new_mazes,
        "log":         new_log,
        "total_score": new_score,
    }

    return new_state, {
        "moved":        True,
        "visited_goal": visited_goal,
        "left_goal":    left_goal,
        "reward":       reward,
    }


def advance_maze(state: dict) -> dict:
    """Increment current_maze_idx. Sets done=True if all mazes complete."""
    new_idx = state["current_maze_idx"] + 1
    return {
        **state,
        "current_maze_idx": new_idx,
        "done": new_idx >= N_MAZES,
    }
