"""
13x13 four-room gridworld for the concept-learning experiment.

Rule: each participant is assigned a hidden rule based on one dimension and one
      value within that dimension (e.g. "all circles" or "all blue goals").
      Goals matching the rule give reward 1, others 0.

Layout per trial:
  - n_goals candidate goals, one placed randomly in each room
  - Goals vary on the active dimensions; inactive dimensions are fixed per maze
  - n_mazes mazes per participant; the rule is constant across all mazes

To add a new dimension: add an entry to DIMENSIONS.
To add a new kind to an existing dimension: append to its list in DIMENSIONS.
"""

import random
import numpy as np

EMPTY = 0
WALL  = 1

GRID_SIZE = 13
N_MAZES   = 4

# Registry of all dimensions and their possible values.
# To extend: add a new key or append to an existing list.
DIMENSIONS = {
    "shape":   ["circle", "square", "triangle", "star", "pentagon", "hexagon", "diamond"],
    "color":   ["blue", "red", "yellow", "green", "purple", "orange", "pink"],
    "texture": ["solid", "striped", "dotted", "outline", "chevron"],
}

DIRECTIONS = {
    "ArrowUp":    (-1,  0),
    "ArrowDown":  ( 1,  0),
    "ArrowLeft":  ( 0, -1),
    "ArrowRight": ( 0,  1),
}

# Experiment modes
MODE_RANDOM    = "random"      # goals are random combos with no constraints
MODE_BIJECTION = "bijection"   # each active-dimension value appears exactly once per maze
MODE_NO_REPEAT = "no-repeat"   # rewarding combos never reappear in later mazes


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

def make_fourroom_grid() -> np.ndarray:
    size = GRID_SIZE
    mid  = size // 2
    grid = np.zeros((size, size), dtype=np.int8)

    grid[0, :]   = WALL;  grid[-1, :] = WALL
    grid[:, 0]   = WALL;  grid[:, -1] = WALL
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
        (1, mid,               1, mid),
        (1, mid,               mid + 1, GRID_SIZE - 1),
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
# Goal attribute generation
# ---------------------------------------------------------------------------

def _active_pool(
    dims: list[str],
    n_kinds: dict[str, int],
    excluded: set[frozenset] | None = None,
) -> list[dict]:
    """
    Build the pool of possible goal attribute dicts for the active dimensions.
    n_kinds maps each dim to how many of its values to use.
    Excluded is a set of frozensets of items representing already-used combos.
    """
    excluded = excluded or set()

    def combos(dims_left, current):
        if not dims_left:
            yield dict(current)
            return
        dim = dims_left[0]
        for v in DIMENSIONS[dim][:n_kinds[dim]]:
            yield from combos(dims_left[1:], current + [(dim, v)])

    return [
        g for g in combos(dims, [])
        if frozenset(g.items()) not in excluded
    ]


def _generate_goal_attrs(
    rng: random.Random,
    n_goals: int,
    dims: list[str],
    n_kinds: dict[str, int],
    mode: str,
    fixed_values: dict,
    excluded: set[frozenset] | None = None,
    rule_dim: str | None = None,
    rule_value: str | None = None,
) -> list[dict]:
    """
    Return n_goals goal attribute dicts.
    - Active dims vary across goals; inactive dims use fixed_values.
    - Bijection: each value in each active dim appears exactly once.
    - Random/no-repeat: sample freely from the pool (minus excluded).
    - No-repeat with rule_dim/rule_value: guarantees at least one rewarding goal
      unless all matching combos are exhausted (in which case falls back silently).
    """
    excluded = excluded or set()

    if mode == MODE_BIJECTION:
        # Sample n_goals values from each dim without replacement, then zip
        per_dim = [rng.sample(DIMENSIONS[d][:n_kinds[d]], n_goals) for d in dims]
        active_attrs = [
            {dims[i]: per_dim[i][j] for i in range(len(dims))}
            for j in range(n_goals)
        ]
    elif mode == MODE_NO_REPEAT and rule_dim is not None and rule_dim in dims:
        pool     = _active_pool(dims, n_kinds, excluded)
        matching = [g for g in pool if g.get(rule_dim) == rule_value]
        if matching:
            # Reserve one matching combo, fill the rest from the full pool
            forced       = rng.choice(matching)
            remaining    = [g for g in pool if g != forced]
            rest         = rng.sample(remaining, min(n_goals - 1, len(remaining)))
            active_attrs = [forced] + rest
        else:
            # All rewarding combos exhausted — sample freely
            active_attrs = rng.sample(pool, n_goals)
    else:
        pool = _active_pool(dims, n_kinds, excluded)
        active_attrs = rng.sample(pool, n_goals)

    goals = []
    for a in active_attrs:
        goal = dict(fixed_values)
        goal.update(a)
        goals.append(goal)
    return goals


# ---------------------------------------------------------------------------
# Episode initialisation
# ---------------------------------------------------------------------------

def _make_maze(
    rng: random.Random,
    n_goals: int,
    dims: list[str],
    n_kinds: dict[str, int],
    mode: str,
    excluded: set[frozenset] | None = None,
    rule_dim: str | None = None,
    rule_value: str | None = None,
) -> dict:
    grid  = make_fourroom_grid()
    rooms = get_room_cells(grid)

    # Place one goal per room first, then fill remaining slots from leftover free cells
    n_rooms = len(rooms)
    room_goals = [list(rng.choice(rooms[i])) for i in range(min(n_goals, n_rooms))]
    goal_set   = {tuple(p) for p in room_goals}
    extra = []
    if n_goals > n_rooms:
        all_free  = [c for c in get_free_cells(make_fourroom_grid()) if c not in goal_set]
        extra     = [list(c) for c in rng.sample(all_free, n_goals - n_rooms)]
        goal_set |= {tuple(p) for p in extra}
    goal_positions = room_goals + extra

    free      = [c for c in get_free_cells(grid) if c not in goal_set]
    agent_pos = list(rng.choice(free))

    # Fix a random value for each inactive dimension (use full list for inactive dims)
    inactive     = [d for d in DIMENSIONS if d not in dims]
    fixed_values = {d: rng.choice(DIMENSIONS[d]) for d in inactive}

    goals = _generate_goal_attrs(
        rng, n_goals, dims, n_kinds, mode, fixed_values, excluded, rule_dim, rule_value
    )
    rng.shuffle(goals)

    return {
        "grid":           grid.tolist(),
        "agent_pos":      agent_pos,
        "goal_positions": goal_positions,
        "goals":          goals,
        "fixed_values":   fixed_values,
        "visited_goals":  [],
        "current_goal":   None,
        "done":           False,
        "step":           0,
    }


def init_episode(
    seed: int | None = None,
    mode: str = MODE_RANDOM,
    dims: list[str] | None = None,
    n_kinds: dict[str, int] | None = None,
    n_goals: int = 4,
) -> dict:
    if dims is None:
        dims = list(DIMENSIONS.keys())

    # Default: 4 kinds per active dimension
    if n_kinds is None:
        n_kinds = {d: 4 for d in dims}

    # Validate
    for d in dims:
        if d not in DIMENSIONS:
            raise ValueError(f"Unknown dimension '{d}'. Valid: {list(DIMENSIONS.keys())}")
    if set(n_kinds.keys()) != set(dims):
        raise ValueError("n_kinds must have exactly one entry per dim in dims")
    if mode == MODE_BIJECTION:
        for d in dims:
            if n_goals > n_kinds[d]:
                raise ValueError(
                    f"Bijection mode requires n_goals <= n_kinds for every dimension. "
                    f"n_goals={n_goals} but n_kinds['{d}']={n_kinds[d]}"
                )
    n_free = len(get_free_cells(make_fourroom_grid()))
    if n_goals > n_free:
        raise ValueError(f"n_goals={n_goals} exceeds number of free cells ({n_free})")

    rng = random.Random(seed)

    rule_dim   = rng.choice(dims)
    rule_value = rng.choice(DIMENSIONS[rule_dim][:n_kinds[rule_dim]])

    if mode == MODE_NO_REPEAT:
        first_maze = _make_maze(
            rng, n_goals, dims, n_kinds, mode,
            excluded=set(), rule_dim=rule_dim, rule_value=rule_value,
        )
        mazes = [first_maze] + [None] * (N_MAZES - 1)
    else:
        mazes = [_make_maze(rng, n_goals, dims, n_kinds, mode) for _ in range(N_MAZES)]

    return {
        "mode":             mode,
        "dims":             dims,
        "n_kinds":          n_kinds,
        "n_goals":          n_goals,
        "rule_dim":         rule_dim,
        "rule_value":       rule_value,
        "current_maze_idx": 0,
        "mazes":            mazes,
        "rng_state":        rng.getstate(),
        "log":              [],
        "total_score":      0,
        "done":             False,
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

    new_maze              = dict(maze)
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

            goal   = maze["goals"][idx]
            reward = int(goal.get(state["rule_dim"]) == state["rule_value"])
            break

    if len(new_maze["visited_goals"]) == state["n_goals"]:
        new_maze["done"] = True

    new_mazes            = list(state["mazes"])
    new_mazes[maze_idx]  = new_maze

    new_log   = list(state["log"])
    new_score = state["total_score"]

    if visited_goal is not None:
        new_score += reward
        log_entry = {
            "maze_idx":  maze_idx,
            "goal_idx":  visited_goal,
            "reward":    reward,
            "maze_step": new_maze["step"],
        }
        log_entry.update(maze["goals"][visited_goal])
        new_log.append(log_entry)

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
    """Increment current_maze_idx. Sets done=True if all mazes complete.

    In no-repeat mode, generates the next maze using only rewarding combos
    visited in previous mazes as the exclusion set.
    """
    new_idx = state["current_maze_idx"] + 1
    if new_idx >= N_MAZES:
        return {**state, "current_maze_idx": new_idx, "done": True}

    new_mazes = list(state["mazes"])

    if state.get("mode") == MODE_NO_REPEAT and new_mazes[new_idx] is None:
        dims    = state["dims"]
        n_kinds = state["n_kinds"]   # dict[str, int]
        n_goals = state["n_goals"]
        excluded = {
            frozenset({d: e[d] for d in dims}.items())
            for e in state["log"] if e["reward"]
        }
        rng = random.Random()
        rng.setstate(state["rng_state"])
        new_mazes[new_idx] = _make_maze(
            rng, n_goals, dims, n_kinds, MODE_NO_REPEAT,
            excluded=excluded,
            rule_dim=state.get("rule_dim"),
            rule_value=state.get("rule_value"),
        )
        new_rng_state = rng.getstate()
    else:
        new_rng_state = state.get("rng_state")

    return {
        **state,
        "current_maze_idx": new_idx,
        "mazes":            new_mazes,
        "rng_state":        new_rng_state,
        "done":             False,
    }
