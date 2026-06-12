"""SVG rendering for the four-room gridworld."""

CELL = 40          # px per grid cell
WALL_COLOR = "#4a4a4a"
EMPTY_COLOR = "#f5f0e8"
DOOR_COLOR = "#f5f0e8"
AGENT_COLOR = "#2563eb"   # blue
TRUE_GOAL_COLOR = "#16a34a"  # green (only visible after success)
GOAL_COLOR = "#f59e0b"    # amber for candidate goals
VISITED_COLOR = "#d1d5db"  # grey — already visited non-goal
FONT_COLOR = "#1e1b4b"
GRID_SIZE = 13


def render_svg(state: dict, show_true_goal: bool = False) -> str:
    grid = state["grid"]
    agent = state["agent_pos"]
    goals = state["goal_positions"]
    true_idx = state["true_goal_idx"]
    visited = set(state.get("visited_goals", []))

    size = GRID_SIZE * CELL
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{size}" height="{size}" '
        f'style="border: 2px solid #333; display:block; margin:auto;">'
    ]

    # Cells
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            x, y = c * CELL, r * CELL
            color = WALL_COLOR if grid[r][c] == 1 else EMPTY_COLOR
            lines.append(
                f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" '
                f'fill="{color}" stroke="#ccc" stroke-width="0.5"/>'
            )

    # Candidate goals
    for idx, (gr, gc) in enumerate(goals):
        cx = gc * CELL + CELL // 2
        cy = gr * CELL + CELL // 2
        r_circle = CELL // 2 - 4

        if show_true_goal and idx == true_idx:
            fill = TRUE_GOAL_COLOR
        elif idx in visited:
            fill = VISITED_COLOR
        else:
            fill = GOAL_COLOR

        lines.append(
            f'<circle cx="{cx}" cy="{cy}" r="{r_circle}" fill="{fill}" '
            f'stroke="#555" stroke-width="1.5"/>'
        )

    # Agent
    ar, ac = agent
    ax = ac * CELL + CELL // 2
    ay = ar * CELL + CELL // 2
    lines.append(
        f'<circle cx="{ax}" cy="{ay}" r="{CELL // 2 - 5}" '
        f'fill="{AGENT_COLOR}" stroke="white" stroke-width="2"/>'
    )
    # Arrow indicator on agent
    lines.append(
        f'<text x="{ax}" y="{ay + 5}" text-anchor="middle" '
        f'font-size="14" fill="white">&#9650;</text>'
    )

    lines.append("</svg>")
    return "\n".join(lines)
