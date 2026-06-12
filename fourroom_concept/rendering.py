"""SVG renderer for the concept-learning gridworld.

Goals are drawn as colored shapes:
  circle   — filled circle
  square   — filled square
  triangle — filled upward-pointing triangle

Color palette:
  blue   → #3b82f6
  red    → #ef4444
  yellow → #eab308

Visited goals (agent has left the cell) are drawn in grey.
"""

CELL       = 40
WALL_COLOR  = "#4a4a4a"
EMPTY_COLOR = "#f5f0e8"
AGENT_COLOR = "#2563eb"
GRAY_FILL   = "#9ca3af"
GRAY_STROKE = "#6b7280"
GRID_SIZE   = 13

COLOR_FILL = {
    "blue":   "#3b82f6",
    "red":    "#ef4444",
    "yellow": "#eab308",
}
COLOR_STROKE = {
    "blue":   "#1d4ed8",
    "red":    "#b91c1c",
    "yellow": "#a16207",
}


def _goal_svg(cx: int, cy: int, shape: str, color: str, grayed: bool) -> str:
    fill   = GRAY_FILL   if grayed else COLOR_FILL[color]
    stroke = GRAY_STROKE if grayed else COLOR_STROKE[color]
    r = CELL // 2 - 6   # 14px

    if shape == "circle":
        return (
            f'<circle cx="{cx}" cy="{cy}" r="{r}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
        )
    elif shape == "square":
        return (
            f'<rect x="{cx - r}" y="{cy - r}" '
            f'width="{2 * r}" height="{2 * r}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
        )
    elif shape == "triangle":
        # Equilateral triangle pointing up
        pts = f"{cx},{cy - r}  {cx - r},{cy + r}  {cx + r},{cy + r}"
        return (
            f'<polygon points="{pts}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
        )
    return ""


def render_svg(maze: dict) -> str:
    grid      = maze["grid"]
    agent     = maze["agent_pos"]
    positions = maze["goal_positions"]
    shapes    = maze["goal_shapes"]
    colors    = maze["goal_colors"]
    visited   = set(maze["visited_goals"])
    current   = maze.get("current_goal")

    size  = GRID_SIZE * CELL
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{size}" height="{size}" '
        f'style="border:2px solid #333; display:block; margin:auto;">'
    ]

    # Cells
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            x, y  = c * CELL, r * CELL
            color = WALL_COLOR if grid[r][c] == 1 else EMPTY_COLOR
            lines.append(
                f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" '
                f'fill="{color}" stroke="#ccc" stroke-width="0.5"/>'
            )

    # Goals
    for idx, (gr, gc) in enumerate(positions):
        cx     = gc * CELL + CELL // 2
        cy     = gr * CELL + CELL // 2
        grayed = idx in visited and idx != current
        lines.append(_goal_svg(cx, cy, shapes[idx], colors[idx], grayed))

    # Agent
    ar, ac = agent
    ax = ac * CELL + CELL // 2
    ay = ar * CELL + CELL // 2
    lines.append(
        f'<circle cx="{ax}" cy="{ay}" r="{CELL // 2 - 5}" '
        f'fill="{AGENT_COLOR}" stroke="white" stroke-width="2"/>'
    )
    lines.append(
        f'<text x="{ax}" y="{ay + 5}" text-anchor="middle" '
        f'font-size="14" fill="white">&#9650;</text>'
    )

    lines.append("</svg>")
    return "\n".join(lines)
