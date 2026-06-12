"""SVG renderer for the four-room fruit-collection gridworld.

Apples  — red circle with a small green stem
Bananas — yellow rotated ellipse
Grayed  — same shape in light grey once the agent has left the cell
Agent   — blue circle with an up-arrow indicator
"""

CELL        = 40
WALL_COLOR  = "#4a4a4a"
EMPTY_COLOR = "#f5f0e8"
GRAY_COLOR  = "#d1d5db"
AGENT_COLOR = "#2563eb"
GRID_SIZE   = 13

FRUIT_COLORS = {
    "apple":  {"fill": "#dc2626", "stroke": "#991b1b"},   # red
    "banana": {"fill": "#fbbf24", "stroke": "#b45309"},   # amber-yellow
}


def _apple_svg(cx: int, cy: int, fill: str, stroke: str) -> str:
    r = CELL // 2 - 5
    lines = [
        f'<circle cx="{cx}" cy="{cy}" r="{r}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>',
        # Stem
        f'<line x1="{cx}" y1="{cy - r}" x2="{cx}" y2="{cy - r - 5}" '
        f'stroke="#15803d" stroke-width="2" stroke-linecap="round"/>',
    ]
    return "\n".join(lines)


def _banana_svg(cx: int, cy: int, fill: str, stroke: str) -> str:
    rx, ry = CELL // 2 - 4, CELL // 2 - 9
    lines = [
        f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" '
        f'transform="rotate(-30 {cx} {cy})" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>',
    ]
    return "\n".join(lines)


def _fruit_svg(cx: int, cy: int, fruit_type: str, grayed: bool) -> str:
    if grayed:
        fill, stroke = GRAY_COLOR, "#9ca3af"
    else:
        c = FRUIT_COLORS[fruit_type]
        fill, stroke = c["fill"], c["stroke"]

    if fruit_type == "apple":
        return _apple_svg(cx, cy, fill, stroke)
    else:
        return _banana_svg(cx, cy, fill, stroke)


def render_svg(state: dict) -> str:
    grid        = state["grid"]
    agent       = state["agent_pos"]
    positions   = state["fruit_positions"]
    types       = state["fruit_types"]
    collected   = set(state["collected_fruits"])
    current     = state.get("current_fruit")

    size   = GRID_SIZE * CELL
    lines  = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{size}" height="{size}" '
        f'style="border:2px solid #333; display:block; margin:auto;">'
    ]

    # Grid cells
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            x, y  = c * CELL, r * CELL
            color = WALL_COLOR if grid[r][c] == 1 else EMPTY_COLOR
            lines.append(
                f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" '
                f'fill="{color}" stroke="#ccc" stroke-width="0.5"/>'
            )

    # Fruits
    for idx, (fr, fc) in enumerate(positions):
        cx = fc * CELL + CELL // 2
        cy = fr * CELL + CELL // 2
        # Gray out only after agent has LEFT the cell
        grayed = idx in collected and idx != current
        lines.append(_fruit_svg(cx, cy, types[idx], grayed))

    # Agent
    ar, ac = agent
    ax = ac * CELL + CELL // 2
    ay = ar * CELL + CELL // 2
    lines.append(
        f'<circle cx="{ax}" cy="{ay}" r="{CELL // 2 - 5}" '
        f'fill="{AGENT_COLOR}" stroke="white" stroke-width="2" zorder="5"/>'
    )
    lines.append(
        f'<text x="{ax}" y="{ay + 5}" text-anchor="middle" '
        f'font-size="14" fill="white">&#9650;</text>'
    )

    lines.append("</svg>")
    return "\n".join(lines)
