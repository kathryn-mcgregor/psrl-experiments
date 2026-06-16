"""SVG renderer for the concept-learning gridworld.

Goals are drawn as colored shapes with textures.
- To add a new shape: add a draw function and register it in SHAPE_DRAW.
- To add a new color: add entries to COLOR_FILL and COLOR_STROKE.
- To add a new texture: add a branch in _make_fill() and a <defs> entry in _make_defs().
"""

import math

CELL        = 40
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
    "green":  "#22c55e",
    "purple": "#a855f7",
    "orange": "#f97316",
    "pink":   "#ec4899",
}
COLOR_STROKE = {
    "blue":   "#1d4ed8",
    "red":    "#b91c1c",
    "yellow": "#a16207",
    "green":  "#15803d",
    "purple": "#7c3aed",
    "orange": "#c2410c",
    "pink":   "#be185d",
}


# ---------------------------------------------------------------------------
# Texture helpers
# ---------------------------------------------------------------------------

def _pattern_id(color: str, texture: str) -> str:
    return f"pat-{color}-{texture}"


def _make_defs(color: str, texture: str, fill_color: str) -> str:
    """Return a <pattern> SVG snippet for pattern-based textures, or '' for solid/outline."""
    pid = _pattern_id(color, texture)

    if texture == "striped":
        return (
            f'<pattern id="{pid}" patternUnits="userSpaceOnUse" '
            f'width="5" height="5" patternTransform="rotate(45)">'
            f'<line x1="0" y1="0" x2="0" y2="5" '
            f'stroke="{fill_color}" stroke-width="2.5"/>'
            f'</pattern>'
        )
    elif texture == "dotted":
        return (
            f'<pattern id="{pid}" patternUnits="userSpaceOnUse" width="6" height="6">'
            f'<circle cx="3" cy="3" r="1.5" fill="{fill_color}"/>'
            f'</pattern>'
        )
    elif texture == "chevron":
        return (
            f'<pattern id="{pid}" patternUnits="userSpaceOnUse" width="8" height="8">'
            f'<polyline points="0,4 4,0 8,4" fill="none" '
            f'stroke="{fill_color}" stroke-width="1.5"/>'
            f'<polyline points="0,8 4,4 8,8" fill="none" '
            f'stroke="{fill_color}" stroke-width="1.5"/>'
            f'</pattern>'
        )
    return ""


def _make_fill(color: str, texture: str, fill_color: str, grayed: bool) -> tuple[str, str]:
    """
    Return (fill_value, stroke_width) appropriate for the texture.
    fill_value is either a hex color or url(#pattern-id).
    """
    if grayed:
        return GRAY_FILL, "1.5"

    if texture == "solid":
        return fill_color, "1.5"
    elif texture == "outline":
        return "none", "3"
    else:
        return f"url(#{_pattern_id(color, texture)})", "1.5"


# ---------------------------------------------------------------------------
# Shape draw functions — each returns an SVG string
# ---------------------------------------------------------------------------

def _draw_circle(cx, cy, r, fill, stroke, sw):
    return (
        f'<circle cx="{cx}" cy="{cy}" r="{r}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )

def _draw_square(cx, cy, r, fill, stroke, sw):
    return (
        f'<rect x="{cx - r}" y="{cy - r}" '
        f'width="{2 * r}" height="{2 * r}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )

def _draw_triangle(cx, cy, r, fill, stroke, sw):
    pts = f"{cx},{cy - r}  {cx - r},{cy + r}  {cx + r},{cy + r}"
    return (
        f'<polygon points="{pts}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )

def _draw_star(cx, cy, r, fill, stroke, sw):
    outer, inner = r, int(r * 0.4)
    pts_list = []
    for i in range(10):
        angle = math.pi / 5 * i - math.pi / 2
        rad   = outer if i % 2 == 0 else inner
        pts_list.append(f"{cx + int(rad * math.cos(angle))},{cy + int(rad * math.sin(angle))}")
    pts = " ".join(pts_list)
    return (
        f'<polygon points="{pts}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )

def _draw_pentagon(cx, cy, r, fill, stroke, sw):
    pts_list = []
    for i in range(5):
        angle = math.pi / 2.5 * i - math.pi / 2
        pts_list.append(f"{cx + int(r * math.cos(angle))},{cy + int(r * math.sin(angle))}")
    pts = " ".join(pts_list)
    return (
        f'<polygon points="{pts}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )

def _draw_hexagon(cx, cy, r, fill, stroke, sw):
    pts_list = []
    for i in range(6):
        angle = math.pi / 3 * i - math.pi / 6
        pts_list.append(f"{cx + int(r * math.cos(angle))},{cy + int(r * math.sin(angle))}")
    pts = " ".join(pts_list)
    return (
        f'<polygon points="{pts}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )

def _draw_diamond(cx, cy, r, fill, stroke, sw):
    pts = f"{cx},{cy - r}  {cx + r},{cy}  {cx},{cy + r}  {cx - r},{cy}"
    return (
        f'<polygon points="{pts}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )

# Registry: add new shapes here
SHAPE_DRAW = {
    "circle":   _draw_circle,
    "square":   _draw_square,
    "triangle": _draw_triangle,
    "star":     _draw_star,
    "pentagon": _draw_pentagon,
    "hexagon":  _draw_hexagon,
    "diamond":  _draw_diamond,
}


# ---------------------------------------------------------------------------
# Goal SVG — returns (defs_str, shape_str)
# ---------------------------------------------------------------------------

def _goal_svg(cx: int, cy: int, goal: dict, grayed: bool) -> tuple[str, str]:
    shape   = goal.get("shape", "circle")
    color   = goal.get("color", "blue")
    texture = goal.get("texture", "solid")
    r       = CELL // 2 - 6   # 14px

    fill_color  = COLOR_FILL.get(color, "#888")
    stroke_color = GRAY_STROKE if grayed else COLOR_STROKE.get(color, "#555")

    fill, sw = _make_fill(color, texture, fill_color, grayed)
    defs     = "" if grayed else _make_defs(color, texture, fill_color)

    draw_fn = SHAPE_DRAW.get(shape)
    if draw_fn is None:
        return "", ""
    return defs, draw_fn(cx, cy, r, fill, stroke_color, sw)


# ---------------------------------------------------------------------------
# Full maze SVG
# ---------------------------------------------------------------------------

def render_svg(maze: dict) -> str:
    grid      = maze["grid"]
    agent     = maze["agent_pos"]
    positions = maze["goal_positions"]
    goals     = maze["goals"]
    visited   = set(maze["visited_goals"])
    current   = maze.get("current_goal")

    size = GRID_SIZE * CELL

    # Collect defs and goal shapes separately
    all_defs  = {}   # pid → def string (deduplicated)
    goal_svgs = []
    for idx, (gr, gc) in enumerate(positions):
        cx     = gc * CELL + CELL // 2
        cy     = gr * CELL + CELL // 2
        grayed = idx in visited and idx != current
        defs_str, shape_str = _goal_svg(cx, cy, goals[idx], grayed)
        if defs_str:
            pid = _pattern_id(goals[idx].get("color", ""), goals[idx].get("texture", ""))
            all_defs[pid] = defs_str
        goal_svgs.append(shape_str)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{size}" height="{size}" '
        f'style="border:2px solid #333; display:block; margin:auto;">'
    ]

    if all_defs:
        lines.append("<defs>" + "".join(all_defs.values()) + "</defs>")

    # Cells
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            x, y  = c * CELL, r * CELL
            color = WALL_COLOR if grid[r][c] == 1 else EMPTY_COLOR
            lines.append(
                f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" '
                f'fill="{color}" stroke="#ccc" stroke-width="0.5"/>'
            )

    lines.extend(goal_svgs)

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
