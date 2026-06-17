"""
Experiment stages for the four-room concept-learning task.

Stages:
  1. Instructions
  2. GridWorld  (N_MAZES mazes, handled within a single stage)
  3. Debrief    (reveals the hidden rule)
"""

import asyncio
from datetime import datetime, timezone

from nicegui import app, ui

import nicewebrl
from nicewebrl import Stage, get_logger
from nicewebrl.stages import Block
from nicewebrl.experiment import SimpleExperiment

import gridworld as gw
import rendering
import db

logger = get_logger(__name__)

# Display helpers — extend these when adding new shapes/colors
SHAPE_SYMBOL = {
    "circle":   "●",
    "square":   "■",
    "triangle": "▲",
    "star":     "★",
    "pentagon": "⬠",
    "hexagon":  "⬡",
    "diamond":  "◆",
}
DIM_COLOR = {
    "color": {
        "blue":   "#3b82f6",
        "red":    "#ef4444",
        "yellow": "#ca8a04",
        "green":  "#16a34a",
        "purple": "#9333ea",
        "orange": "#ea580c",
        "pink":   "#db2777",
    },
    "shape": {
        "circle":   "#6b7280",
        "square":   "#6b7280",
        "triangle": "#6b7280",
        "star":     "#6b7280",
        "pentagon": "#6b7280",
        "hexagon":  "#6b7280",
        "diamond":  "#6b7280",
    },
    "texture": {
        "solid":   "#6b7280",
        "striped": "#6b7280",
        "dotted":  "#6b7280",
        "outline": "#6b7280",
        "chevron": "#6b7280",
    },
}


def _goal_label(goal: dict, dims: list[str]) -> str:
    """Short human-readable label for a goal, e.g. 'blue circle'."""
    return " ".join(str(goal.get(d, "?")) for d in dims)


def _goal_symbol(goal: dict) -> str:
    shape = goal.get("shape")
    return SHAPE_SYMBOL.get(shape, "?") if shape else "●"


def _goal_hex_color(goal: dict) -> str:
    color = goal.get("color")
    if color:
        return DIM_COLOR.get("color", {}).get(color, "#374151")
    return "#374151"


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _get_state() -> dict:
    return app.storage.user["gw_state"]


def _set_state(state: dict):
    app.storage.user["gw_state"] = state


async def _log_step(action: str, prev: dict, new: dict, info: dict):
    maze_idx = new["current_maze_idx"]
    await db.insert_step(
        seed         = app.storage.user.get("seed"),
        timestamp    = datetime.now(timezone.utc).isoformat(),
        maze_idx     = maze_idx,
        maze_step    = new["mazes"][maze_idx]["step"],
        action       = action,
        prev_pos     = str(prev["mazes"][prev["current_maze_idx"]]["agent_pos"]),
        new_pos      = str(new["mazes"][maze_idx]["agent_pos"]),
        moved        = info["moved"],
        visited_goal = info["visited_goal"],
        left_goal    = info["left_goal"],
        reward       = info["reward"],
    )


# ---------------------------------------------------------------------------
# Log panel HTML  (persists across all mazes)
# ---------------------------------------------------------------------------

def _log_html(state: dict) -> str:
    log         = state.get("log", [])
    total_score = state.get("total_score", 0)
    current_idx = state.get("current_maze_idx", 0)
    dims        = state.get("dims", list(gw.DIMENSIONS.keys()))

    sections = ""
    for maze_idx in range(gw.N_MAZES):
        maze      = state["mazes"][maze_idx]
        entries   = [e for e in log if e["maze_idx"] == maze_idx]
        is_done   = maze.get("done", False) if maze is not None else False
        is_active = maze_idx == current_idx

        if maze_idx < current_idx or is_done:
            status_dot = '<span style="color:#16a34a;">✓</span>'
        elif is_active:
            status_dot = '<span style="color:#f59e0b;">→</span>'
        else:
            status_dot = ""

        section = (
            f'<div style="font-size:11px; font-weight:600; color:#6b7280; '
            f'margin:{("10px" if maze_idx > 0 else "0")} 0 3px;">'
            f'MAZE {maze_idx + 1} {status_dot}</div>'
        )

        if entries:
            rows = ""
            for e in entries:
                symbol    = _goal_symbol(e)
                hex_color = _goal_hex_color(e)
                label     = _goal_label(e, dims)
                r_str     = "+1" if e["reward"] else "0"
                r_color   = "#16a34a" if e["reward"] else "#9ca3af"
                rows += (
                    f'<tr>'
                    f'<td style="padding:2px 5px;">'
                    f'<span style="color:{hex_color}; font-size:15px;">{symbol}</span>'
                    f'</td>'
                    f'<td style="padding:2px 4px; font-size:11px; color:#374151;">'
                    f'{label}</td>'
                    f'<td style="padding:2px 5px; font-weight:bold; color:{r_color};">'
                    f'{r_str}</td>'
                    f'<td style="padding:2px 4px; font-size:10px; color:#9ca3af;">'
                    f's{e["maze_step"]}</td>'
                    f'</tr>'
                )
            section += f'<table style="border-collapse:collapse;">{rows}</table>'
        else:
            section += (
                '<div style="font-size:11px; color:#d1d5db; padding:2px 0;">'
                'none yet</div>'
            )

        sections += section

    return f"""
    <div style="
        min-width: 190px; max-width: 220px;
        font-family: sans-serif;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 12px;
        background: #f9fafb;
        align-self: flex-start;
    ">
      <div style="font-size:17px; font-weight:bold; margin-bottom:10px;">
        Score: {total_score}
      </div>
      <div style="border-top:1px solid #e5e7eb; padding-top:8px;">
        {sections}
      </div>
    </div>
    """


# ---------------------------------------------------------------------------
# Stage 1 — Instructions
# ---------------------------------------------------------------------------

def _instructions_table(dims: list[str], n_kinds: dict[str, int]) -> str:
    """Build a markdown table showing the active dimension values."""
    headers   = " | ".join(d.capitalize() for d in dims)
    separator = " | ".join("---" for _ in dims)
    max_rows  = max(n_kinds[d] for d in dims)
    rows = []
    for i in range(max_rows):
        cells = []
        for d in dims:
            vals = gw.DIMENSIONS[d][:n_kinds[d]]
            if i < len(vals):
                v   = vals[i]
                sym = SHAPE_SYMBOL.get(v, "")
                cells.append(f"{sym} {v}".strip() if sym else v)
            else:
                cells.append("")
        rows.append(" | ".join(cells))
    return f"| {headers} |\n|{separator}|\n" + "\n".join(f"| {r} |" for r in rows)


async def instruction_display_fn(stage, container):
    nicewebrl.clear_element(container)
    state   = _get_state()
    dims    = state.get("dims", list(gw.DIMENSIONS.keys()))
    n_kinds = state.get("n_kinds", 4)
    n_goals = state.get("n_goals", 4)
    table   = _instructions_table(dims, n_kinds)

    with container.style("align-items: center; max-width: 680px;"):
        ui.markdown("## Shape & Color Navigation Task")
        ui.markdown(
            f"You will navigate through **{gw.N_MAZES} mazes** in sequence.\n\n"
            f"Each maze contains **{n_goals} candidate goals**. "
            f"Goals vary in: **{', '.join(dims)}**.\n\n"
            f"{table}\n\n"
            "There is a **hidden rule** that makes some goals worth a reward of **+1**. "
            "All other goals are worth **0**.\n\n"
            "Navigate to goals to discover the rule. Your results are logged on "
            "the right and **carry over between mazes**.\n\n"
            "**Controls:** arrow keys or the on-screen buttons.\n\n"
            "Press **Start** when you are ready."
        )


instruction_stage = Stage(
    name="Instructions",
    display_fn=instruction_display_fn,
    next_button=True,
)


# ---------------------------------------------------------------------------
# Stage 2 — GridWorld  (all mazes handled here)
# ---------------------------------------------------------------------------

async def gridworld_display_fn(stage, container):
    nicewebrl.clear_element(container)
    await ui.run_javascript("window.gridworld_active = true;")
    state     = _get_state()
    maze_idx  = state["current_maze_idx"]
    maze      = state["mazes"][maze_idx]

    with container.style("align-items: center;"):
        maze_label = ui.label(f"Maze {maze_idx + 1} of {gw.N_MAZES}").style(
            "font-size:16px; font-weight:600; margin-bottom:6px;"
        )

        with ui.row().style(
            "gap:16px; align-items:flex-start; flex-wrap:nowrap; justify-content:center;"
        ):
            # ---- Maze + controls -------------------------------------------
            with ui.column().style("align-items:center;"):
                grid_html = ui.html(rendering.render_svg(maze))

                with ui.grid(columns=3).style(
                    "gap:4px; justify-items:center; margin-top:6px;"
                ):
                    ui.label("")
                    ui.button(
                        "▲",
                        on_click=lambda: _handle_btn(
                            "ArrowUp", stage, grid_html, log_html, maze_label, next_btn
                        )
                    ).props("dense")
                    ui.label("")
                    ui.button(
                        "◄",
                        on_click=lambda: _handle_btn(
                            "ArrowLeft", stage, grid_html, log_html, maze_label, next_btn
                        )
                    ).props("dense")
                    ui.label("")
                    ui.button(
                        "►",
                        on_click=lambda: _handle_btn(
                            "ArrowRight", stage, grid_html, log_html, maze_label, next_btn
                        )
                    ).props("dense")
                    ui.label("")
                    ui.button(
                        "▼",
                        on_click=lambda: _handle_btn(
                            "ArrowDown", stage, grid_html, log_html, maze_label, next_btn
                        )
                    ).props("dense")
                    ui.label("")

                # "Continue" button — shown only between mazes
                next_btn = (
                    ui.button(
                        "Continue to Maze 2 →",
                        on_click=lambda: _handle_next_maze(
                            stage, grid_html, log_html, maze_label, next_btn
                        )
                    )
                    .props("color=primary")
                    .style("display:none; margin-top:10px;")
                )

            # ---- Log panel -------------------------------------------------
            log_html = ui.html(_log_html(state))

        await stage.set_user_data(
            grid_html=grid_html,
            log_html=log_html,
            maze_label=maze_label,
            next_btn=next_btn,
        )


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------

async def _handle_btn(action, stage, grid_html, log_html, maze_label, next_btn):
    await _apply_action(action, stage, grid_html, log_html, maze_label, next_btn)


async def _handle_next_maze(stage, grid_html, log_html, maze_label, next_btn):
    await _do_advance_maze(stage, grid_html, log_html, maze_label, next_btn)


async def _apply_action(action, stage, grid_html, log_html, maze_label, next_btn):
    prev_state = _get_state()
    new_state, info = gw.step(prev_state, action)
    _set_state(new_state)

    await _log_step(action, prev_state, new_state, info)

    maze = new_state["mazes"][new_state["current_maze_idx"]]
    grid_html.content = rendering.render_svg(maze)

    if info["visited_goal"] is not None:
        log_html.content = _log_html(new_state)

    should_show_btn = info["reward"] == 1 or maze["done"]
    if should_show_btn and not stage.get_user_data("btn_shown", False):
        await stage.set_user_data(btn_shown=True)
        await ui.run_javascript("window.gridworld_active = false;")
        next_maze_idx = new_state["current_maze_idx"] + 1
        label = (
            "Complete experiment →"
            if next_maze_idx >= gw.N_MAZES
            else f"Continue to Maze {next_maze_idx + 1} →"
        )
        next_btn.text = label
        next_btn.style("display:block;")


async def _do_advance_maze(stage, grid_html, log_html, maze_label, next_btn):
    await ui.run_javascript("window.gridworld_active = false;")
    state     = _get_state()
    new_state = gw.advance_maze(state)
    _set_state(new_state)

    if new_state["done"]:
        await stage.set_user_data(finished=True)
        local_cb = stage.get_user_data("local_handle_key_press")
        if local_cb is not None:
            await local_cb()
        return

    new_maze_idx = new_state["current_maze_idx"]
    new_maze     = new_state["mazes"][new_maze_idx]

    grid_html.content = rendering.render_svg(new_maze)
    log_html.content  = _log_html(new_state)
    maze_label.text   = f"Maze {new_maze_idx + 1} of {gw.N_MAZES}"
    next_btn.style("display:none;")

    await stage.set_user_data(btn_shown=False)
    await ui.run_javascript("window.gridworld_active = true;")


async def gridworld_handle_key_press(stage, e, container):
    action = e.args.get("key") if hasattr(e, "args") else str(e)
    if action not in gw.DIRECTIONS:
        return
    grid_html  = stage.get_user_data("grid_html")
    log_html   = stage.get_user_data("log_html")
    maze_label = stage.get_user_data("maze_label")
    next_btn   = stage.get_user_data("next_btn")
    if grid_html is None:
        return
    await _apply_action(action, stage, grid_html, log_html, maze_label, next_btn)


gridworld_stage = Stage(
    name="GridWorld",
    display_fn=gridworld_display_fn,
    next_button=False,
)
gridworld_stage._key_handler = gridworld_handle_key_press


# ---------------------------------------------------------------------------
# Stage 3 — Debrief
# ---------------------------------------------------------------------------

async def debrief_display_fn(stage, container):
    nicewebrl.clear_element(container)
    state      = app.storage.user.get("gw_state", {})
    rule_dim   = state.get("rule_dim", "?")
    rule_value = state.get("rule_value", "?")
    total      = state.get("total_score", 0)
    log        = state.get("log", [])
    n_steps    = sum(m["step"] for m in state.get("mazes", []) if m is not None)

    symbol    = SHAPE_SYMBOL.get(rule_value, "")
    color_hex = DIM_COLOR.get(rule_dim, {}).get(rule_value, "#000")
    rule_display = (
        f'<span style="color:{color_hex}; font-size:1.2em;">{symbol} {rule_value}</span>'
        if symbol
        else f'<span style="color:{color_hex}; font-size:1.2em;">{rule_value}</span>'
    )

    with container.style("align-items: center;"):
        last_maze = next((m for m in reversed(state["mazes"]) if m is not None), None)
        if last_maze is None:
            last_maze = state["mazes"][state.get("current_maze_idx", 0)]
        ui.html(
            f'<div style="filter:grayscale(1) opacity(0.3); pointer-events:none;">'
            f'{rendering.render_svg(last_maze)}</div>'
        )
        with ui.card().style(
            "margin-top:-20px; z-index:10; max-width:500px; "
            "text-align:center; padding:1.5rem; background:rgba(255,255,255,0.95);"
        ):
            ui.markdown("## All mazes complete — thank you!")
            ui.html(
                f'<p>The hidden rule was: all <strong>{rule_dim}s</strong> '
                f'matching {rule_display}.</p>'
                f'<p><strong>Score: {total}</strong> across {len(log)} goals tested '
                f'in {n_steps} total steps.</p>'
                f'<p>Your data has been saved. You may close this window.</p>'
            )

    await stage.finish_stage()


debrief_stage = Stage(
    name="Debrief",
    display_fn=debrief_display_fn,
    next_button=False,
)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

all_stages = [instruction_stage, gridworld_stage, debrief_stage]
experiment  = SimpleExperiment(
    blocks=[Block(stages=[s]) for s in all_stages],
    name="FourRoom Concept Learning",
)
