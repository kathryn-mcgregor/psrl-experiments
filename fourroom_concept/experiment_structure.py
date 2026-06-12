"""
Experiment stages for the four-room concept-learning task.

Stages:
  1. Instructions
  2. GridWorld  (4 mazes, handled within a single stage)
  3. Debrief    (reveals the hidden rule)
"""

import asyncio
from datetime import datetime, timezone

import aiofiles
from nicegui import app, ui

import nicewebrl
from nicewebrl import Stage, get_logger
from nicewebrl.stages import Block
from nicewebrl.experiment import SimpleExperiment
from nicewebrl.utils import write_msgpack_record, user_data_file

import gridworld as gw
import rendering

logger = get_logger(__name__)

# Unicode symbols used in the log panel
SHAPE_SYMBOL = {"circle": "●", "square": "■", "triangle": "▲"}
LOG_COLOR    = {"blue": "#3b82f6", "red": "#ef4444", "yellow": "#ca8a04"}


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _get_state() -> dict:
    return app.storage.user["gw_state"]


def _set_state(state: dict):
    app.storage.user["gw_state"] = state


async def _log_step(action: str, prev: dict, new: dict, info: dict):
    record = {
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "maze_idx":     new["current_maze_idx"],
        "maze_step":    new["mazes"][new["current_maze_idx"]]["step"],
        "action":       action,
        "prev_pos":     prev["mazes"][prev["current_maze_idx"]]["agent_pos"],
        "new_pos":      new["mazes"][new["current_maze_idx"]]["agent_pos"],
        "moved":        info["moved"],
        "visited_goal": info["visited_goal"],
        "left_goal":    info["left_goal"],
        "reward":       info["reward"],
    }
    async with aiofiles.open(user_data_file(), "ab") as f:
        await write_msgpack_record(f, record)


# ---------------------------------------------------------------------------
# Log panel HTML  (persists across all 4 mazes)
# ---------------------------------------------------------------------------

def _log_html(state: dict) -> str:
    log          = state.get("log", [])
    total_score  = state.get("total_score", 0)
    current_idx  = state.get("current_maze_idx", 0)

    sections = ""
    for maze_idx in range(gw.N_MAZES):
        maze     = state["mazes"][maze_idx]
        entries  = [e for e in log if e["maze_idx"] == maze_idx]
        is_done  = maze.get("done", False)
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
                symbol    = SHAPE_SYMBOL.get(e["shape"], "?")
                hex_color = LOG_COLOR.get(e["color"], "#000")
                r_str     = "+1" if e["reward"] else "0"
                r_color   = "#16a34a" if e["reward"] else "#9ca3af"
                rows += (
                    f'<tr>'
                    f'<td style="padding:2px 5px;">'
                    f'<span style="color:{hex_color}; font-size:15px;">{symbol}</span>'
                    f'</td>'
                    f'<td style="padding:2px 4px; font-size:11px; color:#374151;">'
                    f'{e["color"]} {e["shape"]}</td>'
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

async def instruction_display_fn(stage, container):
    nicewebrl.clear_element(container)
    with container.style("align-items: center; max-width: 680px;"):
        ui.markdown("## Shape & Color Navigation Task")
        ui.markdown(
            f"""
You will navigate through **{gw.N_MAZES} mazes** in sequence.

Each maze contains **{gw.N_GOALS} candidate goals**, each a different
combination of shape and color:

| Shapes | Colors |
|--------|--------|
| ● Circle | 🔵 Blue |
| ■ Square | 🔴 Red |
| ▲ Triangle | 🟡 Yellow |

There is a **hidden rule** — either all of one shape or all of one color
is worth a reward of **+1**. All other goals are worth **0**.

Navigate to goals to discover the rule. Your results are logged on
the right and **carry over between mazes**.

**Controls:** arrow keys or the on-screen buttons.

Press **Start** when you are ready.
"""
        )


instruction_stage = Stage(
    name="Instructions",
    display_fn=instruction_display_fn,
    next_button=True,
)


# ---------------------------------------------------------------------------
# Stage 2 — GridWorld  (all 4 mazes handled here)
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

            # ---- Log panel + Continue button -------------------------------
            with ui.column().style("align-items:stretch;"):
                log_html = ui.html(_log_html(state))

                # "Continue" button — shown below the log when maze unlocks
                next_btn = (
                    ui.button(
                        "Continue to Maze 2 →",
                        on_click=lambda: _handle_next_maze(
                            stage, grid_html, log_html, maze_label, next_btn
                        )
                    )
                    .props("color=primary")
                    .style("display:none; margin-top:10px; width:100%;")
                )

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

    # Show Continue button as soon as a rewarding goal is found.
    # Fall back to showing it when all 4 goals are visited (in case no goal
    # in this maze matches the rule).
    should_show_btn = info["reward"] == 1 or maze["done"]
    if should_show_btn and not stage.get_user_data("btn_shown", False):
        await stage.set_user_data(btn_shown=True)
        await ui.run_javascript("window.gridworld_active = false;")

        # Gray out the maze
        current_svg = rendering.render_svg(maze)
        grid_html.content = (
            f'<div style="filter:grayscale(1) opacity(0.4); pointer-events:none;">'
            f'{current_svg}</div>'
        )

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
        # All mazes complete — finish stage
        await stage.set_user_data(finished=True)
        local_cb = stage.get_user_data("local_handle_key_press")
        if local_cb is not None:
            await local_cb()
        return

    new_maze_idx = new_state["current_maze_idx"]
    new_maze     = new_state["mazes"][new_maze_idx]

    grid_html.content = rendering.render_svg(new_maze)   # ungrayed fresh maze
    log_html.content  = _log_html(new_state)
    maze_label.text   = f"Maze {new_maze_idx + 1} of {gw.N_MAZES}"
    next_btn.style("display:none;")

    # Reset the per-maze button flag for the new maze
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
    rule_type  = state.get("rule_type", "?")
    rule_value = state.get("rule_value", "?")
    total      = state.get("total_score", 0)
    log        = state.get("log", [])
    n_steps    = sum(m["step"] for m in state.get("mazes", []))

    symbol = SHAPE_SYMBOL.get(rule_value, "")
    color_hex = LOG_COLOR.get(rule_value, "#000")
    rule_display = (
        f'<span style="color:{color_hex}; font-size:1.2em;">{symbol} {rule_value}</span>'
        if rule_type == "shape"
        else f'<span style="color:{color_hex}; font-size:1.2em;">{rule_value}</span>'
    )

    with container.style("align-items: center;"):
        # Faded final maze in background
        last_maze = state["mazes"][-1]
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
                f'<p>The hidden rule was: all <strong>{rule_type}s</strong> '
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
