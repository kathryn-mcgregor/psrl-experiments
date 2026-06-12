"""
Experiment stages for the four-room fruit-collection task.

Stages:
  1. Instructions — explain the task and reveal which fruit is worth more
  2. GridWorld    — navigate and collect all 14 fruits
  3. Debrief      — show final score and summary
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

APPLE_EMOJI  = "🍎"
BANANA_EMOJI = "🍌"


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _get_state() -> dict:
    return app.storage.user["gw_state"]


def _set_state(state: dict):
    app.storage.user["gw_state"] = state


async def _log_step(action: str, prev_state: dict, new_state: dict, info: dict):
    record = {
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "step":            new_state["step"],
        "action":          action,
        "prev_pos":        prev_state["agent_pos"],
        "new_pos":         new_state["agent_pos"],
        "moved":           info["moved"],
        "collected_fruit": info["collected_fruit"],
        "left_fruit":      info["left_fruit"],
        "score":           new_state["score"],
        "fruit_values":    new_state["fruit_values"],
    }
    async with aiofiles.open(user_data_file(), "ab") as f:
        await write_msgpack_record(f, record)


# ---------------------------------------------------------------------------
# Stage 1 — Instructions
# ---------------------------------------------------------------------------

async def instruction_display_fn(stage, container):
    nicewebrl.clear_element(container)

    with container.style("align-items: center; max-width: 680px;"):
        ui.markdown("## Fruit Collection Task")
        ui.markdown(
            f"""
Navigate the maze and collect **all 14 fruits** to complete the task.

The maze contains {APPLE_EMOJI} **apples** and {BANANA_EMOJI} **bananas**.
Each fruit type has a different point value — figure out which is worth more
as you play!

Your score updates each time you step onto a fruit.

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
# Stage 2 — GridWorld
# ---------------------------------------------------------------------------

def _scoreboard_html(state: dict) -> str:
    history = state.get("score_history", [])
    score   = state.get("score", 0)
    n_left  = gw.N_FRUITS - len(state.get("collected_fruits", []))

    rows = ""
    for entry in reversed(history):
        emoji = APPLE_EMOJI if entry["fruit_type"] == "apple" else BANANA_EMOJI
        rows += (
            f'<tr>'
            f'<td style="padding:2px 8px;">{emoji}</td>'
            f'<td style="padding:2px 8px; color:#16a34a; font-weight:bold;">'
            f'+{entry["points"]}</td>'
            f'<td style="padding:2px 8px; color:#6b7280; font-size:11px;">'
            f'step {entry["step"]}</td>'
            f'</tr>'
        )

    return f"""
    <div style="
        min-width:160px; max-width:180px;
        font-family: sans-serif;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 12px;
        background: #f9fafb;
        align-self: flex-start;
    ">
      <div style="font-size:18px; font-weight:bold; margin-bottom:8px;">
        Score: {score}
      </div>
      <div style="font-size:12px; color:#6b7280; margin-bottom:10px;">
        {n_left} fruit{"s" if n_left != 1 else ""} remaining
      </div>
      <div style="border-top:1px solid #e5e7eb; padding-top:8px;">
        <div style="font-size:11px; color:#9ca3af; margin-bottom:4px;">COLLECTED</div>
        <table style="border-collapse:collapse; width:100%;">
          {rows if rows else '<tr><td style="color:#9ca3af; font-size:12px;">none yet</td></tr>'}
        </table>
      </div>
    </div>
    """


async def gridworld_display_fn(stage, container):
    nicewebrl.clear_element(container)
    await ui.run_javascript("window.gridworld_active = true;")
    state = _get_state()

    with container.style("align-items: center;"):
        ui.markdown("### Collect all the fruits!")

        with ui.row().style("gap: 16px; align-items: flex-start; flex-wrap: nowrap; justify-content: center;"):
            # ---- Maze -------------------------------------------------------
            with ui.column().style("align-items: center;"):
                svg      = rendering.render_svg(state)
                grid_html = ui.html(svg)
                await stage.set_user_data(grid_html=grid_html)

                # On-screen arrow buttons
                with ui.grid(columns=3).style("gap:4px; justify-items:center; margin-top:4px;"):
                    ui.label("")
                    ui.button("▲", on_click=lambda: _handle_btn("ArrowUp",    stage, grid_html, scoreboard_html)).props("dense")
                    ui.label("")
                    ui.button("◄", on_click=lambda: _handle_btn("ArrowLeft",  stage, grid_html, scoreboard_html)).props("dense")
                    ui.label("")
                    ui.button("►", on_click=lambda: _handle_btn("ArrowRight", stage, grid_html, scoreboard_html)).props("dense")
                    ui.label("")
                    ui.button("▼", on_click=lambda: _handle_btn("ArrowDown",  stage, grid_html, scoreboard_html)).props("dense")
                    ui.label("")

            # ---- Scoreboard -------------------------------------------------
            scoreboard_html = ui.html(_scoreboard_html(state))

        await stage.set_user_data(scoreboard_html=scoreboard_html)

        ui.label("").bind_text_from(
            app.storage.user, "gw_state",
            lambda s: "🎉 All fruits collected!" if s.get("done") else ""
        )


def _handle_btn(action: str, stage, grid_html, scoreboard_html):
    asyncio.ensure_future(_apply_action(action, stage, grid_html, scoreboard_html))


async def _apply_action(action: str, stage, grid_html, scoreboard_html):
    prev_state = _get_state()
    new_state, info = gw.step(prev_state, action)
    _set_state(new_state)

    await _log_step(action, prev_state, new_state, info)

    grid_html.content       = rendering.render_svg(new_state)
    scoreboard_html.content = _scoreboard_html(new_state)

    if new_state["done"]:
        await ui.run_javascript("window.gridworld_active = false;")
        await stage.set_user_data(finished=True)
        logger.info(f"All fruits collected in {new_state['step']} steps. "
                    f"Final score: {new_state['score']}")
        local_cb = stage.get_user_data("local_handle_key_press")
        if local_cb is not None:
            await local_cb()


async def gridworld_handle_key_press(stage, e, container):
    action = e.args.get("key") if hasattr(e, "args") else str(e)
    if action not in gw.DIRECTIONS:
        return
    grid_html      = stage.get_user_data("grid_html")
    scoreboard_html = stage.get_user_data("scoreboard_html")
    if grid_html is None:
        return
    await _apply_action(action, stage, grid_html, scoreboard_html)


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
    state  = app.storage.user.get("gw_state", {})
    score  = state.get("score", 0)
    steps  = state.get("step",  0)
    values = state.get("fruit_values", {})
    history = state.get("score_history", [])

    n_apples  = sum(1 for e in history if e["fruit_type"] == "apple")
    n_bananas = sum(1 for e in history if e["fruit_type"] == "banana")
    apple_pts  = values.get("apple",  "?")
    banana_pts = values.get("banana", "?")

    with container.style("align-items: center;"):
        svg = rendering.render_svg(state)
        ui.html(
            f'<div style="filter: grayscale(1) opacity(0.35); pointer-events:none;">'
            f'{svg}</div>'
        )
        with ui.card().style(
            "margin-top:-20px; z-index:10; max-width:480px; "
            "text-align:center; padding:1.5rem; background:rgba(255,255,255,0.95);"
        ):
            ui.markdown("## Task complete — thank you!")
            ui.markdown(
                f"**Final score: {score} points** in {steps} steps\n\n"
                f"You collected {n_apples} {APPLE_EMOJI} "
                f"({apple_pts} pt each) "
                f"and {n_bananas} {BANANA_EMOJI} "
                f"({banana_pts} pt each).\n\n"
                "Your data has been saved. You may close this window."
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
experiment = SimpleExperiment(
    blocks=[Block(stages=[s]) for s in all_stages],
    name="FourRoom Fruit Collection",
)
