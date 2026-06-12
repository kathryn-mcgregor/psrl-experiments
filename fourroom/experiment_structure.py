"""
Experiment definition for the four-room goal-navigation task.

Stages:
  1. Instructions
  2. GridWorld navigation (ends when true goal reached)
  3. Debrief
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_state() -> dict:
    return app.storage.user["gw_state"]


def _set_state(state: dict):
    app.storage.user["gw_state"] = state


async def _log_step(action: str, prev_state: dict, new_state: dict, info: dict):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "step": new_state["step"],
        "action": action,
        "prev_pos": prev_state["agent_pos"],
        "new_pos": new_state["agent_pos"],
        "moved": info["moved"],
        "visited_goal": info["visited_goal"],
        "reached_true": info["reached_true"],
        # stored so analysts can verify correctness offline
        "true_goal_idx": new_state["true_goal_idx"],
    }
    filepath = user_data_file()
    async with aiofiles.open(filepath, "ab") as f:
        await write_msgpack_record(f, record)


# ---------------------------------------------------------------------------
# Stage 1 – Instructions
# ---------------------------------------------------------------------------

async def instruction_display_fn(stage, container):
    nicewebrl.clear_element(container)
    with container.style("align-items: center; max-width: 700px;"):
        ui.markdown("## Navigation Task")
        ui.markdown(
            """
You will be placed in a **gridworld** made up of four connected rooms.

**Your goal:** navigate to a specific hidden target location.

- Use the **arrow keys** (or the on-screen buttons) to move.
- The **amber/yellow circles** are **candidate goal locations** — the true goal
  is one of them, but you don't know which one.
- The experiment ends automatically when you reach the correct goal.
- All of your moves are recorded.

Press **Start** when you are ready.
"""
        )


instruction_stage = Stage(
    name="Instructions",
    display_fn=instruction_display_fn,
    next_button=True,
)


# ---------------------------------------------------------------------------
# Stage 2 – GridWorld
# ---------------------------------------------------------------------------

async def gridworld_display_fn(stage, container):
    nicewebrl.clear_element(container)
    await ui.run_javascript("window.gridworld_active = true;")
    state = _get_state()

    with container.style("align-items: center;"):
        ui.markdown("### Find the hidden goal — navigate to it!")

        with ui.row().style("gap: 8px; margin-bottom: 4px;"):
            ui.label().bind_text_from(
                app.storage.user, "gw_state",
                lambda s: f"Steps taken: {s['step']}"
            )
            ui.label().bind_text_from(
                app.storage.user, "gw_state",
                lambda s: f"  |  Goals visited: {len(s['visited_goals'])}/{gw.N_CANDIDATE_GOALS}"
            )

        # Grid SVG
        svg = rendering.render_svg(state, show_true_goal=state["done"])
        grid_html = ui.html(svg).style("margin: 8px 0;")

        # Persist reference so key-press handler can refresh it
        await stage.set_user_data(grid_html=grid_html)

        # On-screen directional buttons
        with ui.grid(columns=3).style("gap: 4px; justify-items: center;"):
            ui.label("")
            ui.button("▲", on_click=lambda: _handle_btn("ArrowUp", stage, grid_html)).props("dense")
            ui.label("")
            ui.button("◄", on_click=lambda: _handle_btn("ArrowLeft", stage, grid_html)).props("dense")
            ui.label("")
            ui.button("►", on_click=lambda: _handle_btn("ArrowRight", stage, grid_html)).props("dense")
            ui.label("")
            ui.button("▼", on_click=lambda: _handle_btn("ArrowDown", stage, grid_html)).props("dense")
            ui.label("")

        ui.label("").bind_text_from(
            app.storage.user, "gw_state",
            lambda s: "🎉 Goal reached! Well done." if s["done"] else ""
        )


def _handle_btn(action: str, stage, grid_html):
    """Sync wrapper – schedules the async handler."""
    asyncio.ensure_future(_apply_action(action, stage, grid_html))


async def _apply_action(action: str, stage, grid_html):
    prev_state = _get_state()
    new_state, info = gw.step(prev_state, action)
    _set_state(new_state)

    await _log_step(action, prev_state, new_state, info)

    # Refresh only the SVG element
    svg = rendering.render_svg(new_state, show_true_goal=new_state["done"])
    grid_html.content = svg

    if new_state["done"]:
        await ui.run_javascript("window.gridworld_active = false;")
        await stage.set_user_data(finished=True)
        logger.info(
            f"True goal reached at step {new_state['step']} "
            f"(goal #{new_state['true_goal_idx'] + 1})"
        )
        local_cb = stage.get_user_data("local_handle_key_press")
        if local_cb is not None:
            await local_cb()


# Custom key-press handler wired in web_app.py
async def gridworld_handle_key_press(stage, e, container):
    action = e.args.get("key") if hasattr(e, "args") else str(e)
    if action not in gw.DIRECTIONS:
        return
    grid_html = stage.get_user_data("grid_html")
    if grid_html is None:
        return
    await _apply_action(action, stage, grid_html)


gridworld_stage = Stage(
    name="GridWorld",
    display_fn=gridworld_display_fn,
    next_button=False,
)
# Attach the custom key handler
gridworld_stage._key_handler = gridworld_handle_key_press


# ---------------------------------------------------------------------------
# Stage 3 – Debrief
# ---------------------------------------------------------------------------

async def debrief_display_fn(stage, container):
    nicewebrl.clear_element(container)
    state = app.storage.user.get("gw_state", {})
    n_steps = state.get("step", "?")
    true_idx = state.get("true_goal_idx", "?")

    with container.style("align-items: center;"):
        # Grayed-out maze in the background
        svg = rendering.render_svg(state, show_true_goal=True)
        ui.html(
            f'<div style="filter: grayscale(1) opacity(0.35); pointer-events: none;">'
            f'{svg}</div>'
        )

        # Debrief card overlaid below (floated on top visually via margin-top)
        with ui.card().style(
            "margin-top: -20px; z-index: 10; max-width: 480px; "
            "text-align: center; padding: 1.5rem; background: rgba(255,255,255,0.95);"
        ):
            ui.markdown("## Task complete — thank you!")
            if isinstance(true_idx, int):
                ui.markdown(
                    f"The hidden goal was **Goal #{true_idx + 1}**.  \n"
                    f"You found it in **{n_steps} steps**."
                )
            ui.markdown("Your data has been saved. You may close this window.")

    # Mark stage done immediately so run_stage can advance the experiment
    await stage.finish_stage()


debrief_stage = Stage(
    name="Debrief",
    display_fn=debrief_display_fn,
    next_button=False,
)


# ---------------------------------------------------------------------------
# Assemble experiment
# ---------------------------------------------------------------------------

all_stages = [instruction_stage, gridworld_stage, debrief_stage]
experiment = SimpleExperiment(
    blocks=[Block(stages=[s]) for s in all_stages],
    name="FourRoom Goal Navigation",
)
