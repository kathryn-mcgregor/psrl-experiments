"""
NiceGUI web app for the four-room goal-navigation experiment.

Run with:
    python web_app.py
or:
    uvicorn web_app:app --port 8082
"""
import asyncio
import json
import os
import time
from asyncio import Lock

from fastapi import Request
from nicegui import app, ui
from tortoise import Tortoise

import nicewebrl
from nicewebrl.logging import setup_logging, get_logger
from nicewebrl.utils import wait_for_button_or_keypress
from nicewebrl import stages

import gridworld as gw
from experiment_structure import experiment, gridworld_stage

logger = get_logger(__name__)

DATA_DIR = "./data"
DATABASE_FILE = "db.sqlite"

_user_locks: dict = {}


def get_user_lock():
    seed = app.storage.user["seed"]
    if seed not in _user_locks:
        _user_locks[seed] = Lock()
    return _user_locks[seed]


# ---------------------------------------------------------------------------
# DB lifecycle
# ---------------------------------------------------------------------------

async def init_db():
    await Tortoise.init(
        db_url=f"sqlite://{DATA_DIR}/{DATABASE_FILE}",
        modules={"models": ["nicewebrl.stages"]},
    )
    await Tortoise.generate_schemas()


async def close_db():
    await Tortoise.close_connections()


app.on_startup(init_db)
app.on_shutdown(close_db)

setup_logging(DATA_DIR, nicegui_storage_user_key="seed")

if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)


# ---------------------------------------------------------------------------
# Key-press dispatch
# ---------------------------------------------------------------------------

async def global_handle_key_press(e, container):
    if experiment.finished():
        return

    stage = await experiment.get_stage()
    if stage.get_user_data("finished", False):
        return

    # Delegate to stage's custom key handler if present
    handler = getattr(stage, "_key_handler", None)
    if handler is not None:
        await handler(stage, e, container)
    else:
        await stage.handle_key_press(e, container)

    # Check completion after each key press
    if stage.get_user_data("finished", False):
        pass  # stage_over_event set inside run_stage via local_handle_key_press


# ---------------------------------------------------------------------------
# Experiment flow
# ---------------------------------------------------------------------------

async def run_stage(stage, container):
    stage_over_event = asyncio.Event()

    async def local_handle_key_press():
        async with get_user_lock():
            if stage.get_user_data("finished", False):
                logger.info(f"Stage finished: {stage.name}")
                stage_over_event.set()

    await stage.set_user_data(local_handle_key_press=local_handle_key_press)

    async def handle_button_press():
        if stage.get_user_data("finished", False):
            return
        await stage.handle_button_press(container)
        async with get_user_lock():
            if stage.get_user_data("finished", False):
                stage_over_event.set()

    with container.style("align-items: center;"):
        await stage.activate(container)

    if stage.get_user_data("finished", False):
        stage_over_event.set()

    if stage.next_button:
        with container:
            button = ui.button("Start")
            await wait_for_button_or_keypress(button)
            await handle_button_press()

    await stage_over_event.wait()


async def start_experiment(meta_container, stage_container):
    # Register global key handler
    ui.on("key_pressed", lambda e: global_handle_key_press(e, stage_container))

    # Initialise gridworld state once per user (persists across page refreshes)
    if "gw_state" not in app.storage.user:
        seed = app.storage.user.get("seed")
        app.storage.user["gw_state"] = gw.init_episode(seed=seed)
        logger.info(
            f"New episode: true_goal_idx={app.storage.user['gw_state']['true_goal_idx']}"
        )

    while not experiment.finished():
        stage = await experiment.get_stage()
        await run_stage(stage, stage_container)
        await stage.finish_saving_user_data()
        await experiment.advance()

    await finish_experiment(meta_container)


async def finish_experiment(container):
    # Debrief stage already rendered the final screen — just save data silently.
    state = app.storage.user.get("gw_state", {})
    goal_positions = state.get("goal_positions", [])
    true_goal_idx = state.get("true_goal_idx")
    metadata = {
        "finished": True,
        "seed": app.storage.user.get("seed"),
        "steps": state.get("step"),
        "grid": state.get("grid"),
        "goal_positions": goal_positions,
        "true_goal_idx": true_goal_idx,
        "true_goal_pos": goal_positions[true_goal_idx] if isinstance(true_goal_idx, int) else None,
        "visited_goals": state.get("visited_goals", []),
    }
    meta_path = f"{DATA_DIR}/user_meta_{app.storage.user['seed']}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {meta_path}")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

@ui.page("/")
async def index(request: Request):
    nicewebrl.initialize_user(request=request)
    await experiment.initialize()

    basic_js = nicewebrl.basic_javascript_file()
    with open(basic_js) as f:
        ui.add_body_html("<script>" + f.read() + "</script>")

    # Separate listener for gridworld arrow keys (bypasses the accept_keys gate)
    ui.add_body_html("""
    <script>
    window.gridworld_active = false;
    document.addEventListener('keydown', async function(e) {
        if (!window.gridworld_active) return;
        if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key)) {
            e.preventDefault();
            await emitEvent('key_pressed', {key: e.key, keydownTime: new Date(), imageSeenTime: null});
        }
    }, true);
    </script>
    """)

    card = (
        ui.card(align_items=["center"])
        .classes("fixed-center")
        .style(
            "width: 90vw;"
            "max-height: 95vh;"
            "overflow: auto;"
            "display: flex;"
            "flex-direction: column;"
            "justify-content: flex-start;"
            "align-items: center;"
            "padding: 1rem;"
        )
    )
    with card:
        meta_container = ui.column()
        with meta_container.style("align-items: center;"):
            stage_container = ui.column()

        with meta_container.style("align-items: center;"):
            await start_experiment(meta_container, stage_container)


ui.run(
    storage_secret="fourroom_secret_42",
    show=False,
    reload=True,
    title="Four-Room Navigation",
    port=8082,
)
