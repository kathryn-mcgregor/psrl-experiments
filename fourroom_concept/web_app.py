"""
NiceGUI web app for the four-room concept-learning experiment.

Run with:
    python web_app.py                                      # random mode, all dims, 4 kinds, 4 goals
    python web_app.py --mode bijection                     # bijection (requires n-kinds == n-goals)
    python web_app.py --mode no-repeat                     # rewarding combos excluded from later mazes
    python web_app.py --dims shape color --n-kinds 3 2       # 3 shapes, 2 colors
    python web_app.py --dims color --n-kinds 2 --n-goals 2  # only 2 colors, 2 goals per maze
"""

import argparse
import asyncio
import json
import os
from asyncio import Lock
from datetime import datetime, timezone

from fastapi import Request
from nicegui import app, ui
from tortoise import Tortoise

import nicewebrl
from nicewebrl.logging import setup_logging, get_logger
from nicewebrl.utils import wait_for_button_or_keypress

import gridworld as gw
import db
from experiment_structure import experiment

# Parse experiment parameters before NiceGUI takes over sys.argv
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument(
    "--mode",
    choices=[gw.MODE_RANDOM, gw.MODE_BIJECTION, gw.MODE_NO_REPEAT],
    default=gw.MODE_RANDOM,
)
_parser.add_argument(
    "--dims",
    nargs="+",
    choices=list(gw.DIMENSIONS.keys()),
    default=list(gw.DIMENSIONS.keys()),
    metavar="DIM",
    help=f"Dimensions goals vary on. Choices: {list(gw.DIMENSIONS.keys())}",
)
_parser.add_argument(
    "--n-kinds",
    nargs="+",
    type=int,
    dest="n_kinds",
    metavar="N",
    help="Number of values per active dimension, in the same order as --dims.",
)
_parser.add_argument(
    "--n-goals",
    type=int,
    default=4,
    dest="n_goals",
    help="Number of goals per maze (max 4, one per room).",
)
_args, _ = _parser.parse_known_args()

EXPERIMENT_MODE   = _args.mode
EXPERIMENT_DIMS   = _args.dims
EXPERIMENT_NGOALS = _args.n_goals

# Build n_kinds dict; default to 4 per dim if not specified
if _args.n_kinds is None:
    EXPERIMENT_NKINDS = {d: 4 for d in EXPERIMENT_DIMS}
else:
    if len(_args.n_kinds) != len(EXPERIMENT_DIMS):
        raise SystemExit(
            f"--n-kinds must have one value per --dims entry. "
            f"Got {len(_args.n_kinds)} value(s) for {len(EXPERIMENT_DIMS)} dim(s): {EXPERIMENT_DIMS}"
        )
    EXPERIMENT_NKINDS = dict(zip(EXPERIMENT_DIMS, _args.n_kinds))

logger   = get_logger(__name__)
DATA_DIR = "./data"
DB_FILE  = "db.sqlite"

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
    db_url = os.environ.get("DATABASE_URL", f"sqlite://{DATA_DIR}/{DB_FILE}")
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    # Tortoise for nicewebrl's own models
    await Tortoise.init(
        db_url=db_url,
        modules={"models": ["nicewebrl.stages"]},
    )
    await Tortoise.generate_schemas()
    # Our custom tables via direct SQL (avoids Tortoise context issues)
    await db.init_tables()


async def close_db():
    await Tortoise.close_connections()


app.on_startup(init_db)
app.on_shutdown(close_db)

setup_logging(DATA_DIR, nicegui_storage_user_key="seed")
os.makedirs(DATA_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Key-press dispatch
# ---------------------------------------------------------------------------

async def global_handle_key_press(e, container):
    if experiment.finished():
        return
    stage = await experiment.get_stage()
    if stage.get_user_data("finished", False):
        return

    handler = getattr(stage, "_key_handler", None)
    if handler is not None:
        await handler(stage, e, container)
    else:
        await stage.handle_key_press(e, container)

    local_cb = stage.get_user_data("local_handle_key_press")
    if local_cb is not None:
        await local_cb()


# ---------------------------------------------------------------------------
# Stage runner
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


# ---------------------------------------------------------------------------
# Experiment flow
# ---------------------------------------------------------------------------

async def start_experiment(meta_container, stage_container):
    ui.on("key_pressed", lambda e: global_handle_key_press(e, stage_container))

    # Discard stale state from old format (pre-refactor used goal_shapes/goal_colors)
    existing = app.storage.user.get("gw_state")
    if existing is not None:
        first_maze = next((m for m in existing.get("mazes", []) if m is not None), None)
        if first_maze is not None and "goals" not in first_maze:
            logger.info("Discarding stale session state (old format)")
            del app.storage.user["gw_state"]

    if "gw_state" not in app.storage.user:
        seed  = app.storage.user.get("seed")
        state = gw.init_episode(
            seed=seed,
            mode=EXPERIMENT_MODE,
            dims=EXPERIMENT_DIMS,
            n_kinds=EXPERIMENT_NKINDS,
            n_goals=EXPERIMENT_NGOALS,
        )
        app.storage.user["gw_state"] = state
        logger.info(
            f"New episode: mode={EXPERIMENT_MODE} dims={EXPERIMENT_DIMS} "
            f"n_kinds={EXPERIMENT_NKINDS} n_goals={EXPERIMENT_NGOALS} "
            f"rule={state['rule_dim']}='{state['rule_value']}'"
        )

    while not experiment.finished():
        stage = await experiment.get_stage()
        await run_stage(stage, stage_container)
        await stage.finish_saving_user_data()
        await experiment.advance()

    await finish_experiment(meta_container)


async def finish_experiment(container):
    state = app.storage.user.get("gw_state", {})
    seed  = app.storage.user.get("seed")

    mazes_data = [
        {
            "grid":           m["grid"],
            "agent_pos":      m["agent_pos"],
            "goal_positions": m["goal_positions"],
            "goals":          m["goals"],
            "fixed_values":   m.get("fixed_values", {}),
        }
        if m is not None else None
        for m in state.get("mazes", [])
    ]

    await db.upsert_session(
        seed         = seed,
        completed_at = datetime.now(timezone.utc).isoformat(),
        mode         = state.get("mode"),
        dims         = state.get("dims"),
        n_kinds      = state.get("n_kinds"),
        n_goals      = state.get("n_goals"),
        rule_dim     = state.get("rule_dim"),
        rule_value   = state.get("rule_value"),
        total_score  = state.get("total_score"),
        total_steps  = sum(m["step"] for m in state.get("mazes", []) if m is not None),
        log          = state.get("log", []),
        mazes        = mazes_data,
    )
    logger.info(f"Session saved to database → seed={seed}")


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

    ui.add_body_html("""
    <script>
    window.gridworld_active = false;
    document.addEventListener('keydown', async function(e) {
        if (!window.gridworld_active) return;
        if (['ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].includes(e.key)) {
            e.preventDefault();
            await emitEvent('key_pressed', {
                key: e.key,
                keydownTime: new Date(),
                imageSeenTime: null
            });
        }
    }, true);
    </script>
    """)

    card = (
        ui.card(align_items=["center"])
        .classes("fixed-center")
        .style(
            "width:95vw; max-height:95vh; overflow:auto;"
            "display:flex; flex-direction:column;"
            "justify-content:flex-start; align-items:center;"
            "padding:1rem;"
        )
    )
    with card:
        meta_container = ui.column()
        with meta_container.style("align-items: center;"):
            stage_container = ui.column()
        with meta_container.style("align-items: center;"):
            await start_experiment(meta_container, stage_container)


ui.run(
    storage_secret="fourroom_concept_42",
    show=False,
    reload=True,
    title="Concept Navigation",
    port=8084,
)
