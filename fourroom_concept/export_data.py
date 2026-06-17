"""
Export participant data from the database to local files.

Usage:
    # Export all sessions and steps to CSV
    python export_data.py

    # Export a single participant
    python export_data.py --seed <seed>

    # Connect to Heroku Postgres instead of local SQLite
    DATABASE_URL=<heroku-url> python export_data.py

Output (written to ./export/):
    sessions.csv   — one row per completed participant
    steps.csv      — one row per action taken
    session_<seed>.json — full maze layout + log per participant
"""

import asyncio
import csv
import json
import os
import sys

import db

EXPORT_DIR = "./export"


async def export(seed: str | None = None):
    await db.init_tables()
    os.makedirs(EXPORT_DIR, exist_ok=True)

    sessions = await db.fetch_sessions(seed=seed)
    steps    = await db.fetch_steps(seed=seed)

    # ---- Sessions -----------------------------------------------------------
    with open(os.path.join(EXPORT_DIR, "sessions.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seed", "completed_at", "mode", "dims", "n_kinds", "n_goals",
            "rule_dim", "rule_value", "total_score", "total_steps",
        ])
        for s in sessions:
            writer.writerow([
                s["seed"],
                s["completed_at"],
                s["mode"],
                json.dumps(s["dims"]),
                json.dumps(s["n_kinds"]),
                s["n_goals"],
                s["rule_dim"],
                s["rule_value"],
                s["total_score"],
                s["total_steps"],
            ])
    print(f"Exported {len(sessions)} session(s) → {EXPORT_DIR}/sessions.csv")

    # ---- Steps --------------------------------------------------------------
    with open(os.path.join(EXPORT_DIR, "steps.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seed", "timestamp", "maze_idx", "maze_step", "action",
            "prev_pos", "new_pos", "moved", "visited_goal", "left_goal", "reward",
        ])
        for s in steps:
            writer.writerow([
                s["seed"], s["timestamp"], s["maze_idx"], s["maze_step"],
                s["action"], s["prev_pos"], s["new_pos"], s["moved"],
                s["visited_goal"], s["left_goal"], s["reward"],
            ])
    print(f"Exported {len(steps)} step(s) → {EXPORT_DIR}/steps.csv")

    # ---- Per-session JSON (maze layouts + full log) -------------------------
    for s in sessions:
        path = os.path.join(EXPORT_DIR, f"session_{s['seed']}.json")
        with open(path, "w") as f:
            json.dump(s, f, indent=2)
    print(f"Exported {len(sessions)} JSON file(s) → {EXPORT_DIR}/session_<seed>.json")


if __name__ == "__main__":
    seed_filter = None
    if "--seed" in sys.argv:
        idx = sys.argv.index("--seed")
        seed_filter = sys.argv[idx + 1]
    asyncio.run(export(seed=seed_filter))
