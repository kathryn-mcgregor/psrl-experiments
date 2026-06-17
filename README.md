# PSRL Web Experiments

Three browser-based navigation experiments built with [NiceWebRL](https://github.com/KempnerInstitute/nicewebrl).
Each runs as a local web server or on Heroku and records participant data to a database.

---

## Experiments

| Experiment | Port | Description |
|---|---|---|
| `fourroom` | 8082 | Navigate a 4-room maze to find a hidden goal among 15 candidates |
| `fourroom_fruits` | 8083 | Collect apples and bananas — one type is worth more points |
| `fourroom_concept` | 8084 | Discover a hidden rule across 4 mazes using configurable goal dimensions |

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/psrl-experiments.git
cd psrl-experiments
```

### 2. Create the conda environment

```bash
conda env create -f environment.yml
conda activate psrl-experiments
```

> **Note:** this installs `nicewebrl` directly from GitHub, which requires `git` to be on your PATH.

### 3. Run an experiment

Each experiment is a self-contained directory. `cd` into it and run `web_app.py`:

```bash
# Goal-finding experiment
cd fourroom
python web_app.py
# → open http://localhost:8082

# Fruit collection experiment
cd fourroom_fruits
python web_app.py
# → open http://localhost:8083

# Concept-learning experiment
cd fourroom_concept
python web_app.py
# → open http://localhost:8084
```

To run multiple experiments simultaneously, open a separate terminal for each.

---

## Data

`fourroom_concept` stores participant data in a database with two tables:

| Table | Contents |
|---|---|
| `steps` | One row per action (maze, step number, action, position, reward, timestamp) |
| `sessions` | One row per completed participant (rule, score, maze layouts, full goal-visit log) |

**Locally** the database is a SQLite file at `fourroom_concept/data/db.sqlite`, created automatically on first run.

**On Heroku** the database is a hosted Postgres instance. The app reads the `DATABASE_URL` environment variable provided automatically by Heroku Postgres and connects to it instead.

### Exporting data

Run `export_data.py` from inside the `fourroom_concept` directory to pull data out of the database:

```bash
cd fourroom_concept

# Export from local SQLite
python export_data.py

# Export from Heroku Postgres
DATABASE_URL=$(heroku config:get DATABASE_URL -a your-app-name) python export_data.py

# Export a single participant
python export_data.py --seed <seed>
```

Output is written to `fourroom_concept/export/`:

| File | Contents |
|---|---|
| `sessions.csv` | One row per completed participant |
| `steps.csv` | One row per action taken |
| `session_<seed>.json` | Full session data including maze layouts and goal-visit log |

### Visualising paths (`fourroom` and `fourroom_fruits`)

```bash
# Single participant (interactive)
python visualize.py data/user_meta_<seed>.json

# All participants → data/path-vis/path_<seed>.png  (skips existing)
python visualize.py
```

---

## Experiment details

### `fourroom` — Goal finding
- 13×13 four-room maze with 15 amber candidate goals
- One is randomly selected as the true goal (hidden from participant)
- Ends when the agent reaches the true goal
- Includes `run_psrl.py` to compare human paths against a PSRL agent

### `fourroom_fruits` — Fruit collection
- 7 apples and 7 bananas distributed across the maze
- One fruit type is worth 5 points, the other 1 point (randomly assigned, not revealed)
- Ends when all 14 fruits are collected
- Live scoreboard shows accumulated points next to the maze

### `fourroom_concept` — Rule discovery
- 4 mazes per participant; number of goals per maze is configurable (default 4)
- Goals vary along configurable dimensions: shape, color, and/or texture
- A hidden rule — one value on one dimension — gives reward +1; all others give 0
- Continue button appears once a rewarding goal is found; maze grays out
- Log panel persists across all 4 mazes
- Goal placement, dimension kinds, and sampling mode are all controlled via command-line flags (see below)

---

## `fourroom_concept` flags

All flags are optional. Defaults produce a random-mode experiment with shape and color dimensions, 4 kinds each, and 4 goals per maze.

### `--mode`

Controls how goal combinations are selected across mazes.

| Value | Behaviour |
|---|---|
| `random` (default) | Goals are sampled freely from all available combinations |
| `bijection` | Each value in each active dimension appears exactly once per maze (no repeats within a maze). Requires `n-goals ≤ n-kinds` for every dimension |
| `no-repeat` | Combinations that gave a reward in a previous maze are excluded from all later mazes. Guarantees at least one rewarding goal per maze |

```bash
python web_app.py --mode bijection
python web_app.py --mode no-repeat
```

---

### `--dims`

Which dimensions goals vary on. Any subset of `shape`, `color`, `texture`. Dimensions not listed are fixed to a single randomly-chosen value for each maze (visible to the participant but not part of the rule).

| Dimension | Available kinds |
|---|---|
| `shape` | circle, square, triangle, star, pentagon, hexagon, diamond |
| `color` | blue, red, yellow, green, purple, orange, pink |
| `texture` | solid, striped, dotted, outline, chevron |

```bash
python web_app.py --dims shape color          # default
python web_app.py --dims shape color texture  # all three dimensions
python web_app.py --dims color                # color only; shape and texture are fixed
```

---

### `--n-kinds`

How many values to use from each active dimension, given in the same order as `--dims`. Determines which prefix of each dimension's list is available — e.g. `--n-kinds 3` on `shape` uses circle, square, triangle.

Must provide one number per dimension listed in `--dims`.

```bash
python web_app.py --dims shape color --n-kinds 4 4       # 4 shapes, 4 colors (default)
python web_app.py --dims shape color --n-kinds 3 2       # 3 shapes, 2 colors
python web_app.py --dims shape color texture --n-kinds 4 4 3  # 4 shapes, 4 colors, 3 textures
```

---

### `--n-goals`

Number of candidate goals placed in each maze. Defaults to 4 (one per room). Can exceed 4 — the first 4 goals are placed one per room, and any additional goals are placed randomly in remaining free cells.

```bash
python web_app.py --n-goals 4   # default
python web_app.py --n-goals 8
```

---

### Combined examples

```bash
# Bijection with 3 shapes and 3 colors, 3 goals per maze
python web_app.py --mode bijection --dims shape color --n-kinds 3 3 --n-goals 3

# No-repeat with all three dimensions, 4 of each kind, 8 goals per maze
python web_app.py --mode no-repeat --dims shape color texture --n-kinds 4 4 4 --n-goals 8

# Color only, random mode, 2 colors, 2 goals
python web_app.py --dims color --n-kinds 2 --n-goals 2
```

---

## Deploying on Heroku

Heroku hosts the experiment on a public URL so participants can access it from anywhere. These steps cover `fourroom_concept`; the same process applies to `fourroom` and `fourroom_fruits`.

### Prerequisites

- A [Heroku account](https://heroku.com) and the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli) installed
- The repo pushed to GitHub (see Setup above)

### 1. Create a Heroku app

```bash
heroku login
heroku create your-app-name
```

### 2. Add the Heroku Postgres add-on

```bash
heroku addons:create heroku-postgresql:essential-0 -a your-app-name
```

This automatically sets the `DATABASE_URL` environment variable on your app. The `essential-0` plan (~$5/month) supports up to 10 million rows.

### 3. Add a `Procfile`

Create a file named `Procfile` in the `fourroom_concept` directory (no extension):

```
web: python web_app.py
```

To pass flags (e.g. mode), include them in the Procfile:

```
web: python web_app.py --mode no-repeat --dims shape color --n-kinds 4 4 --n-goals 4
```

### 4. Add a `requirements.txt`

Heroku uses `requirements.txt` instead of `environment.yml`. Create one in `fourroom_concept`:

```bash
cd fourroom_concept
pip freeze > requirements.txt
```

### 5. Deploy

Heroku deploys from git. Push the `fourroom_concept` directory as a subtree:

```bash
git subtree push --prefix fourroom_concept heroku main
```

### 6. Set a storage secret

NiceGUI requires a storage secret for session cookies. Set it as a Heroku config var:

```bash
heroku config:set NICEGUI_STORAGE_SECRET=some-long-random-string -a your-app-name
```

Then update `web_app.py` to read it from the environment:

```python
storage_secret = os.environ.get("NICEGUI_STORAGE_SECRET", "fourroom_concept_42")
ui.run(storage_secret=storage_secret, ...)
```

### 7. Open the app

```bash
heroku open -a your-app-name
```

Participants visit `https://your-app-name.herokuapp.com`.

### Viewing logs

```bash
heroku logs --tail -a your-app-name
```

---

## Troubleshooting

**Page loads but nothing happens after clicking Start**
Clear browser storage (or use a private/incognito window) to reset session state from a previous run.

**`ModuleNotFoundError: No module named 'nicewebrl'`**
Make sure you activated the conda environment: `conda activate psrl-experiments`

**`ModuleNotFoundError: No module named 'gridworld'`**
Run `web_app.py` from inside the experiment directory, not from the repo root:
```bash
cd fourroom_concept   # not: python fourroom_concept/web_app.py
python web_app.py
```
