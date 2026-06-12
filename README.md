# PSRL Web Experiments

Three browser-based navigation experiments built with [NiceWebRL](https://github.com/KempnerInstitute/nicewebrl).
Each runs as a local web server and records participant data to the `data/` folder.

---

## Experiments

| Experiment | Port | Description |
|---|---|---|
| `fourroom` | 8082 | Navigate a 4-room maze to find a hidden goal among 15 candidates |
| `fourroom_fruits` | 8083 | Collect apples and bananas — one type is worth more points |
| `fourroom_concept` | 8084 | Discover a hidden rule (shape or color) across 4 mazes |

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

Participant data is saved to `data/` inside each experiment folder:

| File | Contents |
|---|---|
| `data/user_data_<seed>.msgpack` | One record per step (action, position, reward, timestamp) |
| `data/user_meta_<seed>.json` | Summary (maze layout, score, rule, etc.) |

Data files are excluded from git via `.gitignore`.

### Reading data

```bash
# Print a step-by-step trace for one participant
cd fourroom
python read_data.py data/user_data_<seed>.msgpack
```

### Visualising paths

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
- 4 mazes per participant, 4 candidate goals per maze (one per room)
- Goals vary in shape (●▲■) and color (blue/red/yellow)
- A hidden rule — either a shape or a color — gives reward +1
- Continue button appears once a rewarding goal is found; maze grays out
- Log panel persists across all 4 mazes

---

## Troubleshooting

**Page loads but nothing happens after clicking Start**
Clear browser storage (or use a private/incognito window) to reset session state from a previous run.

**`ModuleNotFoundError: No module named 'nicewebrl'`**
Make sure you activated the conda environment: `conda activate psrl-experiments`

**`ModuleNotFoundError: No module named 'gridworld'`**
Run `web_app.py` from inside the experiment directory, not from the repo root:
```bash
cd fourroom   # not: python fourroom/web_app.py
python web_app.py
```
