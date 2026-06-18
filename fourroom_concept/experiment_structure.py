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
# Stage 0 — Welcome
# ---------------------------------------------------------------------------

async def welcome_display_fn(stage, container):
    nicewebrl.clear_element(container)
    with container.style("align-items: center; max-width: 680px;"):
        ui.markdown(
            "## Welcome!\n\n"
            "Thank you for your interest in our study. "
            "Please advance to the next screen to read our consent form and begin the experiment."
        )


welcome_stage = Stage(
    name="Welcome",
    display_fn=welcome_display_fn,
    next_button=True,
)


# ---------------------------------------------------------------------------
# Stage 1 — Consent
# ---------------------------------------------------------------------------

async def consent_display_fn(stage, container):
    nicewebrl.clear_element(container)
    with container.style("align-items: center; max-width: 760px;"):
        ui.html("""
        <div style="font-family: 'Times New Roman', Times, serif; max-width: 740px; font-size: 14px; line-height: 1.5; color: #000;">

          <div style="text-align:center; margin-bottom: 24px;">
            <div style="font-size: 20px; font-weight: bold;">ADULT CONSENT FORM</div>
            <div style="font-size: 18px; font-weight: bold; color: #e77500;">PRINCETON UNIVERSITY</div>
          </div>

          <p style="margin: 6px 0;"><strong>TITLE OF RESEARCH:</strong> <em>Computational Cognitive Science</em></p>
          <p style="margin: 6px 0;"><strong>PRINCIPAL INVESTIGATOR:</strong> Thomas Griffiths</p>
          <p style="margin: 6px 0;"><strong>PRINCIPAL INVESTIGATOR'S DEPARTMENT:</strong> <em>Psychology</em></p>

          <p style="margin: 16px 0;">You are being invited to take part in a research study. Before you decide to participate in this study, it is important that you understand why the research is being done and what it will involve. Please take the time to read the following information carefully. Please ask the researcher if there is anything that is not clear or if you need more information.</p>

          <p style="margin: 12px 0 4px;"><strong><u>Purpose of the research:</u></strong><br>
          This project aims to collect data that can be used to evaluate formal accounts of causal learning, categorization, and language learning and to track how knowledge about these areas is transformed when passed from person to person.</p>

          <p style="margin: 12px 0 4px;"><strong><u>Study Procedures:</u></strong><br>
          You will be presented with some information (e.g., a written narrative, hypothetical scenarios, or scientific data) and will then be asked to make one or more judgments about that information, or decisions based upon it. In some cases you will be asked to provide explanations or justifications for your responses, typically in the form of a short paragraph. The task will not involve deception or emotionally disturbing materials - just simple questions about categories, causal relationships, and languages.
          The task you will perform will be one or more of the following: 1. Being shown a set of members of a category, and then asked to indicate which other objects are likely to belong to the category. 2. Being presented a sequence of pictures or sounds, and being asked to predict the next item in the sequence. 3. Being told a set of words in a language, and then making judgments about whether other words belong to the language. 4. Being shown statistical information about the interaction of causes and effects, and then making judgments about the causal relationships involved. 5. Observing a set of events or reading a description or information, and then evaluating the probability of events or statements. 6. Being presented with one or more visual puzzles and re-arranging elements on a screen to solve them. 7. Being presented with visual stimuli and making a quantitative or qualitative judgment on the basis of said stimuli. 8. Interacting with one or more participants or Large Language Models about shared stimuli using a chat box. 9. Making a decision about an action to take based on quantitative or qualitative information. 10. Interacting with other participants in a joint task, or solving a task, making a decision, or making a judgment that gets passed down to another participant. 11. Viewing a video and responding to it. 12. Being shown a grid, maze, or graph of edges and nodes containing objects that can be moved around with a mouse or the keyboard. 13. Being shown a list of words and/or pictures, and then being asked to recall them. 14. Being asked to solve one or more logic puzzles.</p>

          <p style="margin: 12px 0;">The answers you provide in the task may be used as stimuli for future participants. However, data that would identify you will not be shared with other participants.</p>

          <p style="margin: 12px 0;">The study duration will vary from 5 to 60 minutes.</p>

          <p style="text-align:center; font-weight:bold; margin: 16px 0; border-top: 1px solid #000; padding-top: 8px;">This study has been approved by the Institutional Review Board for Human Subjects</p>

          <hr style="border: none; border-top: 2px solid #000; margin: 16px 0;">

          <p style="margin: 12px 0 4px;"><strong><u>Benefits and Risks:</u></strong><br>
          There are no direct benefits to you as a participant; however, by furthering our understanding of human cognition, this research will benefit society by helping with the development of automated systems that can better solve problems that are computationally challenging, but that people can solve easily with little formal guidance.</p>

          <p style="margin: 12px 0;">Risks associated with participation in this study are minimal. You may feel slight discomfort answering some questions, but you may refrain from answering any questions that make you uncomfortable and may withdraw your participation at any time even after completing the experiment without penalty. In studies where participants interact with large language models, we cannot verify precisely what the model will discuss with participants, though we have engineered our prompts to the model such that it generates highly topically relevant content in a friendly manner. We can be reasonably confident that the model will not generate inappropriate or harmful content, based both on our experience with the model and on the fact that the model has been extensively trained to prevent such outputs.</p>

          <p style="margin: 12px 0 4px;"><strong><u>Confidentiality:</u></strong><br>
          We will not be asking for any personally identifying information, and we will handle responses as confidentially as possible. Your name, or your Worker IDs will never be tied to your responses on this survey. However, we cannot guarantee the confidentiality of information transmitted over the Internet. To minimize this risk, data containing anything that might be personally identifiable (e.g. Worker IDs) will be encrypted on transfer and storage and will only be accessible to qualified lab personnel. We will be keeping data collected as part of this experiment indefinitely. This anonymized data (containing neither Worker IDs nor IP addresses) may be shared with the scientific community and corporate sponsors. At the end of the survey you may be asked if you would like to provide your email or MTurk ID to be contacted to participate in future studies., If you choose to provide this information, you may be contacted for participation in future studies.</p>

          <p style="margin: 12px 0 4px;"><strong><u>Compensation:</u></strong><br>
          For your participation, you will receive <span style="background-color: #ffff00;">[variable payment amount equivalent to $10-$15/hr for both online and in person participation]</span>. If you are on MTurk or Prolific and for any reason you do not complete the study (e.g. technical difficulties, or a desire to stop), we will only be able to pay you if you send an email through MTurk/Prolific, or by emailing the researcher, at <a href="mailto:cocosci-lab@princeton.edu">cocosci-lab@princeton.edu</a>. If you have any questions about the study, feel free to contact the researcher or the Principal Investigator, Thomas Griffiths, at <a href="mailto:tomg@princeton.edu">tomg@princeton.edu</a>.</p>

          <p style="margin: 12px 0 4px;"><strong><u>Who to contact with questions:</u></strong></p>
          <ol style="margin: 4px 0 12px 24px; padding: 0;">
            <li style="margin-bottom: 8px;">PRINCIPAL INVESTIGATOR:<br>
              <span style="display:inline-block; margin-left: 48px;">Dr. Thomas Griffiths<br>
              <a href="mailto:tomg@princeton.edu">tomg@princeton.edu</a></span>
            </li>
            <li style="margin-bottom: 8px;">If you have questions regarding your rights as a research subject, or if problems arise which you do not feel you can discuss with the Investigator, please contact the Institutional Review Board at:<br>
              <span style="display:inline-block; margin-left: 48px; margin-top: 4px;">Assistant Director, Research Integrity and Assurance<br>
              Phone: (609) 258-8543<br>
              Email: <a href="mailto:irb@princeton.edu">irb@princeton.edu</a></span>
            </li>
          </ol>

          <p style="text-align:center; font-weight:bold; border-top: 1px solid #000; padding-top: 8px;">This study has been approved by the Institutional Review Board for Human Subjects</p>

          <hr style="border: none; border-top: 2px solid #000; margin: 16px 0;">

          <ol start="3" style="margin: 0 0 12px 16px; padding: 0;">
            <li style="margin-bottom: 12px;">I understand the information that was presented and that:
              <ol type="A" style="margin: 8px 0 0 24px; padding: 0;">
                <li style="margin-bottom: 8px;">My participation is voluntary, and I may withdraw my consent and discontinue participation in the project at any time.&nbsp; My refusal to participate will not result in any penalty.</li>
                <li style="margin-bottom: 8px;">I do not waive any legal rights or release Princeton University, its agents, or you from liability for negligence.</li>
              </ol>
            </li>
            <li>I hereby give my consent to be the subject of your research.</li>
          </ol>

          <p style="text-align:center; font-weight:bold; border-top: 1px solid #000; padding-top: 8px; margin-top: 16px;">This study has been approved by the Institutional Review Board for Human Subjects</p>

        </div>
        """)

        with ui.row().style("gap: 16px; margin-top: 16px; justify-content: center;"):
            ui.button(
                "I Agree",
                on_click=lambda: _handle_consent(stage, agreed=True),
            ).props("color=primary")
            ui.button(
                "I Do Not Agree",
                on_click=lambda: _handle_consent(stage, agreed=False),
            ).props("color=negative outlined")


async def _handle_consent(stage, agreed: bool):
    if not agreed:
        await stage.set_user_data(declined=True)
    await stage.finish_stage()
    local_cb = stage.get_user_data("local_handle_key_press")
    if local_cb is not None:
        await local_cb()


consent_stage = Stage(
    name="Consent",
    display_fn=consent_display_fn,
    next_button=False,
)


# ---------------------------------------------------------------------------
# Stage 1 — Instructions
# ---------------------------------------------------------------------------

def _instructions_table(dims: list[str], n_kinds: dict[str, int]) -> str:
    """Build a bullet list showing the active dimension values."""
    lines = []
    for d in dims:
        vals = gw.DIMENSIONS[d][:n_kinds[d]]
        formatted = []
        for v in vals:
            sym = SHAPE_SYMBOL.get(v, "")
            formatted.append(f"{sym} {v}".strip() if sym else v)
        lines.append(f"- **{d.capitalize()}:** {', '.join(formatted)}")
    return "\n".join(lines)


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
                f'<p>Your data has been saved.</p>'
                f'<p>Your completion code is: <strong>CVCPWSQG</strong></p>'
                f'<p>Or click the button below to be rerouted to Prolific.</p>'
            )
            ui.button(
                "Return to Prolific",
                on_click=lambda: ui.navigate.to(
                    "https://app.prolific.com/submissions/complete?cc=CVCPWSQG",
                    new_tab=False,
                )
            ).props("color=primary")

    await stage.finish_stage()


debrief_stage = Stage(
    name="Debrief",
    display_fn=debrief_display_fn,
    next_button=False,
)


# ---------------------------------------------------------------------------
# Consent declined screen
# ---------------------------------------------------------------------------

async def declined_display_fn(stage, container):
    nicewebrl.clear_element(container)
    with container.style("align-items: center; max-width: 600px;"):
        ui.markdown(
            "## Thank you for your time\n\n"
            "You have indicated that you do not wish to participate in this study. "
            "You did not consent to participate and the experiment has ended.\n\n"
            "You may close this window."
        )
    await stage.finish_stage()


declined_stage = Stage(
    name="Declined",
    display_fn=declined_display_fn,
    next_button=False,
)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

all_stages = [welcome_stage, consent_stage, instruction_stage, gridworld_stage, debrief_stage]
experiment  = SimpleExperiment(
    blocks=[Block(stages=[s]) for s in all_stages],
    name="FourRoom Concept Learning",
)
