"""
src/graph/debate_graph.py
--------------------------
LangGraph orchestration for the DebateBench debate pipeline.

LANGGRAPH FEATURES USED:
  1. StateGraph(DebateState)    — typed shared state
  2. Conditional edges          — consensus bypass, debate loop, NEI routing,
                                  jury enable/disable toggle
  3. Append reducer             — transcript and jury_assessments accumulate
  4. Pure routing nodes         — consensus_check, early_stop_check, jury_check

NODE FLOW (jury enabled):
  load_case → debater_a_initial → debater_b_initial → consensus_check
    ├─(skip) → judge_verdict → jury_check
    └─(debate) → [rounds] → judge_verdict → jury_check
                                               ├─(jury off) → evaluate_result → END
                                               └─(jury on)
                                                   juror_1 → juror_2 → juror_3
                                                     → jury_deliberate → evaluate_result → END
"""

import time
from langgraph.graph import StateGraph, END

from src.utils.schemas import (
    DebateState, TranscriptTurn,
    STANCE_NEI, VALID_STANCES,
)
from src.models.university_client import make_clients
from src.agents.debater import Debater, SYSTEM_DEBATER_A, SYSTEM_DEBATER_B
from src.agents.judge import Judge
from src.agents.jury import JuryPanel


# ══════════════════════════════════════════════════════════════════════
# Node factory — binds agents to config once at graph build time
# ══════════════════════════════════════════════════════════════════════

def make_nodes(config: dict):
    """
    Create all node functions, closing over config and agent instances.
    Returns a dict of node_name → callable.
    """
    mc = config["models"]
    dc = config["debate"]

    # Build all three clients from config
    debater_client, judge_client, jury_client = make_clients(config)

    debater_a = Debater("Debater A", "debater_a.txt", debater_client,
                        system_message=SYSTEM_DEBATER_A)
    debater_b = Debater("Debater B", "debater_b.txt", debater_client,
                        system_message=SYSTEM_DEBATER_B)
    judge     = Judge(judge_client)
    jury      = JuryPanel(jury_client, judge_client=judge_client)  # judge as arbiter

    # Helper: format transcript for jury prompts
    def _format_transcript_text(turns: list) -> str:
        parts = []
        for t in turns:
            if t["agent"] == "Judge":
                continue   # jury evaluates debate, not single judge's verdict
            label = f"Round {t['round']} — {t['agent']} [{t['stance']}, conf={t['confidence']}]"
            parts.append(f"{label}:\n  {t['reasoning']}")
            if t.get("counter"):
                parts.append(f"  Counter: {t['counter']}")
        return "\n\n".join(parts) if parts else "(No debate turns)"

    # ── NODE: load_case ────────────────────────────────────────────────
    # Initializes timing. State already has all fields from make_initial_state.
    def node_load_case(state: DebateState) -> dict:
        return {"_start_time": time.time()}

    # ── NODE: debater_a_initial ────────────────────────────────────────
    def node_debater_a_initial(state: DebateState) -> dict:
        output, turn = debater_a.argue(
            state["claim"], state["evidence_snippets"], [], 0
        )
        return {
            "a_output": output,
            "a_stance": output["stance"],
            "transcript": [turn],   # APPENDED via operator.add reducer
        }

    # ── NODE: debater_b_initial ────────────────────────────────────────
    # Phase 1 requirement: both debaters generate initial positions
    # INDEPENDENTLY — without seeing the other's response.
    # Pass [] (empty transcript) so B cannot see A's opening argument.
    # B will see A's opening in round 1 of Phase 2 via the full transcript.
    def node_debater_b_initial(state: DebateState) -> dict:
        output, turn = debater_b.argue(
            state["claim"], state["evidence_snippets"],
            [], 0    # empty transcript — B sees no prior turns at init
        )
        return {
            "b_output": output,
            "b_stance": output["stance"],
            "transcript": [turn],
        }

    # ── NODE: consensus_check ──────────────────────────────────────────
    # PURE LOGIC — no LLM call. Determines routing after init.
    # This node itself returns nothing; routing is done by the edge.
    def node_consensus_check(state: DebateState) -> dict:
        return {}   # routing purely via conditional edge below

    # ── NODE: debater_a_rebuttal ───────────────────────────────────────
    # Always increment at the START of a new A/B cycle
    def node_debater_a_rebuttal(state: DebateState) -> dict:
        new_round = state["current_round"] + 1
        output, turn = debater_a.argue(
            state["claim"], state["evidence_snippets"],
            state["transcript"], new_round
        )
        updates = {
            "a_output":      output,
            "a_stance":      output["stance"],
            "a_epistemic":   output.get("epistemic_status", "CERTAIN"),
            "transcript":    [turn],
            "current_round": new_round,  # Sync point
        }
        # First concession sets the round marker
        if output.get("conceded") and state.get("concession_round", -1) == -1:
            updates["concession_round"] = new_round
        return updates

    # ── NODE: debater_b_rebuttal ───────────────────────────────────────
    # B stays in the SAME round as A to complete the "Exchange"
    def node_debater_b_rebuttal(state: DebateState) -> dict:
        output, turn = debater_b.argue(
            state["claim"], state["evidence_snippets"],
            state["transcript"], state["current_round"]  # Uses same round as A
        )
        updates = {
            "b_output":    output,
            "b_stance":    output["stance"],
            "b_epistemic": output.get("epistemic_status", "CERTAIN"),
            "transcript":  [turn],
        }
        if output.get("conceded") and state.get("concession_round", -1) == -1:
            updates["concession_round"] = state["current_round"]
        return updates

    # ── NODE: early_stop_check ─────────────────────────────────────────
    # PURE LOGIC — checks stopping conditions. No LLM call.
    # Detects: (a) consecutive same stance, (b) [CONCEDE] token from either debater
    def node_early_stop_check(state: DebateState) -> dict:
        a = state["a_stance"]
        b = state["b_stance"]
        same = (a == b) and (a in VALID_STANCES)
        new_consec = state["consecutive_same"] + 1 if same else 0

        # [CONCEDE] from either debater triggers immediate stop
        # Set consecutive_same high enough to fire the early stop edge
        a_conceded = state.get("a_output", {}) and state["a_output"].get("conceded", False)
        b_conceded = state.get("b_output", {}) and state["b_output"].get("conceded", False)
        if a_conceded or b_conceded:
            new_consec = 999  # force immediate stop

        return {"consecutive_same": new_consec}

    # ── NODE: judge_verdict ────────────────────────────────────────────
    def node_judge_verdict(state: DebateState) -> dict:
        output = judge.evaluate(
            state["claim"],
            state["evidence_snippets"],
            state["transcript"],
        )
        # Add judge turn to transcript
        turn = TranscriptTurn(
            round=state["current_round"] + 1,
            agent="Judge",
            stance=output["final_verdict"],
            reasoning=output["reasoning"],
            counter="",
            confidence=output["confidence"],
        )
        return {
            "judge_output": output,
            "final_verdict": output["final_verdict"],
            "transcript": [turn],
        }

    # ── NODE: evaluate_result ──────────────────────────────────────────
    # PURE LOGIC — compare verdict to ground truth.
    def node_evaluate_result(state: DebateState) -> dict:
        # Use jury verdict if available, otherwise single judge
        verdict  = state.get("jury_verdict") or state["final_verdict"]
        correct  = verdict == state["ground_truth"]
        total    = len(state["transcript"])
        elapsed  = time.time() - state.get("_start_time", time.time())
        return {
            "judge_correct":    state["final_verdict"] == state["ground_truth"],
            "jury_correct":     state.get("jury_verdict", "") == state["ground_truth"],
            "total_turns":      total,
            "duration_seconds": round(elapsed, 2),
            "debate_finished":  True,
        }

    # ── NODE: jury_check ───────────────────────────────────────────────
    # PURE LOGIC — no LLM call. Routing only.
    def node_jury_check(state: DebateState) -> dict:
        return {}

    # ── NODES: juror_1_assess, juror_2_assess, juror_3_assess ─────────
    # Stub nodes — routing waypoints only.
    # All actual jury work (Phase 1 + consensus check + Phase 2) is done
    # in node_jury_deliberate via jury.run().
    def node_juror_1(state: DebateState) -> dict:
        return {}

    def node_juror_2(state: DebateState) -> dict:
        return {}

    def node_juror_3(state: DebateState) -> dict:
        return {}

    # ── NODE: jury_deliberate ──────────────────────────────────────────
    # Calls jury.run() which handles Phase 1, consensus check, Phase 2.
    # All jury logic lives in jury.py — graph just routes here.
    def node_jury_deliberate(state: DebateState) -> dict:
        transcript_text = _format_transcript_text(state["transcript"])
        result = jury.run(
            claim=state["claim"],
            evidence_snippets=state["evidence_snippets"],
            transcript_text=transcript_text,
        )
        # jury_assessments must be a list for the append reducer
        # but jury.run returns the full lists — store directly
        return {
            "jury_assessments":   result["jury_assessments"],
            "jury_final_votes":   result["jury_final_votes"],
            "jury_verdict":       result["jury_verdict"],
            "jury_disagreement":  result["jury_disagreement"],
            "_p1_disagreement":   result.get("_p1_disagreement", 0.0),
            "_p2_disagreement":   result.get("_p2_disagreement", 0.0),
            "_minds_changed":     result.get("_minds_changed", 0),
            "_consensus_at_init": result.get("_consensus_at_init", False),
            "_ambiguity_flag":    result.get("_ambiguity_flag", False),
            "_arbiter_used":      result.get("_arbiter_used", False),
            "_deliberation_changed": result.get("_deliberation_changed", False),
        }

    return {
        "load_case":            node_load_case,
        "debater_a_initial":   node_debater_a_initial,
        "debater_b_initial":   node_debater_b_initial,
        "consensus_check":     node_consensus_check,
        "debater_a_rebuttal":  node_debater_a_rebuttal,
        "debater_b_rebuttal":  node_debater_b_rebuttal,
        "early_stop_check":    node_early_stop_check,
        "judge_verdict":       node_judge_verdict,
        "jury_check":          node_jury_check,
        "juror_1_assess":      node_juror_1,
        "juror_2_assess":      node_juror_2,
        "juror_3_assess":      node_juror_3,
        "jury_deliberate":     node_jury_deliberate,
        "evaluate_result":     node_evaluate_result,
    }


# ══════════════════════════════════════════════════════════════════════
# Conditional edge functions
# ══════════════════════════════════════════════════════════════════════

def make_edge_after_consensus(config: dict):
    """
    After consensus_check:
      - Both debaters agreed at init → skip debate, go to judge
      - Disagreement → start rebuttal rounds

    NOTE: We do NOT route based on ground_truth. The system never sees the
    gold label during inference. Routing is purely based on debater outputs.
    Early versions used ground_truth==NEI as a routing heuristic (ablation),
    but the main reported system uses only debater consensus as the signal.
    """
    def edge(state: DebateState) -> str:
        a, b = state["a_stance"], state["b_stance"]
        # Both debaters opened with the same stance → no debate needed
        if a == b and a in VALID_STANCES:
            return "skip_debate"
        return "start_debate"
    return edge


def make_edge_should_continue(config: dict):
    """
    After early_stop_check:
      - FIX: early_stop only fires AFTER min_rounds are completed
      - Hard ceiling at max_rounds regardless
      - Otherwise continue
    """
    min_rounds   = config["debate"]["min_rounds"]
    max_rounds   = config["debate"]["max_rounds"]
    early_consec = config["debate"]["early_stop_consecutive"]

    def edge(state: DebateState) -> str:
        r      = state["current_round"]
        consec = state["consecutive_same"]

        # Hard ceiling
        if r >= max_rounds:
            return "go_to_judge"
        # Early stop — ONLY after min_rounds met
        if r >= min_rounds and consec >= early_consec:
            return "go_to_judge"
        # Must keep going
        return "continue_debate"

    return edge


# ══════════════════════════════════════════════════════════════════════
# Graph assembly
# ══════════════════════════════════════════════════════════════════════

def build_graph(config: dict):
    """
    Build and compile the LangGraph StateGraph.
    Returns a compiled app ready for .invoke(initial_state).
    Jury is enabled/disabled via config["jury"]["enabled"].
    """
    nodes = make_nodes(config)
    graph = StateGraph(DebateState)

    for name, fn in nodes.items():
        graph.add_node(name, fn)

    graph.set_entry_point("load_case")

    # Fixed edges — debate pipeline (unchanged)
    graph.add_edge("load_case",           "debater_a_initial")
    graph.add_edge("debater_a_initial",   "debater_b_initial")
    graph.add_edge("debater_b_initial",   "consensus_check")
    graph.add_edge("debater_a_rebuttal",  "debater_b_rebuttal")
    graph.add_edge("debater_b_rebuttal",  "early_stop_check")

    # Single judge always runs — jury is optional extension after it
    graph.add_edge("judge_verdict",       "jury_check")

    # Jury pipeline — sequential deliberation
    graph.add_edge("juror_1_assess",      "juror_2_assess")
    graph.add_edge("juror_2_assess",      "juror_3_assess")
    graph.add_edge("juror_3_assess",      "jury_deliberate")
    graph.add_edge("jury_deliberate",     "evaluate_result")
    graph.add_edge("evaluate_result",     END)

    # Conditional: after consensus check
    graph.add_conditional_edges(
        "consensus_check",
        make_edge_after_consensus(config),
        {
            "skip_debate":  "judge_verdict",
            "start_debate": "debater_a_rebuttal",
        },
    )

    # Conditional: debate loop or stop
    graph.add_conditional_edges(
        "early_stop_check",
        make_edge_should_continue(config),
        {
            "continue_debate": "debater_a_rebuttal",
            "go_to_judge":     "judge_verdict",
        },
    )

    # Conditional: jury enabled or skip straight to evaluate
    jury_enabled = config.get("jury", {}).get("enabled", False)
    graph.add_conditional_edges(
        "jury_check",
        lambda state: "run_jury" if jury_enabled else "skip_jury",
        {
            "run_jury":  "juror_1_assess",
            "skip_jury": "evaluate_result",
        },
    )

    return graph.compile()