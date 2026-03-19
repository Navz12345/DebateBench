"""
src/utils/schemas.py
--------------------
All shared types for DebateBench.

DESIGN NOTES:
  - DebateState is the single TypedDict that flows through LangGraph.
  - Every field has a default so make_initial_state() can initialize cleanly.
  - Annotated[list, operator.add] fields ACCUMULATE across graph nodes —
    transcript turns append automatically without manual merging.
  - All LLM outputs are validated through DebaterOutput and JudgeOutput
    before touching the state. Invalid values are normalized, never raised.
"""

from __future__ import annotations
from typing import TypedDict, Annotated, Optional
import operator


# ══════════════════════════════════════════════════════════════════════
# Enums (as string constants — avoids import overhead)
# ══════════════════════════════════════════════════════════════════════

STANCE_SUPPORT = "SUPPORT"
STANCE_REFUTE  = "REFUTE"
STANCE_NEI     = "NOT_ENOUGH_INFO"
VALID_STANCES  = {STANCE_SUPPORT, STANCE_REFUTE, STANCE_NEI}

WINNER_A       = "Debater A"
WINNER_B       = "Debater B"
WINNER_TIE     = "TIE"


# ══════════════════════════════════════════════════════════════════════
# LLM Output Schemas (parsed + validated from raw LLM JSON)
# ══════════════════════════════════════════════════════════════════════

class DebaterOutput(TypedDict):
    """
    Validated output from one debater call.
    All fields have safe defaults — never raises on bad LLM output.
    """
    stance:              str          # SUPPORT | REFUTE | NOT_ENOUGH_INFO
    epistemic_status:    str          # CERTAIN | LEANING | DOUBTFUL | CONCEDED
    reasoning:           str          # chain-of-thought explanation
    evidence_used:       list[int]    # indices into evidence_snippets
    counter_to_opponent: str          # rebuttal to other debater
    confidence:          int          # 1-5
    conceded:            bool         # True if [CONCEDE] token present in reasoning


class JudgeOutput(TypedDict):
    """
    Validated output from the judge.
    Expanded to capture per-debater strongest/weakest arguments
    for richer analysis and assignment compliance.
    """
    final_verdict:           str    # SUPPORT | REFUTE | NOT_ENOUGH_INFO
    winning_side:            str    # "Debater A" | "Debater B" | "TIE"
    reasoning:               str    # step-by-step evaluation (CoT)
    strongest_argument_from_a: str  # best argument Debater A made
    strongest_argument_from_b: str  # best argument Debater B made
    weakest_argument_from_a:   str  # weakest argument Debater A made
    weakest_argument_from_b:   str  # weakest argument Debater B made
    confidence:              int    # 1-5


class JurorOutput(TypedDict):
    """
    Output from one juror in one phase.
    Richer schema than single judge — captures role-specific analysis.
    """
    juror_id:                str    # "Juror_1" | "Juror_2" | "Juror_3"
    role:                    str    # "evidence" | "logic" | "calibration"
    phase:                   int    # 1=independent, 2=post-deliberation
    verdict:                 str    # SUPPORT | REFUTE | NOT_ENOUGH_INFO
    reasoning:               str
    confidence:              int    # 1-5
    strongest_arg_support:   str    # best argument made for SUPPORT side
    strongest_arg_refute:    str    # best argument made for REFUTE side
    evidence_alignment:      int    # 1-5: how well debaters cited evidence
    self_reflection:         str    # Phase 2 only: weakness in own Phase 1 reasoning
    changed_mind:            bool   # True if Phase 2 verdict differs from Phase 1


def validate_juror_output(raw: dict, juror_id: str, phase: int,
                          phase1_verdict: str = "",
                          role: str = "") -> "JurorOutput":
    """Normalize raw LLM dict into a valid JurorOutput."""
    verdict = str(raw.get("verdict", "")).upper().strip()
    if verdict not in VALID_STANCES:
        raw_text = raw.get("_raw_text", "")
        if raw_text:
            from src.models.university_client import UniversityClient
            extracted = UniversityClient.extract_stance_from_text(raw_text)
            verdict = extracted if extracted in VALID_STANCES else STANCE_NEI
        else:
            verdict = STANCE_NEI

    try:
        conf = int(raw.get("confidence", 3))
        conf = max(1, min(5, conf))
    except (TypeError, ValueError):
        conf = 3

    try:
        ev_align = int(raw.get("evidence_alignment", 3))
        ev_align = max(1, min(5, ev_align))
    except (TypeError, ValueError):
        ev_align = 3

    reasoning = str(raw.get("reasoning", "")).strip()
    if not reasoning:
        reasoning = raw.get("_raw_text", "")[:400]

    return JurorOutput(
        juror_id=juror_id,
        role=role,
        phase=phase,
        verdict=verdict,
        reasoning=reasoning,
        confidence=conf,
        strongest_arg_support=str(raw.get("strongest_arg_support", "")).strip(),
        strongest_arg_refute=str(raw.get("strongest_arg_refute",  "")).strip(),
        evidence_alignment=ev_align,
        self_reflection=str(raw.get("self_reflection", "")).strip(),
        changed_mind=(phase == 2 and verdict != phase1_verdict),
    )


class TranscriptTurn(TypedDict):
    """One turn in the debate transcript."""
    round:       int
    agent:       str
    stance:      str
    reasoning:   str
    counter:     str
    confidence:  int
    """One turn in the debate transcript."""
    round:       int     # 0 = init, 1+ = rebuttal rounds
    agent:       str     # "Debater A" | "Debater B" | "Judge"
    stance:      str
    reasoning:   str
    counter:     str
    confidence:  int


# ══════════════════════════════════════════════════════════════════════
# LangGraph State
# ══════════════════════════════════════════════════════════════════════

class DebateState(TypedDict):
    """
    Shared state that flows through every LangGraph node.

    REDUCER NOTES:
      transcript: Annotated[list, operator.add]
        → Each node appends [new_turn]; LangGraph merges automatically.
          Without this you'd need to pass the full list and manually append.

      All other fields: replace semantics (last write wins).
    """

    # ── Input ──────────────────────────────────────────────────────────
    case_id:           str
    claim:             str
    evidence_snippets: list[str]
    ground_truth:      str          # SUPPORT | REFUTE | NOT_ENOUGH_INFO
    debater_model:     str
    judge_model:       str

    # ── Phase 1: Initialization ────────────────────────────────────────
    a_output:          Optional[DebaterOutput]
    b_output:          Optional[DebaterOutput]
    a_stance:          str
    b_stance:          str

    # ── Phase 2: Debate rounds ─────────────────────────────────────────
    transcript:        Annotated[list, operator.add]   # accumulates turns
    current_round:     int
    max_rounds:        int
    min_rounds:        int
    consecutive_same:  int          # consecutive rounds both hold same stance
    early_stopped:     bool
    debate_finished:   bool
    concession_round:  int          # round where first [CONCEDE] appeared (-1 = none)
    a_epistemic:       str          # latest epistemic status from Debater A
    b_epistemic:       str          # latest epistemic status from Debater B

    # ── Phase 3: Judge ─────────────────────────────────────────────────
    judge_output:      Optional[JudgeOutput]
    final_verdict:     str          # single judge verdict

    # ── Phase 3b: Jury (only populated when jury_enabled=True) ─
    jury_assessments:  list                # Phase 1 votes (set by jury_deliberate)
    jury_final_votes:  list                # Phase 2 votes after deliberation
    jury_verdict:      str           # majority verdict from jury
    jury_disagreement: float         # 0.0=unanimous, 1.0=maximum disagreement
    jury_correct:      bool          # jury verdict == ground_truth
    _consensus_at_init: bool         # True = all jurors agreed at Phase 1
    _ambiguity_flag:   bool          # True = NEI safeguard triggered
    _arbiter_used:     bool          # True = arbiter judge called
    _minds_changed:    int           # count of jurors who changed verdict in Phase 2
    _p1_disagreement:  float         # Phase 1 disagreement score (pre-deliberation)
    _p2_disagreement:  float         # Phase 2 disagreement score (post-deliberation)
    _deliberation_changed: bool      # True if Phase 2 majority differs from Phase 1

    # ── Phase 4: Evaluation ────────────────────────────────────────────
    judge_correct:     bool
    total_turns:       int
    duration_seconds:  float
    error:             Optional[str]


def make_initial_state(
    case_id:           str,
    claim:             str,
    evidence_snippets: list[str],
    ground_truth:      str,
    debater_model:     str,
    judge_model:       str,
    min_rounds:        int = 3,
    max_rounds:        int = 5,
) -> DebateState:
    """Factory: returns a fresh DebateState with all fields initialized."""
    return DebateState(
        case_id=case_id,
        claim=claim,
        evidence_snippets=evidence_snippets,
        ground_truth=ground_truth,
        debater_model=debater_model,
        judge_model=judge_model,
        a_output=None,
        b_output=None,
        a_stance="",
        b_stance="",
        transcript=[],
        current_round=0,
        max_rounds=max_rounds,
        min_rounds=min_rounds,
        consecutive_same=0,
        early_stopped=False,
        debate_finished=False,
        concession_round=-1,
        a_epistemic="CERTAIN",
        b_epistemic="CERTAIN",
        judge_output=None,
        final_verdict="",
        jury_assessments=[],
        jury_final_votes=[],
        jury_verdict="",
        jury_disagreement=0.0,
        jury_correct=False,
        _consensus_at_init=False,
        _ambiguity_flag=False,
        _arbiter_used=False,
        _minds_changed=0,
        _p1_disagreement=0.0,
        _p2_disagreement=0.0,
        _deliberation_changed=False,
        judge_correct=False,
        total_turns=0,
        duration_seconds=0.0,
        error=None,
    )


# ══════════════════════════════════════════════════════════════════════
# Output Validation / Normalization
# ══════════════════════════════════════════════════════════════════════

def validate_debater_output(raw: dict, num_snippets: int) -> DebaterOutput:
    """
    Normalize raw LLM dict into a valid DebaterOutput.
    If JSON parse failed, raw contains {"_raw_text": "..."} — we try
    keyword extraction on the prose as a last resort.
    """
    # Stance — JSON field first
    stance = str(raw.get("stance", "")).upper().strip()
    if stance not in VALID_STANCES:
        # Text fallback: model returned prose instead of JSON
        raw_text = raw.get("_raw_text", "")
        if raw_text:
            from src.models.university_client import UniversityClient
            extracted = UniversityClient.extract_stance_from_text(raw_text)
            stance = extracted if extracted in VALID_STANCES else STANCE_NEI
        else:
            stance = STANCE_NEI

    # Evidence indexes — validate range, drop bad values
    raw_ev = raw.get("evidence_used", [])
    if not isinstance(raw_ev, list):
        raw_ev = []
    evidence_used = [
        int(i) for i in raw_ev
        if isinstance(i, (int, float)) and 0 <= int(i) < num_snippets
    ]

    # Confidence
    try:
        conf = int(raw.get("confidence", 3))
        conf = max(1, min(5, conf))
    except (TypeError, ValueError):
        conf = 3

    # Reasoning — check for empty string (model returned blank field)
    reasoning = str(raw.get("reasoning", "")).strip()
    if not reasoning:
        # Empty reasoning means model declined to argue — use raw_text fallback
        reasoning = raw.get("_raw_text", "")[:500]
        # If still empty and stance is valid, keep stance but flag low confidence
        if not reasoning and stance in VALID_STANCES:
            reasoning = f"[No reasoning provided — stance={stance}]"

    # Epistemic status — validate against allowed values
    VALID_EPISTEMIC = {"CERTAIN", "LEANING", "DOUBTFUL", "CONCEDED"}
    epistemic = str(raw.get("epistemic_status", "CERTAIN")).upper().strip()
    if epistemic not in VALID_EPISTEMIC:
        epistemic = "CERTAIN"

    # Concession detection — [CONCEDE] token in reasoning OR epistemic=CONCEDED
    conceded = "[CONCEDE]" in reasoning or epistemic == "CONCEDED"

    # If conceded, trust the stated stance (may have switched)
    # If epistemic=CONCEDED but stance not updated, keep stance as-is
    # (the graph will handle routing)

    return DebaterOutput(
        stance=stance,
        epistemic_status=epistemic,
        reasoning=reasoning,
        evidence_used=evidence_used,
        counter_to_opponent=str(raw.get("counter_to_opponent", "")).strip(),
        confidence=conf,
        conceded=conceded,
    )


def validate_judge_output(raw: dict) -> JudgeOutput:
    """Normalize raw LLM dict into a valid JudgeOutput."""
    verdict = str(raw.get("final_verdict", "")).upper().strip()
    if verdict not in VALID_STANCES:
        raw_text = raw.get("_raw_text", "")
        if raw_text:
            from src.models.university_client import UniversityClient
            extracted = UniversityClient.extract_stance_from_text(raw_text)
            verdict = extracted if extracted in VALID_STANCES else STANCE_NEI
        else:
            verdict = STANCE_NEI

    winner = str(raw.get("winning_side", WINNER_TIE)).strip()
    if winner not in (WINNER_A, WINNER_B, WINNER_TIE):
        winner = WINNER_TIE

    try:
        conf = int(raw.get("confidence", 3))
        conf = max(1, min(5, conf))
    except (TypeError, ValueError):
        conf = 3

    reasoning = str(raw.get("reasoning", "")).strip()
    if not reasoning:
        reasoning = raw.get("_raw_text", "")[:500]

    # Support both old field names (strongest_argument) and new per-debater fields
    # Old: strongest_argument / weakest_argument (backward compat with v1 logs)
    # New: strongest_argument_from_a/b / weakest_argument_from_a/b
    def _get_arg(new_key, old_key):
        return str(raw.get(new_key) or raw.get(old_key) or "").strip()

    return JudgeOutput(
        final_verdict=verdict,
        winning_side=winner,
        reasoning=reasoning,
        strongest_argument_from_a=_get_arg("strongest_argument_from_a", "strongest_argument"),
        strongest_argument_from_b=_get_arg("strongest_argument_from_b", "strongest_argument"),
        weakest_argument_from_a=  _get_arg("weakest_argument_from_a",   "weakest_argument"),
        weakest_argument_from_b=  _get_arg("weakest_argument_from_b",   "weakest_argument"),
        confidence=conf,
    )