"""
src/agents/jury.py
------------------
Role-specialized jury panel with smart two-phase deliberation.
Implements the full recommended logic:

STEP 1: 3 jurors vote independently (Phase 1)
STEP 2: Unanimous → finalize immediately
STEP 3: Disagreement + NEI present → focused NEI deliberation round
STEP 4: After deliberation:
         - calibration juror conf=5 NEI + others weak → arbiter judge
         - decisive majority + confident → finalize
         - otherwise → arbiter
STEP 5: Log ambiguity_flag, deliberation_changed, arbiter_used

AGGREGATION SAFEGUARD:
  Old rule: 2 REFUTE vs 1 NEI → REFUTE wins always
  New rule: if NEI juror has high confidence (>=4) AND
            majority jurors have low/medium confidence (<=3)
            → NOT_ENOUGH_INFO wins as ambiguity safeguard

CALIBRATION VETO:
  If calibration juror votes NEI with confidence=5 → ambiguity flag raised
  System does not finalize without arbiter review
"""

import os
import random
from collections import Counter
from src.models.university_client import UniversityClient
from src.utils.schemas import (
    JurorOutput, validate_juror_output,
    VALID_STANCES, STANCE_NEI,
)

# ── System messages ────────────────────────────────────────────────────

SYSTEM_JUROR_EVIDENCE = (
    "You are the Evidence Judge on a three-person jury. "
    "Your job is to evaluate how accurately each debater cited "
    "the provided evidence snippets. You prioritize verbatim accuracy "
    "over rhetorical skill. A debater who misquotes or paraphrases "
    "incorrectly loses your vote, even if their conclusion seems right."
)

SYSTEM_JUROR_LOGIC = (
    "You are the Logic Judge on a three-person jury. "
    "Your job is to evaluate the quality of reasoning and internal "
    "consistency across rounds. You check for contradictions, unsupported "
    "leaps, and whether each debater genuinely addressed the opponent's "
    "strongest points or just repeated their opening claim."
)

SYSTEM_JUROR_CALIBRATION = (
    "You are the Calibration Judge on a three-person jury. "
    "Your job is to verify the verdict is proportionate to the evidence — "
    "neither overclaiming nor underclaiming. "
    "You rule SUPPORT when evidence clearly supports the claim, "
    "REFUTE when evidence clearly contradicts it, and "
    "NOT_ENOUGH_INFO only when the evidence genuinely does not address "
    "the claim at all. NOT_ENOUGH_INFO is a last resort, not a default."
)

JUROR_ROLES = [
    ("Juror_1", "evidence",    SYSTEM_JUROR_EVIDENCE,    "juror_evidence.txt",    0.2),
    ("Juror_2", "logic",       SYSTEM_JUROR_LOGIC,       "juror_logic.txt",       0.3),
    ("Juror_3", "calibration", SYSTEM_JUROR_CALIBRATION, "juror_calibration.txt", 0.4),
]

SYSTEM_JUROR = SYSTEM_JUROR_EVIDENCE   # generic fallback alias


def _format_evidence(snippets: list[str]) -> str:
    return "\n".join(f"[{i}] {s}" for i, s in enumerate(snippets))


def _format_phase1_votes(assessments: list[JurorOutput]) -> str:
    lines = []
    for a in assessments:
        lines.append(
            f"{a['juror_id']} ({a['role'].upper()} JUDGE) — "
            f"Verdict: {a['verdict']}  Confidence: {a['confidence']}/5\n"
            f"  Reasoning: {a['reasoning'][:350]}\n"
            f"  Best SUPPORT arg: {a['strongest_arg_support'][:200]}\n"
            f"  Best REFUTE arg:  {a['strongest_arg_refute'][:200]}"
        )
    return "\n\n".join(lines)


class JuryPanel:
    """
    Panel of 3 role-specialized jurors with smart aggregation.
    """

    # Confidence thresholds for ambiguity detection
    NEI_VETO_CONF     = 5    # calibration juror NEI at this conf → veto
    NEI_FLAG_CONF     = 4    # any juror NEI at this conf → ambiguity flag
    WEAK_CONF_CEILING = 3    # majority jurors at or below this → ambiguity

    def __init__(self, jury_client: UniversityClient,
                 judge_client: UniversityClient = None):
        """
        jury_client:  model used for all 3 jurors
        judge_client: used as arbiter when ambiguity flag is raised
                      (falls back to jury_client if not provided)
        """
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        prompts_dir = os.path.join(base, "prompts")

        self.jurors = []
        for juror_id, role, system_msg, prompt_file, temp in JUROR_ROLES:
            client = UniversityClient(
                base_url=jury_client._client.base_url,
                api_key=jury_client._client.api_key,
                model=jury_client.model,
                temperature=temp,
                max_tokens=jury_client.max_tokens,
                timeout=jury_client.timeout,
            )
            with open(os.path.join(prompts_dir, prompt_file)) as f:
                phase1_template = f.read()
            self.jurors.append({
                "juror_id":        juror_id,
                "role":            role,
                "system_message":  system_msg,
                "phase1_template": phase1_template,
                "client":          client,
            })

        with open(os.path.join(prompts_dir, "juror_phase2.txt")) as f:
            self.phase2_template = f.read()
        with open(os.path.join(prompts_dir, "juror_phase2_nei.txt")) as f:
            self.phase2_nei_template = f.read()

        # Arbiter: judge model for escalation
        self.arbiter_client = judge_client or jury_client

    # ── Phase 1: Independent assessment ────────────────────────────────
    def assess_independently(self, claim, evidence_snippets, transcript_text):
        assessments = []
        evidence = _format_evidence(evidence_snippets)
        for j in self.jurors:
            prompt = j["phase1_template"].format(
                CLAIM=claim, EVIDENCE=evidence, TRANSCRIPT=transcript_text,
            )
            raw = j["client"].generate_json(prompt, system=j["system_message"])
            out = validate_juror_output(raw, j["juror_id"], phase=1, role=j["role"])
            assessments.append(out)
        return assessments

    # ── Consensus check ─────────────────────────────────────────────────
    @staticmethod
    def is_unanimous(assessments):
        return len(set(a["verdict"] for a in assessments)) == 1

    # ── Ambiguity detection ─────────────────────────────────────────────
    def has_nei_signal(self, votes: list[JurorOutput]) -> bool:
        """
        Returns True only if a juror voted NEI with meaningful confidence (>=3).
        Low-confidence NEI (1-2) means the juror was uncertain about NEI itself
        — not a strong enough signal to trigger NEI-focused deliberation.
        Using conf>=3 prevents the aggressive two-stage prompt from converting
        high-confidence REFUTE jurors to NEI based on a weak NEI signal.
        """
        return any(
            v["verdict"] == STANCE_NEI and v["confidence"] >= 3
            for v in votes
        )

    def calibration_vetoes(self, votes: list[JurorOutput]) -> bool:
        """
        True if the calibration juror voted NEI with conf >= NEI_VETO_CONF.
        This juror is specifically designed to catch evidence insufficiency,
        so a high-confidence NEI from it is a strong signal.
        """
        for v in votes:
            if v["role"] == "calibration" and \
               v["verdict"] == STANCE_NEI and \
               v["confidence"] >= self.NEI_VETO_CONF:
                return True
        return False

    def ambiguity_flag(self, votes: list[JurorOutput]) -> bool:
        """
        True when the old majority rule would override a high-confidence NEI.
        Condition: NEI juror has high conf AND majority jurors have weak conf.
        """
        nei_votes     = [v for v in votes if v["verdict"] == STANCE_NEI]
        non_nei_votes = [v for v in votes if v["verdict"] != STANCE_NEI]
        if not nei_votes or not non_nei_votes:
            return False
        nei_high    = any(v["confidence"] >= self.NEI_FLAG_CONF for v in nei_votes)
        others_weak = all(v["confidence"] <= self.WEAK_CONF_CEILING for v in non_nei_votes)
        return nei_high and others_weak

    # ── Smart aggregation ───────────────────────────────────────────────
    @staticmethod
    def majority_verdict(votes: list[JurorOutput]) -> tuple[str, float]:
        """Plain majority (used internally). Returns (verdict, disagreement)."""
        verdicts = [v["verdict"] for v in votes]
        counts   = Counter(verdicts)
        majority = counts.most_common(1)[0][0]
        disagree = {1: 0.0, 2: 0.5, 3: 1.0}.get(len(counts), 0.5)
        return majority, disagree

    def smart_verdict(self, votes: list[JurorOutput]) -> tuple[str, float, bool]:
        """
        Ambiguity-aware aggregation.
        Returns (verdict, disagreement, ambiguity_flag_raised).

        Safeguard: if ambiguity_flag() is True, NEI wins over REFUTE/SUPPORT
        to prevent the calibration juror's signal from being silently overridden.
        """
        plain_verdict, disagree = self.majority_verdict(votes)

        # If calibration juror vetoes or ambiguity flag → NEI safeguard
        if self.calibration_vetoes(votes) or self.ambiguity_flag(votes):
            return STANCE_NEI, disagree, True

        return plain_verdict, disagree, False

    # ── Phase 2: NEI-focused deliberation ───────────────────────────────
    def deliberate_nei_focused(self, claim, evidence_snippets,
                                transcript_text, phase1_assessments):
        """
        Focused deliberation when NEI signal was raised.
        Each juror explicitly answers: is evidence sufficient for a decision?
        """
        final_votes = []
        evidence       = _format_evidence(evidence_snippets)
        phase1_summary = _format_phase1_votes(phase1_assessments)

        for juror_out, j in zip(phase1_assessments, self.jurors):
            others = [a for a in phase1_assessments
                      if a["juror_id"] != juror_out["juror_id"]]
            random.shuffle(others)
            shuffled = [juror_out] + others
            summary  = _format_phase1_votes(shuffled)

            prompt = self.phase2_nei_template.format(
                CLAIM=claim, EVIDENCE=evidence, TRANSCRIPT=transcript_text,
                PHASE1_VOTES=summary,
                MY_PHASE1_VERDICT=juror_out["verdict"],
                MY_PHASE1_REASONING=juror_out["reasoning"],
            )
            raw = j["client"].generate_json(prompt, system=j["system_message"])
            out = validate_juror_output(
                raw, j["juror_id"], phase=2,
                phase1_verdict=juror_out["verdict"], role=j["role"],
            )
            final_votes.append(out)
        return final_votes

    # ── Phase 2: General deliberation ──────────────────────────────────
    def deliberate(self, claim, evidence_snippets, transcript_text,
                   phase1_assessments):
        """General deliberation round with shuffled review order."""
        final_votes = []
        evidence    = _format_evidence(evidence_snippets)

        for juror_out, j in zip(phase1_assessments, self.jurors):
            others = [a for a in phase1_assessments
                      if a["juror_id"] != juror_out["juror_id"]]
            random.shuffle(others)
            summary = _format_phase1_votes([juror_out] + others)

            prompt = self.phase2_template.format(
                CLAIM=claim, EVIDENCE=evidence, TRANSCRIPT=transcript_text,
                PHASE1_VOTES=summary,
                MY_PHASE1_VERDICT=juror_out["verdict"],
                MY_PHASE1_REASONING=juror_out["reasoning"],
            )
            raw = j["client"].generate_json(prompt, system=j["system_message"])
            out = validate_juror_output(
                raw, j["juror_id"], phase=2,
                phase1_verdict=juror_out["verdict"], role=j["role"],
            )
            final_votes.append(out)
        return final_votes

    # ── Arbiter call ────────────────────────────────────────────────────
    def call_arbiter(self, claim, evidence_snippets, transcript_text) -> str:
        """
        Ask the single judge model to make a final verdict.
        Used when jury deliberation is inconclusive or ambiguity flag persists.
        """
        from src.agents.judge import Judge
        evidence = _format_evidence(evidence_snippets)
        # Use a minimal transcript summary for arbiter
        from src.utils.schemas import TranscriptTurn
        arbiter_judge = Judge(self.arbiter_client)
        # Build minimal turn list from transcript text
        turns = [TranscriptTurn(
            round=0, agent="Summary", stance="DISPUTED",
            reasoning=transcript_text[:1000], counter="", confidence=3,
        )]
        result = arbiter_judge.evaluate(claim, evidence_snippets, turns)
        return result.get("final_verdict", STANCE_NEI)

    # ── Confidence helpers ──────────────────────────────────────────────
    @staticmethod
    def confidence_spread(votes):
        confs = [v["confidence"] for v in votes]
        return float(max(confs) - min(confs))

    @staticmethod
    def weighted_verdict(votes):
        from collections import defaultdict
        totals: dict = defaultdict(int)
        for v in votes:
            totals[v["verdict"]] += v["confidence"]
        return max(totals, key=lambda k: totals[k])

    # ── Main run() ──────────────────────────────────────────────────────
    def run(self, claim, evidence_snippets, transcript_text) -> dict:
        """
        Smart jury deliberation pipeline.

        STEP 1: Phase 1 independent votes
        STEP 2: Unanimous → finalize
        STEP 3: Disagreement + NEI present → NEI-focused deliberation
        STEP 4: After deliberation:
                 - ambiguity flag + calibration veto → arbiter
                 - decisive majority + confident → finalize
        STEP 5: Return full log (ambiguity_flag, deliberation_changed, arbiter_used)
        """
        # ── STEP 1: Independent ─────────────────────────────────────────
        phase1 = self.assess_independently(claim, evidence_snippets, transcript_text)
        p1_verdict, p1_disagree = self.majority_verdict(phase1)

        # ── STEP 2: Unanimous → done ────────────────────────────────────
        if self.is_unanimous(phase1):
            return {
                "jury_assessments":   phase1,
                "jury_final_votes":   phase1,
                "jury_verdict":       p1_verdict,
                "jury_disagreement":  0.0,
                "_p1_disagreement":   0.0,
                "_minds_changed":     0,
                "_consensus_at_init": True,
                "_ambiguity_flag":    False,
                "_arbiter_used":      False,
                "_deliberation_changed": False,
                "_weighted_verdict":  p1_verdict,
                "_confidence_spread": self.confidence_spread(phase1),
            }

        # ── STEP 3: Disagreement + NEI → focused deliberation ───────────
        nei_present = self.has_nei_signal(phase1)
        if nei_present:
            phase2 = self.deliberate_nei_focused(
                claim, evidence_snippets, transcript_text, phase1
            )
            deliberation_type = "nei_focused"
        else:
            phase2 = self.deliberate(
                claim, evidence_snippets, transcript_text, phase1
            )
            deliberation_type = "general"

        changed        = sum(1 for v in phase2 if v["changed_mind"])
        p2_old_verdict, p2_disagree = self.majority_verdict(phase2)

        # ── STEP 4: Smart aggregation + arbiter decision ─────────────────
        p2_verdict, p2_disagree, flag = self.smart_verdict(phase2)
        arbiter_used = False

        if flag:
            # Ambiguity flag raised — check if arbiter needed
            # Arbiter used if: calibration veto OR jury still split after NEI deliberation
            needs_arbiter = (
                self.calibration_vetoes(phase2) or
                (p2_disagree >= 0.5 and nei_present)
            )
            if needs_arbiter:
                arbiter_verdict = self.call_arbiter(
                    claim, evidence_snippets, transcript_text
                )
                p2_verdict   = arbiter_verdict
                arbiter_used = True

        return {
            "jury_assessments":      phase1,
            "jury_final_votes":      phase2,
            "jury_verdict":          p2_verdict,
            "jury_disagreement":     p1_disagree,   # report Phase 1 disagreement — Phase 1 reflects true diversity before deliberation pressure
            "_p1_disagreement":      p1_disagree,
            "_p2_disagreement":      p2_disagree,   # also log Phase 2 for analysis
            "_minds_changed":        changed,
            "_consensus_at_init":    False,
            "_ambiguity_flag":       flag,
            "_arbiter_used":         arbiter_used,
            "_deliberation_changed": (p2_old_verdict != p1_verdict),
            "_weighted_verdict":     self.weighted_verdict(phase2),
            "_confidence_spread":    self.confidence_spread(phase2),
        }