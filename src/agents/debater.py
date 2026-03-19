"""
src/agents/debater.py
---------------------
Single Debater agent class. Instantiated twice (as A and B) with
different prompt files. Role diversity comes from prompt design,
not from different model classes — this keeps the architecture clean
and the comparison fair (same base model for both debaters).

PROMPT ITERATION LOG (for blog post Section 4):
  v1: Just "argue for/against this claim" — outputs were generic,
      no evidence grounding, frequent JSON parse failures.
  v2: Added explicit JSON schema — parse success improved but agents
      still reasoned from world knowledge, not provided snippets.
  v3: Added evidence index requirement — agents now cite snippets.
  v4: Added OPPONENT_LAST + counter_to_opponent — rebuttal quality improved.
  v5 (current): Added system message (persona/role separation) + verbatim
      quoting instruction + enabled Qwen3 thinking mode for deep reasoning.
"""

import os
from src.models.university_client import UniversityClient
from src.utils.schemas import (
    DebaterOutput, TranscriptTurn,
    validate_debater_output, STANCE_NEI
)

# System messages establish the model's persona before seeing any task.
# Separating "who you are" (system) from "what to do" (user) improves
# instruction-following and reduces role confusion across rounds.
SYSTEM_DEBATER_A = (
    "You are a skilled debate advocate assigned to argue the SUPPORT side. "
    "Think of yourself as a lawyer building the strongest possible case for "
    "your client. Your job is advocacy — find and present the best reading "
    "of the evidence that supports the claim. Always argue SUPPORT. "
    "You must always produce a non-empty reasoning and never return blank fields."
)

SYSTEM_DEBATER_B = (
    "You are a skilled debate advocate assigned to argue the REFUTE side. "
    "Think of yourself as a lawyer identifying every flaw in the opposing "
    "case. Your job is advocacy — find scope mismatches, contradictions, and "
    "overgeneralizations in how the claim relates to the evidence. "
    "Always argue REFUTE. You must always produce a non-empty reasoning."
)


def _format_evidence(snippets: list[str]) -> str:
    return "\n".join(f"[{i}] {s}" for i, s in enumerate(snippets))


def _format_transcript(turns: list[TranscriptTurn]) -> str:
    if not turns:
        return "(No previous turns)"
    parts = []
    for t in turns:
        label = f"Round {t['round']} — {t['agent']} [{t['stance']}]"
        parts.append(f"{label}:\n  Reasoning: {t['reasoning'][:400]}")
        if t.get("counter"):
            parts.append(f"  Counter: {t['counter'][:250]}")
    return "\n".join(parts)


def _get_opponent_last(turns: list[TranscriptTurn], my_agent: str) -> str:
    for t in reversed(turns):
        if t["agent"] != my_agent:
            return t["reasoning"][:500]
    return ""


class Debater:
    """
    One debater agent. Instantiate as A (SUPPORT) or B (REFUTE).
    Same UniversityClient/model for both — role comes from system message
    + prompt file. System message is passed separately so Qwen3's thinking
    mode and Llama's system-message handling both work correctly.
    """

    def __init__(
        self,
        name: str,
        prompt_file: str,
        client: UniversityClient,
        system_message: str = "",
    ):
        self.name           = name
        self.client         = client
        self.system_message = system_message

        prompt_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "prompts",
            prompt_file,
        )
        with open(prompt_path) as f:
            self.prompt_template = f.read()

    def _build_prompt(
        self,
        claim: str,
        evidence_snippets: list[str],
        transcript: list[TranscriptTurn],
        round_num: int,
    ) -> str:
        opponent_last = _get_opponent_last(transcript, self.name)
        return self.prompt_template.format(
            CLAIM=claim,
            EVIDENCE=_format_evidence(evidence_snippets),
            TRANSCRIPT=_format_transcript(transcript),
            OPPONENT_LAST=opponent_last if opponent_last else "(First round — no opponent argument yet)",
        )

    def argue(
        self,
        claim: str,
        evidence_snippets: list[str],
        transcript: list[TranscriptTurn],
        round_num: int,
    ) -> tuple[DebaterOutput, TranscriptTurn]:
        """
        Generate one argument. System message passed separately so
        Qwen3 can use it alongside thinking mode.
        Never raises — invalid output normalised to safe defaults.
        """
        prompt = self._build_prompt(claim, evidence_snippets, transcript, round_num)
        raw    = self.client.generate_json(
            prompt,
            system=self.system_message or None,
        )
        output = validate_debater_output(raw, len(evidence_snippets))

        turn = TranscriptTurn(
            round=round_num,
            agent=self.name,
            stance=output["stance"],
            reasoning=output["reasoning"],
            counter=output["counter_to_opponent"],
            confidence=output["confidence"],
        )
        return output, turn