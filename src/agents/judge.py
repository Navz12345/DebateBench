"""
src/agents/judge.py
-------------------
Judge agent. Uses Llama-3.1-8B-Instruct (separate from debater model)
for independent evaluation.

System message establishes the judge's evaluator persona before the
task arrives, which improves verdict consistency across claims.
"""

import os
from src.models.university_client import UniversityClient
from src.utils.schemas import (
    JudgeOutput, TranscriptTurn,
    validate_judge_output,
)

SYSTEM_JUDGE = (
    "You are an impartial scientific judge with expertise in evaluating "
    "evidence-based claims. You assess arguments strictly on the accuracy "
    "of their evidence citations and the soundness of their logic. "
    "You are not swayed by confident tone — only by correct reading of "
    "the provided evidence snippets."
)


def _format_evidence(snippets: list[str]) -> str:
    return "\n".join(f"[{i}] {s}" for i, s in enumerate(snippets))


def _format_full_transcript(turns: list[TranscriptTurn]) -> str:
    if not turns:
        return "(Empty transcript)"
    parts = []
    for t in turns:
        label = f"Round {t['round']} — {t['agent']} [{t['stance']}, conf={t['confidence']}]"
        parts.append(f"{label}:\n  {t['reasoning']}")
        if t.get("counter"):
            parts.append(f"  Counter: {t['counter']}")
    return "\n\n".join(parts)


class Judge:
    def __init__(self, client: UniversityClient):
        self.client = client
        prompt_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "prompts",
            "judge.txt",
        )
        with open(prompt_path) as f:
            self.prompt_template = f.read()

    def evaluate(
        self,
        claim: str,
        evidence_snippets: list[str],
        transcript: list[TranscriptTurn],
    ) -> JudgeOutput:
        """
        Evaluate the full debate transcript and return a structured verdict.
        System message passed separately for clean role/task separation.
        Never raises — invalid output normalised to safe defaults.
        """
        prompt = self.prompt_template.format(
            CLAIM=claim,
            EVIDENCE=_format_evidence(evidence_snippets),
            TRANSCRIPT=_format_full_transcript(transcript),
        )
        raw = self.client.generate_json(prompt, system=SYSTEM_JUDGE)
        return validate_judge_output(raw)