"""
src/agents/baselines.py
-----------------------
Baseline methods for comparison.

Baseline 1 — Direct QA (CoT):
  Single LLM call with chain-of-thought, no debate.

Baseline 2 — Self-Consistency (Wang et al., 2023):
  N=9 samples, majority vote.
  N=9 matches total LLM calls in a 3-round debate:
    2 (init) + 3*2 (rounds) + 1 (judge) = 9 calls

Both baselines use the DEBATER model (Qwen), not the judge model.
"""

import time
from collections import Counter
from src.models.ollama_client import OllamaClient
from src.utils.schemas import VALID_STANCES, STANCE_NEI


DIRECT_QA_PROMPT = """\
You are a scientific fact-checker.
Determine whether the following evidence SUPPORTS or REFUTES the claim,
or if there is NOT_ENOUGH_INFO.

CLAIM: {CLAIM}

EVIDENCE:
{EVIDENCE}

Think step by step, then respond ONLY with valid JSON:
{{
  "verdict": "SUPPORT",
  "reasoning": "your reasoning here",
  "confidence": 4
}}
"""

SC_SAMPLE_PROMPT = """\
Does the evidence SUPPORT or REFUTE this claim, or is there NOT_ENOUGH_INFO?

CLAIM: {CLAIM}
EVIDENCE: {EVIDENCE}

Respond ONLY with valid JSON:
{{"verdict": "SUPPORT", "reasoning": "brief"}}
"""


def _fmt_evidence(snippets: list[str]) -> str:
    return "\n".join(f"[{i}] {s}" for i, s in enumerate(snippets))


def _parse_verdict(raw: dict) -> str:
    v = str(raw.get("verdict", "")).upper().strip()
    return v if v in VALID_STANCES else STANCE_NEI


class DirectQABaseline:
    def __init__(self, client: OllamaClient):
        self.client = client

    def run(self, record: dict) -> dict:
        prompt = DIRECT_QA_PROMPT.format(
            CLAIM=record["claim"],
            EVIDENCE=_fmt_evidence(record["evidence_snippets"]),
        )
        start = time.time()
        raw   = self.client.generate_json(prompt)
        dur   = round(time.time() - start, 2)

        verdict = _parse_verdict(raw)
        return {
            "baseline":     "direct_qa",
            "case_id":      record["case_id"],
            "claim":        record["claim"],
            "verdict":      verdict,
            "reasoning":    raw.get("reasoning", ""),
            "confidence":   raw.get("confidence", 3),
            "ground_truth": record["ground_truth"],
            "correct":      verdict == record["ground_truth"],
            "duration_seconds": dur,
        }


class SelfConsistencyBaseline:
    def __init__(self, client: OllamaClient, num_samples: int = 9):
        self.client     = client
        self.num_samples = num_samples

    def run(self, record: dict) -> dict:
        prompt = SC_SAMPLE_PROMPT.format(
            CLAIM=record["claim"],
            EVIDENCE=_fmt_evidence(record["evidence_snippets"]),
        )
        start   = time.time()
        samples = []
        for i in range(self.num_samples):
            raw     = self.client.generate_json(prompt, temperature=0.9)
            verdict = _parse_verdict(raw)
            samples.append({"sample": i + 1, "verdict": verdict})
        dur = round(time.time() - start, 2)

        verdicts = [s["verdict"] for s in samples]
        majority = Counter(verdicts).most_common(1)[0][0]
        correct  = majority == record["ground_truth"]

        return {
            "baseline":     "self_consistency",
            "case_id":      record["case_id"],
            "claim":        record["claim"],
            "num_samples":  self.num_samples,
            "samples":      samples,
            "verdict":      majority,
            "ground_truth": record["ground_truth"],
            "correct":      correct,
            "duration_seconds": dur,
        }