"""
src/utils/data_loader.py
------------------------
Loads debate cases from demo JSON or real SciFact dataset.
Returns a flat list of records ready for make_initial_state().
"""

import json
import os
import random
from typing import Optional
from src.utils.schemas import STANCE_SUPPORT, STANCE_REFUTE, STANCE_NEI


# ── Label mapping ──────────────────────────────────────────────────────

SCIFACT_LABEL_MAP = {
    "SUPPORT":     STANCE_SUPPORT,
    "CONTRADICT":  STANCE_REFUTE,
    "NOINFO":      STANCE_NEI,
    "NOT_ENOUGH_INFO": STANCE_NEI,
}


def load_demo_cases(path: str) -> list[dict]:
    """Load hand-crafted demo cases from JSON."""
    with open(path) as f:
        cases = json.load(f)
    return [
        {
            "case_id":          c["id"],
            "claim":            c["claim"],
            "evidence_snippets":c["evidence_snippets"],
            "ground_truth":     c["label"],
            "source":           c.get("source", "demo"),
        }
        for c in cases
    ]


def load_scifact(
    corpus_path: str,
    claims_path: str,
    num_samples: int = 100,
    seed: int = 42,
) -> list[dict]:
    """
    Load SciFact and join claims with their evidence abstracts.
    Falls back to demo cases if files not found.
    """
    if not os.path.exists(corpus_path) or not os.path.exists(claims_path):
        print(f"[Loader] SciFact files not found — use demo cases instead.")
        return []

    # Load corpus
    corpus = {}
    with open(corpus_path) as f:
        for line in f:
            doc = json.loads(line.strip())
            corpus[doc["doc_id"]] = doc

    # Load claims
    records = []
    with open(claims_path) as f:
        for line in f:
            c = json.loads(line.strip())

            # ── LABEL EXTRACTION (critical fix) ────────────────────────
            # Real SciFact has NO top-level "label" field.
            # The label is nested inside evidence[doc_id][0]["label"].
            # c.get("label","") always returns "" → always maps to NEI.
            #
            # Real structure:
            # {
            #   "id": 1,
            #   "claim": "...",
            #   "cited_doc_ids": [12345],
            #   "evidence": {
            #     "12345": [{"sentences": [0], "label": "CONTRADICT"}]
            #   }
            # }
            evidence_dict = c.get("evidence", {})
            raw_label = ""
            for doc_id, ev_list in evidence_dict.items():
                # ev_list can be a list of dicts or a dict directly
                if isinstance(ev_list, list) and ev_list:
                    raw_label = ev_list[0].get("label", "")
                elif isinstance(ev_list, dict):
                    raw_label = ev_list.get("label", "")
                if raw_label:
                    break

            # Claims with no annotated evidence = NOT_ENOUGH_INFO
            if not raw_label and not evidence_dict:
                raw_label = "NOINFO"

            label = SCIFACT_LABEL_MAP.get(raw_label.upper(), STANCE_NEI)
            # ────────────────────────────────────────────────────────────

            cited = c.get("cited_doc_ids", [])
            if not cited:
                continue
            doc = corpus.get(cited[0])
            if not doc:
                continue
            abstract = doc.get("abstract", [])
            records.append({
                "case_id":          str(c["id"]),
                "claim":            c["claim"],
                "evidence_snippets":abstract[:5],   # up to 5 sentences
                "ground_truth":     label,
                "source":           "scifact",
            })

    random.seed(seed)
    random.shuffle(records)
    sampled = records[:num_samples]

    sup  = sum(1 for r in sampled if r["ground_truth"] == STANCE_SUPPORT)
    ref  = sum(1 for r in sampled if r["ground_truth"] == STANCE_REFUTE)
    nei  = sum(1 for r in sampled if r["ground_truth"] == STANCE_NEI)
    print(f"[Loader] SciFact: {len(sampled)} claims | "
          f"SUPPORT={sup} REFUTE={ref} NEI={nei}")
    return sampled


def load_dataset(config: dict) -> list[dict]:
    """Main loader. Returns demo cases if SciFact not available."""
    scifact = load_scifact(
        corpus_path=config["dataset"]["scifact_corpus"],
        claims_path=config["dataset"]["scifact_claims"],
        num_samples=config["dataset"]["num_samples"],
    )
    if scifact:
        return scifact
    print("[Loader] Using demo cases.")
    return load_demo_cases(config["dataset"]["demo_path"])