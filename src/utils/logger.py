"""
src/utils/logger.py
-------------------
Saves every debate run as a structured JSON log.
Logging happens OUTSIDE the LangGraph — called from run_experiment.py
after app.invoke() returns, not as a graph node.

WHY NOT A GRAPH NODE:
  Logging is deterministic I/O with no state dependency.
  A logging node can fail silently (file I/O error) and interrupt
  graph execution. Keeping it outside the graph makes the graph
  cleaner and easier to test independently.
"""

import json
import os
import csv
from datetime import datetime
from src.utils.schemas import DebateState


class DebateLogger:
    def __init__(self, log_dir: str = "logs"):
        self.run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(log_dir, f"run_{self.run_id}")
        os.makedirs(self.run_dir, exist_ok=True)
        self._summary_rows: list[dict] = []
        print(f"[Logger] Saving to: {self.run_dir}")

    # ── Single debate log ──────────────────────────────────────────────
    def log_debate(self, state: dict, index: int) -> str:
        filename = f"debate_{index:04d}_{state.get('case_id', 'unknown')}.json"
        path = os.path.join(self.run_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, default=str, ensure_ascii=False)

        # Append to in-memory summary
        self._summary_rows.append({
            "index":              index,
            "case_id":            state.get("case_id", ""),
            "claim":              state.get("claim", "")[:100],
            "ground_truth":       state.get("ground_truth", ""),
            "final_verdict":      state.get("final_verdict", ""),
            "judge_correct":      state.get("judge_correct", False),
            "judge_confidence":   state.get("judge_output", {}).get("confidence", 0)
                                  if state.get("judge_output") else 0,
            "jury_verdict":       state.get("jury_verdict", ""),
            "jury_correct":       state.get("jury_correct", False),
            "jury_disagreement":  state.get("jury_disagreement", 0.0),
            "minds_changed":      state.get("_minds_changed", 0),
            "consensus_at_init":  state.get("_consensus_at_init", False),
            "p1_disagreement":    state.get("_p1_disagreement", 0.0),
            "p2_disagreement":    state.get("_p2_disagreement", 0.0),
            "deliberation_changed": state.get("_deliberation_changed", False),
            "ambiguity_flag":     state.get("_ambiguity_flag", False),
            "arbiter_used":       state.get("_arbiter_used", False),
            "total_rounds":       state.get("current_round", 0),
            "early_stopped":      state.get("early_stopped", False),
            "concession_round":   state.get("concession_round", -1),
            "a_epistemic":        state.get("a_epistemic", "CERTAIN"),
            "b_epistemic":        state.get("b_epistemic", "CERTAIN"),
            "duration_seconds":   state.get("duration_seconds", 0),
            "error":              state.get("error", ""),
        })
        return path

    # ── Baseline log ───────────────────────────────────────────────────
    def log_baseline(self, results: list, name: str) -> str:
        path = os.path.join(self.run_dir, f"baseline_{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)
        return path

    # ── Summary files ──────────────────────────────────────────────────
    def save_summary(self) -> str:
        json_path = os.path.join(self.run_dir, "summary.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self._summary_rows, f, indent=2, ensure_ascii=False)

        if self._summary_rows:
            csv_path = os.path.join(self.run_dir, "summary.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(self._summary_rows[0].keys()))
                w.writeheader()
                w.writerows(self._summary_rows)

        print(f"[Logger] Summary saved → {json_path}")
        return json_path

    @property
    def run_directory(self) -> str:
        return self.run_dir