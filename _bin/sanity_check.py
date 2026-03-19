"""
scripts/sanity_check.py
-----------------------
Run this script to verify the setup before running the run_experiment.py.
Makes 3 real LLM calls (one per agent role) and verifies the stack.

Usage:
  python scripts/sanity_check.py
  python scripts/sanity_check.py --debater qwen3:8b --judge deepseek-r1:8b
"""

import sys
import os
import argparse
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check(label, fn):
    print(f"  {'··'} {label}...", end=" ", flush=True)
    try:
        msg = fn()
        print(f"✓  {msg}")
        return True
    except Exception as e:
        print(f"✗  FAILED\n       {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debater", default=None)
    parser.add_argument("--judge",   default=None)
    args = parser.parse_args()

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    if args.debater:
        cfg["models"]["debater"] = args.debater
    if args.judge:
        cfg["models"]["judge"]   = args.judge

    debater_model = cfg["models"]["debater"]
    judge_model   = cfg["models"]["judge"]
    url           = cfg["models"]["base_url"]

    print(f"\n{'='*58}")
    print(f"  DebateBench Sanity Check")
    print(f"  Debater: {debater_model}  |  Judge: {judge_model}")
    print(f"{'='*58}\n")

    ok = True

    # 1. Connectivity
    print("[1] Ollama connectivity")
    def test_conn():
        import urllib.request
        with urllib.request.urlopen(url + "/api/tags", timeout=5): pass
        return "Server reachable"
    if not check("Ollama server at " + url, test_conn):
        print("\n  Fix: run `ollama serve` in another terminal\n")
        sys.exit(1)

    # 2. Debater model
    print("\n[2] Debater model (Qwen)")
    from src.models.ollama_client import OllamaClient
    debater_client = OllamaClient(
        base_url=url, model=debater_model, temperature=0, max_tokens=128
    )
    def test_debater_model():
        r = debater_client.generate("Reply with exactly the word: READY")
        if not r.strip():
            raise ValueError("Empty response")
        return f"Got response ({len(r)} chars)"
    ok &= check(f"Model '{debater_model}' responds", test_debater_model)

    # 3. Judge model
    print("\n[3] Judge model (DeepSeek)")
    judge_client = OllamaClient(
        base_url=url, model=judge_model, temperature=0, max_tokens=128
    )
    def test_judge_model():
        r = judge_client.generate("Reply with exactly the word: READY")
        if not r.strip():
            raise ValueError("Empty response")
        return f"Got response ({len(r)} chars)"
    ok &= check(f"Model '{judge_model}' responds", test_judge_model)

    # 4. Debater A role (real prompt)
    print("\n[4] Agent roles (3 LLM calls)")
    from src.agents.debater import Debater
    from src.agents.judge import Judge

    TEST_CLAIM = "Regular exercise reduces symptoms of depression."
    TEST_EV    = [
        "Meta-analysis of 25 RCTs showed exercise has large effect on depression (d=0.82).",
        "Exercise was comparable to antidepressant medication in mild-to-moderate depression.",
        "Both aerobic and resistance training showed similar antidepressant effects.",
    ]

    debater_a = Debater("Debater A", "debater_a.txt", debater_client)
    debater_b = Debater("Debater B", "debater_b.txt", debater_client)
    judge_agent = Judge(judge_client)

    def test_debater_a():
        out, turn = debater_a.argue(TEST_CLAIM, TEST_EV, [], 0)
        if out["stance"] not in ("SUPPORT", "REFUTE", "NOT_ENOUGH_INFO"):
            raise ValueError(f"Bad stance: {out['stance']}")
        return f"stance={out['stance']}, conf={out['confidence']}"
    ok &= check("Debater A (SUPPORT framing)", test_debater_a)

    def test_debater_b():
        out, turn = debater_b.argue(TEST_CLAIM, TEST_EV, [], 0)
        if out["stance"] not in ("SUPPORT", "REFUTE", "NOT_ENOUGH_INFO"):
            raise ValueError(f"Bad stance: {out['stance']}")
        return f"stance={out['stance']}, conf={out['confidence']}"
    ok &= check("Debater B (REFUTE framing)", test_debater_b)

    def test_judge():
        from src.utils.schemas import TranscriptTurn
        turns = [
            TranscriptTurn(round=0,agent="Debater A",stance="SUPPORT",
                           reasoning="Evidence shows strong effect.",counter="",confidence=4),
            TranscriptTurn(round=0,agent="Debater B",stance="REFUTE",
                           reasoning="Effect sizes vary widely.",counter="",confidence=3),
        ]
        out = judge_agent.evaluate(TEST_CLAIM, TEST_EV, turns)
        if out["final_verdict"] not in ("SUPPORT", "REFUTE", "NOT_ENOUGH_INFO"):
            raise ValueError(f"Bad verdict: {out['final_verdict']}")
        return f"verdict={out['final_verdict']}, conf={out['confidence']}"
    ok &= check("Judge role", test_judge)

    # 5. LangGraph compile
    print("\n[5] LangGraph pipeline")
    def test_graph():
        from src.graph.debate_graph import build_graph
        app = build_graph(cfg)
        return "Graph compiled"
    ok &= check("StateGraph compiles", test_graph)

    # 6. Dataset
    print("\n[6] Dataset")
    def test_data():
        from src.utils.data_loader import load_demo_cases
        cases = load_demo_cases(cfg["dataset"]["demo_path"])
        return f"{len(cases)} demo cases loaded"
    ok &= check("Demo cases load", test_data)

    # Summary
    print(f"\n{'='*58}")
    if ok:
        print("  All checks passed — ready to run!\n")
        print("  Quick test:   python run_experiment.py --quick 3")
        print("  Full run:     python run_experiment.py")
        print("  UI:           streamlit run app/streamlit_app.py")
    else:
        print("  Some checks failed. Fix above before running.")
    print(f"{'='*58}\n")


if __name__ == "__main__":
    main()
