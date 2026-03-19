"""
scripts/sanity_check.py
-----------------------
Run this before run_experiment.py.
Checks all 3 model endpoints, agent roles, graph compile, and dataset.

Three models:
  Debater : Qwen3-8B          @ ARC network (no VPN needed)
  Judge   : GPT-OSS-20B       @ UTSA VPN (10.100.1.212)
  Jury    : Llama-3.1-70B     @ UTSA VPN (10.246.100.230)

Usage:
  python scripts/sanity_check.py
"""

import sys
import os
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check(label, fn):
    print(f"  ·· {label}...", end=" ", flush=True)
    try:
        msg = fn()
        print(f"✓  {msg}")
        return True
    except Exception as e:
        print(f"✗  FAILED\n       {e}")
        return False


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    mc          = cfg["models"]
    debater_cfg = mc["debater"]
    judge_cfg   = mc["judge"]
    jury_cfg    = mc.get("jury_model", judge_cfg)

    print(f"\n{'='*62}")
    print(f"  DebateBench Sanity Check")
    print(f"  Debater : {debater_cfg['name']}")
    print(f"  Judge   : {judge_cfg['name']}")
    print(f"  Jury    : {jury_cfg['name']}")
    print(f"{'='*62}\n")

    ok = True

    from src.models.university_client import UniversityClient

    # ── 1. Debater endpoint (Qwen3) ────────────────────────────────────
    print("[1] Debater  (Qwen3-8B — ARC, no VPN needed)")
    debater_client = UniversityClient(
        base_url=debater_cfg["base_url"], api_key=debater_cfg["api_key"],
        model=debater_cfg["name"], temperature=0, max_tokens=512,
        timeout=debater_cfg.get("timeout", 240),
    )
    def test_debater():
        r = debater_client.generate("Reply with exactly the word: READY", temperature=0)
        if not r.strip(): raise ValueError("Empty response")
        return f"'{r.strip()[:40]}'  ({len(r)} chars)"
    if not check(debater_cfg["base_url"], test_debater):
        print("  Check that the ARC network is reachable.")
        ok = False

    # ── 2. Judge endpoint (GPT-OSS-20B) ───────────────────────────────
    print("\n[2] Judge    (GPT-OSS-20B — requires UTSA VPN)")
    judge_client = UniversityClient(
        base_url=judge_cfg["base_url"], api_key=judge_cfg["api_key"],
        model=judge_cfg["name"], temperature=0, max_tokens=64,
        timeout=60,   # GPT-OSS needs more time than a models-list ping
    )
    def test_judge():
        r = judge_client.generate("Reply with exactly the word: READY", temperature=0)
        if not r.strip(): raise ValueError("Empty response")
        return f"'{r.strip()[:40]}'  ({len(r)} chars)"
    judge_ok = check(judge_cfg["base_url"], test_judge)
    if not judge_ok:
        print("  ⚠️  Connect to UTSA VPN: vpn.utsa.edu  then re-run.")
        judge_client_for_tests = debater_client   # fallback for agent tests
    else:
        judge_client_for_tests = judge_client

    # ── 3. Jury endpoint (Llama 70B) ───────────────────────────────────
    print("\n[3] Jury     (Llama-3.1-70B — requires UTSA VPN)")
    jury_client = UniversityClient(
        base_url=jury_cfg["base_url"], api_key=jury_cfg["api_key"],
        model=jury_cfg["name"], temperature=0, max_tokens=64,
        timeout=20,
    )
    def test_jury():
        r = jury_client.generate("Reply with exactly the word: READY", temperature=0)
        if not r.strip(): raise ValueError("Empty response")
        return f"'{r.strip()[:40]}'  ({len(r)} chars)"
    jury_ok = check(jury_cfg["base_url"], test_jury)
    if not jury_ok:
        print("  ⚠️  Connect to UTSA VPN: vpn.utsa.edu  then re-run.")

    # ── 4. Agent roles ─────────────────────────────────────────────────
    print("\n[4] Agent roles  (3 LLM calls via Qwen3 debater)")
    from src.agents.debater import Debater, SYSTEM_DEBATER_A, SYSTEM_DEBATER_B
    from src.agents.judge   import Judge

    debater_test = UniversityClient(
        base_url=debater_cfg["base_url"], api_key=debater_cfg["api_key"],
        model=debater_cfg["name"], temperature=0.7, max_tokens=2048,
        timeout=debater_cfg.get("timeout", 240),
    )
    judge_test = UniversityClient(
        base_url=judge_client_for_tests._client.base_url,
        api_key=judge_cfg["api_key"],
        model=judge_client_for_tests.model,
        temperature=0.3, max_tokens=1024,
        timeout=judge_cfg.get("timeout", 120),
    )

    TEST_CLAIM = "Regular exercise reduces symptoms of depression."
    TEST_EV    = [
        "Meta-analysis of 25 RCTs showed exercise has large effect on depression (d=0.82).",
        "Exercise was comparable to antidepressant medication in mild-to-moderate depression.",
        "Both aerobic and resistance training showed similar antidepressant effects.",
    ]

    deb_a = Debater("Debater A", "debater_a.txt", debater_test, system_message=SYSTEM_DEBATER_A)
    deb_b = Debater("Debater B", "debater_b.txt", debater_test, system_message=SYSTEM_DEBATER_B)
    judge_agent = Judge(judge_test)

    def test_debater_a():
        out, _ = deb_a.argue(TEST_CLAIM, TEST_EV, [], 0)
        if out["stance"] not in ("SUPPORT", "REFUTE", "NOT_ENOUGH_INFO"):
            raise ValueError(f"Bad stance: '{out['stance']}'")
        if not out["reasoning"].strip() or out["reasoning"].startswith("[No reasoning"):
            raise ValueError("Empty reasoning — thinking mode needs more tokens")
        return f"stance={out['stance']}  conf={out['confidence']}"
    ok &= check("Debater A  (SUPPORT framing)", test_debater_a)

    def test_debater_b():
        out, _ = deb_b.argue(TEST_CLAIM, TEST_EV, [], 0)
        if out["stance"] not in ("SUPPORT", "REFUTE", "NOT_ENOUGH_INFO"):
            raise ValueError(f"Bad stance: '{out['stance']}'")
        if not out["reasoning"].strip() or out["reasoning"].startswith("[No reasoning"):
            raise ValueError("Empty reasoning — check prompt format")
        return f"stance={out['stance']}  conf={out['confidence']}"
    ok &= check("Debater B  (REFUTE framing)", test_debater_b)

    def test_judge_role():
        from src.utils.schemas import TranscriptTurn
        turns = [
            TranscriptTurn(round=0, agent="Debater A", stance="SUPPORT",
                           reasoning="Evidence shows strong effect.", counter="", confidence=4),
            TranscriptTurn(round=0, agent="Debater B", stance="REFUTE",
                           reasoning="Effect sizes vary widely.", counter="", confidence=3),
        ]
        out = judge_agent.evaluate(TEST_CLAIM, TEST_EV, turns)
        if out["final_verdict"] not in ("SUPPORT", "REFUTE", "NOT_ENOUGH_INFO"):
            raise ValueError(f"Bad verdict: '{out['final_verdict']}'")
        return f"verdict={out['final_verdict']}  conf={out['confidence']}"
    ok &= check("Judge role", test_judge_role)

    # ── 5. LangGraph compile ───────────────────────────────────────────
    print("\n[5] LangGraph pipeline")
    def test_graph():
        from src.graph.debate_graph import build_graph
        app = build_graph(cfg)
        n = len([x for x in app.get_graph().nodes if not x.startswith("__")])
        return f"Graph compiled  ({n} nodes)"
    ok &= check("StateGraph compiles", test_graph)

    # ── 6. Dataset ─────────────────────────────────────────────────────
    print("\n[6] Dataset")
    def test_data():
        from src.utils.data_loader import load_dataset
        records = load_dataset(cfg)
        labels  = {}
        for r in records:
            labels[r["ground_truth"]] = labels.get(r["ground_truth"], 0) + 1
        dist = "  ".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{len(records)} claims loaded  [{dist}]"
    ok &= check("Dataset loads", test_data)

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    all_endpoints = judge_ok and jury_ok
    if ok and all_endpoints:
        print("✅  All checks passed — ready to run!\n")
        print("  Quick test : python run_experiment.py --quick 3")
        print("  Full run   : python run_experiment.py")
    elif ok and not all_endpoints:
        missing = []
        if not judge_ok: missing.append("Judge  (GPT-OSS-20B)  @ 10.100.1.212")
        if not jury_ok:  missing.append("Jury   (Llama-3.1-70B) @ 10.246.100.230")
        print("⚠️   Core checks passed but these VPN endpoints are unreachable:")
        for m in missing:
            print(f"     · {m}")
        print("\n  Fix: Connect to UTSA VPN (vpn.utsa.edu) then re-run this check.")
    else:
        print("❌  Some checks failed — fix the issues above.\n")
        print("  Common causes:")
        print("  · Debater unreachable  → ARC endpoint down")
        print("  · Judge/Jury down      → connect to UTSA VPN")
        print("  · Empty reasoning      → max_tokens too low in config")
        print("  · 14 nodes not found   → debate_graph.py not updated")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()