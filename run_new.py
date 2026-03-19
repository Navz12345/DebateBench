"""
run_experiment.py
-----------------
Main entry point. Runs the full debate pipeline + baselines.

Usage:
  python run_experiment.py                      # full run
  python run_experiment.py --quick 5            # 5 cases for testing
  python run_experiment.py --skip-baselines     # debate only
  python run_experiment.py --eval-only logs/run_X
"""

import argparse
import sys
import time
import yaml

from src.utils.schemas import make_initial_state
from src.utils.data_loader import load_dataset
from src.utils.logger import DebateLogger
from src.graph.debate_graph import build_graph
from src.models.university_client import UniversityClient, make_clients
from src.agents.baselines import DirectQABaseline, SelfConsistencyBaseline
from evaluation.evaluate import evaluate


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def check_endpoints(config: dict):
    """Verify all three university endpoints are reachable."""
    debater_client, judge_client, jury_client = make_clients(config)
    mc = config["models"]

    for name, client, cfg in [
        ("Debater (Qwen3-8B)",          debater_client, mc["debater"]),
        ("Judge  (GPT-OSS-20B)",         judge_client,   mc["judge"]),
        ("Jury   (Llama-3.1-70B)",       jury_client,    mc.get("jury_model", mc["judge"])),
    ]:
        try:
            resp = client.generate("Reply with exactly: READY", temperature=0)
            print(f"✓  {name} — {cfg['base_url']}  →  '{resp[:30]}'")
        except Exception as e:
            print(f"\n❌  {name} endpoint unreachable: {e}")
            print(f"    URL: {cfg['base_url']}")
            if "localhost:9999" in cfg["base_url"]:
                print("    Start SSH tunnel on your laptop:")
                print("    ssh -R 9999:10.100.1.212:8888 hqv754@arc.utsa.edu")
            else:
                print("    Connect to UTSA VPN: vpn.utsa.edu")
            sys.exit(1)


def run_one(app, record: dict, config: dict) -> dict:
    """Run one debate. Returns final state dict."""
    mc = config["models"]
    state = make_initial_state(
        case_id=record["case_id"],
        claim=record["claim"],
        evidence_snippets=record["evidence_snippets"],
        ground_truth=record["ground_truth"],
        debater_model=mc["debater"]["name"],
        judge_model=mc["judge"]["name"],
        min_rounds=config["debate"]["min_rounds"],
        max_rounds=config["debate"]["max_rounds"],
    )
    try:
        result = app.invoke(state)
        return dict(result)
    except Exception as e:
        import traceback
        print(f"\n!!! NODE CRASHED: {e}")
        print(traceback.format_exc())
        err = dict(state)
        err["error"] = str(e)
        err["traceback"] = traceback.format_exc()
        err["final_verdict"] = "ERROR"
        err["judge_correct"] = False
        return err


def main():
    parser = argparse.ArgumentParser(description="DebateBench Experiment Runner")
    parser.add_argument("--config",          default="config.yaml")
    parser.add_argument("--quick",           type=int, default=None)
    parser.add_argument("--skip-baselines",  action="store_true")
    parser.add_argument("--eval-only",       default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    if args.eval_only:
        evaluate(args.eval_only,
                 results_dir=config["evaluation"]["results_dir"],
                 figures_dir=config["evaluation"]["figures_dir"])
        return

    print("\n" + "="*60)
    print("  DebateBench — LangGraph Fact Verification Pipeline")
    print("="*60)
    check_endpoints(config)

    jury_enabled = config.get("jury", {}).get("enabled", False)
    if jury_enabled:
        print(f"  Jury panel: ENABLED (3 jurors, two-phase deliberation)")
    else:
        print(f"  Jury panel: DISABLED")

    # Dataset
    print("\n[1/5] Loading dataset...")
    records = load_dataset(config)
    if args.quick:
        records = records[:args.quick]
        print(f"      Quick mode: {len(records)} cases")

    # Build graph
    print(f"\n[2/5] Building LangGraph pipeline...")
    app = build_graph(config)
    nodes = [n for n in app.get_graph().nodes if not n.startswith("__")]
    print(f"      Graph compiled ✓  ({len(nodes)} nodes)")

    logger = DebateLogger(config["logging"]["log_dir"])

    # Run debates
    print(f"\n[3/5] Running debate on {len(records)} cases...")
    total = len(records)
    for i, record in enumerate(records, 1):
        print(f"\n  [{i}/{total}] {record['claim'][:70]}...")
        state = run_one(app, record, config)
        logger.log_debate(state, i)

        verdict  = state.get("final_verdict", "?")
        gt       = state.get("ground_truth", "?")
        correct  = "✓" if state.get("judge_correct") else "✗"
        rounds   = state.get("current_round", 0)

        # ── Debater stances ───────────────────────────────────────────
        a_out = state.get("a_output") or {}
        b_out = state.get("b_output") or {}
        a_ep  = state.get("a_epistemic") or a_out.get("epistemic_status", "?")
        b_ep  = state.get("b_epistemic") or b_out.get("epistemic_status", "?")
        a_con = " ← [CONCEDE]" if a_out.get("conceded") else ""
        b_con = " ← [CONCEDE]" if b_out.get("conceded") else ""
        print(f"    Debater A → stance={a_out.get('stance','?'):<18} conf={a_out.get('confidence','?')}  epistemic={a_ep}{a_con}")
        print(f"    Debater B → stance={b_out.get('stance','?'):<18} conf={b_out.get('confidence','?')}  epistemic={b_ep}{b_con}")

        # ── Judge verdict ─────────────────────────────────────────────
        j_out = state.get("judge_output") or {}
        print(f"    Judge     → verdict={verdict:<17} conf={j_out.get('confidence','?')}  {correct} GT={gt}")

        # ── Jury verdict (if enabled) ─────────────────────────────────
        if jury_enabled and state.get("jury_verdict"):
            jury_v    = state.get("jury_verdict", "?")
            jury_ok   = "✓" if state.get("jury_correct") else "✗"
            disagree  = state.get("jury_disagreement", 0.0)
            dis_label = {0.0:"unanimous", 0.5:"split", 1.0:"divided"}.get(disagree, "?")
            amb_flag  = " AMB" if state.get("_ambiguity_flag") else ""
            arbiter   = " →ARB" if state.get("_arbiter_used")   else ""

            # Phase 1 votes (what jurors said independently)
            p1_votes = state.get("jury_assessments", [])
            for a in p1_votes:
                role = a.get("role","?")[:4]
                print(f"    Juror({role}) Phase1 → {a.get('verdict','?'):<18} conf={a.get('confidence','?')}")

            # Phase 2 votes (after deliberation) — only show if different from Phase 1
            p2_votes = state.get("jury_final_votes", [])
            p1_verdicts = [a.get("verdict") for a in p1_votes]
            p2_verdicts = [a.get("verdict") for a in p2_votes]
            if p1_verdicts != p2_verdicts:
                for a in p2_votes:
                    role = a.get("role","?")[:4]
                    changed = " ← CHANGED" if a.get("changed_mind") else ""
                    print(f"    Juror({role}) Phase2 → {a.get('verdict','?'):<18} conf={a.get('confidence','?')}{changed}")

            # Aggregation summary
            p1_dis = state.get("_p1_disagreement", disagree)
            p2_dis = state.get("_p2_disagreement", disagree)
            p2_label = {0.0:"unanimous", 0.5:"split", 1.0:"divided"}.get(p2_dis, f"{p2_dis:.1f}")
            p2_summary = "/".join(a.get("verdict","?")[:3] for a in p2_votes) if p2_votes else "—"
            minds = state.get("_minds_changed", 0)
            print(f"    Aggregation: Phase1={dis_label}({p1_dis}) → Phase2={p2_label}({p2_dis}) [{p2_summary}] changed={minds}{amb_flag}{arbiter}")
            print(f"    Jury  {jury_ok} Final verdict={jury_v:<17} GT={gt}")
        print(f"    Rounds={rounds}")

    # Baselines — use debater model (Qwen3) for direct comparison
    debater_client, _, _ = make_clients(config)
    debater_cfg    = config["models"]["debater"]
    baseline_client = UniversityClient(
        base_url=   debater_cfg["base_url"],
        api_key=    debater_cfg["api_key"],
        model=      debater_cfg["name"],
        temperature=debater_cfg.get("temperature", 0.7),
        max_tokens= 512,
        timeout=    debater_cfg.get("timeout", 120),
    )

    if not args.skip_baselines and config["baselines"]["direct_qa_enabled"]:
        print(f"\n[4a/5] Running Direct QA baseline...")
        dqa = DirectQABaseline(baseline_client)
        dqa_results = []
        for i, r in enumerate(records, 1):
            res = dqa.run(r)
            dqa_results.append(res)
            if i % 5 == 0:
                acc = sum(1 for x in dqa_results if x["correct"]) / len(dqa_results)
                print(f"      {i}/{total}  running acc={acc:.1%}")
        logger.log_baseline(dqa_results, "direct_qa")

    if not args.skip_baselines and config["baselines"]["self_consistency_enabled"]:
        n_sc = config["baselines"]["self_consistency_samples"]
        print(f"\n[4b/5] Running Self-Consistency (N={n_sc})...")
        sc = SelfConsistencyBaseline(baseline_client, num_samples=n_sc)
        sc_results = []
        for i, r in enumerate(records, 1):
            res = sc.run(r)
            sc_results.append(res)
            if i % 5 == 0:
                acc = sum(1 for x in sc_results if x["correct"]) / len(sc_results)
                print(f"      {i}/{total}  running acc={acc:.1%}")
        logger.log_baseline(sc_results, "self_consistency")

    logger.save_summary()

    print(f"\n[5/5] Evaluating results...")
    evaluate(logger.run_directory,
             results_dir=config["evaluation"]["results_dir"],
             figures_dir=config["evaluation"]["figures_dir"])

    print(f"\n{'='*60}")
    print(f"✓  Done! Logs → {logger.run_directory}")
    print(f"✓  Re-eval: python run_experiment.py --eval-only {logger.run_directory}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()