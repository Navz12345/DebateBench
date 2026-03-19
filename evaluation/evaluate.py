"""
evaluation/evaluate.py
-----------------------
Reads summary logs and produces:
  1. Overall accuracy — debate (single judge) vs baselines
  2. Jury vs single judge comparison
  3. Per-label breakdown (SUPPORT / REFUTE / NEI)
  4. Disagreement vs difficulty analysis
  5. Deliberation quality — how often jurors change minds
  6. Confidence calibration
  7. Rounds analysis
  8. McNemar's test for statistical significance
  9. Matplotlib figures (6 total including jury figures)
"""

import json
import os
import math
import argparse
from collections import defaultdict


def load_summary(run_dir: str) -> list:
    path = os.path.join(run_dir, "summary.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No summary.json in {run_dir}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_baseline(run_dir: str, name: str) -> list:
    path = os.path.join(run_dir, f"baseline_{name}.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def accuracy(rows: list, key: str = "correct") -> float:
    return sum(1 for r in rows if r.get(key)) / len(rows) if rows else 0.0


def conf_stats(rows: list, key: str) -> dict:
    vals = [r[key] for r in rows if r.get(key) not in (None, 0, "")]
    if not vals:
        return {"mean": 0.0, "std": 0.0}
    m = sum(vals) / len(vals)
    s = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
    return {"mean": round(m, 3), "std": round(s, 3)}


def mcnemar(a_correct: list, b_correct: list,
            a_ids: list = None, b_ids: list = None) -> dict:
    """
    McNemar's Test with Edwards' continuity correction.

    Supports two calling modes:
      1. ID-aware (recommended): pass a_ids and b_ids — only claims present
         in both lists are compared. Prevents index-shift bugs when one system
         has fewer rows than another (e.g. a failed API call drops a row).
      2. Index-aligned fallback: if no IDs provided, compares by position
         (safe only when both lists come from the same source rows).

    a_correct / b_correct: list of bool (True = correct prediction)
    a_ids / b_ids:         list of str case_ids in same order as correct lists

    Returns:
      n           — number of matched pairs actually compared
      b           — A correct, B wrong  (A "wins")
      c           — A wrong, B correct  (B "wins")
      chi2        — Edwards-corrected chi-square statistic
      p           — "< 0.05" or ">= 0.05"
      significant — bool
      warning     — set if b+c < 10 (small discordant pairs, unreliable)
    """
    # ── Mode 1: ID-aware pairing ──────────────────────────────────────
    if a_ids is not None and b_ids is not None:
        map_a = {id_: correct for id_, correct in zip(a_ids, a_correct)}
        map_b = {id_: correct for id_, correct in zip(b_ids, b_correct)}
        shared = set(map_a.keys()) & set(map_b.keys())
        if not shared:
            return {"n": 0, "chi2": 0.0, "p": ">= 0.05",
                    "significant": False, "b": 0, "c": 0,
                    "warning": "No shared case_ids found"}
        pairs = [(map_a[i], map_b[i]) for i in shared]
    # ── Mode 2: Index-aligned fallback ───────────────────────────────
    else:
        n_pairs = min(len(a_correct), len(b_correct))
        pairs = [(a_correct[i], b_correct[i]) for i in range(n_pairs)]

    n  = len(pairs)
    b  = sum(1 for a, bv in pairs if     a and not bv)   # A correct, B wrong
    c  = sum(1 for a, bv in pairs if not a and     bv)   # A wrong, B correct
    bc = b + c

    # No discordant pairs — test has no power
    if bc == 0:
        return {"n": n, "chi2": 0.0, "p": ">= 0.05",
                "significant": False, "b": 0, "c": 0}

    # Edwards' continuity correction: (|b - c| - 1)^2 / (b + c)
    # Conservative on small samples — prevents false positives
    chi2 = (abs(b - c) - 1) ** 2 / bc
    sig  = chi2 > 3.841   # critical value at p=0.05, df=1

    result = {
        "n":           n,
        "chi2":        round(chi2, 3),
        "p":           "< 0.05" if sig else ">= 0.05",
        "significant": sig,
        "b":           b,   # times A beat B
        "c":           c,   # times B beat A
    }
    # Warn if discordant pairs are too few to trust the statistic
    if bc < 10:
        result["warning"] = f"Only {bc} discordant pairs — interpret cautiously"
    return result


def fleiss_kappa(ratings: list[list[str]]) -> dict:
    """
    Fleiss's kappa for inter-rater reliability across jurors.
    Measures agreement beyond chance — directly from VERDICT methodology.

    ratings: list of [juror1_verdict, juror2_verdict, juror3_verdict] per claim.
    Returns kappa with Landis & Koch (1977) interpretation.
      < 0.0  = poor      0.2-0.4 = fair      0.6-0.8 = substantial
      0.0-0.2 = slight   0.4-0.6 = moderate  0.8-1.0 = almost perfect
    """
    if not ratings:
        return {"kappa": 0.0, "interpretation": "no data"}

    from collections import Counter
    n          = len(ratings)
    k          = len(ratings[0])
    categories = sorted(set(v for row in ratings for v in row))

    counts = [{cat: Counter(row).get(cat, 0) for cat in categories}
              for row in ratings]

    # p_j = proportion of all assignments that are category j
    total = n * k
    p_j   = {cat: sum(counts[i][cat] for i in range(n)) / total
             for cat in categories}

    # P̄e = expected agreement by chance
    P_e = sum(p ** 2 for p in p_j.values())

    # P̄ = mean observed pairwise agreement
    P_bar = sum(
        sum(counts[i][cat] * (counts[i][cat] - 1) for cat in categories)
        / (k * (k - 1))
        for i in range(n)
    ) / n

    kappa = 1.0 if abs(1 - P_e) < 1e-10 else round((P_bar - P_e) / (1 - P_e), 4)

    if   kappa < 0:    interp = "poor (< 0)"
    elif kappa < 0.2:  interp = "slight (0.0–0.2)"
    elif kappa < 0.4:  interp = "fair (0.2–0.4)"
    elif kappa < 0.6:  interp = "moderate (0.4–0.6)"
    elif kappa < 0.8:  interp = "substantial (0.6–0.8)"
    else:              interp = "almost perfect (0.8–1.0)"

    return {"kappa": kappa, "interpretation": interp,
            "P_bar": round(P_bar, 4), "P_e": round(P_e, 4)}

def calculate_calibration_gap(rows: list) -> dict:
    """Measures the 'Certainty Gap' between right and wrong answers."""
    correct_conf = [r.get("judge_confidence", 0) for r in rows if r.get("judge_correct") and r.get("judge_confidence")]
    wrong_conf = [r.get("judge_confidence", 0) for r in rows if not r.get("judge_correct") and r.get("judge_confidence")]
    
    mean_correct = sum(correct_conf) / len(correct_conf) if correct_conf else 0
    mean_wrong = sum(wrong_conf) / len(wrong_conf) if wrong_conf else 0
    
    return {
        "mean_correct_conf": round(mean_correct, 3),
        "mean_wrong_conf": round(mean_wrong, 3),
        "gap": round(mean_correct - mean_wrong, 3)
    }

def confidence_comparison(jury_details: list[dict]) -> dict:
    """
    Compare Phase 1 (independent) vs Phase 2 (post-deliberation) confidence.
    Answers: 'did deliberation make jurors more certain when correct?'
    Data source: individual debate JSON files (jury_assessments + jury_final_votes).
    """
    p1_all, p2_all = [], []
    role_p1: dict  = {}
    role_p2: dict  = {}

    for d in jury_details:
        for a in d.get("jury_assessments", []):
            c, r = a.get("confidence", 0), a.get("role", "unknown")
            if c > 0:
                p1_all.append(c)
                role_p1.setdefault(r, []).append(c)
        for a in d.get("jury_final_votes", []):
            c, r = a.get("confidence", 0), a.get("role", "unknown")
            if c > 0:
                p2_all.append(c)
                role_p2.setdefault(r, []).append(c)

    pre  = round(sum(p1_all) / len(p1_all), 3) if p1_all else 0.0
    post = round(sum(p2_all) / len(p2_all), 3) if p2_all else 0.0

    per_role = {}
    for role in set(list(role_p1) + list(role_p2)):
        p1r = role_p1.get(role, [])
        p2r = role_p2.get(role, [])
        pre_r  = round(sum(p1r) / len(p1r), 3) if p1r else 0.0
        post_r = round(sum(p2r) / len(p2r), 3) if p2r else 0.0
        per_role[role] = {"pre": pre_r, "post": post_r,
                          "delta": round(post_r - pre_r, 3)}

    return {"pre_mean":  pre, "post_mean": post,
            "delta":     round(post - pre, 3), "per_role": per_role}


def load_jury_details(run_dir: str) -> list[dict]:
    """
    Load per-juror vote data from individual debate JSON files.
    Summary.json only has aggregated jury fields; per-juror
    verdicts/confidences live in the individual debate files.
    """
    details = []
    for fname in sorted(os.listdir(run_dir)):
        if not fname.startswith("debate_") or not fname.endswith(".json"):
            continue
        path = os.path.join(run_dir, fname)
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            if d.get("jury_assessments"):
                details.append(d)
        except Exception:
            pass
    return details


def evaluate(run_dir: str, results_dir: str = "evaluation/results",
             figures_dir: str = "evaluation/figures", plots: bool = True) -> dict:

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    summary = load_summary(run_dir)
    dqa     = load_baseline(run_dir, "direct_qa")
    sc      = load_baseline(run_dir, "self_consistency")
    n       = len(summary)

    # Detect if jury was run
    has_jury = any(r.get("jury_verdict") for r in summary)

    print(f"\n{'='*65}")
    print(f"  EVALUATION  |  n={n}  |  {os.path.basename(run_dir)}")
    if has_jury:
        print(f"  JURY PANEL ENABLED")
    print(f"{'='*65}")

    # ── 1. Overall accuracy ────────────────────────────────────────────
    debate_acc = accuracy(summary, "judge_correct")
    jury_acc   = accuracy(summary, "jury_correct") if has_jury else None
    dqa_acc    = accuracy(dqa, "correct")
    sc_acc     = accuracy(sc,  "correct")

    print(f"\n  {'Method':<40} {'Accuracy':>10}  N")
    print(f"  {'-'*55}")
    print(f"  {'LLM Debate (Single Judge)':<40} {debate_acc:>10.1%}  {n}")
    if has_jury:
        print(f"  {'LLM Debate (Jury Panel — 3 jurors)':<40} {jury_acc:>10.1%}  {n}")
    print(f"  {'Direct QA (CoT)':<40} {dqa_acc:>10.1%}  {len(dqa)}")
    print(f"  {'Self-Consistency (N=13)':<40} {sc_acc:>10.1%}  {len(sc)}")

    # ── 2. Jury vs single judge analysis ───────────────────────
    jury_report = {}
    if has_jury:
        jury_rows = [r for r in summary if r.get("jury_verdict")]

        # Cases where jury and judge disagree
        disagree_verdict = [r for r in jury_rows
                            if r.get("jury_verdict") != r.get("final_verdict")]
        # Cases where jury was right but judge was wrong
        jury_better = [r for r in jury_rows
                       if r.get("jury_correct") and not r.get("judge_correct")]
        # Cases where judge was right but jury was wrong
        judge_better = [r for r in jury_rows
                        if r.get("judge_correct") and not r.get("jury_correct")]

        print(f"\n  Jury vs Single Judge:")
        print(f"    Jury accuracy:          {jury_acc:.1%}")
        print(f"    Single judge accuracy:  {debate_acc:.1%}")
        delta = (jury_acc - debate_acc) * 100
        sign  = "+" if delta >= 0 else ""
        print(f"    Delta:                  {sign}{delta:.1f} pp")
        print(f"    Jury/Judge disagree:    {len(disagree_verdict)}/{n} ({len(disagree_verdict)/n:.0%})")
        print(f"    Jury better than judge: {len(jury_better)} cases")
        print(f"    Judge better than jury: {len(judge_better)} cases")

        # ── 3. Disagreement vs difficulty ─────────────────────────────
        print(f"\n  Disagreement vs Difficulty:")
        for level, label in [(0.0,"Unanimous (0.0)"),(0.5,"One dissenter (0.5)"),(1.0,"All differ (1.0)")]:
            rows = [r for r in jury_rows if r.get("jury_disagreement") == level]
            if rows:
                judge_acc = accuracy(rows, "judge_correct")
                jury_a    = accuracy(rows, "jury_correct")
                print(f"    {label:<25} n={len(rows):>3}  "
                      f"judge={judge_acc:.1%}  jury={jury_a:.1%}")

        # ── 4. Consensus early-stopping stats ─────────────────────────
        consensus_rows = [r for r in jury_rows if r.get("consensus_at_init")]
        print(f"\n  Consensus at Phase 1 (skipped deliberation): "
              f"{len(consensus_rows)}/{len(jury_rows)} "
              f"({len(consensus_rows)/len(jury_rows):.0%})")
        if consensus_rows:
            print(f"    Accuracy when unanimous at init: "
                  f"{accuracy(consensus_rows,'jury_correct'):.1%}")
        deliberated = [r for r in jury_rows if not r.get("consensus_at_init")]
        if deliberated:
            print(f"    Accuracy after deliberation:     "
                  f"{accuracy(deliberated,'jury_correct'):.1%}  "
                  f"n={len(deliberated)}")

        # ── 4b. Ambiguity flag + arbiter stats ────────────────────────
        amb_rows     = [r for r in jury_rows if r.get("ambiguity_flag")]
        arbiter_rows = [r for r in jury_rows if r.get("arbiter_used")]
        print(f"\n  Ambiguity safeguard triggered: {len(amb_rows)}/{len(jury_rows)} "
              f"({len(amb_rows)/len(jury_rows):.0%})")
        if amb_rows:
            print(f"    Accuracy when flag raised: "
                  f"{accuracy(amb_rows,'jury_correct'):.1%}  n={len(amb_rows)}")
        print(f"  Arbiter judge called:          {len(arbiter_rows)}/{len(jury_rows)} "
              f"({len(arbiter_rows)/len(jury_rows):.0%})")
        if arbiter_rows:
            print(f"    Accuracy when arbiter used: "
                  f"{accuracy(arbiter_rows,'jury_correct'):.1%}  n={len(arbiter_rows)}")

        # ── 5. Deliberation quality ────────────────────────────────────
        total_jurors = len(jury_rows) * 3
        changed = sum(r.get("minds_changed", 0) for r in jury_rows)
        print(f"\n  Deliberation quality:")
        print(f"    Jurors who changed mind: {changed}/{total_jurors} "
              f"({changed/total_jurors:.0%})")

        # Did changing mind help or hurt?
        changed_rows   = [r for r in jury_rows if r.get("minds_changed", 0) > 0]
        unchanged_rows = [r for r in jury_rows if r.get("minds_changed", 0) == 0]
        if changed_rows:
            print(f"    Jury acc when changed:   "
                  f"{accuracy(changed_rows,'jury_correct'):.1%}  n={len(changed_rows)}")
        if unchanged_rows:
            print(f"    Jury acc when unchanged: "
                  f"{accuracy(unchanged_rows,'jury_correct'):.1%}  n={len(unchanged_rows)}")

        # ── 6. Confidence spread analysis ─────────────────────────────
        # High spread = one juror very certain, another very uncertain
        # Logged for report — does NOT affect the verdict (plain majority wins)
        conf_spread_vals = [r.get("_confidence_spread", 0.0)
                            for r in jury_rows if r.get("_confidence_spread") is not None]
        avg_spread = round(sum(conf_spread_vals)/len(conf_spread_vals), 3) if conf_spread_vals else 0
        print(f"\n  Confidence spread (0=all equal, 4=max spread):")
        print(f"    Mean spread: {avg_spread:.2f}  "
              f"(high spread = jurors disagree on certainty, not just verdict)")
        high_spread_rows = [r for r in jury_rows if r.get("_confidence_spread", 0) >= 2.0]
        if high_spread_rows:
            print(f"    High-spread cases (≥2.0): {len(high_spread_rows)}  "
                  f"jury_acc={accuracy(high_spread_rows,'jury_correct'):.1%}")

        # ── 6b. Advanced Calibration Gap ──────────────────────────────────
        cal_gap = calculate_calibration_gap(summary)
        print(f"\n  Advanced Calibration (Certainty Gap):")
        print(f"    Mean Conf (Correct): {cal_gap['mean_correct_conf']}")
        print(f"    Mean Conf (Wrong):   {cal_gap['mean_wrong_conf']}")
        print(f"    Calibration Gap:     {cal_gap['gap']}  "
              f"({'Healthy' if cal_gap['gap'] > 0.5 else 'Under-calibrated'})")
        # Add to the report dictionary so it saves to JSON
        jury_report["calibration_gap"] = cal_gap["gap"]

        # ── 7. Human-in-the-loop: flag hardest cases ──────────────────
        # Top-10 hardest claims by jury disagreement + confidence spread
        # for manual inspection and qualitative analysis in REPORT.md
        hard_cases = sorted(
            [r for r in jury_rows if r.get("jury_disagreement", 0) > 0],
            key=lambda r: (r.get("jury_disagreement", 0),
                           r.get("_confidence_spread", 0)),
            reverse=True
        )[:10]

        if hard_cases:
            print(f"\n  ── Hard Cases for Manual Inspection ───────────────────")
            for i, r in enumerate(hard_cases, 1):
                j_sym = "✓" if r.get("judge_correct") else "✗"
                u_sym = "✓" if r.get("jury_correct")  else "✗"
                print(f"  {i:2}. [{j_sym}judge/{u_sym}jury] "
                      f"GT={r.get('ground_truth','?'):<18} "
                      f"disagree={r.get('jury_disagreement',0):.1f}  "
                      f"spread={r.get('_confidence_spread',0):.1f}")
                print(f"      {r.get('claim','')[:80]}")
            print(f"  ────────────────────────────────────────────────────────")

        # McNemar: jury vs single judge — both from same jury_rows, index-safe
        judge_c = [r.get("judge_correct", False) for r in jury_rows]
        jury_c  = [r.get("jury_correct",  False) for r in jury_rows]
        mj = mcnemar(judge_c, jury_c)
        warn_mj = f"  ⚠ {mj['warning']}" if mj.get("warning") else ""
        print(f"\n  McNemar (jury vs single judge): "
              f"χ²={mj['chi2']}  p{mj['p']}  Significant={mj['significant']}"
              f"  n={mj['n']}  (b={mj['b']} judge>jury, c={mj['c']} jury>judge){warn_mj}")

        # ── Pre vs Post Deliberation Confidence ───────────────────────
        # Compare Phase 1 (independent) vs Phase 2 (post-deliberation) confidence.
        # Only computed for cases where deliberation actually ran (non-consensus).
        jury_details        = load_jury_details(run_dir)
        conf_report         = {}
        deliberated_details = [d for d in jury_details
                               if not d.get("_consensus_at_init", False)]
        if deliberated_details:
            conf = confidence_comparison(deliberated_details)
            print(f"\n  Pre vs Post Deliberation Confidence:")
            print(f"    Phase 1 mean confidence:  {conf['pre_mean']:.2f}  (independent assessment)")
            print(f"    Phase 2 mean confidence:  {conf['post_mean']:.2f}  (after deliberation)")
            sign_c = "+" if conf["delta"] >= 0 else ""
            print(f"    Delta: {sign_c}{conf['delta']:.3f}  "
                  f"({'deliberation increased certainty' if conf['delta'] > 0 else 'no increase from deliberation'})")
            if conf.get("per_role"):
                print(f"    Per-role confidence change:")
                for role, vals in sorted(conf["per_role"].items()):
                    sign_r = "+" if vals["delta"] >= 0 else ""
                    print(f"      {role:<15} {vals['pre']:.2f} → {vals['post']:.2f}  "
                          f"({sign_r}{vals['delta']:.3f})")
            conf_report = {
                "confidence_pre_mean":  conf["pre_mean"],
                "confidence_post_mean": conf["post_mean"],
                "confidence_delta":     conf["delta"],
            }

        jury_report = {
            "jury_accuracy":           round(jury_acc, 4),
            "jury_vs_judge_delta_pp":  round(delta, 2),
            "verdict_disagreements":   len(disagree_verdict),
            "jury_better_cases":      len(jury_better),
            "judge_better_cases":     len(judge_better),
            "minds_changed_pct":       round(changed/total_jurors, 4) if total_jurors else 0,
            "avg_confidence_spread":   avg_spread,
            "hard_cases_flagged":      len(hard_cases),
            "ambiguity_flag_count":    len(amb_rows),
            "arbiter_used_count":      len(arbiter_rows),
            "mcnemar_jury_vs_judge":   mj,
            **conf_report,
        }

    # ── 5. Per-label breakdown ─────────────────────────────────────────
    labels = ["SUPPORT", "REFUTE", "NOT_ENOUGH_INFO"]
    print(f"\n  Per-label accuracy (single judge):")
    label_accs = {}
    for lbl in labels:
        rows = [r for r in summary if r.get("ground_truth") == lbl]
        acc  = accuracy(rows, "judge_correct")
        label_accs[lbl] = acc
        jury_lbl = accuracy(rows, "jury_correct") if has_jury else None
        jury_str = f"  jury={jury_lbl:.1%}" if jury_lbl is not None else ""
        print(f"    {lbl:<22} {acc:>6.1%}  n={len(rows)}{jury_str}")

    # ── 6. Confidence calibration ──────────────────────────────────────
    correct_rows   = [r for r in summary if r.get("judge_correct")]
    incorrect_rows = [r for r in summary if not r.get("judge_correct")]
    print(f"\n  Confidence calibration (single judge):")
    print(f"    Correct predictions:   {conf_stats(correct_rows,  'judge_confidence')}")
    print(f"    Incorrect predictions: {conf_stats(incorrect_rows,'judge_confidence')}")

    # ── 7. Rounds analysis ─────────────────────────────────────────────
    by_rounds = defaultdict(list)
    for r in summary:
        by_rounds[r.get("total_rounds", 0)].append(r.get("judge_correct", False))
    print(f"\n  Accuracy by rounds:")
    for rnd in sorted(by_rounds):
        acc = sum(by_rounds[rnd]) / len(by_rounds[rnd])
        print(f"    {rnd} rounds: {acc:.1%}  n={len(by_rounds[rnd])}")

    early = [r for r in summary if r.get("early_stopped")]
    print(f"\n  Early stopped: {len(early)}/{n} ({len(early)/n:.0%})")

    # Concession analysis
    conceded_rows = [r for r in summary if r.get("concession_round", -1) >= 0]
    if conceded_rows:
        mean_concession_round = sum(r["concession_round"] for r in conceded_rows) / len(conceded_rows)
        concede_correct = sum(1 for r in conceded_rows if r.get("judge_correct"))
        print(f"\n  Concession analysis:")
        print(f"    Claims with concession:    {len(conceded_rows)}/{n} ({len(conceded_rows)/n:.0%})")
        print(f"    Mean rounds to concession: {mean_concession_round:.1f}")
        print(f"    Accuracy after concession: {concede_correct/len(conceded_rows):.1%}")
        # Epistemic drift — how many ended in DOUBTFUL without conceding
        doubtful_rows = [r for r in summary if r.get("a_epistemic") == "DOUBTFUL"
                         or r.get("b_epistemic") == "DOUBTFUL"]
        print(f"    Claims ending DOUBTFUL:    {len(doubtful_rows)}/{n} (debate was unsettled)")

    # ── 8. Statistical significance ────────────────────────────────────
    if dqa and len(dqa) >= 1:
        # Build ID-aware lists — only compare claims present in both systems
        debate_ids = [r.get("case_id", "") for r in summary]
        debate_c   = [r.get("judge_correct", False) for r in summary]
        dqa_ids    = [r.get("case_id", "") for r in dqa]
        dqa_c      = [r.get("correct",       False) for r in dqa]
        sc_ids     = [r.get("case_id", "") for r in sc]
        sc_c       = [r.get("correct",       False) for r in sc]

        print(f"\n  McNemar's test (α=0.05):")
        m1 = mcnemar(debate_c, dqa_c, debate_ids, dqa_ids)
        warn1 = f"  ⚠ {m1['warning']}" if m1.get("warning") else ""
        print(f"    Debate vs Direct QA:    χ²={m1['chi2']}  p{m1['p']}  "
              f"Significant={m1['significant']}  n={m1['n']}{warn1}")
        if sc:
            m2 = mcnemar(debate_c, sc_c, debate_ids, sc_ids)
            warn2 = f"  ⚠ {m2['warning']}" if m2.get("warning") else ""
            print(f"    Debate vs Self-Con:     χ²={m2['chi2']}  p{m2['p']}  "
                  f"Significant={m2['significant']}  n={m2['n']}{warn2}")
        if has_jury:
            jury_ids = [r.get("case_id", "") for r in summary]
            jury_c2  = [r.get("jury_correct", False) for r in summary]
            m3 = mcnemar(jury_c2, dqa_c, jury_ids, dqa_ids)
            warn3 = f"  ⚠ {m3['warning']}" if m3.get("warning") else ""
            print(f"    Jury vs Direct QA:      χ²={m3['chi2']}  p{m3['p']}  "
                  f"Significant={m3['significant']}  n={m3['n']}{warn3}")

    print(f"{'='*65}\n")

    # ── Save report ────────────────────────────────────────────────────
    # Build mcnemar results for report
    mcnemar_report = {}
    if dqa:
        debate_ids_r = [r.get("case_id", "") for r in summary]
        debate_c_r   = [r.get("judge_correct", False) for r in summary]
        dqa_ids_r    = [r.get("case_id", "") for r in dqa]
        dqa_c_r      = [r.get("correct", False) for r in dqa]
        m1r = mcnemar(debate_c_r, dqa_c_r, debate_ids_r, dqa_ids_r)
        mcnemar_report["debate_vs_dqa"] = m1r
        if sc:
            sc_ids_r = [r.get("case_id", "") for r in sc]
            sc_c_r   = [r.get("correct", False) for r in sc]
            m2r = mcnemar(debate_c_r, sc_c_r, debate_ids_r, sc_ids_r)
            mcnemar_report["debate_vs_sc"] = m2r
        if has_jury:
            jury_ids_r = [r.get("case_id", "") for r in summary]
            jury_c_r   = [r.get("jury_correct", False) for r in summary]
            mjr = mcnemar(judge_c, jury_c)
            mcnemar_report["jury_vs_judge"] = mjr
            m3r = mcnemar(jury_c_r, dqa_c_r, jury_ids_r, dqa_ids_r)
            mcnemar_report["jury_vs_dqa"] = m3r

    report = {
        "n":               n,
        "debate_accuracy": round(debate_acc, 4),
        "dqa_accuracy":    round(dqa_acc,    4),
        "sc_accuracy":     round(sc_acc,     4),
        "per_label":       {k: round(v, 4) for k, v in label_accs.items()},
        "by_rounds":       {str(k): {"n": len(v), "acc": round(sum(v)/len(v), 4)}
                            for k, v in by_rounds.items()},
        "mcnemar":         mcnemar_report,
        **jury_report,
    }
    out = os.path.join(results_dir, f"report_{os.path.basename(run_dir)}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[Eval] Report → {out}")

    if plots:
        try:
            _make_plots(report, summary, dqa, sc, figures_dir, has_jury)
        except ImportError:
            print("[Eval] matplotlib not available — skipping plots.")
    return report


def _make_plots(report, summary, dqa, sc, figures_dir, has_jury=False):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE   = "#3B82F6"
    ORANGE = "#F59E0B"
    GREEN  = "#10B981"
    RED    = "#EF4444"
    GRAY   = "#6B7280"
    PURPLE = "#8B5CF6"

    # ── Figure 1: Accuracy comparison (includes jury if present) ──────
    methods = ["Single\nJudge", "Direct QA\n(CoT)", "Self-Con.\n(N=13)"]
    accs    = [report["debate_accuracy"], report["dqa_accuracy"], report["sc_accuracy"]]
    colors  = [BLUE, ORANGE, GREEN]
    if has_jury and report.get("jury_accuracy"):
        methods.insert(1, "Jury Panel\n(3 jurors)")
        accs.insert(1, report["jury_accuracy"])
        colors.insert(1, PURPLE)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(methods, [a*100 for a in accs], color=colors,
                  edgecolor="white", width=0.5)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy: Debate Pipeline vs Baselines", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.spines[["top","right"]].set_visible(False)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                f"{acc:.1%}", ha="center", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig1_accuracy.png"), dpi=150)
    plt.close()

    # ── Figure 2: Per-label accuracy ───────────────────────────────────
    pl = report.get("per_label", {})
    if pl:
        fig, ax = plt.subplots(figsize=(7, 4))
        lbls  = list(pl.keys())
        laccs = [pl[l]*100 for l in lbls]
        ax.bar(lbls, laccs, color=[GREEN, RED, GRAY], edgecolor="white", width=0.5)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_title("Per-Label Accuracy (Single Judge)", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 110)
        ax.spines[["top","right"]].set_visible(False)
        for i, acc in enumerate(laccs):
            ax.text(i, acc+1.5, f"{acc:.1f}%", ha="center",
                    fontsize=10, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "fig2_per_label.png"), dpi=150)
        plt.close()

    # ── Figure 3: Confidence distribution ──────────────────────────────
    confs = [r.get("judge_confidence", 0) for r in summary if r.get("judge_confidence")]
    if confs:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(confs, bins=[0.5,1.5,2.5,3.5,4.5,5.5],
                color=BLUE, edgecolor="white", alpha=0.85)
        ax.set_xlabel("Confidence (1-5)"); ax.set_ylabel("Count")
        ax.set_title("Judge Confidence Distribution", fontsize=12, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "fig3_confidence.png"), dpi=150)
        plt.close()

    # ── Figure 4: Disagreement vs Accuracy ─────────────────────
    if has_jury:
        jury_rows = [r for r in summary if r.get("jury_verdict")]
        if jury_rows:
            disagree_levels  = [0.0, 0.5, 1.0]
            labels_d         = ["Unanimous\n(0.0)", "One dissenter\n(0.5)",
                                 "All differ\n(1.0)"]
            judge_by_dis, jury_by_dis, counts_dis = [], [], []
            for level in disagree_levels:
                rows = [r for r in jury_rows if r.get("jury_disagreement") == level]
                counts_dis.append(len(rows))
                judge_by_dis.append(accuracy(rows, "judge_correct") * 100 if rows else 0)
                jury_by_dis.append(accuracy(rows,  "jury_correct")  * 100 if rows else 0)

            x    = range(len(disagree_levels))
            w    = 0.3
            fig, ax1 = plt.subplots(figsize=(8, 5))
            ax1.bar([i-w/2 for i in x], judge_by_dis, w, label="Single Judge",
                    color=BLUE,   edgecolor="white")
            ax1.bar([i+w/2 for i in x], jury_by_dis,  w, label="Jury Panel",
                    color=PURPLE, edgecolor="white")
            ax1.set_xticks(list(x))
            ax1.set_xticklabels(labels_d, fontsize=10)
            ax1.set_ylabel("Accuracy (%)", fontsize=11)
            ax1.set_title("Accuracy by Jury Disagreement Level\n"
                          "(High disagreement = harder claim)",
                          fontsize=12, fontweight="bold")
            ax1.set_ylim(0, 110)
            ax1.legend(fontsize=10)
            ax1.spines[["top","right"]].set_visible(False)

            # Annotate with n=
            for i, c in enumerate(counts_dis):
                ax1.text(i, 5, f"n={c}", ha="center", fontsize=9, color="white",
                         fontweight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, "fig4_disagreement_vs_accuracy.png"),
                        dpi=150)
            plt.close()

        # ── Figure 5: Jury vs Judge verdict comparison ──────────
        both_correct = sum(1 for r in jury_rows
                           if r.get("judge_correct") and r.get("jury_correct"))
        only_judge   = sum(1 for r in jury_rows
                           if r.get("judge_correct") and not r.get("jury_correct"))
        only_jury    = sum(1 for r in jury_rows
                           if not r.get("judge_correct") and r.get("jury_correct"))
        neither      = sum(1 for r in jury_rows
                           if not r.get("judge_correct") and not r.get("jury_correct"))

        fig, ax = plt.subplots(figsize=(6, 5))
        cats   = ["Both\ncorrect", "Judge\nonly", "Jury\nonly", "Neither"]
        vals   = [both_correct, only_judge, only_jury, neither]
        cols   = [GREEN, BLUE, PURPLE, RED]
        bars   = ax.bar(cats, vals, color=cols, edgecolor="white", width=0.5)
        ax.set_ylabel("Number of claims", fontsize=11)
        ax.set_title("Single Judge vs Jury Panel\nCorrectness Overlap",
                     fontsize=12, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                        str(v), ha="center", fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "fig5_jury_vs_judge_overlap.png"),
                    dpi=150)
        plt.close()

        # ── Figure 6: Mind-change rate ──────────────────────────
        changed_counts = [r.get("minds_changed", 0) for r in jury_rows]
        from collections import Counter
        mc_dist = Counter(changed_counts)
        fig, ax = plt.subplots(figsize=(6, 4))
        xs = sorted(mc_dist.keys())
        ax.bar([str(x) for x in xs], [mc_dist[x] for x in xs],
               color=PURPLE, edgecolor="white", width=0.5)
        ax.set_xlabel("Jurors who changed mind per claim", fontsize=11)
        ax.set_ylabel("Number of claims", fontsize=11)
        ax.set_title("Deliberation: Mind-Change Distribution",
                     fontsize=12, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "fig6_mind_changes.png"), dpi=150)
        plt.close()

    print(f"[Eval] Figures → {figures_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    evaluate(args.run_dir, plots=not args.no_plots)