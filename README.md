# DebateBench
**Assignment 2 — LLM & Agentic Systems | Dr. Peyman Najafirad | UTSA**

Scientific fact verification via LangGraph multi-agent debate pipeline.

---

## Network Requirements

This project uses UTSA ARC GPU endpoints — **not local models**.

| Model | Endpoint | Access |
|---|---|---|
| Qwen3-8B (Debater) | `http://149.165.171.140:8888/v1` | ARC network |
| GPT-OSS-20B (Judge + Jury) | `http://10.100.1.212:8888/v1` | **UTSA VPN required** |

Connect to UTSA VPN (`vpn.utsa.edu`) before running any experiment.

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Connect to UTSA VPN
#    Open GlobalProtect → vpn.utsa.edu

# 3. Verify all 3 endpoints are reachable
python scripts/sanity_check.py

# 4. Quick test (5 claims, ~10 minutes)
python run_experiment.py --quick 5

# 5. Full experiment (100 claims, ~3-5 hours)
python run_experiment.py

# 6. Re-run evaluation on existing logs
python run_experiment.py --eval-only logs/run_YYYYMMDD_HHMMSS

# 7. Web UI
streamlit run app/streamlit_app.py
```

---

## Project Structure

```
debatebench/
├── config.yaml                    ← all hyperparameters
├── run_experiment.py              ← main entry point
├── requirements.txt
├── REPORT.md                      ← full written report
├── src/
│   ├── graph/
│   │   └── debate_graph.py        ← 14-node LangGraph pipeline
│   ├── agents/
│   │   ├── debater.py             ← Debater A + B (Qwen3-8B)
│   │   ├── judge.py               ← Single judge (GPT-OSS-20B)
│   │   ├── jury.py                ← Smart 3-juror panel + arbiter
│   │   └── baselines.py           ← Direct QA + Self-Consistency
│   ├── prompts/
│   │   ├── debater_a.txt          ← SUPPORT advocate + epistemic status
│   │   ├── debater_b.txt          ← REFUTE advocate + epistemic status
│   │   ├── judge.txt              ← Impartial judge + 5-step CoT
│   │   ├── juror_evidence.txt     ← Evidence citation accuracy
│   │   ├── juror_logic.txt        ← Reasoning quality
│   │   ├── juror_calibration.txt  ← Devil's Advocate (stress-tester)
│   │   ├── juror_phase2.txt       ← General deliberation
│   │   └── juror_phase2_nei.txt   ← NEI-focused deliberation
│   ├── models/
│   │   └── university_client.py   ← JSON-first client, retry, thinking-mode strip
│   ├── utils/
│   │   ├── schemas.py             ← DebateState TypedDict + validators
│   │   ├── logger.py              ← Per-case JSON + CSV summary
│   │   └── data_loader.py         ← SciFact loader
│   └── data/
├── app/
│   └── streamlit_app.py           ← Web UI (round-by-round debate view)
├── evaluation/
│   └── evaluate.py                ← Metrics, figures (fig1-fig6), McNemar
├── scripts/
│   └── sanity_check.py            ← 3-endpoint pre-flight check
└── logs/                          ← Auto-created per run
    └── run_YYYYMMDD_HHMMSS/
        ├── debate_0001.json        ← Per-case full transcript
        ├── baseline_direct_qa.json
        ├── baseline_self_consistency.json
        └── summary.json
```

---

## Pipeline Architecture (14 nodes)

```
load_case → debater_a_initial → debater_b_initial → consensus_check
    ├─ skip_debate  → judge_verdict → jury_check
    └─ start_debate → [debater_a_rebuttal → debater_b_rebuttal
                        → early_stop_check (loop up to 6 rounds,
                          or immediate stop on [CONCEDE] token)]
                          └─ go_to_judge → judge_verdict → jury_check
                                               ├─ jury disabled → evaluate_result
                                               └─ jury enabled
                                                   → juror_1_assess → juror_2_assess
                                                   → juror_3_assess → jury_deliberate
                                                   → evaluate_result
```

**Phase 1 independence:** Both debaters generate opening arguments without seeing each other's response. The full transcript is only shared from round 1 of Phase 2 onwards.

---

## Smart Jury Logic

1. **Phase 1:** 3 jurors (Evidence / Logic / Devil's Advocate) vote independently
2. **Unanimous → finalize** (skip Phase 2)
3. **Disagreement + NEI signal → NEI-focused deliberation** — jurors re-read snippets directly, not debater performance
4. **Smart aggregation after deliberation:**
   - Calibration juror conf=5 NEI → **calibration veto** → arbiter judge
   - NEI juror conf≥4 + majority conf≤3 → **ambiguity safeguard** → NEI wins
   - Decisive majority + confident → finalize
5. **Logged:** `_ambiguity_flag`, `_arbiter_used`, `_deliberation_changed`, `_minds_changed`

The Calibration Juror acts as a **Devil's Advocate** — its mandate is to stress-test the majority view, not reach consensus. It evaluates evidence snippets directly and ignores debater confidence as a proxy for evidence quality.

---

## Key Configuration (config.yaml)

| Parameter | Value | Reason |
|---|---|---|
| Debater temperature | 0.7 | Adversarial argument diversity |
| Judge temperature | 0.3 | Deterministic verdict |
| Evidence juror temp | 0.2 | Verbatim accuracy checking |
| Calibration juror temp | 0.4 | Uncertainty estimation |
| Debater max_tokens | 6144 | Qwen3 thinking mode budget |
| Min debate rounds | 3 | Assignment requirement (N ≥ 3) |
| Max debate rounds | 6 | Hard compute ceiling |
| Early stop consecutive | 2 | Stop after 2 rounds of agreement |
| Self-consistency N | 13 | Matches avg debate call count |

---

## LangGraph Features Used

| Feature | Where | What it does |
|---|---|---|
| `StateGraph` | `debate_graph.py` | Typed shared state across all nodes |
| Append reducer | `schemas.py` | `transcript` accumulates turns automatically |
| Conditional edges | `debate_graph.py` | Consensus bypass, debate loop, early stop, jury routing |
| Pure routing nodes | `debate_graph.py` | `consensus_check`, `early_stop_check` are logic-only, no LLM calls |

---

## Results (n=100, SciFact)

| Method | Accuracy | N |
|---|---|---|
| LLM Debate (Single Judge) | 55.0% | 100 |
| LLM Debate (Jury Panel v2) | 51.0% | 100 |
| Direct QA (CoT) | **61.0%** | 100 |
| Self-Consistency (N=13) | 54.0% | 100 |

**Per-label (single judge):** SUPPORT=37.5%, REFUTE=66.7%, NEI=61.7%

**McNemar test (paired, ID-aware):**
- Jury v1 vs judge: χ²=11.025 p<0.05 — jury significantly worse before fix
- Jury v2 vs judge: χ²=0.643 p>0.05 — jury competitive after Devil's Advocate redesign

**Ambiguity safeguard:** triggered on 30/100 cases, achieved 73.3% accuracy on flagged cases (vs 55% overall)

See `REPORT.md` for full analysis, transcript cases, and prompt engineering details.