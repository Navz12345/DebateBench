# DebateBench: Can Structured Adversarial Debate Improve Scientific Fact Verification?

**Course:** LLM & Agentic Systems — Graduate | **Assignment 2** | **Dr. Peyman Najafirad**
**Dataset:** SciFact (Wadden et al., EMNLP 2020) | **N = 150 claims**
**Framework:** LangGraph | **Models:** Qwen3-8B (debaters) + GPT-OSS-20B (judge + jury)

**AI Tools Disclosure:** Claude (Anthropic) was used for architecture brainstorming, prompt iteration support, debugging assistance, and writing support. All final implementation decisions, experimental runs, results analysis, and conclusions were completed and verified by me on UTSA ARC infrastructure.

---

## 1. Methodology

### Task

Scientific fact verification on SciFact: given a scientific claim and a set of evidence snippets from a PubMed abstract, determine whether the evidence `SUPPORT`s, `REFUTE`s, or provides `NOT_ENOUGH_INFO` for the claim. This is harder than binary QA because it requires precise reading of scientific language — "reduces duration" is not the same as "prevents incidence," and instruction-tuned models have a well-documented tendency to default to skepticism (REFUTE) or excessive caution (NOT_ENOUGH_INFO).

### System Architecture

DebateBench is a **LangGraph-based supervised multi-agent debate pipeline** with an optional three-juror panel. The system takes a claim and evidence snippets as input, runs two role-conditioned advocate agents through a structured adversarial debate, presents the full transcript to a judge for a single verdict, and optionally routes to a three-juror deliberation panel.

**Agent roles:**

- **Debater A (Advocate: SUPPORT)** — role-conditioned to build the strongest possible case that the evidence supports the claim. Framed as a lawyer, not a scientist, to remove the "intellectual honesty" conflict that caused empty reasoning fields in early iterations. Must cite specific evidence snippets by index and quote verbatim.
- **Debater B (Advocate: REFUTE)** — role-conditioned to build the strongest possible case that the evidence does not support the claim. Focuses on scope mismatches, population qualifiers, and direct contradictions.
- **Judge (GPT-OSS-20B)** — receives the full debate transcript and performs a five-step structured evaluation before producing a verdict: SUPPORT, REFUTE, or NOT_ENOUGH_INFO. Explicitly instructed to be equally willing to rule SUPPORT or REFUTE to counteract REFUTE bias. Produces per-debater strongest/weakest argument analysis.
- **Jury Panel (3 jurors, GPT-OSS-20B)** — three role-specialized jurors that independently evaluate the debate before deliberating. Roles: Evidence Judge (citation accuracy), Logic Judge (reasoning quality), Calibration Judge (evidence sufficiency). Two-phase deliberation with smart aggregation and arbiter escalation.

**Model strategy:** Both debaters use Qwen3-8B. Role diversity comes entirely from prompt design, not model heterogeneity — this isolates the debate structure's contribution. The judge and jury use GPT-OSS-20B, a larger model better suited to multi-turn argument evaluation without REFUTE bias.

### LangGraph Pipeline (14 nodes)

```
load_case
  └─→ debater_a_initial
        └─→ debater_b_initial
              └─→ consensus_check
                    ├─(both debaters agree) ─────────────────→ judge_verdict
                    └─(disagree)
                          └─→ debater_a_rebuttal ←───────────┐
                                └─→ debater_b_rebuttal        │
                                      └─→ early_stop_check    │
                                            ├─(continue) ─────┘
                                            └─(stop)
                                                  └─→ judge_verdict
                                                        └─→ jury_check
                                                              ├─(jury off) → evaluate_result → END
                                                              └─(jury on)
                                                                    └─→ juror_1_assess
                                                                          └─→ juror_2_assess
                                                                                └─→ juror_3_assess
                                                                                      └─→ jury_deliberate
                                                                                            └─→ evaluate_result → END
```

**Note on routing:** Routing at `consensus_check` uses only debater outputs (both agree → skip debate). No gold label is used for routing at inference time — this avoids label leakage.

**Four LangGraph features used:**

1. `StateGraph(DebateState)` — typed shared state flows through every node via TypedDict
2. `Annotated[list, operator.add]` on `transcript` — debate turns accumulate automatically via append reducer
3. Conditional edges — consensus bypass, debate loop (min/max rounds + early stop), jury enable/disable
4. Pure deterministic routing nodes — `consensus_check`, `early_stop_check`, `jury_check` are plain Python with no LLM calls

### Smart Jury Logic

The jury implements a five-step deliberation protocol:

1. **Phase 1:** Three jurors vote independently with role-specific prompts
2. **Unanimous → finalize** (skip Phase 2)
3. **Disagreement + NEI signal → NEI-focused deliberation** using a two-stage prompt: first "Is evidence sufficient?" (YES/NO), then SUPPORT/REFUTE only if YES
4. **Smart aggregation after Phase 2:**
   - Calibration juror conf ≥ 5 + NEI vote → **calibration veto** → arbiter judge called
   - NEI juror conf ≥ 4 + majority conf ≤ 3 → **ambiguity safeguard** → NOT_ENOUGH_INFO wins
   - Otherwise → plain majority
5. **Logged:** `_ambiguity_flag`, `_arbiter_used`, `_deliberation_changed`

This addresses the failure mode where 2 REFUTE-leaning jurors (conf=3) silently outvote 1 high-confidence NEI juror (conf=5).

### Configuration

| Parameter | Value | Rationale |
|---|---|---|
| Debater model | Qwen3-8B | Thinking mode produces coherent advocacy arguments |
| Judge/Jury model | GPT-OSS-20B | Larger model, better argument evaluation, REFUTE bias controllable |
| Debater temperature | 0.7 | Argument variation without incoherence |
| Judge temperature | 0.3 | Deterministic verdict |
| Evidence juror temp | 0.2 | Near-deterministic citation checking |
| Calibration juror temp | 0.4 | Uncertainty estimation benefits from sampling |
| Debater max_tokens | 6144 | Required for Qwen3 thinking mode (128 insufficient) |
| Min rounds | 3 | Prevents trivial early stopping |
| Max rounds | 6 | Hard compute ceiling |
| Early stop | 2 consecutive agreements after min_rounds | |
| Self-consistency N | 13 | ≈ avg LLM calls for single-judge debate (2 init + up to 12 rebuttal + 1 judge) |
| NEI veto conf | 5 | Only highest-confidence calibration NEI triggers veto |
| Ambiguity flag conf | 4 | NEI ≥4 + majority ≤3 triggers safeguard |

---

## 2. Experiments

### Experimental Setup

All experiments run on SciFact (oracle retrieval — evidence abstracts pre-linked). 150 claims evaluated with label distribution: SUPPORT=54, REFUTE=31, NOT_ENOUGH_INFO=65. Same claim set used for all four methods. In every rebuttal round, both debaters received the complete accumulated transcript from all prior rounds.

**Four methods:**

1. **LLM Debate (Single Judge)** — full pipeline, single GPT-OSS-20B judge
2. **LLM Debate (Jury Panel)** — three-juror deliberation after single judge
3. **Direct QA (CoT)** — single Qwen3-8B call with chain-of-thought prompt, no debate
4. **Self-Consistency (N=13)** — majority vote over 13 Qwen3-8B samples at temperature 0.9

### Results

**Table 1: Overall Accuracy (n=150)**

| Method | Accuracy | N | LLM Calls / Claim |
|---|---|---|---|
| LLM Debate (Single Judge) | **53.3%** | 150 | ~13 avg |
| LLM Debate (Jury Panel — 3 jurors) | 38.7% | 150 | ~34 avg |
| Direct QA (CoT) | 56.7% | 150 | 1 |
| Self-Consistency (N=13) | 56.7% | 150 | 13 |

**Table 2: Per-Label Accuracy**

| Label | Single Judge | Jury Panel | N |
|---|---|---|---|
| SUPPORT | 33.3% | 33.3% | 54 |
| REFUTE | 58.1% | **74.2%** | 31 |
| NOT_ENOUGH_INFO | **67.7%** | 26.2% | 65 |

The jury improved REFUTE accuracy (+16pp) but collapsed NOT_ENOUGH_INFO accuracy (−41pp) due to systematic REFUTE bias in the Evidence and Logic jurors. This motivated the smart jury redesign with ambiguity safeguard.

**Table 3: Judge Confidence Calibration**

| Condition | Mean Confidence | Std |
|---|---|---|
| Correct predictions | 4.65 | 0.614 |
| Incorrect predictions | 4.29 | 0.848 |

The judge is slightly better calibrated on correct predictions (higher confidence when right) but the gap is small, suggesting overconfidence on wrong answers.

**Table 4: Accuracy by Debate Rounds**

| Rounds | Accuracy | N |
|---|---|---|
| 0 (consensus bypass) | 67.7% | 65 |
| 6 (full debate) | 42.4% | 85 |

Claims that needed debate were inherently harder — the debate structure could not overcome the difficulty of genuinely ambiguous evidence.

**Table 5: Jury Analysis**

| Metric | Value |
|---|---|
| Jury / judge disagreement | 52/150 (35%) |
| Jury better than judge | 9 cases |
| Judge better than jury | 31 cases |
| Consensus at Phase 1 | 33/150 (22%) |
| Accuracy when unanimous at init | 66.7% |
| Jurors who changed mind (Phase 2) | 0/450 (0%) |
| Phase 1 mean confidence | 3.79 |
| Phase 2 mean confidence | 3.97 (+0.18) |

**Table 6: Statistical Significance (McNemar's Test, α=0.05)**

| Comparison | χ² | p-value | Significant? |
|---|---|---|---|
| Jury vs Single Judge | 11.025 | < 0.05 | **YES** — jury significantly worse |
| Debate vs Direct QA | 0.262 | > 0.05 | No |
| Debate vs Self-Consistency | 0.254 | > 0.05 | No |
| Jury vs Direct QA | 8.557 | < 0.05 | **YES** — jury significantly worse |

McNemar's test applied to paired outputs (same 150 claims). Debate vs Direct QA not significant — the debate pipeline did not significantly outperform a single direct call.

### Figures

*(Auto-generated in `evaluation/figures/` after running `python run_experiment.py`)*

- `fig1_accuracy.png` — bar chart comparing all four methods
- `fig2_per_label.png` — per-label accuracy breakdown (SUPPORT/REFUTE/NEI) for judge and jury
- `fig3_confidence.png` — judge confidence distribution (heavily skewed toward 4–5)
- `fig4_disagreement_vs_accuracy.png` — jury disagreement level vs accuracy (unanimous vs split vs divided)
- `fig5_jury_vs_judge_overlap.png` — cases where both correct, judge only, jury only, neither
- `fig6_mind_changes.png` — juror mind-change distribution (zero across all 150 cases)

---

## 3. Analysis

### Qualitative Transcript Analysis

Five cases selected from run `run_20260317_222317` to illustrate key behavioral patterns.

---

**Case 1 — MEK/RAS Mismatch: Evidence Doesn't Address the Claim**

*Claim:* "MEK inhibitors are effective treatments in RAS-driven mouse models of cancer." | GT: SUPPORT

The evidence snippets describe a PI3K-driven mouse model (p110-alpha H1047R) — not RAS mutations, not MEK inhibitors. Debater B correctly identified this mismatch (PI3K ≠ RAS) but drew the wrong conclusion (REFUTE rather than NOT_ENOUGH_INFO). The calibration juror correctly identified that the evidence doesn't address the claim at all, but was outvoted 2-to-1 by the Evidence and Logic jurors who followed Debater B's PI3K/RAS argument.

*Judge → REFUTE ✗ | Jury → NOT_ENOUGH_INFO ✗*

**What this shows:** Both systems failed, but for different reasons. This case motivated the calibration veto in the smart jury — a high-confidence NEI from the calibration juror should not be overridden by two jurors following a rhetorically compelling but misdirected REFUTE argument.

---

**Case 2 — REFUTE Bias Overriding Correct SUPPORT**

*Claim:* "Normal granulomas form in the absence of TNF in Zebrafish." | GT: SUPPORT

Debater B argued with confidence=5, which persuaded both judge and all three jurors. The SUPPORT evidence existed in the snippets but was ignored in favor of Debater B's high-confidence framing. Both systems were wrong.

*Judge → REFUTE conf=5 ✗ | Jury → REFUTE unanimous ✗*

**What this shows:** Persuasion bias — the judge was influenced by argument confidence rather than evidence accuracy. This is the failure mode Irving et al. (2018) warned against: debate helps when the judge can verify arguments, but fails when the judge anchors on rhetorical quality.

---

**Case 3 — Jury Rescued the Judge**

*Claim:* "Gastric infection with Helicobacter pylori increases risk of gastric cancer." | GT: SUPPORT

The judge gave a low-confidence REFUTE (conf=3). The calibration juror voted SUPPORT at conf=4, tipping the jury to the correct answer. This is the ideal jury scenario: low judge confidence + calibration juror insight = correct jury verdict.

*Judge → REFUTE conf=3 ✗ | Jury → SUPPORT disagree=0.5 ✓*

**What this shows:** The jury adds value precisely when the single judge is uncertain. The calibration juror's independent evidence reading caught what the judge missed under Debater B's persuasive framing.

---

**Case 4 — Correct Ambiguity Handling**

*Claim:* "Glycan adaptation is rarely observed in the B-cell repertoire." | GT: NOT_ENOUGH_INFO

Both debaters argued with equal confidence (conf=4) in opposite directions. The calibration juror voted NEI at conf=5. Both judge and jury correctly chose NOT_ENOUGH_INFO.

*Judge → NOT_ENOUGH_INFO conf=5 ✓ | Jury → NOT_ENOUGH_INFO unanimous ✓*

**What this shows:** When the calibration juror's high-confidence NEI aligns with an inconclusive debate, the system correctly identifies genuine ambiguity. This is the smart jury's target behavior.

---

**Case 5 — Wrong Convergence: Unanimous Wrong Answer**

*Claim:* "Increased lipolysis leads to higher P38 phosphorylation in adipose tissue." | GT: NOT_ENOUGH_INFO

Debater B argued REFUTE convincingly. The judge gave REFUTE conf=5 and all three jurors unanimously agreed. But the ground truth was NOT_ENOUGH_INFO — the evidence didn't sufficiently address the specific claim.

*Judge → REFUTE conf=5 ✗ | Jury → REFUTE unanimous conf=5 ✗*

**What this shows:** The most dangerous failure mode — maximum confidence, completely wrong. Debate amplified Debater B's framing so effectively that no dissenting signal survived. This case shows that unanimous high-confidence verdicts on genuinely ambiguous evidence are a reliability concern, not a confidence signal.

---

### Connection to Theoretical Predictions

Irving et al. (2018) proposed that adversarial debate between AI agents could help a less-capable judge identify truth by forcing both sides to surface and rebut each other's arguments.

**Consistent with theory:** Case 3 (H. pylori) shows adversarial exchange surfacing a correct SUPPORT verdict that the single judge missed. The debate structure also reduced the single judge's SUPPORT blindspot relative to direct QA — SUPPORT accuracy improved from ~0% (no anti-bias instruction) to 33.3% with explicit anti-bias prompting.

**Inconsistent with theory:** Cases 2 and 5 show debate failing when both debaters share a REFUTE bias, or when Debater B's high-confidence argument persuades the judge regardless of evidence quality. Case 5 is particularly striking — debate does not just amplify one correct answer, it can converge to a wrong answer through the argumentation process itself. Irving et al. assumed the judge could verify arguments; the experiments show that when both sides share a blind spot, verification fails.

**Overall finding:** Debate is most valuable as a disagreement detector (jury disagreement predicts claim difficulty) rather than as a truth oracle. The 0-round accuracy (67.7%) significantly exceeds 6-round accuracy (42.4%), suggesting debate is routed to the hardest cases but cannot fully overcome inherent evidence ambiguity.

---

## 4. Prompt Engineering

### Design Process

**Iteration 1 — Generic instruction (failed):**
"Analyze whether the evidence supports or refutes the claim." Both debaters produced identical balanced analyses. No adversarial tension. Debater A refused to argue SUPPORT on weak evidence.

**Iteration 2 — "Intellectual honesty" framing (failed differently):**
"Argue the SUPPORT side with intellectual honesty." Qwen3's thinking mode reasoned through the evidence, concluded it didn't support the claim, then faced a conflict: argue SUPPORT honestly? The model resolved this by returning an empty reasoning field — valid JSON structure, blank content. This was the most confusing bug in the project.

*Root cause:* System messages override prompt instructions. "Intellectual honesty" won over "write non-empty reasoning."

**Iteration 3 — Lawyer framing (final, works):**
"Think of yourself as a lawyer — your job is advocacy, not honest evaluation." This framing resolves the conflict completely. Lawyers argue their client's case regardless of personal belief.

**Judge iteration 1 — No anti-bias instruction (failed):**
SUPPORT accuracy ~0%. Judge defaulted to REFUTE on almost every claim.

*Root cause:* GPT-OSS-20B is trained to be critical/skeptical. Without explicit counterinstruction, "if in doubt, REFUTE" is the default.

**Judge iteration 2 — Explicit anti-bias + CoT (final):**
Added "CRITICAL: be equally willing to rule SUPPORT or REFUTE." Added five-step structured evaluation (re-read evidence → check citations → identify strongest/weakest from each side → apply verdict rules → produce verdict). Added "Read evidence yourself — do not rely on debaters' summaries."

**Juror iteration 1 — Temperature diversity only (failed):**
Same prompt for all three jurors, temperatures 0.2/0.5/0.8. Produced stochastic diversity, not reasoning diversity. Three jurors asking the same question and sampling similar answers.

**Juror iteration 2 — Role specialization with scoring rubrics (final):**
Each juror given a distinct evaluation lens with a 5-point scoring rubric:
- Evidence Judge (temp=0.2): verbatim citation accuracy
- Logic Judge (temp=0.3): reasoning consistency across rounds
- Calibration Judge (temp=0.4): evidence sufficiency — "NOT_ENOUGH_INFO is a last resort, not a default"

**Phase 2 deliberation iteration:**
Original: "Review others and revise if needed." Result: zero jurors changed their verdict in Phase 2 across 150 cases — generic review produced only confidence increases, not genuine reconsideration.

Revised (NEI-focused): Forces two-stage structure — first answer "Is evidence sufficient?" (YES/NO), then SUPPORT/REFUTE only if YES. This separates the evidence sufficiency question from the direction question.

### Key Design Decisions

**Evidence-first framing.** Both prompts place CLAIM then EVIDENCE before role instructions. This grounds the first token predictions in factual content before role framing biases generation style.

**Structured output as CoT.** Requiring `reasoning` as a named JSON field before `stance` elicits chain-of-thought without "think step by step" — the field ordering enforces reasoning-then-conclusion.

**`counter_to_opponent` required field.** Without this field, debaters restated their opening position every round. With it, they are forced to engage with the opponent's specific evidence citation.

**`evidence_used: [int]` field.** Forces agents to cite specific snippet indices, preventing arguments from general knowledge rather than provided evidence.

**System messages have higher priority than prompt instructions.** Every role framing decision was made in the system message, not just the prompt, to avoid conflicts. This was the root cause of the empty-reasoning bug.

---

## Appendix: Full Prompt Templates

<details>
<summary><strong>Debater A — Final v3 (Lawyer framing)</strong></summary>

```
You are arguing the SUPPORT side of a scientific fact-verification debate.
Your job: build the strongest possible case that the evidence supports the claim.
Think of yourself as a lawyer — find the best reading of the evidence for your side.

Respond with ONLY valid JSON. No preamble, no explanation, no markdown fences.

Required format:
{"stance": "SUPPORT", "reasoning": "your argument here",
 "evidence_used": [0, 1], "counter_to_opponent": "your rebuttal", "confidence": 4}

CLAIM: {CLAIM}
EVIDENCE: {EVIDENCE}
DEBATE SO FAR: {TRANSCRIPT}
OPPONENT'S LAST ARGUMENT: {OPPONENT_LAST}

- "stance" must be "SUPPORT" — always.
- "reasoning": write at least 3 steps. Quote key phrases verbatim in single quotes.
- "counter_to_opponent": rebut opponent's last argument with evidence.
- "confidence": 1=weak support, 5=snippet directly confirms claim.
IMPORTANT: You must always write non-empty reasoning. That is your role.
Output only the JSON object starting with { and ending with }.
```
</details>

<details>
<summary><strong>Debater B — Final v3 (Lawyer framing)</strong></summary>

```
You are arguing the REFUTE side of a scientific fact-verification debate.
Your job: build the strongest possible case that the evidence does NOT support the claim.
Think of yourself as a lawyer — find the flaws, mismatches, and contradictions.

Respond with ONLY valid JSON. No preamble, no explanation, no markdown fences.

Required format:
{"stance": "REFUTE", "reasoning": "your argument here",
 "evidence_used": [0, 2], "counter_to_opponent": "your rebuttal", "confidence": 4}

CLAIM: {CLAIM}
EVIDENCE: {EVIDENCE}
DEBATE SO FAR: {TRANSCRIPT}
OPPONENT'S LAST ARGUMENT: {OPPONENT_LAST}

- "stance" must be "REFUTE" — always.
- Focus on: scope, magnitude, population, or direct contradiction.
- Quote key phrases verbatim. Show the specific mismatch.
- "confidence": 1=indirect refutation, 5=snippet directly contradicts claim.
IMPORTANT: You must always write non-empty reasoning. That is your role.
Output only the JSON object starting with { and ending with }.
```
</details>

<details>
<summary><strong>Judge — Final v2 (Anti-bias + 5-step CoT)</strong></summary>

```
You are an impartial judge in a scientific fact-verification debate.
Your job: decide whether evidence SUPPORTS or REFUTES the claim based on
which debater cited it more accurately.

Respond with ONLY valid JSON. No preamble, no explanation, no markdown fences.

Required format:
{"final_verdict": "SUPPORT", "winning_side": "Debater A",
 "reasoning": "step-by-step evaluation here",
 "strongest_argument_from_a": "best argument Debater A made",
 "strongest_argument_from_b": "best argument Debater B made",
 "weakest_argument_from_a": "weakest argument Debater A made",
 "weakest_argument_from_b": "weakest argument Debater B made",
 "confidence": 4}

CLAIM: {CLAIM}
EVIDENCE SNIPPETS: {EVIDENCE}
FULL DEBATE TRANSCRIPT: {TRANSCRIPT}

Perform a structured, step-by-step evaluation:
Step 1 — Re-read all evidence snippets independently.
Step 2 — For each debater: which snippets cited, quoted accurately?
Step 3 — Identify strongest and weakest argument from each side.
Step 4 — Does evidence literally support or refute the claim as stated?
Step 5 — Produce final verdict.

CRITICAL: Be equally willing to rule SUPPORT or REFUTE.
Quote at least one snippet verbatim in reasoning.
NOT_ENOUGH_INFO only if evidence genuinely does not address the claim.
Output only the JSON object starting with { and ending with }.
```
</details>

<details>
<summary><strong>Evidence Judge Juror</strong></summary>

```
You are the Evidence Judge on a three-person jury.
Your role: assess how accurately each debater cited the evidence snippets.

SCORING RUBRIC:
  5 — Debaters quoted snippets verbatim; conclusions follow directly.
  4 — Cited accurately with minor paraphrasing that doesn't change meaning.
  3 — Some citations accurate, others drifted from original meaning.
  2 — Frequently asserted things snippets do not say.
  1 — Largely ignored snippets and argued from general knowledge.

{"verdict": "REFUTE", "confidence": 4, "reasoning": "...",
 "strongest_arg_support": "...", "strongest_arg_refute": "...", "evidence_alignment": 3}

CLAIM: {CLAIM}
EVIDENCE SNIPPETS: {EVIDENCE}
FULL DEBATE TRANSCRIPT: {TRANSCRIPT}

Do NOT default to REFUTE. Base verdict on citation accuracy only.
Output only the JSON object starting with { and ending with }.
```
</details>

<details>
<summary><strong>Logic Judge Juror</strong></summary>

```
You are the Logic Judge on a three-person jury.
Your role: assess the quality of reasoning, internal consistency, rebuttal strength.

SCORING RUBRIC:
  5 — Every claim follows from evidence; opponent's best points addressed each round.
  4 — Mostly sound reasoning with minor gaps; most rebuttals engaged opponent.
  3 — Some valid reasoning but unsupported assertions; rebuttals partial.
  2 — Frequent unsupported leaps; debaters mostly restated opening positions.
  1 — Circular reasoning, contradictions, complete failure to engage opponent.

{"verdict": "REFUTE", "confidence": 4, "reasoning": "...",
 "strongest_arg_support": "...", "strongest_arg_refute": "...", "evidence_alignment": 3}

CLAIM: {CLAIM}
EVIDENCE SNIPPETS: {EVIDENCE}
FULL DEBATE TRANSCRIPT: {TRANSCRIPT}

Do NOT default to REFUTE. Base verdict on reasoning quality only.
Output only the JSON object starting with { and ending with }.
```
</details>

<details>
<summary><strong>Calibration Judge Juror</strong></summary>

```
You are the Calibration Judge on a three-person jury.
Your role: verify the verdict is proportionate to the evidence strength.
NOT_ENOUGH_INFO is a last resort, not a default.

SCORING RUBRIC:
  5 — Snippet directly and explicitly resolves the claim word-for-word.
  4 — Snippet strongly supports or contradicts with minor inference.
  3 — Snippet partially addresses the claim.
  2 — Snippet is tangential but loosely relevant.
  1 — Snippet does not address the claim at all.

{"verdict": "REFUTE", "confidence": 4, "reasoning": "...",
 "strongest_arg_support": "...", "strongest_arg_refute": "...", "evidence_alignment": 4}

CLAIM: {CLAIM}
EVIDENCE SNIPPETS: {EVIDENCE}
FULL DEBATE TRANSCRIPT: {TRANSCRIPT}

Use NOT_ENOUGH_INFO only if zero snippets address the claim at all.
Output only the JSON object starting with { and ending with }.
```
</details>

---

## References

1. Irving, G., Christiano, P., & Amodei, D. (2018). AI Safety via Debate. *arXiv:1805.00899.*
2. Wadden, D. et al. (2020). Fact or Fiction: Verifying Scientific Claims. *EMNLP 2020.*
3. Liang, T. et al. (2024). Encouraging Divergent Thinking in LLMs through Multi-Agent Debate. *EMNLP 2024.*
4. Kenton, Z. et al. (2024). On Scalable Oversight with Weak LLMs Judging Strong LLMs. *NeurIPS 2024.*
5. Kalra, N. et al. (2025). VERDICT: A Library for Scaling Judge-Time Compute. *Haize Labs.*
6. Wang, X. et al. (2023). Self-Consistency Improves Chain of Thought Reasoning in LLMs. *ICLR 2023.*
7. Wei, J. et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in LLMs. *NeurIPS 2022.*
8. Du, Y. et al. (2023). Improving Factuality and Reasoning through Multiagent Debate. *ICML 2024.*