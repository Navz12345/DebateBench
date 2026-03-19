import json, glob, os

run_dirs = sorted(glob.glob('logs/run_*/'))
run_dir = run_dirs[-1]
files = sorted(glob.glob(f'{run_dir}debate_*.json'))
print(f'Run dir: {run_dir}  ({len(files)} debates)')

interesting = []
for f in files:
    try:
        with open(f, encoding='utf-8') as fh:
            d = json.load(fh)
        jury_v   = d.get('jury_verdict', '')
        judge_v  = d.get('final_verdict', '')
        disagree = d.get('jury_disagreement', 0)
        if jury_v and judge_v and jury_v != judge_v:
            interesting.append((disagree, f, d))
    except Exception:
        pass

interesting.sort(reverse=True)

if not interesting:
    print('No jury-judge disagreement cases found.')
    print('Showing highest-disagreement case instead...')
    all_cases = []
    for f in files:
        try:
            with open(f, encoding='utf-8') as fh:
                d = json.load(fh)
            disagree = d.get('jury_disagreement', 0)
            all_cases.append((disagree, f, d))
        except Exception:
            pass
    all_cases.sort(reverse=True)
    interesting = all_cases[:1]

_, best_file, best = interesting[0]
print()
print('=' * 60)
print('CLAIM:', best['claim'])
print('GROUND TRUTH:', best['ground_truth'])
print('JUDGE VERDICT:', best['final_verdict'])
print('JURY VERDICT:', best.get('jury_verdict', 'N/A'))
print('JURY DISAGREEMENT:', best.get('jury_disagreement', 0))
print()
print('EVIDENCE SNIPPETS:')
for i, e in enumerate(best.get('evidence_snippets', [])):
    print(f'  [{i}] {e}')
print()
print('JUROR PHASE 1 VOTES:')
for a in best.get('jury_assessments', []):
    print(f'  {a.get("juror_id")} ({a.get("role")}): {a.get("verdict")} conf={a.get("confidence")}')
    print(f'    reasoning: {str(a.get("reasoning",""))[:200]}')
print()
print('TRANSCRIPT (first 4 turns):')
for t in best.get('transcript', [])[:4]:
    print(f'  [{t.get("agent")} Round {t.get("round")}] stance={t.get("stance")}')
    print(f'    {str(t.get("reasoning",""))[:200]}')
    if t.get('counter'):
        print(f'    counter: {str(t.get("counter",""))[:150]}')
print()
print('JUDGE REASONING:')
judge_out = best.get('judge_output') or {}
print(f'  {str(judge_out.get("reasoning",""))[:300]}')