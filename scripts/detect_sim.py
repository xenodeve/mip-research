"""#307 anti-loop accuracy: replay the loop detector at other thresholds and check cadences over the stored A100 traces of every arm with
no closer (so the trace is the model's own text to the end). For each variant: on answers that finished (right / wrong) any fire is a false
positive; on answers the run ended `stuck`, how many tokens earlier the variant fires than the run's own stop.

Variants (rounds of an exact cycle, rounds past 512 tokens, line_back count, where it is checked):
  V0 20 / 8 / 16, every 1024 tokens            (the runner now)
  V1 20 / 8 / 16, every line end
  V2 10 / 4 / 8,  every line end
  V3 6  / 3 / 5,  every line end
  V4 6  / 3 / 5,  every line end, and the cycle must span 500+ tokens (Circular Reasoning's numeric-loop rule)
The counting check (digits masked, --guard-progress) stays as it is, every 1024 tokens, in every variant.
python detect_sim.py ARM [ARM...]   -> detect_sim.json, and a table."""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, r'D:\Github\Strata\.worktrees\gs-dev')
LINES = (198, 271)
THINK_CLOSE = 248069   # set from the tokenizer in main
VARIANTS = {'V0': (20, 8, 16, 'chunk', 0), 'V1': (20, 8, 16, 'line', 0), 'V2': (10, 4, 8, 'line', 0), 'V3': (6, 3, 5, 'line', 0),
            'V4': (6, 3, 5, 'line', 500),
            # the cycle check alone moved to every line end and fewer rounds; line_back kept as the runner has it (16, at chunk ends)
            'V5': (10, 4, 16, 'cycle-line', 0), 'V6': (6, 3, 16, 'cycle-line', 0), 'V7': (6, 3, 16, 'cycle-line', 500)}


def cycle_rounds(t, i, longest=4096):
    """[(period, rounds)] of exact cycles ending at t[i]: rounds = how many times the period repeats back from i."""
    out = []
    for p in range(1, min(longest, i) + 1):
        if t[i] != t[i - p]:
            continue
        j = i
        while j - p >= 0 and t[j] == t[j - p]:
            j -= 1
        rounds = (i - j) // p + 1
        if rounds >= 3:
            out.append((p, rounds))
    return out


def scan(job):
    from tools.igc.external_score import counting
    key, ids, think_end = job
    t = ids[:think_end]
    fired = {}
    lines = {}
    line = []
    for i, tok in enumerate(t):
        at_line = tok in LINES
        line_count = 0
        if at_line:
            if len(line) >= 12:
                k = tuple(line)
                lines[k] = lines.get(k, 0) + 1
                line_count = lines[k]
            line = []
        else:
            line.append(tok)
        at_chunk = (i + 1) % 1024 == 0
        if not (at_line or at_chunk):
            continue
        cyc = None
        for name, (r_short, r_long, lb, where, span) in VARIANTS.items():
            if name in fired:
                continue
            if where == 'chunk' and not at_chunk:
                continue
            if cyc is None:
                cyc = cycle_rounds(t, i)
            hit = None
            for p, rounds in cyc:
                need = r_short if p <= 512 else r_long
                if rounds >= need and p * rounds >= span:
                    hit = 'cycle'
                    break
            if hit is None and line_count >= lb and (where != 'cycle-line' or at_chunk):
                hit = 'line_back'
            if hit is None and at_chunk and counting(t[:i + 1], progress=True):
                hit = 'counting'
            if hit:
                fired[name] = (i + 1, hit)
        if len(fired) == len(VARIANTS):
            break
    return key, fired


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(r'D:\models\Qwen3.5-4B')
    close = tok.convert_tokens_to_ids('</think>')
    jobs, meta = [], {}
    for arm in sys.argv[1:]:
        recs = {(r['id'], r['sample']): r for r in json.loads((HERE / 'out' / arm / 'records.json').read_text(encoding='utf-8'))}
        for line in (HERE / 'out' / arm / 'ids.jsonl').read_text(encoding='utf-8').splitlines():
            d = json.loads(line)
            r = recs[(d['id'], d['sample'])]
            ids = d['out_ids']
            end = ids.index(close) if close in ids else len(ids)
            key = f"{arm}|{d['id']}|{d['sample']}"
            meta[key] = dict(arm=arm, finish=r['finish'], right=r['right'], think_end=end, tokens=len(ids))
            jobs.append((key, ids, end))
    jobs.sort(key=lambda j: len(j[1]))
    results = {}
    with ProcessPoolExecutor(max_workers=4) as ex:
        for key, fired in ex.map(scan, jobs, chunksize=4):
            results[key] = fired
    (HERE / 'detect_sim.json').write_text(json.dumps(dict(meta=meta, fired=results)), encoding='utf-8')
    for name in VARIANTS:
        fp_right = [k for k, m in meta.items() if m['finish'] == 'stop' and m['right'] and name in results[k]]
        fp_wrong = [k for k, m in meta.items() if m['finish'] == 'stop' and not m['right'] and name in results[k]]
        stuck = [k for k, m in meta.items() if m['finish'] == 'stuck' and m['think_end'] == m['tokens']]
        caught = [k for k in stuck if name in results[k]]
        saved = sum(meta[k]['think_end'] - results[k][name][0] for k in caught)   # against the run's own stop
        print(f"{name}: fires on finished-right {len(fp_right)}/{sum(1 for m in meta.values() if m['finish']=='stop' and m['right'])}, "
              f"finished-wrong {len(fp_wrong)}, catches stuck-in-thinking {len(caught)}/{len(stuck)}, tokens before the run's stop on them {saved:,}")
        for k in fp_right[:30]:
            print('    FP right', k, results[k][name])


if __name__ == '__main__':
    main()
