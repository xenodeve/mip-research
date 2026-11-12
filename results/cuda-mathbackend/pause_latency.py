"""pause_latency.py OUT: where the pause of an inline put goes, per arm, from tir4b's TIR4B_TIMING lines (scripts/pc/pause_latency.sh).

A pause starts when llama-server's reply for the request that ended at `=` comes back and ends when the next request's first new token
exists. Split:
  mip_host      the reply taken by tir4b: the readers, the arithmetic, the put, encoding (take_s), plus the gap before the next request
  of it device  the CUDAMathBackend calls inside it (arm cuda; 0 in arm py), with their count
  resume_read   the server reading the tokens added since the cached prefix (the put and what came with it): the next prompt_ms
  http          the rest of the next request's round trip besides reading and generating (HTTP, queueing, slot pick)
  first_token   one token's share of the next request's generation (predicted_ms / predicted_n)
pause = mip_host + resume_read + http + first_token. Not measured: the time from the `=` token to the server's reply (inside the first
request's round trip, the same in both arms). Stops at `=` with no put are reported apart: the same machinery, nothing put."""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def load(path):
    by = defaultdict(list)
    for line in open(path, encoding='utf-8'):
        r = json.loads(line)
        by[(r['row'], r['sample'])].append(r)
    return by


def pauses(by):
    put, bare = [], []
    for recs in by.values():
        for a, b in zip(recs, recs[1:]):
            if a.get('stopping_word') != '=' and '=' not in (a.get('stopping_word') or ''):
                continue
            if not b.get('predicted_n'):
                continue
            host = a['take_s'] + (b['host_before_s'] or 0)
            read = (b['prompt_ms'] or 0) / 1000
            gen = (b['predicted_ms'] or 0) / 1000
            http = max(b['http_s'] - read - gen, 0)
            first = gen / b['predicted_n']
            row = dict(host=host, dev=a['dev_s'], calls=a['dev_calls'], mismatch=a['dev_mismatch'], fallback=a['dev_fallback'],
                       read=read, read_n=b['prompt_n'] or 0, http=http, first=first, total=host + read + http + first)
            (put if 'inline value' in (a.get('events') or []) else bare).append(row)
    return put, bare


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float('nan')


def show(name, rows):
    if not rows:
        print(f'  {name}: none')
        return
    ms = lambda k, f=statistics.median: f([r[k] for r in rows]) * 1000   # noqa: E731
    print(f'  {name}: {len(rows)} pauses')
    print(f'    total        median {ms("total"):8.2f} ms   p10 {q([r["total"] for r in rows], .1) * 1000:8.2f}   p90 {q([r["total"] for r in rows], .9) * 1000:8.2f}   mean {ms("total", statistics.mean):8.2f}')
    for k, label in (('host', 'mip_host'), ('dev', 'of it device'), ('read', 'resume_read'), ('http', 'http'), ('first', 'first_token')):
        print(f'    {label:12s} median {ms(k):8.3f} ms   mean {ms(k, statistics.mean):8.3f}   p90 {q([r[k] for r in rows], .9) * 1000:8.3f}')
    calls = [r['calls'] for r in rows]
    print(f'    device calls per pause: median {statistics.median(calls)}, mean {statistics.mean(calls):.1f}, max {max(calls)}; '
          f'mismatches {sum(r["mismatch"] for r in rows)}, fallbacks {sum(r["fallback"] for r in rows)}; '
          f'tokens read on resume: median {statistics.median([r["read_n"] for r in rows])}')


def main():
    out = Path(sys.argv[1])
    print(f'{out} (commit {(out / "commit").read_text().strip()[:12]}); {(out / "setup").read_text().strip()}')
    for arm in ('py', 'cuda'):
        f = out / f'{arm}.timing.jsonl'
        if not f.exists():
            print(f'arm {arm}: no timing file')
            continue
        by = load(f)
        put, bare = pauses(by)
        reqs = sum(len(v) for v in by.values())
        dev = sum(r['dev_s'] for v in by.values() for r in v)
        calls = sum(r['dev_calls'] for v in by.values() for r in v)
        print(f'arm {arm}: {len(by)} answers, {reqs} requests; device calls {calls} ({dev:.2f} s in all)')
        show('stops at = with a put', put)
        show('stops at = without a put', bare)


if __name__ == '__main__':
    main()
