"""rescore.py: round 9's stored answers (4B NVFP4 W4A8 · llama.cpp · PC layer) re-scored by the current rule, beside the stored rule and
the API reader. Every answer whose rule result changes is listed.

python results/exact-answer-r9/rescore.py <pc dir> <rows> [<rows> ...]
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.igc.select_rows import right  # noqa: E402


def main():
    pc = Path(sys.argv[1])
    rows = {}
    for f in sys.argv[2:]:
        for line in Path(f).read_text(encoding='utf-8').splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r['id']] = r
    for arm in ('gen', 'ctrl'):
        was = now = api = 0
        for half in ('a', 'b'):
            d = pc / f'r9{half}-{arm}'
            verdicts = json.loads((d / 'verdicts.json').read_text(encoding='utf-8'))
            for rec in json.loads((d / 'records.json').read_text(encoding='utf-8')):
                key = f"{rec['id']}#{rec['sample']}"
                new = right(rows[rec['id']], rec['text'])
                verdict = (verdicts.get(key) or {}).get('verdict')
                was, now, api = was + bool(rec['right']), now + bool(new), api + (verdict == 'same')
                if bool(new) != bool(rec['right']):
                    print(f'  {arm} {key}: rule {rec["right"]} -> {new}, api {verdict}')
        print(arm, 'rule before', was, 'rule now', now, 'api', api)


if __name__ == '__main__':
    main()
