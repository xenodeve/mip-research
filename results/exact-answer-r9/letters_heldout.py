"""letters_heldout.py: the letter-count reader (P7) where it was not written from. Lists fixed 2026-11-11 before the run.
1. string-letter questions in other wordings and tasks (should read the string named);
2. questions that mention letters but quote no string, or quote no letters question (should read nothing);
3. reach over MATH-500, GSM8K test and every stored row: every read is printed (a read there must be a real letters question).
python results/exact-answer-r9/letters_heldout.py <MATH-500 test.jsonl> <gsm8k parquet>
"""
import glob, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding='utf-8')
from tools.igc.letters import letter_case  # noqa: E402

SHOULD = [
    ('How many times does the letter e appear in "nevertheless"?', 'nevertheless'),
    ("Count the s's in mississippi.", 'mississippi'),
    ('Which letter appears most often in the string abracadabra?', 'abracadabra'),
    ('How many distinct letters are in the word "bookkeeper"?', 'bookkeeper'),
    ('A letter is chosen at random from the word PROBABILITY. What is the chance it is a B?', 'PROBABILITY'),
    ('Arrange the letters of `banana`: how many distinct orders are there?', 'banana'),
    ('Three letters picked without replacement from qqrrqsrq. What is prob of picking 2 q and 1 r?', 'qqrrqsrq'),
    ("How many characters are in 'hello'?", 'hello'),
]
SHOULD_NOT = [
    'How many letters are in the English alphabet?',
    'Each letter of the alphabet is assigned a value. Find the sum.',
    'Write a letter to your friend about your holiday.',
    'The letters A, B and C stand for digits. Find A + B + C.',
    'Let the characters in the play be counted. How many are there?',
    'What is the sum of 3 and 4?',
]


def main():
    print('1. should read the string')
    for q, want in SHOULD:
        got = letter_case(q)
        print(f'   {"ok  " if got and got[0] == want else "MISS"} {want:14s} got {got[0] if got else None!s:14s} | {q}')
    print('2. should read nothing')
    for q in SHOULD_NOT:
        got = letter_case(q)
        print(f'   {"ok  " if got is None else "READ"} got {got[0] if got else None!s:14s} | {q}')
    print('3. reach')
    math = [json.loads(x) for x in Path(sys.argv[1]).read_text(encoding='utf-8').splitlines() if x.strip()]
    import polars as pl
    gsm = pl.read_parquet(sys.argv[2])
    stored = {}
    for f in glob.glob('C:/Users/xenod/mip-work/a100/*-rows.jsonl'):
        for line in open(f, encoding='utf-8'):
            if line.strip():
                r = json.loads(line)
                stored[r['id']] = r.get('question') or ''
    for name, qs in (('MATH-500', [r['problem'] for r in math]), ('GSM8K', list(gsm['question'])), ('stored', list(stored.values()))):
        reads = [(q, letter_case(q)) for q in qs]
        reads = [(q, g) for q, g in reads if g is not None]
        print(f'   {name}: {len(qs)} questions, {len(reads)} read')
        for q, g in reads:
            print(f'      {g[0]:16s} | {q[:110]!r}')


if __name__ == '__main__':
    main()
