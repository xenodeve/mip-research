"""spec_probe.py: before a speculative-decoding flag is used under tir4b, check that llama-server still hands back exactly what tir4b
reads. tir4b stops the server on words (`\\n`, `=`, ...) and takes `tokens` from the reply as the text the model wrote; a server that
accepted draft tokens past the stop and returned them would make software put its value in the wrong place.

For each of N requests (dev-row prompts, T 0.6, the stops tir4b sends): the returned tokens, decoded, must hold the stop word only at
their end, and `content` + the stop word must equal the decoded tokens. Prints the draft counts the server reports (`timings`).
python scripts/pc/spec_probe.py --server http://127.0.0.1:8099 [--n 24]
"""
import argparse, json, sys, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding='utf-8')

W = Path('C:/Users/xenod/mip-work/a100')


def post(server, body):
    req = urllib.request.Request(server.rstrip('/') + '/completion', data=json.dumps(body).encode('utf-8'),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as reply:
        return json.loads(reply.read().decode('utf-8'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--server', required=True)
    ap.add_argument('--n', type=int, default=24)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    from tools.igc.tir4b import prompt_text
    tok = AutoTokenizer.from_pretrained('D:/models/Qwen3.5-4B')
    rows = [json.loads(x) for x in (W / 'pc9-dev-rows.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    bad = 0
    drafted = accepted = 0
    for i in range(args.n):
        row = rows[i % len(rows)]
        msgs = [{'role': 'user', 'content': prompt_text(row)}]
        # text first, then ids: this transformers returns a BatchEncoding from tokenize=True, which json cannot send
        prompt = tok.encode(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), add_special_tokens=False)
        # a long thinking prefix first, so the probe sees the speculative path inside a long context too
        warm = post(args.server, dict(prompt=prompt, n_predict=400 + 50 * i, temperature=0.6, top_p=0.95, top_k=20, seed=i,
                                      cache_prompt=True, return_tokens=True))
        prefix = prompt + list(warm['tokens'])
        for stops in (['\n'], ['='], ['=', '\n']):
            r = post(args.server, dict(prompt=prefix, n_predict=256, stop=stops, temperature=0.6, top_p=0.95, top_k=20, seed=1000 + i,
                                       cache_prompt=True, return_tokens=True))
            text = tok.decode(r['tokens'])
            t = r.get('timings') or {}
            drafted += t.get('draft_n', 0) or 0
            accepted += t.get('draft_n_accepted', 0) or 0
            if r.get('stop_type') == 'word':
                # the tokens spell the content, then the stop word, which ends inside the last token (that token may hold more,
                # `\n\n` for `\n`); no token may follow the one that holds the stop word
                whole = r['content'] + r.get('stopping_word', '')
                ok = r.get('stopping_word') in stops and text.startswith(whole) and len(tok.decode(r['tokens'][:-1])) < len(whole)
            else:
                ok = not any(w in text for w in stops)
            if not ok:
                bad += 1
                print(f'BAD row {row["id"]} stops {stops!r}: stop_type {r.get("stop_type")} word {r.get("stopping_word")!r} '
                      f'content {r["content"][-60:]!r} tokens-text {text[-80:]!r}')
    print(f'{args.n * 3} stopped requests, {bad} bad; draft tokens {drafted}, accepted {accepted}'
          f'{f" ({accepted / drafted:.0%})" if drafted else ""}')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
