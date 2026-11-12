"""gate1_parity.py: gate 1 of the token channel (docs/plans/2026-11-12-mip-token-channel-plan.md, task 7). A GPU run on the RTX 5060 Ti.

The stock llama-server (build-w4a8) and the patched one (build-w4a8-mip, MIP_CHANNEL set, no mip_id in the requests) must give the same
tokens: 3 questions x 400 tokens, greedy, and sampled at T 0.6 with seed 1 (the locked sampling). Each server alone, -np 1, the locked
config. Writes results/token-channel/gate1.txt.  python scripts/pc/gate1_parity.py
"""
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GGUF = 'C:/Users/xenod/mip-work/gguf/qwen35-4b-nvfp4-axion.gguf'
BUILDS = {'stock': 'C:/AI/llama.cpp/build-w4a8/bin/llama-server.exe', 'patched': 'C:/AI/llama.cpp/build-w4a8-mip/bin/llama-server.exe'}
PORT = 8099
QUESTIONS = ['What is 17*23? Think step by step.', 'Simplify (172/(-1))/((15/9*(-4)/((180/5)/18))/(5/(-3))).',
             'How many positive divisors does 360 have?']
CUDA = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin'


def prompt(q):
    return f'<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n'


def post(body):
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}/completion', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def run(build):
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='1', PATH=f'{CUDA}/x64;{CUDA};' + os.environ['PATH'])
    if build == 'patched':
        env['MIP_CHANNEL'] = '127.0.0.1:8098'   # set, nothing listening: a request without mip_id must not touch it anyway
    log = open(os.path.join(os.environ['TEMP'], f'gate1-{build}.log'), 'w')
    srv = subprocess.Popen([BUILDS[build], '-m', GGUF, '-ngl', '999', '-c', '262144', '-np', '1', '-kvu', '-fa', 'on', '-ctk', 'q8_0',
                            '-ctv', 'q8_0', '--no-cache-idle-slots', '--host', '127.0.0.1', '--port', str(PORT), '--no-webui'],
                           env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(180):
            if srv.poll() is not None:
                raise RuntimeError(f'{build} server exited {srv.returncode}')
            try:
                urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2).read()
                break
            except OSError:
                time.sleep(1)
        out = {}
        for i, q in enumerate(QUESTIONS):
            g = post(dict(prompt=prompt(q), n_predict=400, temperature=0, cache_prompt=False, return_tokens=True))
            s = post(dict(prompt=prompt(q), n_predict=400, temperature=0.6, top_p=0.95, top_k=20, min_p=0, seed=1, cache_prompt=False,
                          return_tokens=True))
            out[i] = (g['tokens'], s['tokens'])
        return out
    finally:
        srv.terminate()
        srv.wait(60)


def main():
    a, b = run('stock'), run('patched')
    lines = ['gate 1 (token channel): stock build-w4a8 vs patched build-w4a8-mip with MIP_CHANNEL set and no mip_id; 4B NVFP4 W4A8 · '
             'llama.cpp · RTX 5060 Ti; 3 questions x 400 tokens, greedy and T 0.6 seed 1']
    ok = True
    for i in a:
        for k, name in ((0, 'greedy'), (1, 'sampled')):
            x, y = a[i][k], b[i][k]
            same = x == y
            ok &= same
            first = next((j for j, (p, q) in enumerate(zip(x, y)) if p != q), None)
            lines.append(f'question {i} {name}: {len(x)} vs {len(y)} tokens, {"identical" if same else f"differ at {first}"}')
    lines.append('PASS' if ok else 'DIFFERENCES')
    path = ROOT / 'results' / 'token-channel' / 'gate1.txt'
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
