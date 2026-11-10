"""One short streamed request with n_probs: the brake needs a chunk with completion_probabilities and top_logprobs/top_probs."""
import json, sys, urllib.request
body = dict(prompt='The capital of France is', n_predict=3, n_probs=5, stream=True)
req = urllib.request.Request(f'http://127.0.0.1:{sys.argv[1]}/completion', json.dumps(body).encode(), {'Content-Type': 'application/json'})
for raw in urllib.request.urlopen(req, timeout=120):
    line = raw.decode().strip()
    if line.startswith('data:'):
        msg = json.loads(line[5:])
        cp = msg.get('completion_probabilities') or []
        print(json.dumps(cp)[:300])
        ok = bool(cp) and any(k in cp[0] for k in ('top_logprobs', 'top_probs'))
        print('BRAKE FIELDS', 'OK' if ok else 'MISSING')
        sys.exit(0 if ok else 1)
