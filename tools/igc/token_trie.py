"""token_trie.py: the tokens that can spell a value, step by step (token channel design, 2026-11-12).

MIP steers a value it computes by allowing, at each step, only the tokens that can still spell the value's remaining text, in any
tokenization the vocabulary has; the model chooses among them (the developer's option "ii": the right path made the only one, the
model's own token choice kept). The structure is a DAG over the text's character offsets: node i is "the first i characters are
written"; an edge from i to i + len(p) for every token whose text p starts the rest at i. Nodes that cannot reach the end are pruned,
so a plan never leaves the sampler a step with no token allowed.

vocab_pieces(tokenizer) -> list[str]      the decoded text of every token id (special tokens as their own text; never spelled)
ValueDag.build(text, pieces) -> ValueDag | None
dag.allowed(node) -> list[int]; dag.step(node, token) -> int | None; dag.end; dag.to_json() -> {"edges": [[[token, node], ...]], "end"}
"""
import re
from collections import defaultdict

_SPECIAL = re.compile(r'^<\|.*\|>$')
_PIECES = {}


def vocab_pieces(tokenizer):
    """The text of every token id, decoded one id at a time; cached per tokenizer."""
    key = (getattr(tokenizer, 'name_or_path', ''), len(tokenizer))
    if key not in _PIECES:
        _PIECES[key] = [tokenizer.decode([i]) for i in range(len(tokenizer))]
    return _PIECES[key]


_INDEX = {}


def _index(pieces):
    """Token ids grouped by their first character, special tokens and empty pieces left out; cached per pieces list."""
    key = id(pieces)
    hit = _INDEX.get(key)
    if hit is not None and hit[0] is pieces:
        return hit[1]
    by_first = defaultdict(list)
    for t, p in enumerate(pieces):
        if p and not _SPECIAL.match(p):
            by_first[p[0]].append(t)
    _INDEX[key] = (pieces, by_first)
    return by_first


class ValueDag:
    def __init__(self, edges, end):
        self.edges, self.end = edges, end   # edges[node] = {token: next node}

    @classmethod
    def build(cls, text, pieces):
        if not text:
            return None
        by_first = _index(pieces)
        n = len(text)
        out = [dict() for _ in range(n + 1)]
        for i in range(n):
            for t in by_first.get(text[i], ()):
                p = pieces[t]
                if text.startswith(p, i):
                    out[i][t] = i + len(p)
        # keep only nodes that reach the end, walking back from it
        live = [False] * (n + 1)
        live[n] = True
        for i in range(n - 1, -1, -1):
            out[i] = {t: j for t, j in out[i].items() if live[j]}
            live[i] = bool(out[i])
        if not live[0]:
            return None
        for i in range(n):
            if not live[i]:
                out[i] = {}
        return cls(out, n)

    def allowed(self, node):
        return list(self.edges[node]) if 0 <= node < len(self.edges) else []

    def step(self, node, token):
        return self.edges[node].get(token) if 0 <= node < len(self.edges) else None

    def to_json(self):
        return dict(edges=[[[t, j] for t, j in e.items()] for e in self.edges], end=self.end)
