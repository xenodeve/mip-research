"""letters.py: the letters of a string the question writes out (round 9 P7, docs/failures.md).

The case, stated without a test set: a question asks about the letters of a string it writes out (how many of a letter, a draw from
its letters, its arrangements). Tokenized models miscount or misquote such strings (The Strawberry Problem, arXiv 2505.14172; counts
grow wrong with length, arXiv 2412.18626); round 9 lost `swr_p_sequence` 3480#0 and 9951#1 this way. Software counts the string as the
question writes it, which is exact under every reading.

The string is read only where the sentence makes it the object of a question about letters:
- quoted (`"mississippi"`, `'abc'`, backticks) in a question that names letters or characters;
- `letters of / in (the word) W`, `the word / string / name W`;
- `picked / chosen / drawn / selected ... from W` in a question about letters;
- `x's ... in W` (`How many r's are in strawberry?`).
A word that only follows `in` or `from` (`in full`, `in alphabetical order`) is not read: the first rule set read 4 GSM8K questions
and 2 stored rows that way (`in kind`, `in full`), true counts with no bearing on the question.

letter_case(question) -> (string, [(letter, count), ...]) or None   counts in order of first appearance
letter_line(string, counts) -> str
"""
import re
from collections import Counter

_ABOUT = re.compile(r"\b(?:letters?|characters?|chars?)\b|\b[a-zA-Z]'s\b", re.I)
_QUOTED = re.compile(r'["`“‘\']([A-Za-z]{2,})["`”’\']')
# the string ends its phrase: punctuation or the end follows it (`from vgcgplcggpgvp.`, `in strawberry?`); a word with another word
# after it is prose (`her name has 4 more`, `from each pair`)
_W = r'([A-Za-z]{3,})(?=\s*(?:[.?!,:;]|$))'
_OBJECT = [
    re.compile(r'\b(?:letters?|characters?)\s+(?:of|in)\s+(?:the\s+(?:word|string|name)\s+)?' + _W, re.I),
    re.compile(r'\b(?:word|string|name)\s+' + _W, re.I),
    re.compile(r'\b(?:picked|chosen|drawn|selected|taken)\b[^.?!]{0,60}?\bfrom\s+(?:the\s+(?:word|string)\s+)?' + _W, re.I),
    re.compile(r"\b[a-zA-Z]'s\b[^.?!]{0,20}?\bin\s+(?:the\s+(?:word|string)\s+)?" + _W, re.I),
]
_NOT_A_STRING = {'the', 'this', 'that', 'which', 'her', 'his', 'their', 'its', 'your', 'our', 'a', 'an', 'it', 'them'}


def letter_case(question):
    q = question.strip().split('\n')[0]
    if not _ABOUT.search(q):
        return None
    m = _QUOTED.search(q)
    word = m.group(1) if m else None
    if word is None:
        for pattern in _OBJECT:
            m = pattern.search(q)
            if m and m.group(1).lower() not in _NOT_A_STRING:
                word = m.group(1)
                break
    if word is None:
        return None
    counts = Counter(word)
    order = []
    for ch in word:
        if ch not in dict(order):
            order.append((ch, counts[ch]))
    return word, order


def letter_line(word, counts):
    parts = ', '.join(f'{ch} {n}' for ch, n in counts)
    return f'The string {word} has {len(word)} letters: {parts}.'
