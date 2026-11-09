"""Was each value --check-hand inserted on an answer that finished right itself right? Each expression is evaluated again, exactly,
and compared with the inserted value (a congruence by its remainder; a decimal or scientific value to the places it gives).
python hand_values.py SIM_DIR SIM_NAME [wrong]   (SIM_DIR holds <half>-<name>.json and snap/<half>/records.json;
add `wrong` for the answers that finished wrong)"""
import json
import re
import sys

sys.path.insert(0, r'D:\Github\mip-research')
from tools.igc.mathexpr import evaluate  # noqa: E402

D, name = sys.argv[1], sys.argv[2]
WANT_RIGHT = not (len(sys.argv) > 3 and sys.argv[3] == 'wrong')
MOD = re.compile(r'(-?\d+) \\pmod\{(\d+)\}')
right = wrong = unchecked = 0
for h in 'ab':
    sim = json.load(open(f'{D}/{h}-{name}.json', encoding='utf-8'))
    recs = json.load(open(f'{D}/snap/{h}/records.json', encoding='utf-8'))
    for rec, row in zip(recs, sim['rows']):
        st = row.get('stop') or {}
        if st.get('kind') != 'hand check' or bool(rec['right']) != WANT_RIGHT:
            continue
        expr = re.sub(r'^.*\$', '', str(st.get('expr'))).strip(' *')
        val = str(st.get('value'))
        good = None
        try:
            e = evaluate(expr)
            m = MOD.fullmatch(val)
            if e is not None and m:
                good = int(e) % int(m.group(2)) == int(m.group(1)) % int(m.group(2))
            elif e is not None:
                # `1.2644 \times 10^{-9}`, and `-0.332 \times 10` (times ten): the places count before the power is applied
                sci = re.fullmatch(r'(-?[\d.]+) \\times 10(?:\^\{(-?\d+)\})?', val)
                v = evaluate(val) if not sci else float(sci.group(1)) * 10 ** int(sci.group(2) or 1)
                if v is not None:
                    places = len(val.split('.')[1].split(' ')[0]) if '.' in val else None
                    if places is None:
                        good = e == v
                    else:   # a decimal is right when it is the exact value rounded or cut at the places it gives
                        scale = 10 ** int(sci.group(2) or 1) if sci else 1
                        good = abs(float(e) - float(v)) <= 10 ** -places * scale
        except Exception:
            good = None
        tag = {True: 'RIGHT value', False: 'WRONG value', None: 'not checked'}[good]
        right += good is True
        wrong += good is False
        unchecked += good is None
        print(f'{tag:12} {rec["id"]}#{rec["sample"]}: {expr[:50]} -> {val}')
print(f"inserted on {'right' if WANT_RIGHT else 'wrong'} answers: {right} right, {wrong} wrong, {unchecked} not checked by this script")
