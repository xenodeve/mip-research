"""Hand labels for the 69 round-9 General answers that had a loop close or answer close (read case by case, 2026-11-10).
Class = what the model was repeating when the close came. Prints the class x outcome table and the appendix rows."""
import json, sys
from collections import Counter, OrderedDict

L = {
    # A: re-checking an equation that is already true
    'A': ['dm_mixed_longer-5053#0', 'dm_mixed_longer-5053#1', 'dm_div_remainder-5053#0', 'dm_mul_div_longer-5942#0',
          'dm_mul_div_longer-5942#1', 'dm_div_big-5942#0', 'stdmath-ed48028c#1',
          'dm_mixed_longer-926#0', 'dm_mixed_longer-926#1', 'dm_mul_div_longer-5070#1', 'dm_div_big-928#1', 'dm_div_remainder-926#0',
          'stdmath-eaaaaeb1#1', 'dm_mixed_longer-5920#1', 'stdmath-ed48028c#0', 'stdmath-5a258a23#0',
          'dm_mul_div_longer-9991#1', 'stdmath-fbf10ed4#1',
          'dm_div_big-5070#1', 'dm_div_big-9991#0', 'dm_div_big-9991#1'],
    # B: doubting how the question's input reads (a sign, a bracket, a string, a phrase)
    'B': ['dm_simplify_surd-3480#0', 'dm_simplify_surd-9937#0', 'dm_swr_p_sequence-3480#1', 'dm_swr_p_sequence-5053#0',
          'dm_mul_div_longer-1395#0', 'dm_mul_div_longer-1395#1', 'dm_div_big-5942#1', 'dm_simplify_surd-1391#0',
          'dm_simplify_surd-1391#1', 'dm_simplify_surd-5920#0', 'dm_simplify_surd-5920#1', 'dm_conversion-1391#0',
          'dm_conversion-1391#1', 'dm_swr_p_sequence-9951#0',
          'dm_simplify_surd-926#1', 'dm_simplify_surd-1501#1', 'dm_simplify_surd-5053#1', 'dm_simplify_surd-2773#0',
          'dm_simplify_surd-9951#0', 'dm_simplify_surd-9951#1',
          'dm_simplify_surd-3480#1', 'dm_swr_p_sequence-3480#0', 'dm_swr_p_sequence-9951#1'],
    # C: the value is found; choosing its form, or re-confirming a settled answer
    'C': ['dm_div_big-1505#0', 'dm_div_big-1505#1', 'stdmath-9b21d63d#0', 'dm_div_big-2783#1', 'stdmath-5dd40b1a#1',
          'dm_div_big-3491#0', 'dm_div_big-9977#0', 'dm_swr_p_sequence-4125#0', 'dm_swr_p_sequence-4125#1',
          'dm_div_big-1395#1', 'dm_div_big-4138#0'],
    # D: searching for a common factor by trial division
    'D': ['dm_div_big-5070#0', 'dm_div_big-1395#0', 'dm_div_big-2783#0', 'dm_div_big-4138#1', 'dm_mixed_longer-9951#0'],
    # E: a run of one token or one empty line
    'E': ['dm_simplify_surd-2773#1', 'stdmath-6fe51888#0', 'stdmath-5e2eb2d9#0'],
    # F: reasoning stuck on a hard problem (a contradiction it cannot resolve)
    'F': ['stdmath-0b7036eb#0', 'stdmath-245aadb1#0', 'stdmath-370853d3#0', 'stdmath-370853d3#1', 'stdmath-97b18f58#0',
          'stdmath-97b18f58#1'],
}
rows = json.load(open(sys.argv[1], encoding='utf-8'))
rec = OrderedDict()
for r in rows:
    rec.setdefault(r['key'], r)
lab = {k: c for c, ks in L.items() for k in ks}
assert len(lab) == sum(map(len, L.values())), 'a key labelled twice'
assert set(lab) == set(rec), (set(rec) - set(lab), set(lab) - set(rec))


def ok(r, p):
    return bool(r[p + '_rule'] or r[p + '_api'] == 'same')


def outcome(r):
    g, c = ok(r, 'gen'), ok(r, 'ctrl')
    return {(True, False): 'helped', (True, True): 'both right', (False, True): 'hurt', (False, False): 'both wrong'}[(g, c)]


t = Counter((lab[k], outcome(r)) for k, r in rec.items())
cols = ['helped', 'both right', 'hurt', 'both wrong']
print('| Class | ' + ' | '.join(cols) + ' | all |')
for c in 'ABCDEF':
    print(f'| {c} | ' + ' | '.join(str(t[(c, o)]) for o in cols) + f' | {sum(t[(c, o)] for o in cols)} |')
print('| all | ' + ' | '.join(str(sum(t[(c, o)] for c in "ABCDEF")) for o in cols) + f' | {len(rec)} |')
print()
for c in 'ABCDEF':
    for k in L[c]:
        r = rec[k]
        kinds = '+'.join(x['kind'].split()[0] for x in rows if x['key'] == k)
        print(f"| `{k}` | {c} | {kinds} | {outcome(r)} | {r['ctrl_finish']} |")
