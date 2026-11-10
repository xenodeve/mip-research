"""tools/igc/tir4b.py and tir4b_train.py without a GPU: a scripted model drives the forced-block state machine through each path the
engine's rules take, and the training labels mask what the kernel printed. python tools/test_igc_tir4b.py"""
import os, sys, types, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.igc import tir4b
from tools.igc.tir4b_train import labels_of

SPECIAL = {'</think>': 5, '<|im_end|>': 6, '```python': 7, '```': 8, '</tool_call>': 9, '<tool_call>': 10, '<|im_start|>': 11, '\n\n': 271, '\n': 198}


class Tok:
    """Longest match over the specials, one id per other character."""
    def encode(self, text, add_special_tokens=False):
        out, i = [], 0
        while i < len(text):
            for s in sorted(SPECIAL, key=len, reverse=True):
                if text.startswith(s, i):
                    out.append(SPECIAL[s])
                    i += len(s)
                    break
            else:
                out.append(1000 + ord(text[i]))
                i += 1
        return out

    def decode(self, ids):
        back = {v: k for k, v in SPECIAL.items()}
        return ''.join(back[i] if i in back else chr(i - 1000) if i >= 1000 else '' for i in ids)

    def convert_tokens_to_ids(self, s):
        return SPECIAL[s]

    def __len__(self):
        return 1000 + 200


class Kernel:
    def __init__(self, replies):
        self.replies = list(replies)
        self.ran = []

    def alive(self):
        return True

    def run(self, code):
        self.ran.append(code)
        return self.replies.pop(0)

    def close(self):
        pass


def play(rollout, script, kernel):
    """The model writes `script` in order; each request takes what fits its limit and stops after a stop token, as vLLM does."""
    stream = TOK.encode(script)
    while True:
        ask = rollout.request()
        if not ask:
            return
        most, stops = ask
        new = []
        while stream and len(new) < most:
            new.append(stream.pop(0))
            if new[-1] in stops:
                break
        if not new:   # the script ran out: the model ends its turn
            new = [SPECIAL['<|im_end|>']]
        if rollout.take(new):
            rollout.run_pending(lambda: kernel)


class WideTok(Tok):
    """Tok plus whole-word tokens after its range, as a real vocabulary has (`//`, `/**`)."""
    def __init__(self, words):
        self.words = list(words)

    def decode(self, ids):
        base = 1000 + 200
        return ''.join(self.words[i - base] if i >= base else super(WideTok, self).decode([i]) for i in ids)

    def __len__(self):
        return 1000 + 200 + len(self.words)


TOK = Tok()
IDS = tir4b.Ids(TOK, [6])
ROW = dict(id='q1', suite='gsm', answer='42')


def rollout(**arm):
    return tir4b.Rollout(ROW, 0, [1, 2, 3], tir4b.Arm(**arm), IDS, TOK, seed=1)


def kinds(r):
    return [e['kind'] for e in r.events]


def _pad():
    """Thinking past CANDIDATE_MIN_TOKENS (one token a character here) with no cycle, no line back and no `=`."""
    import random
    rng = random.Random(7)
    return ''.join(rng.choice('qwertyuiopasdfghjklzxcvbnm ,') for _ in range(1600)) + '\n'


PAD = _pad()


class ForcedBlock(unittest.TestCase):
    def test_ids_are_the_tokenizers(self):
        self.assertEqual(IDS.think_close, 5)
        self.assertEqual(IDS.lines, {198, 271})
        self.assertTrue(IDS.same_lines)
        self.assertEqual(IDS.fence, {7, 8})

    def test_forced_block_runs_and_a_fitting_print_closes(self):
        r = rollout(force_after=10, execute=True, close_fit=True)
        k = Kernel([(True, '42')])
        play(r, 'aaaaaaaaaaaaa rest of line\nprint(6*7)\n```The answer is 42.<|im_end|>', k)
        self.assertEqual(k.ran, ['print(6*7)\n'])
        self.assertEqual(kinds(r), ['forced open', 'output', 'close after a fitting result'])
        text = r.text()
        self.assertEqual(text, 'aaaaaaaaaaaaa rest of line\n' + tir4b.OPEN_LINE + 'print(6*7)\n```\n```output\n42\n```\n\n</think>\n\nThe answer is 42.<|im_end|>')
        self.assertEqual(r.phase, 'answer')
        self.assertEqual(r.finish, 'stop')
        rec = r.record(lambda row, t: '42' in t.rsplit('</think>', 1)[-1])
        self.assertTrue(rec['right'])
        self.assertEqual((rec['runs'], rec['runs_forced'], rec['runs_by_model']), (1, 1, 0))
        self.assertEqual(rec['injected'] + rec['tokens'], len(r.out))
        out = [s for s in r.spans if s[2] == 'output'][0]
        self.assertEqual(TOK.decode(r.out[out[0]:out[1]]), '\n```output\n42\n```\n\n')

    def test_nothing_printed_keeps_thinking(self):
        r = rollout(force_after=4, execute=True, close_fit=True)
        k = Kernel([(True, '')])
        play(r, 'abcdef\nx = 1\n```on by hand</think>\n\n42<|im_end|>', k)
        self.assertIn(tir4b.NOTHING, r.text())
        self.assertIn('on by hand</think>', r.text())
        self.assertEqual(kinds(r), ['forced open', 'output'])
        self.assertFalse(r.events[-1]['fits'])

    def test_an_error_keeps_thinking(self):
        r = rollout(force_after=4, execute=True, close_fit=True)
        play(r, 'abcdef\n1/0\n```fix</think>\n\n42<|im_end|>', Kernel([(False, 'ZeroDivisionError')]))
        self.assertEqual(kinds(r), ['forced open', 'output'])
        self.assertIn('fix</think>', r.text())

    def test_force_opens_once(self):
        r = rollout(force_after=4, execute=True)
        play(r, 'abcdef\nprint(1)\n```' + 'b' * 40 + '\n' + 'c' * 40 + '</think>\n\n1<|im_end|>', Kernel([(True, '1')]))
        self.assertEqual(kinds(r).count('forced open'), 1)

    def test_model_opens_its_own_block(self):
        r = rollout(execute=True, close_fit=True)
        k = Kernel([(True, '7')])
        play(r, 'let me check ```python\nprint(7)\n```</think>\n\n7<|im_end|>', k)
        self.assertEqual(k.ran, ['print(7)\n'])
        self.assertEqual(kinds(r), ['model opened a block', 'output', 'close after a fitting result'])
        self.assertEqual(r.record(lambda *_: True)['runs_by_model'], 1)

    def test_bare_fence_then_python(self):
        r = rollout(execute=True)
        k = Kernel([(True, '3')])
        play(r, 'so ```python\nprint(3)\n```then</think>\n\n3<|im_end|>'.replace('```python', '```' + 'python'), k)
        self.assertEqual(k.ran, ['print(3)\n'])

    def test_bare_fence_not_python_goes_on(self):
        r = rollout(execute=True)
        play(r, 'a ```text``` b</think>\n\nx<|im_end|>', Kernel([]))
        self.assertEqual(r.finish, 'stop')
        self.assertEqual(r.runs, 0)

    def test_plain_arm_runs_nothing(self):
        r = rollout()
        play(r, 'a ```python\nprint(1)\n``` b</think>\n\nx<|im_end|>', Kernel([]))
        self.assertEqual((r.runs, r.events, r.spans), (0, [], []))
        self.assertEqual(r.text(), 'a ```python\nprint(1)\n``` b</think>\n\nx<|im_end|>')

    def test_a_loop_stops_a_plain_arm_and_closes_a_sandbox_arm(self):
        plain = rollout()
        play(plain, 'ab' * 700 + '</think>\n\nx<|im_end|>', Kernel([]))
        self.assertEqual(plain.finish, 'stuck')
        self.assertEqual(len(plain.out), tir4b.CHUNK)   # read between chunks, as the bench reads every 64 tokens
        closed = rollout(execute=True, loop_close=True)
        play(closed, 'ab' * 700, Kernel([]))
        self.assertIn('loop close', kinds(closed))

    def test_end_of_turn_in_thinking_after_a_run_closes(self):
        r = rollout(execute=True)
        play(r, '```python\nprint(2)\n```hm<|im_end|>2<|im_end|>', Kernel([(True, '2')]))
        self.assertIn('end of turn in thinking, closed it', kinds(r))
        self.assertIn('hm\n</think>\n\n2', r.text())

    def test_repeated_program_closes(self):
        r = rollout(execute=True, close_repeat=2)
        k = Kernel([(False, 'e')])
        play(r, '```python\nf()\n```x ```python\nf()\n```y</think>\n\nz<|im_end|>', k)
        self.assertEqual(len(k.ran), 1)
        self.assertIn('repeated program close', kinds(r))

    def test_output_is_cut_like_the_engine(self):
        self.assertEqual(len(tir4b.shown_text(True, 'x' * 5000).encode()), tir4b.SHOWN)
        self.assertEqual(tir4b.shown_text(True, '  '), tir4b.NOTHING)
        self.assertEqual(tir4b.shown_text(False, ''), '')


class InlineCalc(unittest.TestCase):
    def test_value_of_the_expression_before_equals(self):
        v = tir4b.inline_value
        self.assertEqual(v('so 3 \\times 4 \\times 7 '), '84')
        self.assertEqual(v('then 14641 - 2662 - 4235 + 352 + 324 '), '8420')   # a sum the 4B got wrong by hand (4d822b73)
        self.assertEqual(v('$0.1 + 0.2 '), '0.3')                               # exact, not a float, and in the model's own decimals
        self.assertEqual(v('0.9 \\times 5 '), '4.5')                             # dm 6890: 9/2 here broke the model's decimal sum into a loop
        self.assertEqual(v('7.95922042 \\times 5 '), '39.7961021')
        self.assertEqual(v('1/3 + 1 '), '4/3')                                  # no terminating decimal: the reduced fraction
        self.assertEqual(v('0.5 - 0.75 '), '-0.25')                             # a negative decimal keeps its sign
        # long division by hand (dm div_big 5866, 6311, 8989): `21 \div 19 =` asks for a quotient digit, and 21/19 back, then 21/19 again for
        # `21/19 =`, looped until stuck where base finished the division and was right
        self.assertIsNone(v('21 \\div 19 '))                     # an integer quotient that does not come out: the model's own long division
        self.assertIsNone(v('15/13 '))                           # the value spelled as the expression: nothing new, and it loops
        self.assertIsNone(v('\\frac{-2142}{-7037} '))
        self.assertEqual(v('84 \\div 7 '), '12')                 # a division that comes out is still given
        self.assertEqual(v('1 \\div 8 '), '0.125')
        self.assertEqual(v('\\frac{3}{4} + 1/4 '), '1')
        self.assertEqual(v('(140 + 56) / 28 '), '7')
        self.assertEqual(v('2^{10} '), '1024')

    def test_not_arithmetic_is_left_to_the_model(self):
        v = tir4b.inline_value
        self.assertIsNone(v('2x + 3 '))              # algebra
        self.assertIsNone(v('so x '))                # an assignment
        self.assertIsNone(v('a3 + 4 '))              # the tail of a word
        self.assertIsNone(v('9^{999} '))             # too large to compute in a step
        self.assertIsNone(v('\\sqrt{8} + 1 '))       # not rational
        self.assertEqual(v('(3 + 4 '), '7')          # inside a bracket the text opened: `(3 + 4 = 7)` (tools.igc.mathexpr)
        self.assertIsNone(v('17 '))                  # no operator
        self.assertIsNone(v('1.2.3 + 1 '))           # a section number: F('1.2').F('3') killed the whole runner (gen-mt5)
        self.assertEqual(v('(1)(2) + 3 '), '5')      # implicit multiplication
        self.assertEqual(v('so 12,510,204 \\times 100 '), '1251020400')   # read whole: the old reader took 204 x 100 (dm 4617)
        self.assertEqual(v('so 511.276030 \\times 10 '), '5112.7603')   # never started inside a number (the old reader took 030 x 10)

    def test_the_rollout_puts_the_value_in(self):
        r = rollout(inline_calc=True)
        play(r, 'so 6 \\times 7 = and on</think>\n\n42<|im_end|>', Kernel([]))
        self.assertIn('6 \\times 7 = 42 (exact) and on', r.text())
        self.assertEqual(kinds(r), ['inline value'])
        self.assertEqual((r.events[0]['expr'], r.events[0]['value']), ('6 \\times 7', '42'))
        self.assertEqual(r.record(lambda *_: True)['inline_values'], 1)
        out = [s for s in r.spans if s[2] == 'inline value'][0]
        self.assertEqual(TOK.decode(r.out[out[0]:out[1]]), ' 42 (exact)')

    def test_plain_puts_in_only_the_number(self):
        # --inline-plain, as Qwenputer prints: ` (exact)` inside `$...$` left the model's math unclosed, and it copied the tag
        r = rollout(inline_calc=True, inline_plain=True)
        play(r, 'so $6 \\times 7 =$ and on</think>\n\n42<|im_end|>', Kernel([]))
        self.assertIn('$6 \\times 7 = 42$ and on', r.text())

    def test_off_by_default(self):
        r = rollout()
        play(r, 'so 6 \\times 7 = 41</think>\n\nx<|im_end|>', Kernel([]))
        self.assertEqual(r.events, [])

    def test_an_answer_with_no_thinking_gets_values_too(self):
        # --no-think, as Qwenputer measures arithmetic (enable_thinking=False): the reply starts as the answer
        from tools.igc.select_rows import right
        r = tir4b.Rollout(dict(id='a', suite='arith', answer='478128321'), 0, [1], tir4b.Arm(inline_calc=True), IDS, TOK, seed=1, thinking=False)
        self.assertEqual(r.phase, 'answer')
        play(r, '48213 \\times 9917 = so \\boxed{478128321}<|im_end|>', Kernel([]))
        self.assertIn('48213 \\times 9917 = 478128321 (exact) so', r.text())
        self.assertEqual(r.finish, 'stop')
        # the prompt closed the thinking, so the scorer, which reads after </think>, must see the whole reply
        self.assertTrue(r.record(right)['right'])

    def test_arith_rows_are_scored_by_their_box(self):
        from tools.igc.select_rows import right
        row = dict(id='a', suite='arith', answer='-3/7')
        self.assertTrue(right(row, '</think>\n\nso \\boxed{-3/7}'))
        self.assertFalse(right(row, '</think>\n\nso \\boxed{3/7}'))
        self.assertEqual(tir4b.prompt_text(dict(suite='arith', question='What is 2*3?')), 'What is 2*3?')

    def test_answer_close_asks_for_one_value_after_a_loop_in_the_answer(self):
        """#307 loop-lc: 3 of 15 closed loops looped again in the answer (`-5511170.918918...`); there the cue asks for one value."""
        r = rollout(answer_close=True)
        r.take(TOK.encode('x</think>'))   # a request ends at the close, as vLLM's stop does
        r.take(TOK.encode('\n\nso ' + 'ab' * 600))
        self.assertIsNone(r.finish)
        self.assertEqual(kinds(r), ['answer close'])
        self.assertTrue(r.text().endswith('The final answer is $\\boxed'))
        self.assertLess(r.text().count('ab'), 10)   # the cycle cut back to one round before the cue
        most, stops = r.request()
        self.assertLessEqual(most, tir4b.ANSWER_CLOSE_TOKENS)
        self.assertIn(198, stops)
        r.take(TOK.encode('{-3/7}$'))
        r.take(TOK.encode('\n'))
        self.assertEqual(r.finish, 'stop')
        from tools.igc.select_rows import right
        self.assertTrue(right(dict(id='a', suite='arith', answer='-3/7'), r.text()))

    def test_answer_close_cue_ends_where_the_model_own_text_has_a_token_end(self):
        """Qwen writes `{-` as one token, so a cue that ended with `{` left only the unseen `{` + `-` for a negative value: 22 of 28
        gold-negative closes lost the minus (A100 bf16 and PC NVFP4 alike; 9977 boxed -70468 five times, then wrote `70468` after the
        cue). The cue's tokens must be a prefix of the tokens of the cue followed by any value, so the model chooses `{` or `{-` itself."""
        self.assertFalse(tir4b.ANSWER_CLOSE.endswith('{'))
        path = os.environ.get('IGC_QWEN_TOKENIZER')
        if not path:
            self.skipTest('IGC_QWEN_TOKENIZER (a Qwen tokenizer folder) not set')
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(path)
        cue = tok.encode(tir4b.ANSWER_CLOSE, add_special_tokens=False)
        for value in ('{-70468}$.', '{70468}$.', '{-\\frac{3}{4}}$', '{\\frac{3}{4}}$', '{(-1, 2)}$', '{x}$'):
            self.assertEqual(tok.encode(tir4b.ANSWER_CLOSE + value, add_special_tokens=False)[:len(cue)], cue, value)

    def test_answer_close_also_takes_a_loop_of_equations(self):
        """loop-lc surd-2201: `Let's check $19+1$. $19+1 = 20$.` looped in the answer; each request ended at `=`, so the loop was read on
        that path, which ended it `stuck` past the answer close."""
        r = rollout(inline_calc=True, inline_plain=True, answer_close=True)
        play(r, 'x</think>\n\n' + 'Check: $19+1 =$. Okay.\n' * 400, Kernel([]))
        self.assertIn('answer close', kinds(r))
        self.assertEqual(r.finish, 'stop')

    def test_after_the_cue_nothing_else_puts_a_value(self):
        r = rollout(inline_calc=True, inline_plain=True, answer_close=True)
        r.take(TOK.encode('x</think>'))
        r.take(TOK.encode('\n\nso ' + 'ab' * 600))
        puts = kinds(r).count('inline value')
        r.take(TOK.encode('{6 \\times 7 ='))
        r.take(TOK.encode(' 42}$\n'))
        self.assertEqual((kinds(r).count('inline value'), r.finish), (puts, 'stop'))
        self.assertTrue(r.text().endswith('The final answer is $\\boxed{6 \\times 7 = 42}$\n'))

    def test_loop_value_puts_the_questions_computation_before_the_close(self):
        """#307: div_big-2208 divided by hand without end; the question's own fraction, exact, goes in before the loop is closed."""
        row = dict(ROW, id='d', suite='arith', answer='-6681/10943', question='Calculate -6681 divided by 10943.')
        r = tir4b.Rollout(row, 0, [1, 2, 3], tir4b.Arm(loop_close=True, loop_value=True), IDS, TOK, seed=1)
        play(r, 'ab' * 700, Kernel([]))
        self.assertEqual(kinds(r)[:2], ['loop value', 'loop close'])
        self.assertIn('Computed exactly: $', r.text())
        self.assertIn('= -6681/10943$\n\n</think>', r.text())

    def test_loop_value_once_and_only_with_a_computation(self):
        row = dict(ROW, id='w', suite='gsm', answer='3', question='Tom has three apples. How many?')
        r = tir4b.Rollout(row, 0, [1, 2, 3], tir4b.Arm(loop_close=True, loop_value=True), IDS, TOK, seed=1)
        play(r, 'ab' * 700, Kernel([]))
        self.assertNotIn('loop value', kinds(r))   # a word problem states no computation software reads
        row = dict(ROW, id='d', suite='arith', answer='-6681/10943', question='Calculate -6681 divided by 10943.')
        r = tir4b.Rollout(row, 0, [1, 2, 3], tir4b.Arm(loop_close=True, answer_close=True, loop_value=True), IDS, TOK, seed=1)
        play(r, 'ab' * 700 + 'cd' * 1500, Kernel([]))
        self.assertEqual(kinds(r).count('loop value'), 1)
        self.assertIn('answer close', kinds(r))

    def test_early_cycle_reads_six_rounds_over_500_tokens(self):
        """#307 detector replay V7: an exact cycle of 6 rounds (3 past 512 tokens) that spans 500+ tokens; no fire on 915 right answers."""
        unit = list(range(100, 200))   # one round of 100 tokens
        self.assertTrue(tir4b.early_cycle(unit * 6))
        self.assertFalse(tir4b.early_cycle(unit * 5))
        self.assertFalse(tir4b.early_cycle(list(range(50, 100)) * 9))   # 9 rounds of 50: 450 tokens, short of the span
        self.assertTrue(tir4b.early_cycle(list(range(50, 100)) * 10))
        self.assertTrue(tir4b.early_cycle(list(range(1000, 1600)) * 3))   # past 512 three rounds are enough
        self.assertFalse(tir4b.early_cycle(list(range(1000, 1600)) * 2))
        self.assertFalse(tir4b.early_cycle([7] + unit * 6 + [8]))   # the tail is not the cycle

    def test_early_cycle_closes_a_loop_the_bench_check_lets_run(self):
        text = 'Wait, let me recheck the sum of the two parts again, the first part and the second part, carefully now.\n' * 7
        r = rollout(loop_close=True, early_cycle=True)
        play(r, text + 'more', Kernel([]))
        self.assertEqual(kinds(r), ['loop close'])
        self.assertEqual(r.early_cycles, 1)
        thinking = r.text().split('</think>')[0]
        self.assertEqual(thinking.count('carefully now.'), 6)   # cut at the sixth line end, inside the request that wrote the seventh
        self.assertNotIn('more', thinking)
        r = rollout(loop_close=True)
        play(r, text + 'more', Kernel([]))
        self.assertNotIn('loop close', kinds(r))   # 20 rounds: the bench check waits

    def test_early_cycle_on_the_equals_path_needs_no_full_chunk(self):
        """Requests that end at `=` are read for a loop once per CHUNK; the early check reads every one."""
        text = 'Let me write it out once more so that we can see where the value comes from, x =' + ' x.\n'
        r = rollout(inline_calc=True, inline_plain=True, loop_close=True, early_cycle=True)
        play(r, text * 8, Kernel([]))
        self.assertIn('loop close', kinds(r))
        self.assertLess(r.model_tokens(), len(TOK.encode(text * 8)))

    def test_early_cycle_leaves_the_answer_to_the_answer_checks(self):
        text = 'Wait, let me recheck the sum of the two parts again, the first part and the second part, carefully now.\n' * 7
        r = rollout(loop_close=True, early_cycle=True)
        play(r, 'x</think>\n\n' + text + 'done<|im_end|>', Kernel([]))
        self.assertEqual((r.finish, kinds(r)), ('stop', []))

    def test_shape_rounds_reads_numbers_of_any_length_as_one(self):
        """#307 round 7: `(Digit 5243: 7)` digit-by-digit division ran 210k tokens; its numbers change length, so no digit-masked cycle."""
        def digits(n):
            return [15 + int(ch) for ch in str(n)]
        line = lambda n, d: [40, 41] + digits(n) + [42] + digits(d) + [198]   # noqa: E731
        t = [7] + sum((line(5000 + 37 * k, k % 10) for k in range(300)), [])
        t += line(99999, 3)   # a five-digit number among four-digit ones: still the same round
        self.assertEqual(tir4b.shape_rounds(t), 301)
        self.assertLess(tir4b.shape_rounds(t + [50, 51]), 2)

    def test_cover_reads_through_the_odd_rounds_an_exact_cycle_breaks_on(self):
        """#307 round 8: long division where each quotient digit 0 made a shorter round ran to the context end, shape rounds never past 46."""
        def digits(n):
            return [15 + int(ch) for ch in str(n)]
        full = lambda n: [40] + digits(n) + [41, 42] + digits(n * 7) + [198, 43] + digits(n) + [198]   # noqa: E731
        zero = lambda n: [44] + digits(n) + [198, 45, 15, 198]   # noqa: E731
        import random
        rng = random.Random(5)
        t = [7, 8, 9]
        for k in range(3000):   # a 0 digit comes at no fixed period, as in the division
            t += zero(k) if rng.random() < 0.3 else full(k)
        self.assertLess(tir4b.shape_rounds(t), 10)
        c = tir4b.Cover(frozenset(range(15, 25)), 4096)
        self.assertGreaterEqual(c.feed(t, 0), tir4b.COVER_SHARE)
        self.assertGreaterEqual(c.feed(t + [50 + i for i in range(30)], 0), tir4b.COVER_SHARE)   # one odd stretch does not restart it
        self.assertEqual(c.feed(t[:100], 0), 0.0)   # a shorter window starts afresh, and a window not yet full reads 0

    def test_cover_cap_closes_a_loop_and_leaves_plain_reasoning(self):
        text = ''.join(f'Step {n}: {n * 10943} / 10943 gives {n}, carry {n % 7}, digit 0.\n' for n in range(1, 900))
        r = rollout(loop_close=True, guard_progress=True, cover_cap=4096)
        play(r, text, Kernel([]))
        self.assertEqual(kinds(r)[:1], ['loop close'])
        self.assertGreaterEqual(r.cover_loops, 1)
        import random
        rng = random.Random(3)
        vocab = 'the a sum of root prime then so we check each part side angle area half more less next term value set map key row'.split()
        words = ' '.join(rng.choice(vocab) for _ in range(9000))   # 12-grams of random words almost never meet twice
        r = rollout(loop_close=True, guard_progress=True, cover_cap=4096)
        play(r, words, Kernel([]))
        self.assertEqual(r.cover_loops, 0)

    def test_shape_cap_closes_a_counter_loop_the_progress_guard_lets_run(self):
        """#307 round 7: `check ... base 5815. No. Okay.` ran 262k tokens: every round new, so --guard-progress let it go."""
        text = ''.join(f'Check base {n}. No.\n' for n in range(1000, 2400))
        r = rollout(loop_close=True, guard_progress=True, shape_cap=1024)
        play(r, text, Kernel([]))
        self.assertEqual(kinds(r), ['loop close'])
        self.assertEqual(r.shape_loops, 1)
        r = rollout(loop_close=True, guard_progress=True)
        play(r, text, Kernel([]))
        self.assertNotIn('loop close', kinds(r))

    def test_answer_close_ends_a_value_that_keeps_going(self):
        r = rollout(answer_close=True)
        play(r, 'x</think>\n\nso ' + 'ab' * 2000, Kernel([]))
        self.assertEqual(r.finish, 'stop')
        self.assertEqual(len(r.out) - r.answer_closed_at, tir4b.ANSWER_CLOSE_TOKENS)

    def test_answer_close_leaves_a_thinking_loop_and_a_plain_answer_alone(self):
        r = rollout(answer_close=True)
        play(r, 'ab' * 700, Kernel([]))
        self.assertEqual(r.finish, 'stuck')   # the thinking is loop-close's, not this
        r = rollout(answer_close=True)
        play(r, 'x</think>\n\nso \\boxed{2}<|im_end|>', Kernel([]))
        self.assertEqual((r.finish, kinds(r)), ('stop', []))

    def test_a_loop_of_equations_is_stuck(self):
        # every request ends at an `=`, never at a chunk: the loop check must still come round, or it runs to the end of the context
        r = rollout(inline_calc=True)
        play(r, 'so 1 + 1 = ' * 3000, Kernel([]))
        self.assertEqual(r.finish, 'stuck')
        self.assertLess(len(r.out), 3 * tir4b.CHUNK)


CALL = '<tool_call>\n<function=python>\n<parameter=code>\nprint(6*7)\n</parameter>\n</function>\n</tool_call>'


class NativeTool(unittest.TestCase):
    def test_a_call_in_the_reply_gets_a_tool_turn_and_thinks_again(self):
        r = rollout(native_tool=True)
        k = Kernel([(True, '42')])
        play(r, 'plan</think>\n\n' + CALL + 'it printed 42</think>\n\n42<|im_end|>', k)
        self.assertEqual(k.ran, ['print(6*7)'])
        self.assertEqual(r.text(), 'plan</think>\n\n' + CALL + '<|im_end|>\n<|im_start|>user\n<tool_response>\n42\n</tool_response><|im_end|>\n'
                         '<|im_start|>assistant\n<think>\nit printed 42</think>\n\n42<|im_end|>')
        self.assertEqual(r.finish, 'stop')
        rec = r.record(lambda *_: True)
        self.assertEqual((rec['runs'], rec['runs_by_tool']), (1, 1))
        out = [e for e in r.events if e['kind'] == 'output'][0]
        self.assertEqual((out['by'], out['shown']), ('tool', '42'))

    def test_a_call_inside_thinking_is_answered_inside_thinking(self):
        r = rollout(native_tool=True)
        k = Kernel([(True, '42')])
        play(r, 'hm ' + CALL + 'so 42</think>\n\n42<|im_end|>', k)
        self.assertIn(CALL + '\n<tool_response>\n42\n</tool_response>\nso 42</think>', r.text())
        self.assertEqual([e['by'] for e in r.events if e['kind'] == 'output'], ['tool in thinking'])

    def test_a_call_with_no_code_is_told_so(self):
        r = rollout(native_tool=True)
        play(r, 'x</think>\n\n<tool_call>\n<function=python>\n</function>\n</tool_call>ok</think>\n\n1<|im_end|>', Kernel([]))
        self.assertIn(tir4b.NO_CODE, r.text())
        self.assertEqual(r.finish, 'stop')

    def test_the_prompt_offers_the_tool(self):
        self.assertEqual(tir4b.PYTHON_TOOL['function']['name'], 'python')
        self.assertEqual(tir4b.PYTHON_TOOL['function']['parameters']['required'], ['code'])


class HintCalc(unittest.TestCase):
    def test_a_tedious_hand_calculation_becomes_a_program(self):
        r = rollout(hint_calc=True)
        k = Kernel([(True, '56088')])
        play(r, 'think a bit\nso 123 \\times 456 =print(123*456)\n```then done</think>\n\n56088<|im_end|>', k)
        self.assertEqual(k.ran, ['print(123*456)\n'])
        self.assertEqual(r.text(), 'think a bit\n' + tir4b.HINT_OPEN + 'print(123*456)\n```' + tir4b.output_block('56088') + tir4b.TRUST
                         + 'then done</think>\n\n56088<|im_end|>')
        hint = [e for e in r.events if e['kind'] == 'hint open'][0]
        self.assertEqual((hint['expr'], hint['cut_tokens']), ('123 \\times 456', len(TOK.encode('so 123 \\times 456 ='))))
        self.assertEqual([e['by'] for e in r.events if e['kind'] == 'output'], ['hint'])
        self.assertEqual(r.phase, 'answer')

    def test_small_arithmetic_is_left_alone(self):
        r = rollout(hint_calc=True)
        play(r, 'so 1 + 1 = 2</think>\n\n2<|im_end|>', Kernel([]))
        self.assertEqual(r.events, [])

    def test_hints_are_spaced(self):
        r = rollout(hint_calc=True)
        k = Kernel([(True, '56088')])
        play(r, '\n123 \\times 456 =print(1)\n```and 789 \\times 321 = 253269</think>\n\n1<|im_end|>', k)
        self.assertEqual(len(k.ran), 1)
        self.assertIn('789 \\times 321 = 253269', r.text())

    def test_a_hint_that_gets_the_same_program_runs_it_again(self):
        # the repeat close is for a model that reruns its own program; a second hint at the same sum asks for it (gen-mt5: most answers closed)
        r = rollout(hint_calc=True, close_repeat=2)
        k = Kernel([(True, '56088'), (True, '56088')])
        gap = ' '.join(str(i) for i in range(300)) + '\n'   # past HINT_GAP tokens with no cycle, so the loop check stays quiet
        play(r, '\n123 \\times 456 =print(123*456)\n```' + gap + '123 \\times 456 =print(123*456)\n```done</think>\n\n56088<|im_end|>', k)
        self.assertEqual(len(k.ran), 2)
        self.assertNotIn('repeated program close', [e['kind'] for e in r.events])

    def test_the_trust_line_is_the_original_by_default_and_can_be_left_out_or_made_positive(self):
        # lb1: the original line, put after every hinted program, did not stop a 4B from re-checking a printed -530 for 12,000 tokens
        hinted = '\n123 \\times 456 =print(123*456)\n```hm</think>\n\n56088<|im_end|>'
        r = rollout(hint_calc=True)
        play(r, hinted, Kernel([(True, '56088')]))
        self.assertIn(tir4b.TRUST, r.text())
        r = rollout(hint_calc=True, trust='none')
        play(r, hinted, Kernel([(True, '56088')]))
        self.assertNotIn(tir4b.TRUST, r.text())
        self.assertNotIn('trust hint', [e['kind'] for e in r.events])
        self.assertIn('output', [e['kind'] for e in r.events])   # the program still runs and its output is still shown
        r = rollout(hint_calc=True, trust='positive')
        play(r, hinted, Kernel([(True, '56088')]))
        self.assertIn(tir4b.TRUST_LINES['positive'], r.text())
        self.assertNotIn(tir4b.TRUST, r.text())

    def test_no_trust_line_after_an_error(self):
        r = rollout(hint_calc=True)
        play(r, '\n123 \\times 456 =1/0\n```hm</think>\n\n1<|im_end|>', Kernel([(False, 'ZeroDivisionError')]))
        self.assertNotIn(tir4b.TRUST, r.text())


class AutoCall(unittest.TestCase):
    def test_a_function_defined_and_never_called_is_called(self):
        code = 'def find_n():\n    for n in range(5):\n        print(n)\n'
        self.assertEqual(tir4b.with_call(code), (code + '\n_r = find_n()\nif _r is not None:\n    print(_r)\n', 'find_n'))

    def test_left_alone(self):
        for code in ('def f():\n    return 1\nprint(f())\n',          # called
                     'def f(x):\n    return x\n',                      # needs an argument
                     'x = 1\nprint(x)\n',                              # no function
                     'def f(:\n'):                                     # does not parse: the run shows the error
            self.assertEqual(tir4b.with_call(code), (code, None))

    def test_the_last_uncalled_one(self):
        code = 'def a():\n    return 1\ndef b():\n    return a() + 1\n'
        self.assertEqual(tir4b.with_call(code)[1], 'b')

    def test_the_run_records_it(self):
        r = rollout(hint_calc=True)
        k = Kernel([(True, '37')])
        play(r, '\n123 \\times 456 =def f():\n    return 37\n```so</think>\n\n37<|im_end|>', k)
        self.assertTrue(k.ran[0].endswith('_r = f()\nif _r is not None:\n    print(_r)\n'))
        out = [e for e in r.events if e['kind'] == 'output'][0]
        self.assertEqual((out['auto_call'], out['code']), ('f', 'def f():\n    return 37\n'))


class LoopHint(unittest.TestCase):
    def test_the_loop_is_cut_to_one_round_and_a_program_opens(self):
        r = rollout(loop_hint=True)
        k = Kernel([(True, 'True')])
        # 4 + 68 x 15 = 1024 tokens: the loop is read at the end of the first chunk, and what the model writes next answers the hint
        play(r, 'set\n' + 'Wait, is it 5?\n' * 68 + 'print(5 == 5)\n```so 5</think>\n\n5<|im_end|>', k)
        text = r.text()
        self.assertTrue(text.startswith('set\nWait, is it 5?\n' + tir4b.LOOP_HINT))
        self.assertEqual(text.count('Wait, is it 5?'), 1)
        self.assertIn(tir4b.LOOP_HINT + 'print(5 == 5)\n```' + tir4b.output_block('True'), text)
        self.assertEqual(r.finish, 'stop')
        hint = [e for e in r.events if e['kind'] == 'loop hint'][0]
        self.assertGreater(hint['cut_tokens'], 1000)

    def test_a_loop_after_the_last_hint_is_stuck(self):
        r = rollout(loop_hint=True)
        r.loop_hints = tir4b.LOOP_HINTS
        play(r, 'Wait, is it 5?\n' * 400, Kernel([]))
        self.assertEqual([e['kind'] for e in r.events], [])
        self.assertEqual(r.finish, 'stuck')

    def test_loop_start_finds_the_first_round(self):
        tokens = [1, 2, 3] + [7, 8, 9] * 30
        self.assertEqual(tir4b.loop_start(tokens), 6)   # keeps [7, 8, 9] once


class WholeHint(unittest.TestCase):
    """--whole-hint: the software computes the whole step, not the pieces the model would cut it into (dm div_big 5866: `21 \\div 19 =`
    of a long division got 21/19 and looped; the whole -2119381524 / 19 is one value)."""
    def test_the_whole_value_keeps_its_sign_and_its_fraction(self):
        v = tir4b.inline_value
        self.assertEqual(v('$-2119381524 \\div 19 ', whole=True), '-111546396')
        self.assertEqual(v('$-2142 \\div -7037 ', whole=True), '2142/7037')   # dm 8989's gold: an uneven quotient is the answer here
        self.assertEqual(v('so -3 \\times 0.2 ', whole=True), '-0.6')
        self.assertIsNone(v('$-2142 \\div -7037 '))                           # not asked for: still the model's long division

    def test_the_hint_opens_the_answer_and_its_first_value_is_whole(self):
        r = tir4b.Rollout(dict(id='a', suite='arith', answer='-111546396'), 0, [1], tir4b.Arm(inline_calc=True, inline_plain=True, whole_hint=True),
                          IDS, TOK, seed=1, thinking=False)
        play(r, '$-2119381524 \\div 19 =$ so \\boxed{-111546396}<|im_end|>', Kernel([]))
        self.assertTrue(r.text().startswith(tir4b.WHOLE_HINT + '$-2119381524 \\div 19 = -111546396\n$ so'))
        self.assertEqual([e['kind'] for e in r.events], ['whole hint', 'inline value'])
        self.assertTrue(r.events[1]['whole'])
        self.assertTrue(r.record(__import__('tools.igc.select_rows', fromlist=['right']).right)['right'])

    def test_the_plain_hint_and_a_nested_expression(self):
        # dm mul_div_multiple_longer 1020: the question's own nested form, which v1 rewrote as \frac inside \frac and never computed
        r = tir4b.Rollout(dict(id='a', suite='arith', answer='319'), 0, [1], tir4b.Arm(inline_calc=True, inline_plain=True, whole_hint=True,
                          whole_plain=True), IDS, TOK, seed=1, thinking=False)
        play(r, '(-1*-638*(-7)/(-7))/(((-62)/(-279))/(((-170)/3060)/(1/(-2)))) = so \\boxed{319}<|im_end|>', Kernel([]))
        self.assertTrue(r.text().startswith(tir4b.WHOLE_HINT_PLAIN + '(-1*-638*(-7)/(-7))/(((-62)/(-279))/(((-170)/3060)/(1/(-2)))) = 319\n so'))

    def test_the_whole_value_ends_its_line(self):
        # pc4-whole dm div_big 7815: `= 10097` left open, and the model wrote `.000...` after it until the guard stopped the answer
        r = tir4b.Rollout(dict(id='a', suite='arith', answer='10097'), 0, [1], tir4b.Arm(inline_calc=True, inline_plain=True, whole_hint=True,
                          whole_plain=True), IDS, TOK, seed=1, thinking=False)
        play(r, '793836237 / 78621 =\\boxed{10097}<|im_end|>', Kernel([]))
        self.assertIn('793836237 / 78621 = 10097\n\\boxed{10097}', r.text())

    def whole_arm(self, answer, **more):
        return tir4b.Rollout(dict(id='a', suite='arith', answer=answer), 0, [1], tir4b.Arm(inline_calc=True, inline_plain=True,
                             whole_hint=True, whole_plain=True, hand_off=True, **more), IDS, TOK, seed=1, thinking=False)

    def test_a_line_that_only_restates_the_value_closes_the_answer_with_it(self):
        # pc4-whole dm mul_div 1618: `= -18` written again on every line, never boxed; pc6 led it on and it answered 18 (P('-') 0.07).
        # The model's own restated value closes the answer, as it wrote it (developer, 2026-11-04)
        r = self.whole_arm('-18')
        play(r, '(360/27)/4*(-5)/((-20)/4)*-1*3*(-12)/180*-27 =\n= -18\nSo the answer is 18. \\boxed{18}<|im_end|>', Kernel([]))
        self.assertTrue(r.text().endswith('= -18\n\\boxed{-18}<|im_end|>'), r.text()[-60:])
        self.assertEqual(r.finish, 'stop')
        self.assertEqual([e['kind'] for e in r.events].count('hand-off close'), 1)
        self.assertTrue(r.record(__import__('tools.igc.select_rows', fromlist=['right']).right)['right'])

    def test_a_decimal_expansion_of_the_value_is_cut(self):
        # pc4-whole dm div_big 2289: `= -37/18329827` then the model wrote `= -0.0000020186299999...` to the length cap
        r = self.whole_arm('-37/18329827')
        play(r, '333 / -164968443 =\n= -0.00000201862999999999999999999999999999999999\\boxed{-37/18329827}<|im_end|>', Kernel([]))
        self.assertTrue(r.text().endswith('= -37/18329827\n\n\\boxed{-37/18329827}<|im_end|>'), r.text()[-80:])   # closed with the fraction
        self.assertNotIn('0.0000020186299999999', r.text())

    def test_a_rounded_decimal_of_a_given_fraction_closes_with_the_fraction(self):
        # pc8 4104: 1539439/2572 given, then `= 598.5377138413301`, boxed and scored wrong against the fraction (developer, 2026-11-04)
        r = self.whole_arm('1539439/2572')
        play(r, '-1539439 / -2572 =\n= 598.5377138413301\n\\boxed{598.5377138413301}<|im_end|>', Kernel([]))
        self.assertTrue(r.text().endswith('\\boxed{1539439/2572}<|im_end|>'), r.text()[-80:])
        self.assertNotIn('598.5377', r.text())
        self.assertEqual([e['kind'] for e in r.events].count('hand-off close exact'), 1)

    def test_the_question_s_own_computation_replaces_a_wrong_copy(self):
        # pc8 5180: 'What is 3534538042068 less than 1.712?' copied as 3534538042068 - 1.712; the question reads 1.712 - 3534538042068
        row = dict(id='a', suite='arith', answer='-3534538042066.288', question='What is 3534538042068 less than 1.712?')
        r = tir4b.Rollout(row, 0, [1], tir4b.Arm(inline_calc=True, inline_plain=True, whole_hint=True, whole_plain=True, hand_off=True,
                          question_value=True), IDS, TOK, seed=1, thinking=False)
        play(r, '3534538042068 - 1.712 =\\boxed{-3534538042066.288}<|im_end|>', Kernel([]))
        self.assertIn('3534538042068 - 1.712 = -3534538042066.288\n', r.text())
        ev = [e for e in r.events if e['kind'] == 'question value']
        self.assertEqual(ev[0]['copied'], '3534538042066.288')

    def test_a_new_step_after_the_value_is_left_alone(self):
        r = self.whole_arm('95')
        play(r, '190 / 2 =\nso the total is 95 apples\n\\boxed{95}<|im_end|>', Kernel([]))
        self.assertNotIn(tir4b.HAND_OFF, r.text())

    def test_a_whole_line_with_no_equals_gets_its_value(self):
        # pc-whole dm mul_div 3350: the model wrote the whole expression on its own line, with no `=`, then went on in pieces
        r = tir4b.Rollout(dict(id='a', suite='arith', answer='228'), 0, [1], tir4b.Arm(inline_calc=True, inline_plain=True, whole_hint=True,
                          whole_plain=True), IDS, TOK, seed=1, thinking=False)
        play(r, '((( -1 ) / ( -15 )) / ( ( -1 ) / 2 ))\nso \\boxed{-2/15}<|im_end|>', Kernel([]))
        self.assertIn('((( -1 ) / ( -15 )) / ( ( -1 ) / 2 ))\n= -2/15\nso', r.text())
        self.assertTrue(r.events[1]['whole'])

    def test_later_pieces_follow_the_usual_rules(self):
        r = tir4b.Rollout(ROW, 0, [1], tir4b.Arm(inline_calc=True, inline_plain=True, whole_hint=True), IDS, TOK, seed=1, thinking=False)
        play(r, '$-6 \\div 4 =$ then 21 \\div 19 = x<|im_end|>', Kernel([]))
        self.assertIn('$-6 \\div 4 = -1.5\n$', r.text())
        self.assertIn('21 \\div 19 = x', r.text())   # after the whole value, an uneven piece is left alone again


class CheckHand(unittest.TestCase):
    """--check-hand: what the model already worked out by hand is checked, and a wrong value is answered with the right one."""
    def test_a_wrong_hand_value_gets_a_correction(self):
        r = rollout(check_hand=True)
        play(r, 'so 144 - 144 - 572 - 168 + 96 = -744.\nthen 2 + 2 = 4 and</think>\n\nx<|im_end|>', Kernel([]))
        self.assertIn('= -744.\n' + tir4b.CORRECTION.format(expr='144 - 144 - 572 - 168 + 96', right='-644', wrote='-744'), r.text())
        ev = [e for e in r.events if e['kind'] == 'hand check']
        self.assertEqual([(e['wrote'], e['value']) for e in ev], [('-744', '-644')])

    def test_right_values_rounding_and_rewrites_pass(self):
        r = rollout(check_hand=True)
        # e: a long division's quotient digit with its remainder (dm mul_big 4267 would have been "corrected" to 6.5 six times)
        play(r, 'a 9 \\cdot 21 + 7 = 189 + 7 = 196.\nb 151.45 - 51.47 = 99.98\nc 2 / 3 = 0.67\nd 1 + 1 = 2\ne 13 \\div 2 = 6 remainder 1\n'
                '</think>\n\nx<|im_end|>', Kernel([]))
        self.assertEqual([e for e in r.events if e['kind'] == 'hand check'], [])

    def test_a_minus_before_the_expression_is_left_alone(self):
        r = rollout(check_hand=True)   # dm 3578: `-(25 * 130) = -3250` read as (25 * 130) and "corrected" to 3250
        play(r, 'so -(25 * 130) = -3250\nok</think>\n\nx<|im_end|>', Kernel([]))
        self.assertEqual([e for e in r.events if e['kind'] == 'hand check'], [])

    def test_an_uneven_quotient_is_corrected_in_the_places_written(self):
        r = rollout(check_hand=True)   # dm div 7815-like: a decimal written for 5 / 6.744908131, wrong in its 4th place
        play(r, 'so 5 / 6.744908131 = 0.7419\nok</think>\n\nx<|im_end|>', Kernel([]))
        self.assertIn(tir4b.CORRECTION.format(expr='5 / 6.744908131', right='0.7413', wrote='0.7419'), r.text())

    def test_checked_at_the_end_of_its_line_only(self):
        # the value is only known once the model has moved on: a check at the `=` itself would see no value yet
        r = rollout(check_hand=True)
        play(r, '123 \\times 2 = 247 so\nok</think>\n\nx<|im_end|>', Kernel([]))
        self.assertIn('247 so\n' + tir4b.CORRECTION.format(expr='123 \\times 2', right='246', wrote='247'), r.text())


class HttpDrive(unittest.TestCase):
    """drive_http: llama-server's /completion in place of vLLM, so the 5060 Ti on Windows runs the same rules."""
    def test_string_stops_and_the_stop_word_put_back(self):
        scripts = {(1, 200): 'so 6 \\times 7 = and on</think>\n\n42<|im_end|>', (1, 201): 'plain</think>\n\nok<|im_end|>'}
        cursor = dict.fromkeys(scripts, 0)

        def post(body):   # llama-server's /completion: the text up to a stop word, the word given apart, eos at <|im_end|>
            key = tuple(body['prompt'][:2])
            rest = scripts[key][cursor[key]:]
            cut, word = len(rest), ''
            for s in body['stop'] + ['<|im_end|>']:
                j = rest.find(s)
                if 0 <= j < cut:
                    cut, word = j, s
            cursor[key] += cut + len(word)
            if word == '<|im_end|>':
                return dict(content=rest[:cut], stop_type='eos', stopping_word='')
            return dict(content=rest[:cut], stop_type='word' if word else 'limit', stopping_word=word)
        rs = [tir4b.Rollout(ROW, 0, [1, 200], tir4b.Arm(inline_calc=True), IDS, TOK, seed=1),
              tir4b.Rollout(ROW, 0, [1, 201], tir4b.Arm(), IDS, TOK, seed=1)]
        tir4b.drive_http(post, TOK, IDS, rs, lambda: Kernel([]), live_most=2)
        self.assertEqual(rs[0].text(), 'so 6 \\times 7 = 42 (exact) and on</think>\n\n42<|im_end|>')
        self.assertEqual(rs[1].text(), 'plain</think>\n\nok<|im_end|>')
        self.assertTrue(all(r.finish == 'stop' for r in rs))

    def test_the_ids_the_server_returns_are_taken_as_they_are(self):
        # llama-server stopped on '\n' inside the token '\n\n' and returns that token: the text is not encoded again
        def post(body):
            self.assertTrue(body['return_tokens'])
            return dict(content='ab', stop_type='word', stopping_word='\n', tokens=[1000 + ord('a'), 1000 + ord('b'), 271]) \
                if not post.done else dict(content='', stop_type='eos', stopping_word='', tokens=[6])
        post.done = False
        r = tir4b.Rollout(ROW, 0, [1], tir4b.Arm(check_hand=True), IDS, TOK, seed=1, thinking=False)

        def once(body):
            reply = post(body)
            post.done = True
            return reply
        tir4b.drive_http(once, TOK, IDS, [r], lambda: Kernel([]), live_most=1)
        self.assertEqual(r.out, [1000 + ord('a'), 1000 + ord('b'), 271, 6])   # 271 kept, and no second end of turn

    def test_the_logit_bias_goes_to_the_server(self):
        # --no-digit-join on the 5060 Ti: the digit bias after a put travels in the /completion body as llama-server's [[id, bias], ...]
        bodies = []

        def post(body):
            bodies.append(body)
            return dict(content='', stop_type='eos', stopping_word='', tokens=[6])
        r = tir4b.Rollout(ROW, 0, [1], tir4b.Arm(inline_calc=True, inline_plain=True, no_digit_join=True), IDS, TOK, seed=1, thinking=False)
        r.join_guard = True
        want = r.bias()
        self.assertTrue(want)
        tir4b.drive_http(post, TOK, IDS, [r], lambda: Kernel([]), live_most=1)
        self.assertEqual(sorted(map(tuple, bodies[0]['logit_bias'])), sorted(want.items()))
        self.assertEqual(bodies[0]['n_predict'], 1)   # one token carries the bias, as on vLLM

    def test_a_drain_starts_no_new_answer_and_lets_the_running_one_end(self):
        # `<out>/drain`: the answers in flight finish, no answer starts after it, and the run can --resume later
        asked = []

        def post(body):
            asked.append(body['prompt'][:2])
            return dict(content='', stop_type='eos', stopping_word='', tokens=[6])
        rs = [tir4b.Rollout(ROW, 0, [1, 200 + i], tir4b.Arm(), IDS, TOK, seed=1, thinking=False) for i in range(3)]
        flag = []
        tir4b.drive_http(post, TOK, IDS, rs, lambda: Kernel([]), live_most=1, on_round=lambda: flag.append(1), drain=lambda: bool(flag))
        self.assertEqual(asked, [[1, 200]])
        self.assertEqual([bool(r.finish) for r in rs], [True, False, False])

    def test_no_logit_bias_key_without_a_bias(self):
        bodies = []

        def post(body):
            bodies.append(body)
            return dict(content='', stop_type='eos', stopping_word='', tokens=[6])
        r = tir4b.Rollout(ROW, 0, [1], tir4b.Arm(inline_calc=True, inline_plain=True, no_digit_join=True), IDS, TOK, seed=1, thinking=False)
        tir4b.drive_http(post, TOK, IDS, [r], lambda: Kernel([]), live_most=1)
        self.assertNotIn('logit_bias', bodies[0])

    def test_the_stop_words_of_a_request(self):
        r = tir4b.Rollout(ROW, 0, [1], tir4b.Arm(inline_calc=True, execute=True), IDS, TOK, seed=1)
        most, stops = r.request()
        self.assertEqual(sorted(tir4b.stop_words(stops, IDS, TOK)), sorted(['=', '```', '</think>']))


class FakeEngine:
    """vLLM's LLMEngine as drive() uses it: each request finishes on a later step, one at a time, so answers interleave."""
    def __init__(self, scripts):
        self.streams = {tuple(p): TOK.encode(s) for p, s in scripts}
        self.queue = []

    def add_request(self, rid, prompt, params):
        self.queue.append((rid, prompt['prompt_token_ids'], params))

    def step(self):
        if not self.queue:
            return []
        rid, ids, params = self.queue.pop(0)
        stream = next(s for p, s in self.streams.items() if tuple(ids[:len(p)]) == p)
        new = []
        while stream and len(new) < params.max_tokens:
            new.append(stream.pop(0))
            if new[-1] in params.stop_token_ids:
                break
        new = new or [SPECIAL['<|im_end|>']]
        out = types.SimpleNamespace(token_ids=new, stop_reason=None)
        return [types.SimpleNamespace(request_id=rid, finished=True, outputs=[out])]


class Drive(unittest.TestCase):
    def setUp(self):
        fake = types.ModuleType('vllm')
        fake.SamplingParams = lambda **kw: types.SimpleNamespace(**kw)
        fake.TokensPrompt = lambda prompt_token_ids: dict(prompt_token_ids=prompt_token_ids)
        self.saved = sys.modules.get('vllm')
        sys.modules['vllm'] = fake

    def tearDown(self):
        if self.saved is None:
            sys.modules.pop('vllm')
        else:
            sys.modules['vllm'] = self.saved

    def test_continuous_drive_writes_what_one_answer_alone_writes(self):
        scripts = ['a 2 + 3 = and 4 \\times 5 = done</think>\n\n25<|im_end|>',
                   'let me check ```python\nprint(7)\n```</think>\n\n7<|im_end|>',
                   'plain thinking</think>\n\nok<|im_end|>']
        arms = [dict(inline_calc=True), dict(execute=True, close_fit=True), dict()]
        alone = []
        for s, a in zip(scripts, arms):
            r = tir4b.Rollout(ROW, 0, [1], tir4b.Arm(**a), IDS, TOK, seed=1)
            play(r, s, Kernel([(True, '7')]))
            alone.append(r.text())
        rs = [tir4b.Rollout(ROW, 0, [1, 100 + i], tir4b.Arm(**a), IDS, TOK, seed=1) for i, a in enumerate(arms)]
        engine = FakeEngine([(r.prompt, s) for r, s in zip(rs, scripts)])
        saves = []
        tir4b.drive(types.SimpleNamespace(llm_engine=engine), rs, lambda: Kernel([(True, '7')]), live_most=2, on_round=lambda: saves.append(1))
        self.assertEqual([r.text() for r in rs], alone)
        self.assertTrue(all(r.finish == 'stop' for r in rs))
        self.assertTrue(saves)


class Training(unittest.TestCase):
    def test_labels_mask_prompt_and_output(self):
        self.assertEqual(labels_of([1, 2], [3, 4, 5, 6], [[1, 3]]), [-100, -100, 3, -100, -100, 6])

    def test_choose_prefers_a_fitting_run_then_no_call(self):
        def rec(i, s, right, tokens, events, runs=0, finish='stop'):
            return dict(id=i, sample=s, right=right, tokens=tokens, injected=0, events=events, runs=runs, finish=finish)
        fit = [dict(kind='forced open'), dict(kind='output', fits=True), dict(kind='close after a fitting result')]
        records = [rec('a', 0, True, 100, []), rec('a', 1, True, 300, fit, 1), rec('a', 2, False, 50, fit, 1),
                   rec('b', 0, True, 80, []), rec('b', 1, True, 60, [dict(kind='loop close')]),
                   rec('c', 0, False, 10, [])]
        hinted = [dict(kind='hint open'), dict(kind='output', fits=True), dict(kind='trust hint'), dict(kind='loop hint'),
                  dict(kind='output', fits=True)]
        records.append(rec('d', 0, True, 500, hinted, 2))
        chosen = tir4b.choose(records)
        self.assertEqual(chosen['d']['sample'], 0)   # Hint-RFT: the hints are taught as the model's own words
        self.assertEqual(chosen['a']['sample'], 1)   # the run that fitted, though longer
        self.assertEqual(chosen['b']['sample'], 0)   # a loop close is never taught
        self.assertNotIn('c', chosen)


def play_biased(rollout, script, kernel, wins=True):
    """The scripted model with a prediction layer: a request that carries a bias writes the biased token when `wins`, its own when not.
    Either way the model's own token for that step is consumed from the script, as a replaced prediction would be."""
    stream = TOK.encode(script)
    asked = []
    while True:
        ask = rollout.request()
        if not ask:
            return asked
        most, stops = ask
        bias = rollout.bias()
        own = stream.pop(0) if stream else SPECIAL['<|im_end|>']
        if bias:
            asked.append((most, dict(bias)))
            new = [next(iter(bias)) if wins else own]
        else:
            new = [own]
            while stream and len(new) < most and new[-1] not in stops:
                new.append(stream.pop(0))
        rollout.take(new)


class LogitValue(unittest.TestCase):
    """--logit-value (MIP's logit override, soft): the value is not put in as text; the model writes it, with software's token raised."""
    def test_the_prediction_is_raised_and_the_model_writes_the_value(self):
        r = rollout(logit_value=6.0)
        asked = play_biased(r, 'so 6 \\times 7 = 41 and on</think>\n\n42<|im_end|>', Kernel([]))
        self.assertIn('6 \\times 7 = 42 and on', r.text())
        self.assertEqual([a[0] for a in asked], [1, 1, 1])   # ' 42' is three tokens, one request each
        self.assertEqual([list(a[1].values()) for a in asked], [[6.0]] * 3)
        self.assertEqual(kinds(r), ['logit value'])
        e = r.events[0]
        self.assertEqual((e['expr'], e['value'], e['tokens'], e['matched'], e['ok'], e['bias']), ('6 \\times 7', '42', 3, 3, True, 6.0))
        span = [s for s in r.spans if s[2] == 'logit value'][0]
        self.assertEqual(TOK.decode(r.out[span[0]:span[1]]), ' 42')

    def test_when_the_model_wins_against_the_bias_its_token_stays(self):
        r = rollout(logit_value=1.0)
        play_biased(r, 'so 6 \\times 7 = 41 and on</think>\n\n42<|im_end|>', Kernel([]), wins=False)
        self.assertIn('6 \\times 7 = 41 and on', r.text())
        e = r.events[0]
        self.assertEqual((e['matched'], e['tokens'], e['ok']), (2, 3, False))
        span = [s for s in r.spans if s[2] == 'logit value'][0]
        self.assertEqual(TOK.decode(r.out[span[0]:span[1]]), ' 4')   # only what the bias carried is marked, not the model's own token
        self.assertIsNone(r.bias())

    def test_a_sign_is_a_token_of_the_plan(self):
        r = rollout(logit_value=6.0)
        r.start_plan('3 - 21', '-18')
        self.assertEqual(TOK.decode(r.plan), ' -18')
        self.assertEqual(r.bias(), {r.plan[0]: 6.0})

    def test_off_by_default_and_not_text(self):
        r = rollout()
        play_biased(r, 'so 6 \\times 7 = 41</think>\n\nx<|im_end|>', Kernel([]))
        self.assertEqual(r.events, [])
        self.assertIsNone(r.bias())
        r = rollout(logit_value=6.0)
        play_biased(r, 'so 6 \\times 7 = 41</think>\n\n42<|im_end|>', Kernel([]))
        self.assertNotIn('inline value', kinds(r))

    def test_the_wanted_token_own_log_probability_is_kept(self):
        r = rollout(logit_value=6.0)
        r.start_plan('3 - 21', '-18')
        r.out += [r.plan[0]]
        r.last_lp = {r.plan[0]: types.SimpleNamespace(logprob=-0.25)}
        self.assertTrue(r.follow_plan(r.plan[0]))
        self.assertEqual(r.plan_own, [-0.25])


class SuppressRecheck(unittest.TestCase):
    """--suppress-recheck: the re-check words' logits are lowered for a number of steps after a value software gave."""
    def setUp(self):
        self.kept = IDS.recheck
        IDS.recheck = {1000 + ord('W')}   # the test tokenizer has one id per character: `W` stands for Wait

    def tearDown(self):
        IDS.recheck = self.kept

    def test_a_window_follows_an_inline_value(self):
        r = rollout(inline_calc=True, inline_plain=True, suppress_recheck=5)
        self.assertIsNone(r.bias())
        r.take(TOK.encode('so 6 \\times 7 ='))
        self.assertIn('6 \\times 7 = 42', r.text())
        self.assertEqual(r.window, 5)
        self.assertEqual(r.bias(), {1000 + ord('W'): tir4b.SUPPRESS_BIAS})
        self.assertLessEqual(r.request()[0], 5)   # one static bias covers the request
        r.take(TOK.encode('abc'))
        self.assertEqual(r.window, 2)
        r.take(TOK.encode('def'))
        self.assertEqual(r.window, 0)
        self.assertIsNone(r.bias())
        self.assertGreater(r.request()[0], 5)

    def test_a_window_follows_a_value_the_model_wrote_under_the_bias(self):
        r = rollout(logit_value=6.0, suppress_recheck=4)
        r.take(TOK.encode('so 6 \\times 7 ='))
        self.assertEqual(r.window, 0)   # the value is still being written
        while r.plan is not None:
            r.take([next(iter(r.bias()))])
        self.assertEqual(r.window, 4)
        self.assertEqual(set(r.bias().values()), {tir4b.SUPPRESS_BIAS})

    def test_a_lost_bias_opens_no_window(self):
        r = rollout(logit_value=1.0, suppress_recheck=4)
        r.take(TOK.encode('so 6 \\times 7 ='))
        r.take([1000 + ord('x')])   # the model's own token wins at the first step
        self.assertIsNone(r.plan)
        self.assertEqual(r.window, 0)

    def test_off_by_default(self):
        r = rollout(inline_calc=True)
        r.take(TOK.encode('so 6 \\times 7 ='))
        self.assertEqual(r.window, 0)
        self.assertIsNone(r.bias())


class UnifiedMathBackendAndCandidateClose(unittest.TestCase):
    """--mathbackend and --candidate-close: exact 6-module mathbackend inline evaluation and Assumption-Independent CandidateAnswerTracker."""
    def test_mathbackend_evaluates_exact_number_theory_and_combinatorics(self):
        r = rollout(inline_calc=True, inline_plain=True, mathbackend=True)
        play(r, 'so \\gcd(84, 126) + comb(10, 3) = and \\varphi(36) = done</think>\n\n162<|im_end|>', Kernel([]))
        self.assertIn('\\gcd(84, 126) + comb(10, 3) = 162 and \\varphi(36) = 12 done', r.text())

    def test_same_assumption_repeat_does_not_close_early(self):
        r = rollout(inline_calc=True, inline_plain=True, candidate_close=True)
        play(r, PAD + 'by formula 6 \\times 7 + 2 = so the final answer is 44.\nagain 2 + 7 \\times 6 = so the final answer is 44.\nWait, let me double-check.\nactual answer</think>\n\n44<|im_end|>', Kernel([]))
        self.assertNotIn('candidate close', kinds(r))
        self.assertIn('actual answer</think>', r.text())

    def test_a_check_in_words_is_no_second_method(self):
        """#303: the second run of one derivation said `substituting` / `small case check`, and the keyword made it CROSS_METHOD_CONFIRMED."""
        r = rollout(inline_calc=True, inline_plain=True, candidate_close=True)
        play(r, PAD + 'Using the formula, the final answer is 42.\nNow substituting into the same formula again, small case check: the final answer is 42.\n'
                'Wait, let me double-check.\nmore work</think>\n\n42<|im_end|>', Kernel([]))
        self.assertNotIn('candidate close', kinds(r))
        self.assertIn('more work</think>', r.text())

    def test_a_program_that_prints_the_answer_confirms_it(self):
        r = rollout(execute=True, candidate_close=True)
        play(r, PAD + 'so the final answer is 44.\n```python\nprint(40 + 4)\n```\nWait, let me double-check.\n\\boxed{44}<|im_end|>', Kernel([(True, '44\n')]))
        self.assertIn('candidate close', kinds(r))
        ev = [e for e in r.events if e['kind'] == 'candidate close'][0]
        self.assertEqual((ev['value'], ev['status']), ('44', 'CROSS_METHOD_CONFIRMED'))
        self.assertEqual(r.finish, 'stop')

    def test_a_program_that_prints_another_number_falsifies_nothing(self):
        """#303: any printed line was a `search`, so a helper's intermediate value falsified every other answer."""
        r = rollout(execute=True, candidate_close=True)
        play(r, PAD + '```python\nprint(7 * 3)\n```\nso the final answer is 44.\nmore</think>\n\n44<|im_end|>', Kernel([(True, '21\n')]))
        self.assertTrue(r.tracker.has_live_candidate())
        self.assertEqual({o.method for o in r.tracker.observations}, {'METHOD_ALGEBRAIC'})

    def test_loop_with_a_cycle_and_no_candidate_is_nudged_then_closed(self):
        """#304: an exact cycle is cut back and nudged at most LOOP_NUDGES times; then it is closed so the model answers, never `stuck`."""
        r = rollout(execute=True, loop_close=True, candidate_close=True)
        play(r, 'ab' * (tir4b.CHUNK // 2) * (tir4b.LOOP_NUDGES + 1) + '\\boxed{1}<|im_end|>', Kernel([]))   # one loop check a chunk
        self.assertEqual(kinds(r).count('loop nudge'), tir4b.LOOP_NUDGES)
        self.assertIn('loop close', kinds(r))
        self.assertNotEqual(r.finish, 'stuck')
        self.assertEqual(r.record(lambda row, text: False)['loop_nudges'], tir4b.LOOP_NUDGES)

    def test_a_line_back_loop_with_no_candidate_is_closed_not_nudged(self):
        """#304: a cut of 64 tokens left the repeated lines in place; line_back fired at the next line end, three nudges, then `stuck`."""
        import random
        rng = random.Random(3)
        same = 'and so the sum must be even here\n'
        script = ''.join(same + ''.join(rng.choice('qwertyuiopasdfghjklzxcvbnm ') for _ in range(30)) + '\n' for _ in range(16)) + '\\boxed{7}<|im_end|>'
        r = rollout(execute=True, loop_close=True, candidate_close=True)
        play(r, script, Kernel([]))
        self.assertNotIn('loop nudge', kinds(r))
        self.assertIn('loop close', kinds(r))
        self.assertNotEqual(r.finish, 'stuck')

    def test_an_answer_that_restates_its_thinking_is_not_stuck(self):
        """#305: the answer's loop check read the whole history, so a line the thinking wrote 15 times and the answer once more made 16 for
        line_back, and a right answer ended `stuck`. In the answer it reads the answer alone."""
        import random
        rng = random.Random(5)
        same = 'and so the sum must be even here\n'
        thinking = ''.join(same + ''.join(rng.choice('qwertyuiopasdfghjklzxcvbnm ') for _ in range(30)) + '\n' for _ in range(15))
        # line_back runs on a check that ends at a line end: --check-hand ends every request there (any arm does when a chunk does)
        script = thinking + '</think>\n' + same + 'The answer is 7.\n' + PAD * 2 + '\\boxed{7}<|im_end|>'
        for arm in (dict(check_hand=True), dict(check_hand=True, inline_calc=True), dict(check_hand=True, execute=True, loop_close=True)):
            r = rollout(**arm)
            play(r, script, Kernel([]))
            self.assertEqual(r.finish, 'stop', arm)
        r = rollout()
        play(r, '</think>\n' + same * 60 + '\\boxed{7}<|im_end|>', Kernel([]))
        self.assertEqual(r.finish, 'stuck')   # a loop in the answer itself still stops


class RepeatedValues(unittest.TestCase):
    """#301: an expression used three times anywhere in an answer closed or nudged the thinking as a loop, at the third use in 41 of 48
    stored inlineclose answers. Only an echo (values back to back, close together) is left to the model; whether it loops is the loop checks'."""
    def test_one_expression_three_times_in_a_derivation_is_no_loop(self):
        r = rollout(inline_calc=True, inline_plain=True, loop_close=True, candidate_close=True)
        filler = ' then the other side of the figure needs its own length worked out, '
        play(r, 'side 2^2 =' + filler + 'again 2^2 =' + filler + 'and 2^2 =' + filler + 'done</think>\n\n4<|im_end|>', Kernel([]))
        self.assertEqual(r.text().count('2^2 = 4'), 3)
        self.assertNotIn('loop close', kinds(r))
        self.assertNotIn('loop nudge', kinds(r))
        self.assertEqual(r.echo_skips, 0)

    def test_an_echo_is_left_to_the_model_and_closes_nothing(self):
        r = rollout(inline_calc=True, inline_plain=True, loop_close=True)
        play(r, '6 \\times 7 = so 6 \\times 7 = so 6 \\times 7 = 42 ok</think>\n\n42<|im_end|>', Kernel([]))
        self.assertEqual(kinds(r).count('inline value'), 2)
        self.assertEqual(r.echo_skips, 1)
        self.assertNotIn('loop close', kinds(r))
        self.assertIn('42 ok</think>', r.text())


class HeldOutInline(unittest.TestCase):
    """#306: the inline values over arms and models not used before (4B arms with nothing put in, DeepSeek-R1, QwQ-32B), checked against
    what each model wrote next. Each of these put a value the model did not mean there."""
    def test_what_the_text_means_otherwise_is_left_alone(self):
        for before in ('Right: cube root of 8^26',                                  # the argument of a root: 8^(26/3)
                       'but on the left side, √2 raised to (3*0 -1)',              # an exponent
                       'm=3, 64=8*8; m=4,512',                                      # m=4 and 512, not 4512
                       'Then 9√2 -3',                                               # already the value, in another spelling
                       'Total passed in Math:14 boys (10+4) +18 girls (12+6)',     # a bracket that counts the girls
                       'A=24:576.5*7*6',                                            # 24^2 = 576. and its digits' product
                       'Rightmost (10^-7): $0-4$. Borrow. Result **6**.\n10^-6: $0-1-8'):   # a column that borrows
            for mb in (False, True):
                with self.subTest(before=before, mathbackend=mb):
                    self.assertIsNone(tir4b.inline_found(before, use_mathbackend=mb))

    def test_a_point_or_vector_gets_each_component(self):
        for before, said in (('Moving 7 units from A: (5 -35/13, 12 -84/13)', '(30/13, 72/13)'),
                             ('So M - A = (0, 1 - (-1.5), 0 - (-0.5))', '(0, 2.5, 0.5)'),
                             ('Midpoint M: ((2 + 1.5)/2, (0 +1)/2)', '(1.75, 0.5)'),
                             ('$R_2 - 2 R_3 = (4+6, -4-8, -2+4)', '(10, -12, 2)')):
            with self.subTest(before=before):
                self.assertEqual(tir4b.inline_found(before, use_mathbackend=True)[1], said)
        for before in ('    $g(0, 1/2)', 'Q(12-10, 2(10)-6)', 'A is (2, 0)'):   # a call, a named point, a point already
            with self.subTest(before=before):
                self.assertIsNone(tir4b.inline_found(before, use_mathbackend=True))

    def test_a_plus_minus_is_written_back_simplified(self):
        for before, said in (('roots: [6 ± sqrt(36 - 4*1*1)] / 2 = [6 ± sqrt(32)] / 2', '3 ± 2*sqrt(2)'),
                             ('are $d = \\frac{16 \\pm \\sqrt{256 - 240}}{2}', '8 \\pm 2'),
                             ('u=(16 ±√(256 -64))/32=(16 ±√192)/32', '(2 ± √3)/4'),
                             ('dratic formula: x = [-3 ± sqrt(9 + 160)] / 2', '(-3 ± 13)/2'),
                             ('So $y_X = \\pm \\frac{\\sqrt{575}}{\\sqrt{48}}', '\\pm \\frac{5\\sqrt{69}}{12}')):
            with self.subTest(before=before):
                self.assertEqual(tir4b.inline_found(before, use_mathbackend=True)[1], said)
        for before in ('Then g=±1/2', 'b = [ (k -1) ± sqrt(D) ] / 2', ' = \\frac{-6 \\pm \\sqrt{-108}}{2}'):   # alone, a symbol, no real root
            with self.subTest(before=before):
                self.assertIsNone(tir4b.inline_found(before, use_mathbackend=True))

    def test_decimal_coefficients_keep_the_root(self):
        self.assertEqual(tir4b.inline_found('so 7√3 + 17.5√3', use_mathbackend=True)[1], '24.5√3')
        self.assertEqual(tir4b.inline_found('then $0.4 \\cdot 50 \\sqrt{10}', use_mathbackend=True)[1], '20\\sqrt{10}')
        self.assertIsNone(tir4b.inline_found('so $\\sqrt{310.9666}', use_mathbackend=True))   # a decimal under the root wants a decimal

    def test_pi_and_spaced_products(self):
        self.assertEqual(tir4b.inline_found('so E = π/2 - 5π/12', use_mathbackend=True)[1], 'π/12')
        self.assertEqual(tir4b.inline_found('Total area is $\\frac{\\pi}{2} - \\frac{\\pi}{3}', use_mathbackend=True)[1], '\\frac{\\pi}{6}')
        self.assertEqual(tir4b.inline_found('so $160 (1 - 2/3)', use_mathbackend=True)[1], '160/3')
        self.assertEqual(tir4b.inline_found('so |4 - (-19)|', use_mathbackend=True)[1], '23')
        self.assertEqual(tir4b.inline_found("Heron's formula: √[45*20*144]", use_mathbackend=True)[1], '360')   # the bracket is the radicand
        self.assertEqual(tir4b.inline_found('so ~104 + 104', use_mathbackend=True)[1], '208')
        self.assertEqual(tir4b.inline_found('- H=2:2:13→2+1+3', use_mathbackend=True)[1], '6')

    def test_general_style_gaps_closed(self):
        # #307: forms the General Style read nothing at, each closed by the meaning of the notation
        def found(before):
            got = tir4b.inline_found(before, use_mathbackend=True)
            return got and got[1]
        self.assertEqual(found("11: 11 mod25. Let's compute 11^2=121≡21, 11^4=21^2=441≡441-17*25"), '16')   # a relation sign ends it
        self.assertEqual(found('so $3005/6 / 5'), '601/6')               # one reading, by the spaces or left to right
        self.assertIsNone(found('so $-1/4 / 1/2'))                        # two readings
        self.assertIsNone(found('x = 3 / 6/2'))
        self.assertEqual(found('so sqrt(1 +1.21)'), 'sqrt(221)/10 ≈ 1.4866')   # a decimal under a root: a short exact form, and its decimal
        self.assertEqual(found('s = \\sqrt{312.6}'), '17.6804...')    # no short exact form: cut and marked
        self.assertEqual(found('x = 1.530996 / \\pi'), '0.4873...')     # a decimal with pi
        self.assertEqual(found('so sqrt(2.25)'), '1.5')                   # exact where it comes out
        self.assertEqual(found('Area=1/2 |42√5 -6√5|'), '18√5')           # a factor before an opening bar (the shoelace area)
        self.assertEqual(found('Right:3|0 +0 +3*1|'), '9')
        self.assertEqual(found('Multiply by ½: ½ *4'), '2')               # vulgar fractions
        self.assertIsNone(found('so 3½ + 1'))                             # a mixed number reads two ways
        self.assertEqual(found('centroid is (12.684/8,96.564/8)'), '(1.5855, 12.0705)')   # a vector after a word
        self.assertIsNone(found('point Q (12-10, 20-6)'))                 # a named point
        self.assertEqual(found('so ∛2 + 1/(∛2)²'), '1.8898...')           # past the exact field: cut and marked
        self.assertEqual(found('then 2^(1/3) + 2^(-2/3)'), '1.8898...')
        self.assertEqual(found('so 8^(1/3) + 1'), '3')
        self.assertEqual(found('SD: $\\sqrt{264.5}'), '\\frac{23\\sqrt{2}}{2} \\approx 16.2635')
        # an uneven quotient: the simplest true form, never a `0 rem a`
        self.assertEqual(found('Denominator: $-4 / -7'), '4/7')
        self.assertEqual(found('$BD/CD = 200/225'), '8/9')
        self.assertEqual(found('remainder 3, so $4/9'), '0.4444...')
        self.assertEqual(found('$2034 \\div 4'), '508 \\text{ rem } 2')
        # with a sign, never `q rem r` from unsigned digits: these were false (held-out audit)
        self.assertEqual(found('So, $T_3 = \\frac{-1}{1} \\div 6 = -1 \\div 6'), '-0.1666...')
        self.assertEqual(found('2. $20 \\div (-8)'), '-2.5')

    def test_mod_work_and_long_division(self):
        # #306 held-out. The general style writes what is true under every reading; a model's profile writes the form it was measured to
        # want: a remainder's quotient alone, a residue only where its `=` already meant `≡`
        def found(before, style='general'):
            return tir4b.inline_found(before, use_mathbackend=True, style=style)[1]
        chain = '7^10: 7^2=49≡-1 mod25. Then 7^4=(7^2)^2=(-1)^2=1 mod25. Then 7^5=7^4*7'
        latex = 'Product $4 \\times 2 = 8 \\equiv 3 \\pmod 5$.\nSo $S \\equiv 4 \\times 2'
        r1 = '6^7=6*24=144≡144-4*29=144-116=28 mod29.6^8=6*28'
        self.assertEqual(found('Long division: 1091 rem 3. So 13 / 2'), '6 rem 1')
        self.assertEqual(found(chain), '16807 ≡ 7')
        self.assertEqual(found(latex), '8 \\equiv 3')
        self.assertEqual(found(r1), '168 ≡ 23')
        for style in ('qwen3.5-4b', 'deepseek-r1', 'qwq-32b'):
            with self.subTest(style=style):
                self.assertEqual(found('Long division: 1091 rem 3. So 13 / 2', style), '6')
                self.assertEqual(found(chain, style), '7')
                self.assertEqual(found(latex, style), '8')   # it writes the full value, then reduces
                self.assertEqual(found(r1, style), '168')
        with self.assertRaises(KeyError):
            tir4b.inline_found('so 2+3', use_mathbackend=True, style='no-such-model')
        self.assertIsNone(tir4b.inline_found('    601 mod 7: $601 = 560 + 41', use_mathbackend=True))   # taking 601 apart
        self.assertEqual(tir4b.inline_found('the carry: $7+9', use_mathbackend=True)[1], '16')
        self.assertIsNone(tir4b.inline_found('borrow: $0-9', use_mathbackend=True))
        self.assertEqual(tir4b.inline_found('We need it mod 7. Earlier.\n\n$12^2', use_mathbackend=True)[1], '144')

    def test_the_neighbours_still_get_their_value(self):
        for before, said in (('Divide by $2-1', '1'),                               # a step named, then its operand worked out
                             ('then multiply by $7 \\times 2', '14'),
                             ('So P = z1 z2 approaches (-0.5)(-1)', '0.5'),
                             ('so b=(64 ±58)/2=(122)/2=61 or (6)/2', '3'),
                             ('So 2.5*3', '7.5'),
                             ('Sum: $3 + 4', '7')):
            with self.subTest(before=before):
                self.assertEqual(tir4b.inline_found(before, use_mathbackend=True)[1], said)


class NoDigitJoin(unittest.TestCase):
    """#307: 80 of 19,990 A100 puts were run on by the model's own digits into another number (`-5` then `75`); --no-digit-join makes the
    token after a put that ends in a digit a request of its own, every digit held down."""
    def put(self, **arm):
        r = rollout(inline_calc=True, inline_plain=True, **arm)
        stream = TOK.encode('so $6 \\times 7 =')
        r.take(stream)
        return r

    def test_the_next_token_cannot_be_a_digit(self):
        r = self.put(no_digit_join=True)
        self.assertIn('$6 \\times 7 = 42', r.text())
        most, _ = r.request()
        self.assertEqual(most, 1)
        bias = r.bias()
        self.assertEqual(bias[1000 + ord('5')], tir4b.JOIN_BIAS)
        self.assertNotIn(1000 + ord('$'), bias)
        r.take(TOK.encode('$'))
        self.assertIsNone(r.bias())
        self.assertGreater(r.request()[0], 1)
        self.assertEqual(r.record(lambda *_: True)['joins_held'], 1)

    def test_a_slash_cannot_follow_either(self):
        """`-23/11 + 12/11 =` put `-1`, then the model's planned `/11` read `-1/11` (#307)."""
        r = self.put(no_digit_join=True)
        self.assertEqual(r.bias()[1000 + ord('/')], tir4b.JOIN_BIAS)

    def test_only_tokens_that_start_a_division_are_held(self):
        """The 4B has 1,078 tokens starting with `/`; holding them all made 1,106 bias tokens and vLLM killed the engine (#307)."""
        tok = WideTok(['/', '/(', '/\\', '/-', '/7', '//', '/**', '/>', '/usr'])
        held = {tok.decode([i]) for i in tir4b.Ids(tok, [6]).slash_start}
        self.assertEqual(held, {'/', '/(', '/\\', '/-', '/7'})

    def test_a_tokenizer_that_would_overflow_vllms_cap_is_refused(self):
        tok = WideTok([f'{d}x{n}' for d in '123456789' for n in range(120)])   # 1,080 tokens that start with a digit
        with self.assertRaisesRegex(RuntimeError, 'vLLM refuses more than 1024'):
            tir4b.Ids(tok, [6])

    def test_after_a_point_only_digits_are_held(self):
        """`0` then `.75` read `0.75`; `= 42.` ends a sentence. The point is let through, then one token with digits held."""
        r = self.put(no_digit_join=True)
        r.take(TOK.encode('.'))
        self.assertEqual(r.request()[0], 1)
        bias = r.bias()
        self.assertEqual(bias[1000 + ord('7')], tir4b.JOIN_BIAS)
        self.assertNotIn(1000 + ord('/'), bias)
        r.take(TOK.encode('\n'))
        self.assertIsNone(r.bias())
        self.assertGreater(r.request()[0], 1)

    def test_no_point_hold_after_other_text(self):
        r = self.put(no_digit_join=True)
        r.take(TOK.encode('$'))
        r.take(TOK.encode('.'))
        self.assertIsNone(r.bias())

    def test_off_by_default(self):
        r = self.put()
        self.assertGreater(r.request()[0], 1)
        self.assertIsNone(r.bias())

    def test_a_cut_decimal_needs_no_hold(self):
        r = rollout(inline_calc=True, inline_plain=True, mathbackend=True, no_digit_join=True)
        r.take(TOK.encode('so $21/19 ='))
        self.assertIn('1.1052...', r.text())
        self.assertFalse(r.join_guard)


class FamilyProfile(unittest.TestCase):
    """#307: a family profile picks one of the General Style's true forms at a decision point; it never writes a value of its own."""
    def setUp(self):
        self.saved = dict(tir4b.INLINE_STYLES['qwen']['forms'])

    def tearDown(self):
        tir4b.INLINE_STYLES['qwen']['forms'] = self.saved

    def test_each_form_is_the_same_value(self):
        from fractions import Fraction
        tir4b.INLINE_STYLES['qwen']['forms'] = {'mod': 'value', 'quotient-reduces': 'decimal', 'quotient-lowest': 'none',
                                               'quotient-terminates': 'fraction'}
        self.assertEqual(tir4b.inline_found('Product $8 \\equiv 3 \\pmod 5$.\nSo $S \\equiv 4 \\times 2', use_mathbackend=True,
                                            style='qwen')[1], '8')
        self.assertEqual(tir4b.inline_found('so $10044/42', use_mathbackend=True, style='qwen')[1], '239.1428...')
        self.assertIsNone(tir4b.inline_found('so $21/19', use_mathbackend=True, style='qwen'))
        self.assertEqual(tir4b.inline_found('so 3/8', use_mathbackend=True, style='qwen')[1], '3/8')
        self.assertEqual(Fraction(10044, 42), Fraction(1674, 7))

    def test_an_empty_table_is_the_general_style(self):
        tir4b.INLINE_STYLES['qwen']['forms'] = {}
        for before in ('so $10044/42', 'so $21/19', 'so 3+4', 'so 3/8'):
            self.assertEqual(tir4b.inline_found(before, use_mathbackend=True, style='qwen'),
                             tir4b.inline_found(before, use_mathbackend=True))

    def test_a_context_key_wins_over_the_decision(self):
        tir4b.INLINE_STYLES['qwen']['forms'] = {'quotient-reduces': 'decimal', ('quotient-reduces', True, False): 'fraction'}
        self.assertEqual(tir4b.inline_found('so $10044/42', use_mathbackend=True, style='qwen')[1], '\\frac{1674}{7}')
        self.assertEqual(tir4b.inline_found('so 10044/42', use_mathbackend=True, style='qwen')[1], '239.1428...')


class ExactAnswer(unittest.TestCase):
    """--exact-answer: when the question states one computation (question_read), the answer's box holds its exact value. Round 9 (4B
    NVFP4 W4A8, llama.cpp, PC layer): the model wrote `-73388/36903` and then boxed `-1.9886` on purpose; told the exact value (loop value,
    the class-C settle line) it still chose the decimal. Software writes the box; the model's reasoning is untouched."""

    def arm(self, question, answer, **more):
        row = dict(id='dm_div_big-1', suite='arith', answer=answer,
                   question=question + '\n\nGive the final answer as a single number in \\boxed{} at the end.')
        return tir4b.Rollout(row, 0, [1, 2, 3], tir4b.Arm(exact_answer=True, **more), IDS, TOK, seed=1)

    def answered(self, r, box):
        play(r, 'x</think>\n\nSo the result is $\\boxed{' + box + '}$.<|im_end|>', Kernel([]))
        return r

    def acts(self, r):
        return [e for e in r.events if e['kind'] == 'exact answer']

    def test_a_rounded_decimal_of_the_questions_value_is_boxed_exact(self):
        # 1395#0: `-73388/36903 \approx -1.9886`, then `I will provide the decimal value rounded to 4 decimal places`, \boxed{-1.9886}
        r = self.answered(self.arm('Divide -73388 by 36903.', '-73388/36903'), '-1.9886')
        self.assertTrue(r.text().endswith('So the result is $\\boxed{-73388/36903}$.<|im_end|>'), r.text()[-80:])
        self.assertEqual(r.finish, 'stop')
        act, = self.acts(r)
        self.assertEqual((act['copied'], act['value'], act['via']), ('-1.9886', '-73388/36903', 'box'))
        from tools.igc.select_rows import right
        self.assertTrue(right(r.row, r.text()))

    def test_every_shortening_of_the_value_counts(self):
        cases = [('Divide -73388 by 36903.', '-73388/36903', '-1.9887'),          # rounded (1395#1)
                 ('Divide 139241497 by 13.', '139241497/13', '10710884.384615...'),  # cut, with dots
                 ('Divide -139241497 by 13.', '-139241497/13', '-10710884'),        # the integer part (4138#0)
                 ('Divide -171 by 2221234.', '-171/2221234', '0'),                  # the integer part of a small one (2783#0)
                 ('Divide 11294 by 2475526.', '5647/1237763', '0.00456226'),        # the model's own digits drift (5070#1)
                 ('Divide 11294 by 2475526.', '5647/1237763', '0.004562376')]       # 5070#0
        for question, gold, box in cases:
            r = self.answered(self.arm(question, gold), box)
            self.assertEqual(len(self.acts(r)), 1, (question, box))
            self.assertIn('\\boxed{' + gold + '}', r.text())

    def test_a_wrong_value_or_a_lost_sign_is_left_to_the_model(self):
        # not a form of the value: a different number, or the opposite one. Software does not answer for the model there
        # -2 and -1.98 are the value rounded to a whole number and cut at 2 places: shortenings, rewritten; these are not
        for box in ('-1.5', '1.9886', '-3', '-1.97', '2', '-1.95', '-1.9875'):
            r = self.answered(self.arm('Divide -73388 by 36903.', '-73388/36903'), box)
            self.assertEqual(self.acts(r), [], box)
            self.assertIn('\\boxed{' + box + '}', r.text())

    def test_the_value_already_exact_is_left_alone(self):
        for box in ('-73388/36903', '-\\frac{73388}{36903}', '\\frac{-73388}{36903}'):
            r = self.answered(self.arm('Divide -73388 by 36903.', '-73388/36903'), box)
            self.assertEqual(self.acts(r), [], box)

    def test_an_integer_value_is_never_rewritten(self):
        # 9977: the question's value is the integer -70468; whatever the model boxed is its answer, right or wrong
        for box in ('70468', '-70468.0', '-70468'):
            r = self.answered(self.arm('Divide -418157112 by 5934.', '-70468'), box)
            self.assertEqual(self.acts(r), [], box)

    def test_no_act_without_a_computation_the_question_states(self):
        for question in ('Divide 3 by 7. Round to 2 decimal places.', 'Tom has 3 apples and gives away 7. How many per friend?'):
            r = self.answered(self.arm(question, '3/7'), '0.43')
            self.assertEqual(self.acts(r), [], question)
        r = tir4b.Rollout(dict(id='q', suite='arith', answer='-73388/36903', question='Divide -73388 by 36903.'), 0, [1],
                          tir4b.Arm(), IDS, TOK, seed=1)
        self.answered(r, '-1.9886')
        self.assertEqual(self.acts(r), [])   # off without the flag

    def test_only_the_last_box_of_the_answer_is_read(self):
        r = self.arm('Divide -73388 by 36903.', '-73388/36903')
        play(r, 'x</think>\n\nRoughly $\\boxed{-1.9886}$, exactly $\\boxed{-73388/36903}$.<|im_end|>', Kernel([]))
        self.assertEqual(self.acts(r), [])

    def test_the_answer_close_cue_carries_the_exact_value(self):
        # 2783#0: the cue cut `Let's provide the fraction` and the model wrote `{0}`; 5070#0: it wrote `{0.004562376}`. With the question's
        # value known the cue ends with it, and the answer ends there
        r = self.arm('Divide -73388 by 36903.', '-73388/36903', answer_close=True)
        r.take(TOK.encode('x</think>'))
        r.take(TOK.encode('\n\nso ' + 'ab' * 600))
        self.assertEqual(r.finish, 'stop')
        self.assertTrue(r.text().endswith('The final answer is $\\boxed{-73388/36903}$.<|im_end|>'), r.text()[-80:])
        act, = self.acts(r)
        self.assertEqual(act['via'], 'cue')
        r = rollout(answer_close=True, exact_answer=True)   # no question: the cue is as before, the model writes the value
        r.take(TOK.encode('x</think>'))
        r.take(TOK.encode('\n\nso ' + 'ab' * 600))
        self.assertIsNone(r.finish)
        self.assertTrue(r.text().endswith('The final answer is $\\boxed'))


if __name__ == '__main__':
    unittest.main()
