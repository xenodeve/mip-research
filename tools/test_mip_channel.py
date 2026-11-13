"""Tests for tools/igc/mip_channel.py: the service the llama.cpp shim talks to, driven by a fake shim over a real socket."""
import json
import socket
import unittest

from tools.igc import tir4b
from tools.igc.mip_channel import ChannelService
from tools.igc.mip_planner import Planner
from tools.test_igc_tir4b import TOK, IDS, SPECIAL

PIECES = [TOK.decode([i]) for i in range(len(TOK))]
PROMPT = [11, 1001, 1002]   # any prompt; the answer's text is what follows it


def roll(question='Find x.', out=''):
    a = tir4b.Arm(inline_calc=True, inline_plain=True, mathbackend=True, no_digit_join=True, approx_put=True)
    r = tir4b.Rollout(dict(id='q', suite='arith', question=question, answer='1'), 0, PROMPT, a, IDS, TOK, seed=1)
    r.out = TOK.encode(out)
    return r


class Shim:
    """The shim's side of the wire, one JSON object a line."""
    def __init__(self, port):
        self.s = socket.create_connection(('127.0.0.1', port), timeout=5)
        self.f = self.s.makefile('rw', encoding='utf-8', newline='\n')

    def send(self, **m):
        self.f.write(json.dumps(m) + '\n')
        self.f.flush()

    def recv(self):
        return json.loads(self.f.readline())

    def open(self, mip_id, slot, prompt):
        self.send(t='open', mip_id=mip_id, slot=slot, prompt=prompt)

    def toks(self, slot, text):
        for t in TOK.encode(text):
            self.send(t='tok', slot=slot, id=t)

    def ask(self, slot, seq):
        self.send(t='ask', slot=slot, seq=seq)
        m = self.recv()
        assert m['t'] == 'plan' and m['slot'] == slot and m['seq'] == seq, m
        return m


def spell(dag):
    node, out = 0, ''
    while node != dag['end']:
        t, node = min(dag['edges'][node])
        out += PIECES[t]
    return out


class ChannelTests(unittest.TestCase):
    def setUp(self):
        self.svc = ChannelService(triggers=IDS.equals, think_close=SPECIAL['</think>'], pieces=PIECES, tokenizer=TOK)
        self.port = self.svc.start()
        self.shim = Shim(self.port)
        self.hello = self.shim.recv()

    def tearDown(self):
        self.shim.s.close()
        self.svc.stop()

    def test_hello_names_the_triggers(self):
        self.assertEqual(self.hello['t'], 'hello')
        self.assertEqual(set(self.hello['triggers']), set(IDS.equals))
        self.assertEqual(self.hello['think_close'], SPECIAL['</think>'])

    def test_a_value_after_equals(self):
        r = roll()
        self.svc.register('a#0#0', r, Planner(r, PIECES))
        self.shim.open('a#0#0', 0, PROMPT)
        self.assertIsNone(self.shim.ask(0, 0)['dag'])          # not a fixed computation: no line at the start
        self.shim.toks(0, 'So $3 \\times 4 =')
        m = self.shim.ask(0, 1)
        self.assertEqual(spell(m['dag']), ' 12')
        self.assertTrue(set(m['after_ban']) >= set(IDS.digit_start))
        self.shim.toks(0, ' 12')
        self.shim.ask(0, 2)                                   # a reply: every token before it was read
        self.assertEqual(self.svc.stats()['steered'], 1)
        ev = [e for e in r.events if e['kind'] == 'steered value']
        self.assertEqual((ev[0]['value'], ev[0]['done']), ('12', True))

    def test_two_answers_of_one_question_keep_their_own_plans(self):
        r0, r1 = roll(), roll()
        self.svc.register('a#0#0', r0, Planner(r0, PIECES))
        self.svc.register('a#1#0', r1, Planner(r1, PIECES))
        self.shim.open('a#0#0', 0, PROMPT)
        self.shim.open('a#1#0', 1, PROMPT)
        self.shim.toks(0, '$2+2 =')
        self.shim.toks(1, '$5+5 =')
        self.assertEqual(spell(self.shim.ask(1, 1)['dag']), ' 10')
        self.assertEqual(spell(self.shim.ask(0, 1)['dag']), ' 4')

    def test_a_request_that_ends_inside_a_value_goes_on_from_where_it_stopped(self):
        r = roll()
        self.svc.register('a#0#0', r, Planner(r, PIECES))
        self.shim.open('a#0#0', 0, PROMPT)
        self.shim.toks(0, '$3 \\times 4 =')
        self.shim.ask(0, 1)
        self.shim.toks(0, ' 1')                                # the request ends here
        r.out = TOK.encode('$3 \\times 4 = 1')                 # the driver takes the reply
        self.svc.register('a#0#1', r, Planner(r, PIECES))
        self.shim.open('a#0#1', 0, PROMPT + r.out)
        self.assertEqual(spell(self.shim.ask(0, 0)['dag']), '2')   # the rest of the value, not the value again

    def test_an_ask_read_after_the_driver_took_the_reply(self):
        # the request's last token is '=': the shim asks, the server finishes, and the driver may take the reply before the
        # service reads that ask; the plan must still be read on this request's text, placed where the value starts
        r = roll()
        self.svc.register('a#0#0', r, Planner(r, PIECES))
        self.shim.open('a#0#0', 0, PROMPT)
        self.shim.toks(0, '$3 \\times 4 =')
        r.out = TOK.encode('$3 \\times 4 =')                   # the driver took the reply first
        self.assertEqual(spell(self.shim.ask(0, 1)['dag']), ' 12')
        ev = [e for e in r.events if e['kind'] == 'steered value']
        self.assertEqual(ev[0]['at'], len(r.out))
        self.svc.register('a#0#1', r, Planner(r, PIECES))
        self.shim.open('a#0#1', 0, PROMPT + r.out)
        self.assertEqual(spell(self.shim.ask(0, 0)['dag']), ' 12')   # the next request writes it from its start

    def _cut_inside_a_value(self, out_after):
        r = roll()
        self.svc.register('a#0#0', r, Planner(r, PIECES))
        self.shim.open('a#0#0', 0, PROMPT)
        pre = 'Some long reasoning first, then $3 \\times 4 ='
        self.shim.toks(0, pre)
        m = self.shim.ask(0, 1)
        t0 = min(m['dag']['edges'][0])[0]
        self.shim.send(t='tok', slot=0, id=t0)
        self.shim.ask(0, 2)                                    # every token before it was read
        r.out = out_after(pre, t0)
        return r

    def test_no_resume_after_the_thinking_closed(self):
        # the driver cut the text back past the value and closed the thinking: the rest of the value must never be steered
        r = self._cut_inside_a_value(lambda pre, t0: TOK.encode('Some') + [SPECIAL['</think>']])
        r.phase = 'answer'
        self.svc.register('a#0#1', r, Planner(r, PIECES))
        self.shim.open('a#0#1', 1, PROMPT + r.out)
        self.assertIsNone(self.shim.ask(1, 0)['dag'])
        ev = [e for e in r.events if e['kind'] == 'steered value']
        self.assertTrue(all(e.get('dropped') for e in ev))      # every plan of it ended as dropped, none left open

    def test_no_resume_when_the_text_before_the_value_changed(self):
        # cut back and rewritten (a loop close in the thinking): the value's place is gone
        r = self._cut_inside_a_value(lambda pre, t0: TOK.encode('Some long reasoning first, then $3 \\times 5 =') + [t0])
        self.svc.register('a#0#1', r, Planner(r, PIECES))
        self.shim.open('a#0#1', 1, PROMPT + r.out)
        self.assertIsNone(self.shim.ask(1, 0)['dag'])

    def test_an_event_names_its_slot_and_seq(self):
        # gate 3 joins the shim's missed asks (llama.log, by slot and seq) to tell a value the model wrote itself
        r = roll()
        self.svc.register('a#0#0', r, Planner(r, PIECES))
        self.shim.open('a#0#0', 5, PROMPT)
        self.shim.toks(5, 'So $3 \\times 4 =')
        self.shim.ask(5, 3)
        ev = [e for e in r.events if e['kind'] == 'steered value']
        self.assertEqual((ev[0]['slot'], ev[0]['seq']), (5, 3))

    def test_the_fixed_question_line_at_the_start_only(self):
        r = roll(question='What is 2*240?')
        self.svc.register('b#0#0', r, Planner(r, PIECES))
        self.shim.open('b#0#0', 0, PROMPT)
        self.assertEqual(spell(self.shim.ask(0, 0)['dag']), '2*240 = 480\n')
        r.out = TOK.encode('2*240 = 480\nSo')
        self.svc.register('b#0#1', r, Planner(r, PIECES))
        self.shim.open('b#0#1', 0, PROMPT + r.out)
        self.assertIsNone(self.shim.ask(0, 0)['dag'])

    def test_a_pending_plan_is_steered_at_the_next_request(self):
        r = roll(out='So x\n')
        r.steer_queue.append(('The question writes: $2*240$.\n', 'question quote', '2*240', '2*240'))
        self.svc.register('c#0#1', r, Planner(r, PIECES))
        self.shim.open('c#0#1', 0, PROMPT + r.out)
        self.assertEqual(spell(self.shim.ask(0, 0)['dag']), 'The question writes: $2*240$.\n')
        ev = [e for e in r.events if e['kind'] == 'steered value']
        self.assertEqual((ev[0]['what'], ev[0]['at']), ('question quote', len(r.out)))

    def test_nothing_after_the_thinking_closed(self):
        r = roll()
        self.svc.register('a#0#0', r, Planner(r, PIECES))
        self.shim.open('a#0#0', 0, PROMPT)
        self.shim.toks(0, 'ok</think>$3 \\times 4 =')
        self.assertIsNone(self.shim.ask(0, 1)['dag'])

    def test_an_unknown_slot_gets_no_plan(self):
        self.assertIsNone(self.shim.ask(7, 1)['dag'])


if __name__ == '__main__':
    unittest.main()
