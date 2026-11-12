"""Tests for tools/igc/token_trie.py: the token DAG that spells a value MIP steers (token channel design, 2026-11-12)."""
import os
import unittest

from tools.igc.token_trie import ValueDag, vocab_pieces

HF = 'C:/Users/xenod/mip-work/q4b-hf'


class ValueDagTests(unittest.TestCase):
    def test_dag_spells_every_tokenization(self):
        pieces = ['', ' ', '1', '2', '12', ' 1', '=', 'a']
        d = ValueDag.build(' 12', pieces)
        self.assertEqual(sorted(d.allowed(0)), [1, 5])          # ' ' or ' 1'
        self.assertEqual(sorted(d.allowed(d.step(0, 5))), [3])  # after ' 1' only '2'
        self.assertEqual(d.step(d.step(0, 1), 4), d.end)        # ' ' then '12'

    def test_no_dag_when_no_token_can_start(self):
        self.assertIsNone(ValueDag.build(' 7', ['', '1', '2']))

    def test_no_dead_node(self):
        # ' ' and ' 1' start the text but nothing then spells '2': no plan, never a mask with nothing left
        self.assertIsNone(ValueDag.build(' 12', ['', ' ', '1', ' 1']))

    def test_dead_branches_are_pruned(self):
        # ' 1' leads to '2' (spellable); ' ' leads to '1' then '2'? '1' and '2' exist, so both live; 'x' never appears
        d = ValueDag.build(' 12', ['', ' ', '1', '2', ' 1', ' 12x'])
        self.assertEqual(sorted(d.allowed(0)), [1, 4])

    def test_step_outside_the_dag(self):
        d = ValueDag.build(' 12', ['', ' ', '1', '2'])
        self.assertIsNone(d.step(0, 2))

    def test_special_tokens_never_spell(self):
        self.assertIsNone(ValueDag.build('<|im_end|>', ['<|im_end|>']))

    def test_json_shape(self):
        d = ValueDag.build(' 12', ['', ' ', '1', '2', '12'])
        j = d.to_json()
        self.assertEqual(j['end'], d.end)
        self.assertEqual(sorted(t for t, _ in j['edges'][0]), [1])
        self.assertEqual(sorted(t for t, _ in j['edges'][1]), [2, 4])

    @unittest.skipUnless(os.path.isdir(HF), 'the HF tokenizer is not on disk')
    def test_real_tokenizer_digits(self):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(HF)
        pieces = vocab_pieces(tok)
        d = ValueDag.build(' 480', pieces)
        self.assertIsNotNone(d)
        self.assertIn(tok.encode(' 480', add_special_tokens=False)[0], d.allowed(0))
        # every path through the DAG spells exactly the text
        node, out = 0, ''
        while node != d.end:
            t = min(d.allowed(node))
            out += pieces[t]
            node = d.step(node, t)
        self.assertEqual(out, ' 480')


if __name__ == '__main__':
    unittest.main()
