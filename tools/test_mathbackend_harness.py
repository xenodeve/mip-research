"""Tests for the CUDAMathBackend's gate: the canonical encoding (canon.py) and the harness (harness.py). The harness must pass Python
through its own encoding at 100 % (the null backend) and must fail every wrong backend on the operations that backend damages."""
import json
import tempfile
import unittest
from decimal import Decimal
from fractions import Fraction

from tools.igc.mathbackend import BadInput, NotCovered, Surd, canon, harness

OPS = ['numbers.add', 'numbers.sub', 'numbers.mul', 'numbers.div', 'numbers.gcd', 'numbers.xgcd']


class Canon(unittest.TestCase):
    def test_round_trip_every_type(self):
        values = [0, -12345678901234567890123, True, None, Fraction(-2, 6), Fraction(4, 2), Decimal('1.50'), Decimal('-1.5E+3'),
                  'abc', [1, (2, 3)], {'a': 1, 2: Fraction(1, 3)}, {3, 1, 2}, Surd(1, Fraction(1, 2), 8), 0.1, 10 ** 5000]
        for v in values:
            b = canon.encode(v)
            self.assertEqual(canon.encode(canon.decode(b)), b, v)

    def test_forms_that_must_differ(self):
        self.assertNotEqual(canon.encode(Decimal('1.50')), canon.encode(Decimal('1.5')))   # the exponent is part of the value
        self.assertNotEqual(canon.encode((1, 2)), canon.encode([1, 2]))
        self.assertNotEqual(canon.encode(True), canon.encode(1))
        self.assertEqual(canon.encode(Fraction(4, 2)), canon.encode(2))                       # a whole Fraction is an int
        self.assertEqual(canon.encode(Fraction(2, -6)), b'["q","-1","3"]')                     # reduced, sign on the numerator
        self.assertEqual(canon.encode({3, 1, 2}), canon.encode({2, 3, 1}))

    def test_refusals_by_class_only(self):
        self.assertEqual(canon.encode_refusal(NotCovered('past the bound')), canon.encode_refusal(NotCovered('other words')))
        self.assertNotEqual(canon.encode_refusal(NotCovered('x')), canon.encode_refusal(BadInput('x')))
        with self.assertRaises(TypeError):
            canon.encode_refusal(ZeroDivisionError())

    def test_python_int_limit_is_kept(self):
        import sys
        before = sys.get_int_max_str_digits()
        canon.encode(10 ** 5000)
        self.assertEqual(sys.get_int_max_str_digits(), before)   # the reference Python is not changed by the encoding

    def test_unknown_type_is_an_error(self):
        with self.assertRaises(TypeError):
            canon.encode(object())


class Harness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        path = harness.build('numbers', cls.tmp.name, OPS, seed=1, n=30)
        cls.head, cls.rows = harness.load(path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_corpus_is_frozen_with_its_provenance(self):
        self.assertEqual(self.head['ops'], sorted(OPS))
        self.assertEqual(self.head['python_bugs'], [])
        self.assertTrue(self.head['python'])
        again = harness.build('numbers', self.tmp.name + '/again', OPS, seed=1, n=30)
        self.assertEqual(harness.load(again)[1], self.rows)   # the same seed gives the same bytes

    def test_null_backend_reads_100(self):
        rep = harness.check(self.rows, harness.python_backend)
        self.assertTrue(harness.all_agree(rep), {k: v['examples'] for k, v in rep.items() if v['disagree']})
        self.assertGreater(sum(v['refusals_matched'] for v in rep.values()), 0)   # refusals are in the corpus and matched

    def test_every_wrong_backend_fails_where_it_damages(self):
        for how, op in (('off_by_one', 'numbers.add'), ('unreduced', 'numbers.div'), ('wrong_refusal', 'numbers.div'),
                        ('refuse_answer', 'numbers.gcd'), ('reorder', 'numbers.xgcd')):
            rep = harness.check(self.rows, harness.wrong_backend(how))
            self.assertGreater(rep[op]['disagree'], 0, f'{how} not caught on {op}')

    def test_decimal_damage_is_caught(self):
        out = canon.encode(Decimal('1.50'))
        self.assertNotEqual(harness._mutate(out, 'decimal_exponent'), out)

    def test_a_crash_is_a_disagreement(self):
        def boom(name, args):
            raise RuntimeError('device fault')
        rep = harness.check(self.rows[:3], boom)
        self.assertFalse(harness.all_agree(rep))


if __name__ == '__main__':
    unittest.main()
