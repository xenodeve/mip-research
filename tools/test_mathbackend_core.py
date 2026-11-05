"""tools/igc/mathbackend core: the registry, the errors and Surd. python tools/test_mathbackend_core.py (standard library only)."""
import sys
import unittest
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.igc.mathbackend import BadInput, NotCovered, Surd, op, registry, run


class SurdTests(unittest.TestCase):
    def test_normal_form(self):
        self.assertEqual(Surd(0, 1, 8), Surd(0, 2, 2))            # sqrt(8) = 2 sqrt(2)
        self.assertEqual(Surd(1, 3, 1), Surd(4))                  # a radicand 1 is rational
        self.assertEqual(Surd(5, 0, 7), Surd(5))
        self.assertTrue(Surd(0, 1, 9).is_rational())
        self.assertEqual(Surd.sqrt(Fraction(8, 9)), Surd(0, Fraction(2, 3), 2))
        self.assertEqual(Surd.sqrt(16), Surd(4))

    def test_squarefree_fast_cached_and_same_as_trial_division(self):
        # 2026-11-06: a 14-digit prime radicand took 0.6 s by trial division to its square root, paid again by every Surd on it
        import random
        import time
        from tools.igc.mathbackend import _squarefree

        def slow(d):
            s, r, p = 1, d, 2
            while p * p <= r:
                while r % (p * p) == 0:
                    r //= p * p
                    s *= p
                p += 1 if p == 2 else 2
            return s, r
        rs = random.Random(2)
        for d in list(range(1, 3000)) + [rs.randint(1, 10 ** 9) for _ in range(500)] + [rs.randint(1, 3000) ** 2 * rs.randint(1, 3000) for _ in range(300)]:
            self.assertEqual(_squarefree(d), slow(d), d)
        _squarefree.cache_clear()
        t = time.perf_counter()
        self.assertEqual(_squarefree(99999999999973), (1, 99999999999973))          # a 14-digit prime
        self.assertEqual(Surd(0, 1, 4 * 99999999999973), Surd(0, 2, 99999999999973))
        self.assertEqual(_squarefree(1000003 ** 3), (1000003, 1000003))             # past the trial bound: Pollard rho
        self.assertEqual(_squarefree(9999991 ** 2 * 9999973 * 1000003), (9999991, 9999973 * 1000003))
        self.assertLess(time.perf_counter() - t, 1)
        hits = _squarefree.cache_info().hits
        Surd(0, 3, 99999999999973)
        self.assertEqual(_squarefree.cache_info().hits, hits + 1)                     # the second Surd on it does not factor again
        t = time.perf_counter()
        with self.assertRaises(NotCovered):
            _squarefree((10 ** 12 + 39) * (10 ** 12 + 61) * (10 ** 12 + 63))        # three 13-digit primes: rho past its budget
        self.assertLess(time.perf_counter() - t, 2)

    def test_arithmetic(self):
        a, b = Surd(1, 2, 3), Surd(3, -1, 3)
        self.assertEqual(a + b, Surd(4, 1, 3))
        self.assertEqual(a * b, Surd(3 - 6, 6 - 1, 3))            # (1+2r)(3-r) = 3 - r + 6r - 2*3 = -3 + 5r
        self.assertEqual((a * b) / b, a)
        self.assertEqual(1 / Surd(0, 1, 2), Surd(0, Fraction(1, 2), 2))
        self.assertEqual(a - a, Surd(0))
        self.assertEqual(2 * a, Surd(2, 4, 3))
        with self.assertRaises(NotCovered):
            a + Surd(0, 1, 5)
        with self.assertRaises(BadInput):
            a / Surd(0)

    def test_order_is_exact(self):
        self.assertTrue(Surd(0, 1, 2) < Surd(Fraction(3, 2)))     # 1.414 < 1.5
        self.assertTrue(Surd(0, 1, 2) > Surd(Fraction(7, 5)))     # 1.41421 > 1.4
        self.assertTrue(Surd(3, -1, 10) < Surd(0))                # 3 - 3.162 < 0
        self.assertTrue(Surd(-3, 1, 10) > Surd(0))
        self.assertEqual(Surd(0, 1, 2).sign(), 1)
        self.assertEqual((Surd(1, 1, 2) * Surd(1, -1, 2)).sign(), -1)   # 1 - 2 = -1

    def test_decimal(self):
        self.assertEqual(str(Surd(0, 1, 2).to_decimal(10)), '1.414213562')
        self.assertAlmostEqual(float(Surd(1, 1, 5)), 3.2360679775, places=9)

    def test_str(self):
        self.assertEqual(str(Surd(1, 2, 3)), '1 + 2*sqrt(3)')
        self.assertEqual(str(Surd(0, 2, 3)), '2*sqrt(3)')
        self.assertEqual(str(Surd(Fraction(1, 2))), '1/2')


class RegistryTests(unittest.TestCase):
    def test_register_and_run(self):
        @op('core_test.double', kind='test', bound='any')
        def double(x):
            """double(3) -> 6"""
            return 2 * x
        self.assertEqual(run('core_test.double', 3), 6)
        self.assertIn('core_test.double', registry())
        with self.assertRaises(ValueError):
            op('core_test.double', kind='test')(double)
        with self.assertRaises(NotCovered):
            run('core_test.nothing')


if __name__ == '__main__':
    unittest.main()
