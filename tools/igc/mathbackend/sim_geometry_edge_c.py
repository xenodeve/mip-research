"""Edge cases, part 3: triangles and angles (see sim_geometry_edge.py)."""
from fractions import Fraction

from tools.igc.mathbackend.sim_geometry_edge import (BADIN, REFUSE, F, P, case, o_angle, pts)    # noqa: F401
from tools.igc.mathbackend.sim_geometry_edge import Fr, FrPt, MF, N50, S, SPt, Triangle, mp, sp   # noqa: F401


def sq(p, q):
    return sum((a - b) ** 2 for a, b in zip(p, q))


def heron_sq(p, q, r):
    """16 area^2 from the three squared sides (a different formula from the cross product): 2(ab+bc+ca) - (a^2+b^2+c^2) with a, b, c the squared sides."""
    a2, b2, c2 = sq(q, r), sq(p, r), sq(p, q)
    return (2 * (a2 * b2 + b2 * c2 + c2 * a2) - (a2 * a2 + b2 * b2 + c2 * c2)) / 16


def mp_tri(p, q, r):
    """(a, b, c, area, R, r) at 50 digits by Heron on the mpmath side lengths (2D or 3D)."""
    a, b, c = (mp.sqrt(MF(sq(s, t))) for s, t in ((q, r), (p, r), (p, q)))
    s = (a + b + c) / 2
    area = mp.sqrt(s * (s - a) * (s - b) * (s - c))
    return a, b, c, area, a * b * c / (4 * area), area / s


def several_radicals(p, q, r):
    """True when the three side lengths bring in two or more different irrational radicals (square-free parts of the squared sides, by sympy.factorint)."""
    rad = set()
    for s, t in ((q, r), (p, r), (p, q)):
        d2 = sq(s, t)
        n, m = d2.numerator, d2.denominator
        sf = 1
        for pr, e in sp.factorint(n * m).items():
            if e % 2:
                sf *= pr
        if sf != 1:
            rad.add(sf)
    return len(rad) > 1


@case('triangle_area')
def _(add):
    add((P(0, 0), P(1, 1), P(2, 2)), F(0), 'degenerate', 'collinear: area 0 (sympy refuses to build a Triangle, the shoelace sum is 0)')
    add((P(3, 3), P(3, 3), P(3, 3)), F(0), 'degenerate', 'three coincident points')
    add((P(0, 0), P(0, 0), P(5, 7)), F(0), 'degenerate', 'two coincident points')
    add((P(0, 0), P(0, 4), P(3, 0)), Fr(abs(Triangle(SPt(P(0, 0)), SPt(P(0, 4)), SPt(P(3, 0))).area)), 'boundary', 'clockwise order: the area is still positive (6)')
    add((P(0, 0), P(10 ** 20, 0), P(0, 10 ** 20)), Fr(abs(Triangle(SPt(P(0, 0)), SPt(P(10 ** 20, 0)), SPt(P(0, 10 ** 20))).area)), 'boundary', 'huge right triangle')
    add((P(F(1, 2), F(1, 3)), P(F(5, 7), F(1, 9)), P(F(-2, 3), F(4, 5))), Fr(abs(Triangle(SPt(P(F(1, 2), F(1, 3))), SPt(P(F(5, 7), F(1, 9))), SPt(P(F(-2, 3), F(4, 5)))).area)), 'boundary', 'fractions')
    add((P(0, 0), P(1, 0), P(F(1, 2), F(1, 10 ** 20))), F(1, 2 * 10 ** 20), 'boundary', 'a sliver of height 10^-20')
    add((P(0, 0, 0), P(1, 0, 0), P(0, 1, 0)), BADIN, 'bad-input', '3D points (use triangle_area_sq)')
    add((P(0, 0), P(1, 0), (0, 0.5)), BADIN, 'bad-input', 'a float')


@case('triangle_area_sq')
def _(add):
    add((P(0, 0), P(1, 1), P(2, 2)), F(0), 'degenerate', 'collinear: 0')
    add((P(1, 2, 3), P(2, 4, 6), P(3, 6, 9)), F(0), 'degenerate', 'collinear in 3D')
    add((P(1, 2, 3),) * 3, F(0), 'degenerate', 'three coincident points')
    add((P(0, 0, 0), P(1, 0, 0), P(0, 1, 0)), heron_sq(P(0, 0, 0), P(1, 0, 0), P(0, 1, 0)), 'boundary', '3D right triangle: 1/4')
    add((P(1, 0, 0), P(0, 1, 0), P(0, 0, 1)), heron_sq(P(1, 0, 0), P(0, 1, 0), P(0, 0, 1)), 'boundary', 'a face of the unit simplex: area^2 = 3/4')
    add((P(0, 0), P(4, 0), P(0, 3)), F(36), 'boundary', 'the 3-4-5 triangle has area 6, squared 36 (hand)')
    add((P(F(1, 2), F(1, 3), F(1, 5)), P(F(2, 7), 0, 1), P(0, F(3, 4), F(-1, 3))), heron_sq(P(F(1, 2), F(1, 3), F(1, 5)), P(F(2, 7), 0, 1), P(0, F(3, 4), F(-1, 3))), 'boundary', 'fractions in 3D')
    add((P(0, 0), P(1, 0), P(0, 1, 0)), BADIN, 'bad-input', 'mixed dimensions')
    add((P(0, 0), P(1, 0), (0, 0, 0, 0)), BADIN, 'bad-input', 'four coordinates')


@case('triangle_from_sides')
def _(add):
    def sy(a, b, c):
        t = Triangle(sss=(S(a), S(b), S(c)))
        return dict(area=N50(t.area), cr=N50(t.circumradius), ir=N50(t.inradius), a=a, b=b, c=c)
    add((F(1), F(1), F(2)), BADIN, 'degenerate', 'a + b = c: a flat triangle')
    add((F(3), F(4), F(7)), BADIN, 'degenerate', 'a + b = c')
    add((F(7), F(3), F(4)), BADIN, 'degenerate', 'a = b + c (the long side first)')
    add((F(1), F(5), F(1)), BADIN, 'degenerate', 'b > a + c')
    add((F(0), F(1), F(1)), BADIN, 'degenerate', 'a zero side')
    add((F(-3), F(4), F(5)), BADIN, 'bad-input', 'a negative side')
    add((F(1), F(1), F(1)), sy(F(1), F(1), F(1)), 'boundary', 'equilateral')
    add((F(3), F(4), F(5)), sy(F(3), F(4), F(5)), 'boundary', 'right triangle: area 6, R = 5/2, r = 1')
    add((F(10 ** 6), F(10 ** 6), F(1)), sy(F(10 ** 6), F(10 ** 6), F(1)), 'boundary', 'a needle: two long sides, one short')
    add((F(1, 2), F(1, 3), F(1, 4)), sy(F(1, 2), F(1, 3), F(1, 4)), 'boundary', 'fractions')
    add((F(10 ** 9 + 7), F(10 ** 9 + 7), F(10 ** 9 + 7)), REFUSE, 'past-bound', 'an equilateral triangle on a prime side beyond 10^7: the square-free part of 3 a^4 / 16 needs trial division past 10^7')
    add((F(1), F(1), 'x'), BADIN, 'bad-input', 'a non-numeric side')
    add((1.5, F(1), F(1)), BADIN, 'bad-input', 'a float')


@case('triangle_from_sides_sq')
def _(add):
    def sy(a2, b2, c2):
        t = Triangle(sss=(sp.sqrt(S(a2)), sp.sqrt(S(b2)), sp.sqrt(S(c2))))
        return (N50(t.area ** 2), N50(t.circumradius ** 2))
    add((F(1), F(1), F(4)), BADIN, 'degenerate', 'sides 1, 1, 2: 2(1+4+4) - (1+1+16) = 0, flat')
    add((F(2), F(2), F(8)), BADIN, 'degenerate', 'sides sqrt2, sqrt2, 2sqrt2: flat')
    add((F(1), F(4), F(9)), BADIN, 'degenerate', 'sides 1, 2, 3: flat')
    add((F(1), F(1), F(5)), BADIN, 'degenerate', 'the long side is too long')
    add((F(0), F(1), F(1)), BADIN, 'degenerate', 'a zero squared side')
    add((F(-1), F(1), F(1)), BADIN, 'bad-input', 'a negative squared side')
    add((F(1), F(1), F(1)), sy(F(1), F(1), F(1)), 'boundary', 'equilateral: area^2 = 3/16')
    add((F(1), F(1), F(2)), sy(F(1), F(1), F(2)), 'boundary', 'the right isosceles triangle with legs 1')
    add((F(3), F(5), F(7)), sy(F(3), F(5), F(7)), 'boundary', 'three different radicals as sides (area^2 stays rational)')
    add((F(1, 3), F(1, 5), F(1, 7)), sy(F(1, 3), F(1, 5), F(1, 7)), 'boundary', 'fractions')
    add((F(10 ** 30), F(10 ** 30), F(10 ** 30 + 1)), sy(F(10 ** 30), F(10 ** 30), F(10 ** 30 + 1)), 'boundary', 'huge squared sides, exact rational result')
    add((F(1), F(1), 'x'), BADIN, 'bad-input', 'a non-numeric argument')


@case('circumradius_sq')
def _(add):
    def mp_r2(p, q, r):
        a, b, c, area, R, ri = mp_tri(p, q, r)
        return str(R * R)
    add((P(0, 0), P(1, 1), P(2, 2)), BADIN, 'degenerate', 'collinear')
    add((P(1, 1), P(1, 1), P(5, 7)), BADIN, 'degenerate', 'two coincident points')
    add((P(1, 2, 3), P(2, 4, 6), P(3, 6, 9)), BADIN, 'degenerate', 'collinear in 3D')
    add((P(0, 0), P(4, 0), P(0, 3)), mp_r2(P(0, 0), P(4, 0), P(0, 3)), 'boundary', 'right triangle: R is half the hypotenuse, R^2 = 25/4')
    add((P(0, 0, 0), P(1, 0, 0), P(0, 1, 0)), mp_r2(P(0, 0, 0), P(1, 0, 0), P(0, 1, 0)), 'boundary', '3D right triangle: R^2 = 1/2')
    add((P(1, 0, 0), P(0, 1, 0), P(0, 0, 1)), mp_r2(P(1, 0, 0), P(0, 1, 0), P(0, 0, 1)), 'boundary', 'a face of the unit simplex: R^2 = 2/3')
    add((P(0, 0), P(10, 0), P(5, 1)), mp_r2(P(0, 0), P(10, 0), P(5, 1)), 'boundary', 'a very obtuse triangle')
    add((P(F(1, 3), F(1, 4)), P(F(5, 7), F(2, 3)), P(F(-3, 5), F(1, 9))), mp_r2(P(F(1, 3), F(1, 4)), P(F(5, 7), F(2, 3)), P(F(-3, 5), F(1, 9))), 'boundary', 'fractions')
    add((P(0, 0), P(1, 0), P(0, 1, 0)), BADIN, 'bad-input', 'mixed dimensions')


@case('circumradius')
def _(add):
    def sy(p, q, r):
        t = Triangle(SPt(p), SPt(q), SPt(r))
        return (str(sp.N(t.circumradius, 55)), Fr(t.circumradius ** 2))
    add((P(0, 0), P(1, 1), P(2, 2)), BADIN, 'degenerate', 'collinear')
    add((P(0, 0), P(0, 0), P(1, 0)), BADIN, 'degenerate', 'two coincident points')
    add((P(0, 0), P(4, 0), P(0, 3)), sy(P(0, 0), P(4, 0), P(0, 3)), 'boundary', 'right triangle: R = 5/2 exactly')
    add((P(0, 0), P(2, 0), P(1, 1)), sy(P(0, 0), P(2, 0), P(1, 1)), 'boundary', 'right isosceles: R = 1')
    add((P(0, 0), P(10, 0), P(5, 1)), sy(P(0, 0), P(10, 0), P(5, 1)), 'boundary', 'R^2 = 169: R = 13')
    add((P(0, 0), P(1, 0), P(F(1, 2), 3)), sy(P(0, 0), P(1, 0), P(F(1, 2), 3)), 'boundary', 'an isosceles needle with fractions')
    add((P(0, 0), P(10 ** 8, 0), P(0, 10 ** 8 + 1)), REFUSE, 'past-bound', 'R^2 has a prime factor of size 10^16 beyond the trial division')
    add((P(0, 0, 0), P(1, 0, 0), P(0, 1, 0)), (str(sp.N(sp.sqrt(sp.Rational(1, 2)), 55)), F(1, 2)), 'boundary', '3D right isosceles with legs 1: R = sqrt(2)/2')


@case('inradius')
def _(add):
    def orc(p, q, r):
        a, b, c, area, R, ri = mp_tri(p, q, r)
        u, v = [x - y for x, y in zip(q, p)], [x - y for x, y in zip(r, p)]
        return REFUSE if several_radicals(p, q, r) else (str(ri), None)
    def add2(args, tag, desc):
        add(args, orc(*args), tag, 'mpmath Heron r = area / s at 50 digits; ' + desc)
    add((P(0, 0), P(1, 1), P(2, 2)), BADIN, 'degenerate', 'collinear')
    add((P(0, 0), P(0, 0), P(1, 0)), BADIN, 'degenerate', 'two coincident points')
    add2((P(0, 0), P(4, 0), P(0, 3)), 'boundary', 'the 3-4-5 triangle: r = 1')
    add2((P(0, 0), P(1, 0), P(0, 1)), 'boundary', 'right isosceles: r = (2 - sqrt(2))/2 (one radical)')
    add2((P(0, 0), P(5, 0), P(0, 12)), 'boundary', '5-12-13: r = 2')
    add2((P(0, 0, 0), P(3, 0, 0), P(0, 4, 0)), 'boundary', '3D right triangle 3-4-5')
    add2((P(0, 0), P(1, 1), P(3, 0)), 'degenerate', 'sides sqrt2, sqrt5, 3: two radicals, the exact form is refused')
    add2((P(0, 0), P(2, 1), P(1, 3)), 'degenerate', 'sides sqrt5, sqrt5, sqrt10: two radicals')
    add2((P(F(1, 2), 0), P(F(5, 2), 0), P(F(1, 2), F(3, 2))), 'boundary', 'a right triangle with fractions (2 x 3/2 x 5/2)')
    add((P(0, 0), P(10 ** 8, 0), P(0, 10 ** 8 + 1)), REFUSE, 'past-bound', 'the sides need radicals beyond the 10^7 trial division')
    add((P(0, 0), P(1, 0), P(0, 1, 0)), BADIN, 'bad-input', 'mixed dimensions')


@case('inradius_numeric')
def _(add):
    def orc(p, q, r, dg, dps=50):
        with mp.workdps(dps):
            a, b, c, area, R, ri = mp_tri(p, q, r)
            return (str(ri), dg)
    add((P(0, 0), P(1, 1), P(2, 2), 10), BADIN, 'degenerate', 'collinear')
    add((P(1, 1), P(1, 1), P(2, 3), 10), BADIN, 'degenerate', 'two coincident points')
    add((P(0, 0), P(1, 0), P(0, 1), 12), orc(P(0, 0), P(1, 0), P(0, 1), 12), 'boundary', 'right isosceles, one radical')
    add((P(0, 0), P(1, 1), P(3, 0), 12), orc(P(0, 0), P(1, 1), P(3, 0), 12), 'boundary', 'two radicals: the numeric form answers where the exact one refuses')
    add((P(0, 0), P(0, 4), P(3, 0), 12), orc(P(0, 0), P(0, 4), P(3, 0), 12), 'boundary', 'clockwise order (the radius stays positive)')
    add((P(0, 0, 0), P(1, 2, 3), P(-2, 1, 5), 20), orc(P(0, 0, 0), P(1, 2, 3), P(-2, 1, 5), 20), 'boundary', '3D triangle')
    add((P(0, 0), P(1, 0), P(F(1, 2), F(1, 10 ** 10)), 25), orc(P(0, 0), P(1, 0), P(F(1, 2), F(1, 10 ** 10)), 25, 80), 'boundary', 'a sliver of height 10^-10 (Heron at 80 digits)')
    add((P(0, 0), P(1, 0), P(0, 1), 0), orc(P(0, 0), P(1, 0), P(0, 1), 0), 'boundary', 'zero digits: rounds to an integer (0)')
    add((P(0, 0), P(4, 0), P(0, 3), 2000), orc(P(0, 0), P(4, 0), P(0, 3), 2000, 2030), 'past-bound', '2000 digits, exactly the bound: answered (r = 1.000...)')
    add((P(0, 0), P(4, 0), P(0, 3), 2001), REFUSE, 'past-bound', '2001 digits: past the bound')
    add((P(0, 0), P(4, 0), P(0, 3), -1), BADIN, 'bad-input', 'negative digits')
    add((P(0, 0), P(4, 0), P(0, 3), 5.0), BADIN, 'bad-input', 'digits as a float')


@case('incenter')
def _(add):
    def orc(p, q, r):
        if several_radicals(p, q, r):
            return REFUSE
        ic = Triangle(SPt(p), SPt(q), SPt(r)).incenter
        return (N50(ic.x), N50(ic.y))
    add((P(0, 0), P(1, 1), P(2, 2)), BADIN, 'degenerate', 'collinear')
    add((P(1, 1), P(1, 1), P(5, 5)), BADIN, 'degenerate', 'coincident')
    for t, tag, d in [((P(0, 0), P(4, 0), P(0, 3)), 'boundary', 'the 3-4-5 triangle: (1, 1)'), ((P(0, 0), P(1, 0), P(0, 1)), 'boundary', 'right isosceles, one radical'),
                      ((P(0, 0), P(6, 0), P(3, 4)), 'boundary', 'isosceles 5-5-6'), ((P(0, 0), P(1, 1), P(3, 0)), 'degenerate', 'two radicals: refused'),
                      ((P(0, 0), P(2, 1), P(1, 3)), 'degenerate', 'two radicals: refused'), ((P(-5, -5), P(-1, -5), P(-5, -2)), 'boundary', 'a 3-4-5 triangle far from the origin'),
                      ((P(F(1, 2), 0), P(F(5, 2), 0), P(F(1, 2), F(3, 2))), 'boundary', 'a right triangle with fractions')]:
        add(t, orc(*t), tag, 'sympy Triangle.incenter at 55 digits: ' + d)
    add((P(0, 0), P(10 ** 8, 0), P(0, 10 ** 8 + 1)), REFUSE, 'past-bound', 'the sides need radicals beyond the 10^7 trial division')
    add((P(0, 0, 0), P(1, 0, 0), P(0, 1, 0)), BADIN, 'bad-input', '3D points: the incentre is for the plane')


@case('incenter_numeric')
def _(add):
    def orc(p, q, r, dg):
        ic = Triangle(SPt(p), SPt(q), SPt(r)).incenter
        return ((str(sp.N(ic.x, 70)), str(sp.N(ic.y, 70))), dg)
    add((P(0, 0), P(1, 1), P(2, 2), 10), BADIN, 'degenerate', 'collinear')
    add((P(1, 1), P(1, 1), P(5, 5), 10), BADIN, 'degenerate', 'coincident')
    for t, dg, d in [((P(0, 0), P(4, 0), P(0, 3)), 3, 'the 3-4-5 triangle'), ((P(0, 0), P(1, 1), P(3, 0)), 20, 'two radicals'), ((P(0, 0), P(2, 1), P(1, 3)), 30, 'two radicals'),
                     ((P(0, 0), P(0, 4), P(3, 0)), 12, 'clockwise'), ((P(F(1, 2), F(1, 3)), P(F(5, 7), F(1, 9)), P(F(-2, 3), F(4, 5))), 25, 'fractions'),
                     ((P(0, 0), P(1, 0), P(0, 1)), 0, 'zero digits')]:
        add(t + (dg,), orc(*t, dg), 'boundary', 'sympy Triangle.incenter at 70 digits: ' + d)
    add((P(0, 0), P(4, 0), P(0, 3), 2001), REFUSE, 'past-bound', '2001 digits: past the bound')
    add((P(0, 0), P(4, 0), P(0, 3), -1), BADIN, 'bad-input', 'negative digits')
    add((P(0, 0, 0), P(1, 0, 0), P(0, 1, 0), 5), BADIN, 'bad-input', '3D points')


@case('centroid')
def _(add):
    add((P(3, 4),), (F(3), F(4)), 'degenerate', 'one point is its own centroid')
    add((P(1, 1), P(1, 1), P(4, 4)), (F(2), F(2)), 'degenerate', 'a repeated point counts twice: (1 + 1 + 4) / 3 (hand)')
    add((P(0, 0), P(1, 1)), (F(1, 2), F(1, 2)), 'degenerate', 'two points: the midpoint (not floor division)')
    add((P(-1, -1), P(-2, -4), P(0, 0)), (F(-1), F(-5, 3)), 'boundary', 'negative coordinates, a non-integer mean')
    add((P(0, 0, 0), P(1, 2, 3), P(2, 4, 9)), (F(1), F(2), F(4)), 'boundary', '3D, the third coordinate counts')
    add((P(0, 0, 0), P(1, 0, 0), P(0, 1, 0), P(0, 0, 1)), (F(1, 4), F(1, 4), F(1, 4)), 'boundary', 'a tetrahedron: (1/4, 1/4, 1/4)')
    add((P(F(1, 3), F(1, 5)), P(F(1, 6), F(2, 5))), (F(1, 4), F(3, 10)), 'boundary', 'fractions')
    add((), BADIN, 'bad-input', 'no points')
    add((P(0, 0), P(1, 1, 1)), BADIN, 'bad-input', 'mixed dimensions')
    add(tuple(P(i, 0) for i in range(5001)), REFUSE, 'past-bound', '5001 points')
    add(tuple(P(i, -i) for i in range(5000)), (F(4999, 2), F(-4999, 2)), 'past-bound', '5000 points, exactly the bound: the mean of 0..4999 is 4999/2 (hand)')


@case('circumcenter')
def _(add):
    def sy(p, q, r):
        return FrPt(Triangle(SPt(p), SPt(q), SPt(r)).circumcenter)
    add((P(0, 0), P(1, 1), P(2, 2)), BADIN, 'degenerate', 'collinear')
    add((P(2, 2), P(2, 2), P(3, 5)), BADIN, 'degenerate', 'coincident points')
    add((P(0, 0), P(4, 0), P(0, 3)), sy(P(0, 0), P(4, 0), P(0, 3)), 'boundary', 'right angle: the midpoint of the hypotenuse (2, 3/2)')
    add((P(0, 0), P(10, 0), P(5, 1)), sy(P(0, 0), P(10, 0), P(5, 1)), 'boundary', 'obtuse: outside the triangle')
    add((P(-1, 0), P(1, 0), P(0, 10 ** 10)), sy(P(-1, 0), P(1, 0), P(0, 10 ** 10)), 'boundary', 'a tall needle')
    add((P(F(1, 2), 0), P(0, F(1, 3)), P(F(2, 5), F(7, 3))), sy(P(F(1, 2), 0), P(0, F(1, 3)), P(F(2, 5), F(7, 3))), 'boundary', 'fractions')
    add((P(0, 0, 0), P(1, 0, 0), P(0, 1, 0)), BADIN, 'bad-input', '3D points')


@case('orthocenter')
def _(add):
    def sy(p, q, r):
        return FrPt(Triangle(SPt(p), SPt(q), SPt(r)).orthocenter)
    add((P(0, 0), P(1, 1), P(2, 2)), BADIN, 'degenerate', 'collinear')
    add((P(2, 2), P(2, 2), P(3, 5)), BADIN, 'degenerate', 'coincident points')
    add((P(0, 0), P(4, 0), P(0, 3)), sy(P(0, 0), P(4, 0), P(0, 3)), 'boundary', 'right angle: the orthocentre is the right-angle vertex (0, 0)')
    add((P(0, 0), P(10, 0), P(5, 1)), sy(P(0, 0), P(10, 0), P(5, 1)), 'boundary', 'obtuse: outside the triangle')
    add((P(0, 0), P(2, 0), P(1, 5)), sy(P(0, 0), P(2, 0), P(1, 5)), 'boundary', 'isosceles: on the axis of symmetry')
    add((P(F(1, 2), 0), P(0, F(1, 3)), P(F(2, 5), F(7, 3))), sy(P(F(1, 2), 0), P(0, F(1, 3)), P(F(2, 5), F(7, 3))), 'boundary', 'fractions')
    add((P(-10 ** 12, 3), P(10 ** 12, 5), P(7, 10 ** 12)), sy(P(-10 ** 12, 3), P(10 ** 12, 5), P(7, 10 ** 12)), 'boundary', 'huge coordinates')
    add((P(0, 0, 0), P(1, 0, 0), P(0, 1, 0)), BADIN, 'bad-input', '3D points')


# --- angles

def cos_orc(u, v):
    uu, vv, d = sum(a * a for a in u), sum(a * a for a in v), sum(a * b for a, b in zip(u, v))
    return (str(MF(d) / mp.sqrt(MF(uu) * MF(vv))), d, uu * vv)


@case('angle_cos')
def _(add):
    add((P(0, 0), P(1, 1)), (BADIN, 'a zero vector has no direction'), 'degenerate', 'a zero vector has no direction')
    add((P(1, 1), P(0, 0)), (BADIN, 'a zero vector has no direction'), 'degenerate', 'the second vector is zero')
    add((P(0, 0, 0), P(0, 0, 0)), BADIN, 'degenerate', 'both vectors are zero')
    for u, v, d in [((1, 0), (2, 0), 'parallel: cos 1'), ((1, 0), (-3, 0), 'antiparallel: cos -1'), ((1, 0), (0, 5), 'perpendicular: cos 0'),
                    ((1, 2, 3), (-2, -4, -6), 'antiparallel in 3D'), ((1, 1, 0), (0, 1, 1), '60 degrees in 3D: cos 1/2'), ((1, 0), (1, 1), '45 degrees: sqrt(2)/2'),
                    ((1, 0), (-1, 1), '135 degrees'), ((1, 0), (1, F(1, 1000)), 'almost parallel'), ((10 ** 20, 0), (10 ** 20, 10 ** 20), 'huge, 45 degrees')]:
        u, v = P(*u), P(*v)
        add((u, v), cos_orc(u, v), 'boundary', 'mpmath dot / (|u||v|): ' + d)
    add((P(1, 0), (0, 0.5)), BADIN, 'bad-input', 'a float')
    add((P(1, 0), P(0, 1, 0)), BADIN, 'bad-input', 'mixed dimensions')
    add((P(10 ** 9, 1), P(10 ** 9 + 2, 1)), REFUSE, 'past-bound', '|u|^2 |v|^2 is a product of two large primes beyond the 10^7 trial division')


@case('angle_degrees_exact')
def _(add):
    for deg, (u, v) in {0: ((1, 0), (2, 0)), 45: ((1, 0), (1, 1)), 90: ((1, 0), (0, 1)), 135: ((1, 0), (-1, 1)), 180: ((1, 0), (-1, 0)), 150: ((-3, -3, 0), (1, 2, -1)), 30: ((3, 3, 0), (1, 2, -1))}.items():
        add((P(*u), P(*v)), deg, 'boundary', 'the table angle %d (hand: checked against mpmath below in 3D, here by construction)' % deg)
    for u, v in [((1, 1, 0), (1, 0, 1)), ((1, 1, 0), (-1, 0, -1)), ((1, 0, 0), (1, 1, 1)), ((1, 0, 0), (-1, 1, 1)), ((1, 1, 1), (1, 1, 0)), ((1, 1, 1), (-1, -1, 0)),
                 ((0, 1, 1), (2, 1, 1)), ((3, 3, 0), (0, 3, 3)), ((1, 2, 3), (1, 2, 3)), ((1, 2, 3), (-1, -2, -3)), ((1, 2, 3), (2, -1, 0))]:
        u, v = P(*u), P(*v)
        ang = o_angle(u, v)
        hit = next((t for t in (0, 30, 45, 60, 90, 120, 135, 150, 180) if abs(ang - t) < mp.mpf(10) ** -35), None)
        add((u, v), REFUSE if hit is None else hit, 'boundary', 'mpmath atan2 in degrees, matched to the table within 1e-35 (else the backend must refuse)')
    add((P(1, 0), P(1, F(1, 10 ** 20))), REFUSE, 'degenerate', 'almost parallel but not parallel: not in the table')
    add((P(0, 0), P(1, 1)), BADIN, 'degenerate', 'a zero vector')
    add((P(1, 0), P(1, 1, 1)), BADIN, 'bad-input', 'mixed dimensions')
    add((P(1, 0), (0, 1.0)), BADIN, 'bad-input', 'a float')
    add((P(10 ** 9, 1), P(10 ** 9 + 2, 1)), REFUSE, 'past-bound', '|u|^2 |v|^2 beyond the 10^7 trial division (the angle is small and not in the table either)')


@case('angle_degrees_numeric')
def _(add):
    def orc(u, v, dg, dps=50):
        with mp.workdps(dps):
            return (str(o_angle(u, v)), dg)
    add((P(0, 0), P(1, 1), 5), BADIN, 'degenerate', 'a zero vector')
    for u, v, dg, d in [((1, 0), (2, 0), 8, 'parallel: 0'), ((1, 0), (-2, 0), 8, 'antiparallel: 180'), ((1, 0), (0, 3), 8, 'perpendicular: 90'), ((1, 0), (1, 1), 5, '45'),
                        ((1, 2, 3), (4, 5, 6), 20, '3D'), ((1, 0), (1, F(1, 10 ** 12)), 25, 'almost parallel: 5.7e-11 degrees'),
                        ((1, 0), (-1, F(1, 10 ** 12)), 25, 'almost antiparallel: 180 - 5.7e-11'), ((10 ** 20, 0), (10 ** 20, 1), 25, 'huge'), ((1, 0), (0, 1), 0, 'zero digits: 90')]:
        u, v = P(*u), P(*v)
        add((u, v, dg), orc(u, v, dg), 'boundary', 'mpmath atan2(|u x v|, u.v) at 50 digits: ' + d)
    add((P(1, 0), P(0, 1), 2000), orc(P(1, 0), P(0, 1), 2000, 2030), 'past-bound', '2000 digits, exactly the bound: answered (90.000...)')
    add((P(1, 0), P(0, 1), 2001), REFUSE, 'past-bound', '2001 digits: past the bound')
    add((P(1, 0), P(0, 1), -5), BADIN, 'bad-input', 'negative digits')
    add((P(1, 0), P(0, 1, 1), 5), BADIN, 'bad-input', 'mixed dimensions')


@case('triangle_angles_numeric')
def _(add):
    def orc(p, q, r, dg, dps=50):
        with mp.workdps(dps):
            return (tuple(str(o_angle([x - y for x, y in zip(b, a)], [x - y for x, y in zip(c, a)])) for a, b, c in ((p, q, r), (q, p, r), (r, p, q))), dg)
    for t, dg, d in [((P(0, 0), P(1, 1), P(2, 2)), 8, 'collinear: angles 0, 180, 0 (the middle point sees a straight angle)'), ((P(0, 0), P(4, 0), P(0, 3)), 10, 'right triangle'),
                     ((P(0, 0), P(1, 0), P(F(1, 2), F(1, 10 ** 8))), 25, 'a sliver: two angles near 0'), ((P(0, 0, 0), P(1, 2, 3), P(-2, 1, 5)), 20, '3D'),
                     ((P(0, 0), P(0, 4), P(3, 0)), 10, 'clockwise: the angles are unsigned'), ((P(0, 0), P(10, 0), P(5, F(1, 1000))), 30, 'nearly flat, obtuse')]:
        add(t + (dg,), orc(*t, dg), 'boundary' if 'collinear' not in d else 'degenerate', 'mpmath at each vertex: ' + d)
    add((P(1, 1), P(1, 1), P(2, 5), 5), BADIN, 'degenerate', 'two coincident vertices: a zero vector')
    add((P(0, 0), P(1, 0), P(0, 1), 2001), REFUSE, 'past-bound', '2001 digits')
    add((P(0, 0), P(1, 0), P(0, 1, 0), 5), BADIN, 'bad-input', 'mixed dimensions')
