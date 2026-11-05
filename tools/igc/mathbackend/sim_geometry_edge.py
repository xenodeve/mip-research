"""Edge cases for the simulation of tools/igc/mathbackend/geometry.py: the inputs a random generator seldom or never draws, each with a named independent oracle.

Four kinds, written in the case's oracle name as `EDGE[kind] ...` so that the report can count them per operation:
  degenerate   collinear, zero-length, tangent, coincident, concentric, touching, zero-area, contained, parallel (the backend must answer exactly or say BadInput)
  bad-input    malformed input (a float, a bool, wrong dimension, a negative count, a point as a number): BadInput, never a crash and never a wrong number
  past-bound   just past the documented bound of the operation: NotCovered, never a guess (and the case just inside the bound where it is cheap enough to run)
  boundary     values on the edge of a rule (a point on an edge, an angle at a table value, a tangent length of 0, a cosine of +-1)
The oracles are SymPy objects, mpmath at 50 digits, an explicit brute force written in sim_geometry.py, or a hand-worked identity named in the case (a theorem or a
definition, not the algorithm of the backend). `REFUSE` and `BADIN` are the expectations that the backend refuses or rejects the input."""
import math
from decimal import Decimal
from fractions import Fraction

from tools.igc.mathbackend import sim_geometry as G
from tools.igc.mathbackend.sim_geometry import (BADIN, REFUSE, FrPt, Fr, MF, N50, S, SM, SPt, mp, sp, Circle, Line, Line3D, Plane, Point, Point3D, Polygon, Segment, Triangle,
                                                convex_hull)

F = Fraction
B = {}          # operation (without the 'geometry.') -> builder(add)


def case(*names):
    def deco(fn):
        for n in names:
            B[n] = fn
        return fn
    return deco


def P(*c):
    return tuple(F(x) for x in c)


def pts(*rows):
    return [P(*r) for r in rows]


# ----------------------------------------------------------------------------------------------------------------------------- shared oracles

def o_dist(p, q):
    d = Point(*map(S, p)).distance(Point(*map(S, q)))
    return (str(sp.N(d, 55)), sum((a - b) ** 2 for a, b in zip(p, q)))


def o_dpl(p, a, b):
    d = Line(SPt(a), SPt(b)).distance(SPt(p)) if len(p) == 2 else Line3D(SPt(a), SPt(b)).distance(SPt(p))
    return (str(sp.N(d, 55)), Fr(d ** 2))


def o_rel(c1, r1, c2, r2):
    d = mp.sqrt(sum((MF(x) - MF(y)) ** 2 for x, y in zip(c1, c2)))
    R1, R2 = mp.sqrt(MF(r1)), mp.sqrt(MF(r2))
    tol = mp.mpf(10) ** -40
    if tuple(c1) == tuple(c2) and r1 == r2:
        return 'identical'
    if abs(d - (R1 + R2)) < tol:
        return 'external tangent'
    if d > R1 + R2:
        return 'separate'
    if abs(d - abs(R1 - R2)) < tol:
        return 'internal tangent'
    if d < abs(R1 - R2):
        return 'contained'
    return 'intersecting'


def o_overlap(c1, r1, c2, r2, dg):
    d = mp.sqrt(sum((MF(a) - MF(b)) ** 2 for a, b in zip(c1, c2)))
    R1, R2 = mp.sqrt(MF(r1)), mp.sqrt(MF(r2))
    if d >= R1 + R2:
        area = mp.mpf(0)
    elif d <= abs(R1 - R2):
        area = mp.pi * min(R1, R2) ** 2
    else:
        lo, hi = max(-R1, d - R2), min(R1, d + R2)
        x0 = (d * d + R1 * R1 - R2 * R2) / (2 * d)
        f = lambda x: 2 * min(mp.sqrt(max(R1 * R1 - x * x, 0)), mp.sqrt(max(R2 * R2 - (x - d) ** 2, 0)))
        area = mp.quad(f, [lo, x0, hi])
    return (str(area), dg)


def o_circ_pts(inter):
    return [(N50(a), N50(b)) for a, b in (pt.args for pt in inter)]


def o_tri_circ(p, q, r):
    t = Triangle(SPt(p), SPt(q), SPt(r))
    return t


def o_angle(u, v):
    return G.mp_angle_deg(u, v)


# ----------------------------------------------------------------------------------------------------------------------------- points, vectors, lines

@case('dist2')
def _(add):
    add(((0, 0), (0, 0)), F(0), 'degenerate', 'zero length, sympy Point.distance')
    add(((F(1, 3), F(2, 7)), (F(1, 3), F(2, 7))), F(0), 'degenerate', 'zero length with fractions')
    add(((0, 0, 0), (0, 0, 0)), F(0), 'degenerate', 'zero length in 3D')
    add(((10 ** 30, -10 ** 30), (-10 ** 30, 10 ** 30)), Fr(Point(10 ** 30, -10 ** 30).distance(Point(-10 ** 30, 10 ** 30)) ** 2), 'boundary', 'coordinates of 10^30, sympy')
    add((('1/2', 0), (Decimal('1.5'), 0)), F(1), 'boundary', 'a string and a Decimal are read exactly: 1/2 and 3/2 are 1 apart')
    add((('0.1', '0.2'), (0, 0)), Fr(Point(sp.Rational(1, 10), sp.Rational(1, 5)).distance(Point(0, 0)) ** 2), 'boundary', "the decimal string '0.1' is exactly 1/10, not a float")
    add(((0.5, 0), (1, 1)), BADIN, 'bad-input', 'a float is not exact')
    add(((0, 0), (1, 1, 1)), BADIN, 'bad-input', 'a 2D point and a 3D point')
    add(((0, 0, 0, 0), (1, 1, 1, 1)), BADIN, 'bad-input', 'four coordinates')
    add(((0,), (1,)), BADIN, 'bad-input', 'one coordinate')
    add(((True, 0), (1, 1)), BADIN, 'bad-input', 'a bool is no coordinate')
    add(((0, 0), 'x'), BADIN, 'bad-input', 'a string is no point')
    add((('a', 0), (1, 1)), BADIN, 'bad-input', 'a non-numeric string')
    add(((0, 0), None), BADIN, 'bad-input', 'None is no point')


@case('dist')
def _(add):
    add(((0, 0), (0, 0)), o_dist((0, 0), (0, 0)), 'degenerate', 'zero length: the Surd 0')
    add(((0, 0), (3, 4)), o_dist((0, 0), (3, 4)), 'boundary', 'a perfect square (5)')
    add(((0, 0), (12, 0)), o_dist((0, 0), (12, 0)), 'boundary', 'a square factor 144 = 12^2')
    add(((0, 0), (2, 2)), o_dist((0, 0), (2, 2)), 'boundary', '2 sqrt(2): the square factor is taken out')
    add(((F(1, 2), 0), (0, F(1, 3))), o_dist((F(1, 2), 0), (0, F(1, 3))), 'boundary', 'a rational d2 = 13/36: sqrt(13)/6')
    add(((0, 0, 0), (1, 1, 1)), o_dist((0, 0, 0), (1, 1, 1)), 'boundary', 'the diagonal of the unit cube')
    add(((0, 0), (10 ** 7, 1)), o_dist((0, 0), (10 ** 7, 1)), 'boundary', 'd2 = 10^14 + 1: just inside the trial-division bound of 10^7')
    add(((0, 0), (10 ** 7 + 19, 10 ** 7 + 19)), REFUSE, 'past-bound', 'd2 = 2 * (10^7 + 19)^2: the square-free part needs a trial division past 10^7')
    add(((0, 0), (10 ** 8, 1)), REFUSE, 'past-bound', 'd2 = 10^16 + 1')
    add(((0, 0), (10 ** 30, 10 ** 30 + 1)), REFUSE, 'past-bound', 'd2 about 2 * 10^60')
    add(((0, 0), (1.0, 1)), BADIN, 'bad-input', 'a float')
    add(((0, 0), (1, 1, 1)), BADIN, 'bad-input', 'mixed dimensions')


@case('midpoint')
def _(add):
    add(((0, 0), (0, 0)), (F(0), F(0)), 'degenerate', 'the same point twice')
    add(((1, 1), (2, 2)), FrPt(Point(1, 1).midpoint(Point(2, 2))), 'boundary', 'odd integer sums: 3/2 (not floor division)')
    add(((-1, -1), (-2, -4)), FrPt(Point(-1, -1).midpoint(Point(-2, -4))), 'boundary', 'negative odd sums: -3/2, -5/2')
    add(((0, 0, 1), (0, 0, 2)), FrPt(Point(0, 0, 1).midpoint(Point(0, 0, 2))), 'boundary', '3D: the third coordinate counts')
    add(((F(1, 3), 0, 5), (F(2, 3), 1, 6)), FrPt(Point(sp.Rational(1, 3), 0, 5).midpoint(Point(sp.Rational(2, 3), 1, 6))), 'boundary', '3D with fractions')
    add(((0, 0), (1, 1, 1)), BADIN, 'bad-input', 'mixed dimensions')
    add(((0.5, 0), (1, 1)), BADIN, 'bad-input', 'a float')
    add(((0, 0), (1,)), BADIN, 'bad-input', 'one coordinate')


@case('dot')
def _(add):
    add(((0, 0), (5, 7)), F(0), 'degenerate', 'the zero vector')
    add(((1, 0, 0), (0, 0, 5)), F(0), 'boundary', '3D perpendicular, the third coordinate is the only non-zero product')
    add(((1, 2, 3), (-4, 5, -6)), F(1 * -4 + 2 * 5 + 3 * -6), 'boundary', '3D: all three terms count, negative result (no abs)')
    add(((F(1, 2), F(1, 3)), (F(1, 5), F(1, 7))), Fr(sp.Matrix([sp.Rational(1, 2), sp.Rational(1, 3)]).dot(sp.Matrix([sp.Rational(1, 5), sp.Rational(1, 7)]))), 'boundary', 'fractions, sympy dot')
    add(((1, 2), (3, 4, 5)), BADIN, 'bad-input', 'mixed dimensions')
    add(((1, 2), (1.5, 2)), BADIN, 'bad-input', 'a float')


@case('cross')
def _(add):
    add(((2, 4), (1, 2)), F(0), 'degenerate', 'parallel 2D vectors: 0')
    add(((1, 2, 3), (2, 4, 6)), (F(0), F(0), F(0)), 'degenerate', 'parallel 3D vectors: the zero vector')
    add(((0, 0, 0), (1, 2, 3)), (F(0), F(0), F(0)), 'degenerate', 'the zero vector')
    add(((1, 2), (3, 4)), Fr(sp.Matrix([[1, 2], [3, 4]]).det()), 'boundary', '2D scalar is x1 y2 - x2 y1 = -2 (sign)')
    add(((1, 2, 3), (4, 5, 6)), tuple(Fr(c) for c in sp.Matrix([1, 2, 3]).cross(sp.Matrix([4, 5, 6]))), 'boundary', '3D (-3, 6, -3), sympy cross')
    add(((4, 5, 6), (1, 2, 3)), tuple(Fr(c) for c in sp.Matrix([4, 5, 6]).cross(sp.Matrix([1, 2, 3]))), 'boundary', 'swapped: the sign flips')
    add(((1, 2), (1, 2, 3)), BADIN, 'bad-input', 'mixed dimensions')


@case('collinear')
def _(add):
    def sy(*ps):
        return bool(Point.is_collinear(*[SPt(p) for p in ps]))
    add((P(1, 1), P(1, 1), P(1, 1)), True, 'degenerate', 'one point three times (sympy refuses to build it; every set of coincident points is on a line)')
    add((P(1, 1), P(1, 1), P(2, 5)), True, 'degenerate', 'two coincident points and a third')
    add((P(0, 0), P(1, 1), P(2, 2), P(-5, -5), P(10 ** 20, 10 ** 20)), True, 'degenerate', 'five points on y = x, one huge')
    add((P(0, 0), P(1, 1), P(2, 2 + F(1, 10 ** 20))), False, 'boundary', 'off the line by 10^-20: exact arithmetic sees it')
    add((P(0, 0), P(0, 5), P(0, -3), P(0, 9)), True, 'degenerate', 'a vertical line')
    add((P(0, 0, 0), P(1, 2, 3), P(2, 4, 6)), True, 'degenerate', '3D collinear')
    add((P(0, 0, 0), P(1, 2, 3), P(2, 4, 7)), sy(P(0, 0, 0), P(1, 2, 3), P(2, 4, 7)), 'boundary', '3D, off by one in z')
    add((P(0, 0, 0), P(0, 0, 0), P(1, 2, 3), P(2, 4, 6), P(3, 6, 9)), True, 'degenerate', '3D with a repeated point')
    add((P(0, 0), P(1, 1)), BADIN, 'bad-input', 'two points are not enough')
    add((P(0, 0), P(1, 1), P(1, 1, 1)), BADIN, 'bad-input', 'mixed dimensions')
    add(tuple(P(i, 2 * i) for i in range(5000)), True, 'past-bound', '5000 points on y = 2x, exactly the bound: answered (collinear by construction)')
    add(tuple(P(i, 2 * i) for i in range(5001)), REFUSE, 'past-bound', '5001 points: past the bound')


@case('coplanar')
def _(add):
    def sy(*ps):
        return bool(Point3D.are_coplanar(*[SPt(p) for p in ps]))
    add((P(1, 2, 3),) * 4, True, 'degenerate', 'four coincident points')
    add((P(0, 0, 0), P(1, 1, 1), P(2, 2, 2), P(3, 3, 3)), True, 'degenerate', 'four collinear points lie in every plane through the line')
    add((P(0, 0, 0), P(1, 0, 0), P(2, 0, 0), P(0, 0, 7)), sy(P(0, 0, 0), P(1, 0, 0), P(2, 0, 0), P(0, 0, 7)), 'degenerate', 'three collinear and one off: always coplanar')
    add((P(0, 0, 0), P(1, 0, 0), P(0, 1, 0), P(5, 7, 0)), True, 'boundary', 'the plane z = 0')
    add((P(0, 0, 0), P(1, 0, 0), P(0, 1, 0), P(5, 7, F(1, 10 ** 15))), False, 'boundary', 'off the plane by 10^-15')
    add((P(0, 0, 0), P(1, 0, 0), P(0, 1, 0), P(0, 0, 1), P(1, 1, 0)), sy(P(0, 0, 0), P(1, 0, 0), P(0, 1, 0), P(0, 0, 1), P(1, 1, 0)), 'boundary', 'five points, the fourth is off the plane of the others')
    add((P(0, 0), P(1, 0), P(0, 1), P(1, 1)), BADIN, 'bad-input', '2D points: the operation is for space')
    add((P(0, 0, 0), P(1, 0, 0), P(0, 1, 0)), BADIN, 'bad-input', 'three points are not enough')
    add(tuple(P(i, i * i, 0) for i in range(5001)), REFUSE, 'past-bound', '5001 points')
    add(tuple(P(i, i * i, 0) for i in range(5000)), True, 'past-bound', '5000 points on the plane z = 0, exactly the bound: answered')


@case('concyclic')
def _(add):
    def sy(*ps):
        return bool(Point.is_concyclic(*[SPt(p) for p in ps]))
    add((P(0, 0), P(1, 1), P(2, 2), P(3, 3)), True, 'degenerate', 'four collinear points: the determinant is 0 (a line is a circle of infinite radius)')
    add((P(0, 0), P(0, 0), P(1, 0), P(0, 1)), True, 'degenerate', 'two coincident points: three distinct points always lie on a circle')
    add((P(0, 0),) * 4, True, 'degenerate', 'one point four times')
    add((P(5, 0), P(3, 4), P(-5, 0), P(0, -5)), sy(P(5, 0), P(3, 4), P(-5, 0), P(0, -5)), 'boundary', 'four lattice points of x^2 + y^2 = 25')
    add((P(5, 0), P(3, 4), P(-5, 0), P(0, -5 + F(1, 10 ** 12))), sy(P(5, 0), P(3, 4), P(-5, 0), P(0, -5 + F(1, 10 ** 12))), 'boundary', 'the fourth point off the circle by 10^-12')
    add((P(0, 0), P(4, 0), P(0, 3), P(4, 3)), True, 'boundary', 'the corners of a rectangle (circumscribed circle)')
    add((P(0, 0), P(4, 0), P(0, 3), P(4, 4)), False, 'boundary', 'a quadrilateral with unequal diagonals is not cyclic (Ptolemy fails)')
    add((P(0, 0, 0), P(1, 0, 0), P(0, 1, 0), P(1, 1, 0)), BADIN, 'bad-input', '3D points')


@case('line_through')
def _(add):
    def sy(p, q):
        a, b, c = Line(SPt(p), SPt(q)).coefficients
        return (Fr(a), Fr(b), Fr(c))
    add((P(0, 0), P(0, 0)), BADIN, 'degenerate', 'the same point twice has no line')
    add((P(3, 7), P(3, 7)), BADIN, 'degenerate', 'the same point twice')
    add((P(2, 5), P(2, -9)), sy(P(2, 5), P(2, -9)), 'boundary', 'vertical: x = 2 -> (1, 0, 2)')
    add((P(-4, 3), P(8, 3)), sy(P(-4, 3), P(8, 3)), 'boundary', 'horizontal: y = 3 -> (0, 1, 3)')
    add((P(F(1, 2), 0), P(0, F(1, 3))), sy(P(F(1, 2), 0), P(0, F(1, 3))), 'boundary', 'rational points: cleared denominators')
    add((P(0, 0), P(10 ** 20, 3 * 10 ** 20)), sy(P(0, 0), P(10 ** 20, 3 * 10 ** 20)), 'boundary', 'huge coordinates: the coefficients reduce to (3, -1, 0)')
    add((P(1, 1), P(-1, -1)), sy(P(1, 1), P(-1, -1)), 'boundary', 'the first coefficient is made positive')
    add((P(0, 0), P(1)), BADIN, 'bad-input', 'one coordinate')
    add((P(0, 0, 0), P(1, 1, 1)), BADIN, 'bad-input', '3D points')


@case('line_intersection')
def _(add):
    def sy(a, b, c, d):
        r = Line(SPt(a), SPt(b)).intersection(Line(SPt(c), SPt(d)))
        return 'parallel' if not r else ('coincident' if isinstance(r[0], Line) else FrPt(r[0]))
    add((P(0, 0), P(1, 1), P(2, 2), P(5, 5)), 'coincident', 'degenerate', 'the same line given by different point pairs')
    add((P(0, 0), P(1, 1), P(0, 1), P(1, 2)), 'parallel', 'degenerate', 'parallel lines')
    add((P(0, 0), P(0, 1), P(5, 0), P(5, 9)), 'parallel', 'degenerate', 'parallel vertical lines')
    add((P(0, 0), P(0, 1), P(3, 0), P(4, 0)), sy(P(0, 0), P(0, 1), P(3, 0), P(4, 0)), 'boundary', 'a vertical and a horizontal line meet at (0, 0)')
    add((P(0, 0), P(3, 1), P(0, 0), P(1, 3)), sy(P(0, 0), P(3, 1), P(0, 0), P(1, 3)), 'boundary', 'lines through a common given point')
    add((P(0, 0), P(10 ** 9, 1), P(0, 1), P(10 ** 9, 0)), sy(P(0, 0), P(10 ** 9, 1), P(0, 1), P(10 ** 9, 0)), 'boundary', 'nearly parallel lines meet at (10^9/2, 1/2)')
    add((P(F(1, 3), 0), P(F(1, 3), 5), P(0, F(1, 7)), P(1, F(1, 7))), sy(P(F(1, 3), 0), P(F(1, 3), 5), P(0, F(1, 7)), P(1, F(1, 7))), 'boundary', 'fractions, vertical against horizontal')
    add((P(1, 1), P(1, 1), P(0, 0), P(1, 0)), BADIN, 'bad-input', 'the first line has one point')
    add((P(0, 0), P(1, 0), P(2, 2), P(2, 2)), BADIN, 'bad-input', 'the second line has one point')
    add((P(0, 0, 0), P(1, 1, 1), P(0, 1), P(1, 0)), BADIN, 'bad-input', '3D points')


@case('segment_intersection')
def _(add):
    def sy(a, b, c, d):
        if a == b or c == d:
            raise ValueError
        r = Segment(SPt(a), SPt(b)).intersection(Segment(SPt(c), SPt(d)))
        if not r:
            return []
        return sorted([FrPt(r[0].p1), FrPt(r[0].p2)]) if isinstance(r[0], Segment) else [FrPt(r[0])]
    cs = [
        ('endpoints touch (a corner)', P(0, 0), P(2, 0), P(2, 0), P(2, 3)),
        ('T junction: an endpoint on the interior of the other', P(0, 0), P(4, 0), P(2, 0), P(2, 5)),
        ('collinear overlap [1, 2]', P(0, 0), P(2, 0), P(1, 0), P(5, 0)),
        ('collinear, touching at one endpoint only', P(0, 0), P(2, 0), P(2, 0), P(5, 0)),
        ('collinear, disjoint', P(0, 0), P(1, 0), P(2, 0), P(5, 0)),
        ('collinear, one inside the other', P(0, 0), P(10, 10), P(2, 2), P(3, 3)),
        ('identical segments', P(1, 2), P(3, 4), P(1, 2), P(3, 4)),
        ('identical reversed', P(1, 2), P(3, 4), P(3, 4), P(1, 2)),
        ('parallel, disjoint', P(0, 0), P(4, 0), P(0, 1), P(4, 1)),
        ('lines meet outside both segments', P(0, 0), P(1, 1), P(3, 0), P(2, 1)),
        ('lines meet on the extension of one segment', P(0, 0), P(4, 4), P(0, 6), P(2, 4)),
        ('crossing at a fractional point', P(0, 0), P(3, 1), P(0, 1), P(3, 0)),
        ('vertical overlap', P(1, 0), P(1, 5), P(1, 3), P(1, 9)),
        ('diagonal collinear overlap with fractions', P(0, 0), P(F(5, 2), F(5, 2)), P(F(1, 2), F(1, 2)), P(3, 3)),
    ]
    for name, a, b, c, d in cs:
        add((a, b, c, d), sy(a, b, c, d), 'degenerate', 'sympy Segment.intersection: ' + name)
    add((P(1, 1), P(1, 1), P(1, 1), P(1, 1)), [P(1, 1)], 'degenerate', 'two zero-length segments at one point: that point (definition)')
    add((P(1, 1), P(1, 1), P(2, 2), P(2, 2)), [], 'degenerate', 'two zero-length segments at different points: none')
    add((P(1, 1), P(1, 1), P(0, 0), P(2, 2)), [P(1, 1)], 'degenerate', 'a zero-length segment on the other (sympy Segment.contains: %s)' % bool(Segment(SPt(P(0, 0)), SPt(P(2, 2))).contains(SPt(P(1, 1)))))
    add((P(1, 1), P(1, 1), P(0, 0), P(2, 0)), [], 'degenerate', 'a zero-length segment off the other')
    add((P(0, 0), P(2, 2), P(1, 1), P(1, 1)), [P(1, 1)], 'degenerate', 'the second segment has zero length and lies on the first')
    add((P(0, 0, 0), P(1, 1, 1), P(0, 1), P(1, 0)), BADIN, 'bad-input', '3D points')


@case('polyline_intersections')
def _(add):
    def sy(Pp, Q):
        found, overlap = set(), False
        for i in range(len(Pp) - 1):
            for j in range(len(Q) - 1):
                for e in Segment(SPt(Pp[i]), SPt(Pp[i + 1])).intersection(Segment(SPt(Q[j]), SPt(Q[j + 1]))):
                    if isinstance(e, Segment):
                        overlap = True
                    else:
                        found.add(FrPt(e))
        return REFUSE if overlap else sorted(found)
    cs = [
        ('two V shapes meeting at a shared vertex', pts((0, 2), (2, 0), (4, 2)), pts((0, 0), (2, 0), (4, 0))),
        ('a path through a vertex of the other (counted once)', pts((0, 0), (2, 2), (4, 0)), pts((0, 1), (2, 2), (4, 1))),
        ('a vertex lying on an interior point of a segment', pts((0, 0), (2, 2)), pts((0, 4), (1, 1), (4, 4))),
        ('a closed zigzag crossing itself and the other twice', pts((0, 0), (4, 4), (8, 0)), pts((0, 3), (8, 3))),
        ('disjoint paths', pts((0, 0), (1, 1)), pts((5, 5), (6, 7))),
        ('parallel paths', pts((0, 0), (4, 0), (4, 4)), pts((0, 1), (3, 1), (3, 5))),
        ('graphs of |x| and 2 - |x - 1|', [P(x, abs(x)) for x in (-2, 0, 3)], [P(x, 2 - abs(x - 1)) for x in (-2, 1, 4)]),
        ('overlap along a segment (collinear pieces)', pts((0, 0), (4, 0)), pts((2, 0), (6, 0))),
        ('overlap by identical paths', pts((0, 0), (3, 3), (6, 0)), pts((0, 0), (3, 3), (6, 0))),
        ('a path touching the other end to end', pts((0, 0), (1, 1)), pts((1, 1), (3, 0))),
    ]
    for name, Pp, Q in cs:
        add((Pp, Q), sy(Pp, Q), 'degenerate', 'sympy Segment.intersection on every pair: ' + name)
    add((pts((0, 0)), pts((0, 0), (1, 1))), BADIN, 'bad-input', 'a polyline needs two vertices')
    add((pts((0, 0), (1, 1)), pts((0, 0), (1, 1, 5))), BADIN, 'bad-input', 'mixed dimensions')
    zig = lambda n: [P(i, 5 if i % 2 else 0) for i in range(n)]
    flat = lambda n: [P(i, F(5, 2)) for i in range(n)]
    add((zig(1001), flat(1001)), sorted(P(i + F(1, 2), F(5, 2)) for i in range(1000)), 'past-bound',
        'a zigzag of 1000 segments against a line of 1000 segments = 10^6 pairs, exactly the bound: answered, one crossing in the middle of each zigzag segment (symmetry)')
    add((zig(1002), flat(1001)), REFUSE, 'past-bound', '1001 x 1000 = 1,001,000 segment pairs: past the bound')
    add((pts((0, 0), (1, 1)), pts((0, 1), (1, 0))), [P(F(1, 2), F(1, 2))], 'degenerate', 'the two diagonals of the unit square (hand: x = y = 1/2)')


from tools.igc.mathbackend import sim_geometry_edge_b as _b  # noqa: F401, E402
from tools.igc.mathbackend import sim_geometry_edge_c as _c  # noqa: F401, E402


def cases_by_op():
    out = {}
    for short_name, fn in B.items():
        cs = []
        def add(args, orc, kind, desc=''):
            cs.append((args, orc, f'EDGE[{kind}] {desc}'))
        fn(add)
        out[f'geometry.{short_name}'] = cs
    return out

