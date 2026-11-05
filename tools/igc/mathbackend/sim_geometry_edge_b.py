"""Edge cases, part 2: lines to a point, circles, circle angles (see sim_geometry_edge.py)."""
from fractions import Fraction

from tools.igc.mathbackend.sim_geometry_edge import (BADIN, REFUSE, F, P, case, o_circ_pts, o_dpl, o_overlap, o_rel, pts)    # noqa: F401
from tools.igc.mathbackend.sim_geometry_edge import Circle, FrPt, Fr, Line, MF, N50, S, SPt, Triangle, mp, sp                  # noqa: F401


@case('dist_point_line2')
def _(add):
    add((P(1, 1), P(0, 0), P(2, 2)), F(0), 'degenerate', 'the point is on the line (sympy Line.distance = 0)')
    add((P(5, 5), P(0, 0), P(1, 1)), F(0), 'degenerate', 'the point is on the line beyond the segment: distance to a line, not a segment')
    add((P(0, 0), P(0, 0), P(1, 0)), F(0), 'degenerate', 'the point is one of the defining points')
    add((P(3, 4), P(0, 0), P(0, 0)), BADIN, 'degenerate', 'the two defining points coincide: no line')
    add((P(3, 4, 5), P(1, 1, 1), P(1, 1, 1)), BADIN, 'degenerate', 'the two defining points coincide in 3D')
    add((P(0, 5), P(-3, 0), P(3, 0)), Fr(o_dpl(P(0, 5), P(-3, 0), P(3, 0))[1]), 'boundary', 'horizontal line y = 0')
    add((P(1, 2, 3), P(0, 0, 0), P(0, 0, 1)), Fr(o_dpl(P(1, 2, 3), P(0, 0, 0), P(0, 0, 1))[1]), 'boundary', '3D: distance to the z axis = 5')
    add((P(F(1, 3), F(1, 7)), P(F(1, 2), 0), P(0, F(1, 5))), Fr(o_dpl(P(F(1, 3), F(1, 7)), P(F(1, 2), 0), P(0, F(1, 5)))[1]), 'boundary', 'fractions')
    add((P(10 ** 20, 1), P(0, 0), P(1, 0)), F(1), 'boundary', 'a huge foot, distance 1')
    add((P(0, 0), P(1, 1), P(1, 1, 1)), BADIN, 'bad-input', 'mixed dimensions')
    add(((0, 0.5), P(1, 1), P(2, 3)), BADIN, 'bad-input', 'a float')


@case('dist_point_line')
def _(add):
    add((P(1, 1), P(0, 0), P(2, 2)), o_dpl(P(1, 1), P(0, 0), P(2, 2)), 'degenerate', 'on the line: the Surd 0')
    add((P(3, 4), P(0, 0), P(0, 0)), BADIN, 'degenerate', 'coincident defining points')
    add((P(0, 1), P(0, 0), P(1, 1)), o_dpl(P(0, 1), P(0, 0), P(1, 1)), 'boundary', 'sqrt(2)/2')
    add((P(0, 3), P(0, 0), P(1, 0)), o_dpl(P(0, 3), P(0, 0), P(1, 0)), 'boundary', 'a rational distance (3)')
    add((P(1, 2, 3), P(0, 0, 0), P(1, 1, 1)), o_dpl(P(1, 2, 3), P(0, 0, 0), P(1, 1, 1)), 'boundary', '3D, sqrt(2)')
    add((P(0, 10 ** 8), P(0, 0), P(10 ** 8 + 1, 1)), REFUSE, 'past-bound', 'the squared distance has a prime-size factor beyond the 10^7 trial division')
    add(((0, 0.5), P(1, 1), P(2, 3)), BADIN, 'bad-input', 'a float')


@case('circle_through3')
def _(add):
    def sy(p, q, r):
        t = Triangle(SPt(p), SPt(q), SPt(r))
        return (FrPt(t.circumcenter), Fr(t.circumradius ** 2))
    add((P(0, 0), P(1, 1), P(2, 2)), BADIN, 'degenerate', 'collinear points have no circle')
    add((P(0, 0), P(0, 0), P(1, 1)), BADIN, 'degenerate', 'two coincident points')
    add((P(3, 3), P(3, 3), P(3, 3)), BADIN, 'degenerate', 'three coincident points')
    add((P(0, 0), P(0, 5), P(0, -7)), BADIN, 'degenerate', 'collinear on a vertical line')
    add((P(0, 0), P(4, 0), P(0, 3)), sy(P(0, 0), P(4, 0), P(0, 3)), 'boundary', 'right angle: the centre is the midpoint of the hypotenuse')
    add((P(0, 0), P(10, 0), P(5, 1)), sy(P(0, 0), P(10, 0), P(5, 1)), 'boundary', 'a very obtuse triangle: the centre is far below the base')
    add((P(10 ** 15, 3), P(-10 ** 15, 3), P(0, 10 ** 15)), sy(P(10 ** 15, 3), P(-10 ** 15, 3), P(0, 10 ** 15)), 'boundary', 'huge coordinates')
    add((P(F(1, 3), 0), P(0, F(1, 5)), P(F(2, 7), F(3, 11))), sy(P(F(1, 3), 0), P(0, F(1, 5)), P(F(2, 7), F(3, 11))), 'boundary', 'fractions')
    add((P(0, 0, 0), P(1, 0, 0), P(0, 1, 0)), BADIN, 'bad-input', '3D points')
    add((P(0, 0), (1, 0.5), P(2, 1)), BADIN, 'bad-input', 'a float')


@case('circle_line_intersection')
def _(add):
    def sy(c, r2, p, q):
        inter = Circle(SPt(c), sp.sqrt(S(r2))).intersection(Line(SPt(p), SPt(q)))
        return (o_circ_pts(inter), c, r2, p, q)

    def hand(c, r2, p, q, pts_):
        return ([(N50(S(x)), N50(S(y))) for x, y in pts_], c, r2, p, q)
    add((P(0, 0), F(25), P(0, 5), P(1, 5)), sy(P(0, 0), F(25), P(0, 5), P(1, 5)), 'degenerate', 'tangent line y = 5: one point (0, 5)')
    add((P(0, 0), F(2), P(1, 1), P(2, 0)), sy(P(0, 0), F(2), P(1, 1), P(2, 0)), 'degenerate', 'tangent with an irrational circle: x + y = 2 touches x^2 + y^2 = 2 at (1, 1)')
    add((P(0, 0), F(25), P(0, 6), P(1, 6)), sy(P(0, 0), F(25), P(0, 6), P(1, 6)), 'degenerate', 'the line misses the circle: none')
    add((P(0, 0), F(25), P(0, 0), P(1, 0)), sy(P(0, 0), F(25), P(0, 0), P(1, 0)), 'degenerate', 'a line through the centre: the two ends of a diameter')
    add((P(1, 1), F(5), P(1, 0), P(1, 9)), sy(P(1, 1), F(5), P(1, 0), P(1, 9)), 'boundary', 'a vertical line: x = 1 meets the circle at (1, 1 +- sqrt(5))')
    add((P(0, 0), F(1), P(-5, 0), P(5, 0)), sy(P(0, 0), F(1), P(-5, 0), P(5, 0)), 'boundary', 'the unit circle and the x axis, points given outside the circle')
    add((P(F(1, 2), F(1, 3)), F(7, 4), P(0, 0), P(1, 2)), sy(P(F(1, 2), F(1, 3)), F(7, 4), P(0, 0), P(1, 2)), 'boundary', 'a rational centre and squared radius')
    add((P(0, 0), F(0), P(0, 0), P(1, 1)), hand(P(0, 0), F(0), P(0, 0), P(1, 1), [(0, 0)]), 'degenerate', 'radius 0: a line through the centre meets it in the centre (definition)')
    add((P(0, 0), F(0), P(1, 0), P(1, 1)), hand(P(0, 0), F(0), P(1, 0), P(1, 1), []), 'degenerate', 'radius 0: a line missing the centre meets it nowhere (definition)')
    add((P(0, 0), F(25), P(1, 1), P(1, 1)), BADIN, 'degenerate', 'the two points of the line coincide')
    add((P(0, 0), F(-1), P(0, 0), P(1, 0)), BADIN, 'bad-input', 'a negative squared radius')
    add((P(0, 0, 0), F(1), P(0, 0), P(1, 0)), BADIN, 'bad-input', 'a 3D centre')
    add((P(0, 0), F(10 ** 14 + 2), P(0, 1), P(1, 1)), sy(P(0, 0), F(10 ** 14 + 2), P(0, 1), P(1, 1)), 'past-bound', 'the radicand is 10^14 + 1: inside the trial-division bound')
    add((P(0, 0), F(10 ** 16 + 2), P(0, 1), P(1, 1)), REFUSE, 'past-bound', 'the radicand is 10^16 + 1: past the trial-division bound')


@case('circle_circle_intersection')
def _(add):
    def sy(c1, r1, c2, r2):
        inter = Circle(SPt(c1), sp.sqrt(S(r1))).intersection(Circle(SPt(c2), sp.sqrt(S(r2))))
        return (o_circ_pts(inter), c1, r1, c2, r2)
    add((P(0, 0), F(1), P(0, 0), F(1)), REFUSE, 'degenerate', 'identical circles: all points are common')
    add((P(2, 3), F(4), P(2, 3), F(9)), REFUSE, 'degenerate', 'concentric circles of different radii: none')
    add((P(0, 0), F(1), P(2, 0), F(1)), sy(P(0, 0), F(1), P(2, 0), F(1)), 'degenerate', 'external tangent: one point (1, 0)')
    add((P(0, 0), F(9), P(1, 0), F(4)), sy(P(0, 0), F(9), P(1, 0), F(4)), 'degenerate', 'internal tangent: R = 3, r = 2, d = 1: one point (3, 0)')
    add((P(0, 0), F(9), P(1, 0), F(1)), sy(P(0, 0), F(9), P(1, 0), F(1)), 'degenerate', 'one circle inside the other, no contact: none')
    add((P(0, 0), F(1), P(5, 0), F(1)), sy(P(0, 0), F(1), P(5, 0), F(1)), 'degenerate', 'far apart: none')
    add((P(0, 0), F(1), P(1, 0), F(1)), sy(P(0, 0), F(1), P(1, 0), F(1)), 'boundary', 'two unit circles at distance 1: (1/2, +-sqrt(3)/2)')
    add((P(0, 0), F(25), P(0, 7), F(16)), sy(P(0, 0), F(25), P(0, 7), F(16)), 'boundary', 'centres on a vertical line (the radical line is horizontal)')
    add((P(F(1, 2), F(1, 3)), F(7, 2), P(F(-1, 5), 2), F(9, 4)), sy(P(F(1, 2), F(1, 3)), F(7, 2), P(F(-1, 5), 2), F(9, 4)), 'boundary', 'fractions')
    add((P(0, 0), F(0), P(3, 4), F(25)), ([(N50(S(0)), N50(S(0)))], P(0, 0), F(0), P(3, 4), F(25)), 'degenerate', 'radius 0 on the circumference of the other circle: that point (definition)')
    add((P(0, 0), F(-1), P(3, 4), F(25)), BADIN, 'bad-input', 'a negative squared radius')
    add((P(0, 0, 0), F(1), P(1, 0), F(1)), BADIN, 'bad-input', 'a 3D centre')
    add((P(0, 0), F(10 ** 16 + 2), P(0, 2), F(10 ** 16 + 2)), REFUSE, 'past-bound', 'the intersection radicand is about 10^16: past the trial-division bound')


@case('circle_circle_relation')
def _(add):
    cs = [
        (P(0, 0), F(1), P(0, 0), F(1)), (P(0, 0), F(1), P(0, 0), F(4)), (P(0, 0), F(4), P(0, 0), F(1)), (P(0, 0), F(1), P(2, 0), F(1)), (P(0, 0), F(9), P(1, 0), F(4)),
        (P(0, 0), F(4), P(1, 0), F(9)), (P(0, 0), F(1), P(3, 0), F(1)), (P(0, 0), F(1), P(1, 0), F(1)), (P(0, 0), F(9), P(1, 0), F(1)), (P(0, 0), F(2), P(2, 0), F(2)),
        (P(0, 0), F(2), P(1, 1), F(2)), (P(0, 0), F(1), P(3, 4), F(16)), (P(0, 0), F(1), P(3, 4), F(36)), (P(0, 0), F(1), P(3, 4), F(49)),
        (P(0, 0), F(1), P(3, 4), F(4)), (P(0, 0), F(1), P(F(3, 5), F(4, 5)), F(4)), (P(0, 0), F(4), P(0, 0), F(4)), (P(0, 0), F(1, 4), P(F(1, 2), 0), F(1, 4)),
        (P(0, 0), F(10 ** 24), P(10 ** 12 + 1, 0), F(1)), (P(0, 0), F(10 ** 24), P(10 ** 12 - 1, 0), F(1)), (P(0, 0), F(10 ** 24), P(10 ** 12 + 2, 0), F(1)),
    ]
    for c in cs:
        add(c, o_rel(*c), 'degenerate', 'mpmath 50 digits: distance of centres against R1 + R2 and |R1 - R2|, tangent decided at 1e-40')
    add((P(0, 0), F(0), P(1, 0), F(1)), BADIN, 'bad-input', 'a zero squared radius')
    add((P(0, 0), F(1), P(1, 0), F(-1)), BADIN, 'bad-input', 'a negative squared radius')
    add((P(0, 0, 0), F(1), P(1, 0), F(1)), BADIN, 'bad-input', 'a 3D centre')


@case('circle_circle_overlap_area_numeric')
def _(add):
    cs = [
        ((P(0, 0), F(1), P(0, 0), F(1), 12), 'degenerate', 'identical discs: pi r^2'),
        ((P(0, 0), F(4), P(0, 0), F(9), 12), 'degenerate', 'concentric: the smaller disc, 4 pi'),
        ((P(0, 0), F(1), P(2, 0), F(1), 12), 'degenerate', 'external tangent: 0'),
        ((P(0, 0), F(9), P(1, 0), F(4), 12), 'degenerate', 'internal tangent: the smaller disc'),
        ((P(0, 0), F(1), P(5, 0), F(1), 12), 'degenerate', 'far apart: 0'),
        ((P(0, 0), F(9), P(1, 0), F(1), 12), 'degenerate', 'one inside the other without contact'),
        ((P(0, 0), F(1), P(1, 0), F(1), 12), 'boundary', 'the classic lens, 2 pi/3 - sqrt(3)/2'),
        ((P(0, 0), F(25), P(0, 7), F(16), 14), 'boundary', 'a vertical pair'),
        ((P(0, 0), F(2), P(1, 1), F(2), 14), 'boundary', 'two discs of radius sqrt(2) at distance sqrt(2): a lens of 90 degrees'),
        ((P(0, 0), F(1), P(F(1, 10 ** 6), 0), F(1), 14), 'boundary', 'nearly identical discs'),
        ((P(0, 0), F(1), P(F(19999, 10000), 0), F(1), 14), 'boundary', 'nearly tangent discs: a very thin lens'),
    ]
    for args, tag, desc in cs:
        add(args, o_overlap(*args), tag, 'mpmath quad of the chord length: ' + desc)
    add((P(0, 0), F(0), P(1, 0), F(1), 12), BADIN, 'bad-input', 'a zero squared radius')
    add((P(0, 0), F(1), P(1, 0), F(1), 2001), REFUSE, 'past-bound', 'digits past the bound of 2000')
    add((P(0, 0), F(1), P(1, 0), F(1), -1), BADIN, 'bad-input', 'negative digits')
    add((P(0, 0), F(1), P(1, 0), F(1), 2.5), BADIN, 'bad-input', 'digits must be an integer')


@case('power_of_point')
def _(add):
    add((P(3, 4), P(0, 0), F(25)), F(0), 'degenerate', 'a point on the circle: power 0 (sympy distance^2 - r^2)')
    add((P(0, 0), P(0, 0), F(25)), F(-25), 'degenerate', 'the centre: power -r^2')
    add((P(0, 0), P(0, 0), F(0)), F(0), 'degenerate', 'a zero circle at the point')
    add((P(1, 2), P(1, 2), F(7, 3)), F(-7, 3), 'degenerate', 'the centre with a rational r^2')
    add((P(10 ** 20, 0), P(0, 0), F(1)), F(10 ** 40 - 1), 'boundary', 'a huge distance')
    add((P(F(1, 3), F(1, 7)), P(F(1, 2), F(1, 5)), F(1, 100)), Fr(SPt(P(F(1, 3), F(1, 7))).distance(SPt(P(F(1, 2), F(1, 5)))) ** 2 - S(F(1, 100))), 'boundary', 'fractions, sympy')
    add((P(3, 4), P(0, 0, 0), F(25)), BADIN, 'bad-input', 'a 3D centre')
    add((P(3, 4), P(0, 0), 'x'), BADIN, 'bad-input', 'a non-numeric squared radius')


@case('tangent_length')
def _(add):
    def sy(p, c, r2):
        pw = Fr(SPt(p).distance(SPt(c)) ** 2 - S(r2))
        return (str(sp.N(sp.sqrt(S(pw)), 55)), pw)
    add((P(3, 4), P(0, 0), F(25)), sy(P(3, 4), P(0, 0), F(25)), 'boundary', 'on the circle: tangent length 0')
    add((P(5, 0), P(0, 0), F(25)), sy(P(5, 0), P(0, 0), F(25)), 'boundary', 'on the circle at (5, 0)')
    add((P(0, 0), P(0, 0), F(1)), BADIN, 'degenerate', 'the centre is inside')
    add((P(1, 0), P(0, 0), F(4)), BADIN, 'degenerate', 'inside the circle')
    add((P(3, 0), P(0, 0), F(1)), sy(P(3, 0), P(0, 0), F(1)), 'boundary', '2 sqrt(2)')
    add((P(5, 0), P(0, 0), F(0)), sy(P(5, 0), P(0, 0), F(0)), 'degenerate', 'a zero circle: the tangent length is the distance')
    add((P(F(7, 2), F(1, 3)), P(F(1, 5), F(1, 7)), F(1, 3)), sy(P(F(7, 2), F(1, 3)), P(F(1, 5), F(1, 7)), F(1, 3)), 'boundary', 'fractions')
    add((P(10 ** 8, 2), P(0, 0), F(3)), REFUSE, 'past-bound', 'the power is 10^16 + 1: past the trial-division bound')
    add((P(3, 4), P(0, 0, 0), F(1)), BADIN, 'bad-input', 'a 3D centre')


@case('circle_angle')
def _(add):
    cs = [
        (('inscribed', 0), '0', 'degenerate', 'an arc of 0 degrees: the angle is 0 (theorem: half the arc)'),
        (('inscribed', 360), '180', 'degenerate', 'the whole circle: 180 (theorem)'),
        (('inscribed', 180), '90', 'boundary', 'a diameter subtends a right angle (Thales)'),
        (('inscribed', F(141, 2)), '35.25', 'boundary', 'a rational arc 70.5'),
        (('tangent_chord', 90), '45', 'boundary', 'tangent-chord angle = half the arc'),
        (('tangent_chord', 360), '180', 'degenerate', 'a tangent-chord angle facing the whole circle'),
        (('chords_inside', 70, 50), '60', 'boundary', 'two chords crossing: (70 + 50) / 2'),
        (('chords_inside', 180, 180), '180', 'degenerate', 'two arcs filling the circle: 180'),
        (('chords_inside', 0, 0), '0', 'degenerate', 'two arcs of 0'),
        (('outside', 120, 40), '40', 'boundary', 'two secants: (120 - 40) / 2'),
        (('outside', 180, 0), '90', 'degenerate', 'two tangents with the far arc 180 and near arc 0: 90'),
        (('outside', 100, 100), '0', 'degenerate', 'equal arcs: the angle is 0'),
        (('outside', 359, 1), '179', 'boundary', 'a very wide wedge'),
    ]
    for args, orc, tag, desc in cs:
        add(args, orc, tag, 'theorem on the angle of a chord or tangent, hand-worked: ' + desc)
    add(('inscribed', -1), BADIN, 'bad-input', 'a negative arc')
    add(('inscribed', 361), BADIN, 'bad-input', 'an arc past 360')
    add(('chords_inside', 200, 200), BADIN, 'bad-input', 'two arcs together past 360')
    add(('outside', 40, 120), BADIN, 'bad-input', 'the far arc is smaller than the near arc')
    add(('outside', 300, 100), BADIN, 'bad-input', 'two arcs together past 360')
    add(('central', 90), BADIN, 'bad-input', 'an unknown kind')
    add(('inscribed', 'x'), BADIN, 'bad-input', 'a non-numeric arc')
