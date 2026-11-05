"""Mutation test of tools/igc/mathbackend/geometry.py: two deliberate bugs per operation, each applied to a COPY of the source (the file on disk is never changed),
and the simulation of that operation (sim_geometry.py) must report at least one disagreement for every one of them. A mutant the simulation does not catch means the
simulation is missing a case: add one (a degenerate, boundary or past-the-bound case) until it is caught.

    PYTHONPATH=. python tools/mutate_mathbackend_geometry.py [--seed 1] [--n 60] [--ops geometry.dist2,geometry.midpoint] [--timeout 900]

Needs the interpreter that has SymPy and mpmath (the oracles live in the sim file). Each mutant runs in its own process (a mutant that loops is killed by the timeout and
counts as caught-by-timeout, reported as such). The mutation `old` text must occur exactly once in geometry.py, otherwise the table says so and nothing is run for it.
Exit code 1 when a mutant survives."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'tools' / 'igc' / 'mathbackend' / 'geometry.py'

# op -> [(old, new), (old, new)]: each mutation is a small slip a programmer could make (a sign, an off-by-one, a wrong operand, a dropped check, a moved bound).
MUT = {
    'dist2': [("    d = _sub(p, q)\n    return _dot(d, d)", "    d = _sub(p, q)\n    return _dot(d[:2], d[:2])"),
              ("    d = _sub(p, q)\n    return _dot(d, d)", "    d = _sub(p, q)\n    return sum(abs(x) for x in d) ** 2")],
    'dist': [("    return Surd.sqrt(dist2(p, q))", "    return Surd(dist2(p, q))"),
             ("    return Surd.sqrt(dist2(p, q))", "    return Surd.sqrt(dist2(p, q) + 1)")],
    'midpoint': [("return tuple((a + b) / 2 for a, b in zip(p, q))", "return tuple((a + b) // 2 for a, b in zip(p, q))"),
                 ("return tuple((a + b) / 2 for a, b in zip(p, q))", "return tuple((a + b) / 2 for a, b in zip(p[:2], q[:2]))")],
    'dot': [("    _same_dim(u, v)\n    return _dot(u, v)", "    _same_dim(u, v)\n    return _dot(u[:2], v[:2])"),
            ("    _same_dim(u, v)\n    return _dot(u, v)", "    _same_dim(u, v)\n    return abs(_dot(u, v))")],
    'cross': [("return _cross2(u, v) if len(u) == 2 else _cross3(u, v)", "return _cross2(v, u) if len(u) == 2 else _cross3(u, v)"),
              ("return _cross2(u, v) if len(u) == 2 else _cross3(u, v)", "return _cross2(u, v) if len(u) == 2 else _cross3(v, u)")],
    'collinear': [("    if base is None:\n        return True", "    if base is None:\n        return False"),
                  ("    for q in ps:\n        w = _sub(q, p0)", "    for q in ps[:3]:\n        w = _sub(q, p0)")],
    'coplanar': [("    return all(_dot(normal, _sub(q, p0)) == 0 for q in ps)", "    return all(_dot(normal, _sub(q, p0)) == 0 for q in ps[:4])"),
                 ("    if normal is None:\n        return True\n    return all(", "    if normal is None:\n        return False\n    return all(")],
    'concyclic': [("rows = [[x, y, x * x + y * y, Fraction(1)] for x, y in pts]", "rows = [[x, y, x * x + y, Fraction(1)] for x, y in pts]"),
                  ("    return _det(rows) == 0", "    return _det(rows) <= 0")],
    'line_through': [("a, b = q[1] - p[1], p[0] - q[0]", "a, b = q[1] - p[1], q[0] - p[0]"),
                     ("    c = a * p[0] + b * p[1]\n    return _primitive", "    c = a * p[0] - b * p[1]\n    return _primitive")],
    'line_intersection': [("    if p1 == p2 or p3 == p4:\n        raise BadInput('a line needs two different points')\n    return _line_inter",
                           "    if p1 == p2:\n        raise BadInput('a line needs two different points')\n    return _line_inter"),
                          ("return 'coincident' if _cross2(qp, r) == 0 else 'parallel'", "return 'parallel' if _cross2(qp, r) == 0 else 'coincident'")],
    'segment_intersection': [("        if 0 <= t <= 1 and 0 <= u <= 1:", "        if 0 <= t < 1 and 0 <= u <= 1:"),
                             ("    if lo > hi:\n        return []\n    p_lo", "    if lo >= hi:\n        return []\n    p_lo")],
    'polyline_intersections': [("            if max(a[0], b[0]) < min(c[0], d[0]) or max(c[0], d[0]) < min(a[0], b[0]):", "            if max(a[0], b[0]) <= min(c[0], d[0]) or max(c[0], d[0]) <= min(a[0], b[0]):"),
                               ("    if (len(P) - 1) * (len(Q) - 1) > 10 ** 6:", "    if (len(P) - 1) * (len(Q) - 1) > 10 ** 7:")],
    'dist_point_line2': [("    if a == b:\n        raise BadInput('a line needs two different points')\n    u, w = _sub(b, a), _sub(p, a)", "    u, w = _sub(b, a), _sub(p, a)"),
                         ("    return _cross_sq(u, w) / _dot(u, u)", "    return _cross_sq(u, w)")],
    'dist_point_line': [("    return Surd.sqrt(dist_point_line2(p, a, b))", "    return Surd(dist_point_line2(p, a, b))"),
                        ("    return Surd.sqrt(dist_point_line2(p, a, b))", "    return Surd.sqrt(dist_point_line2(p, a, b)) * 2")],
    'circle_through3': [("    d = 2 * _cross2(b, c)", "    d = _cross2(b, c)"),
                        ("    uy = (b[0] * c2 - c[0] * b2) / d", "    uy = (c[0] * b2 - b[0] * c2) / d")],
    'circle_line_intersection': [("    disc = hb * hb - a * (_dot(f, f) - r2)", "    disc = hb * hb - a * (_dot(f, f) + r2)"),
                                 ("    if disc < 0:\n        return []\n    mid", "    if disc <= 0:\n        return []\n    mid")],
    'circle_circle_intersection': [("    k = r1 - r2 - _dot(c1, c1) + _dot(c2, c2)", "    k = r1 - r2 + _dot(c1, c1) - _dot(c2, c2)"),
                                   ("    t = (k - (a * c1[0] + b * c1[1])) / n2", "    t = (k - (a * c2[0] + b * c2[1])) / n2")],
    'circle_circle_relation': [("    if d2 == 0 and r1 == r2:\n        return 'identical'", "    if d2 == 0:\n        return 'identical'"),
                               ("    return 'internal tangent' if inner == 0 else 'contained'", "    return 'contained' if inner == 0 else 'internal tangent'")],
    'circle_circle_overlap_area_numeric': [("        return _round(_dec(q1) * a + _dec(q2) * b - rk / 2, digits)", "        return _round(_dec(q1) * a + _dec(q2) * b - rk, digits)"),
                                           ("            return _round(_pi(ctx.prec + 5) * _dec(min(q1, q2)), digits)", "            return _round(_pi(ctx.prec + 5) * _dec(max(q1, q2)), digits)")],
    'power_of_point': [("    return dist2(p, c) - _q(r2)", "    return _q(r2) - dist2(p, c)"),
                       ("    return dist2(p, c) - _q(r2)", "    return dist2(p, c) + _q(r2)")],
    'tangent_length': [("    if pw < 0:\n        raise BadInput('the point is inside the circle: no tangent')", "    if pw <= 0:\n        raise BadInput('the point is inside the circle: no tangent')"),
                       ("    return Surd.sqrt(pw)\n\n\n@op('geometry.circle_angle'", "    return Surd(pw)\n\n\n@op('geometry.circle_angle'")],
    'circle_angle': [("    if kind in ('inscribed', 'tangent_chord'):\n        return a1 / 2", "    if kind in ('inscribed', 'tangent_chord'):\n        return a1"),
                     ("        return (a1 - a2) / 2", "        return (a1 + a2) / 2")],
    'triangle_area': [("    return abs(_cross2(_sub(q, p), _sub(r, p))) / 2", "    return _cross2(_sub(q, p), _sub(r, p)) / 2"),
                      ("    return abs(_cross2(_sub(q, p), _sub(r, p))) / 2", "    return abs(_cross2(_sub(q, p), _sub(r, p))) // 2")],
    'triangle_area_sq': [("    return _cross_sq(_sub(q, p), _sub(r, p)) / 4", "    return _cross_sq(_sub(q, p), _sub(r, p)) / 16"),
                         ("    p, q, r = _pt(p), _pt(q), _pt(r)\n    _same_dim(p, q, r)\n    return _cross_sq(_sub(q, p), _sub(r, p)) / 4", "    p, q, r = _pt(p), _pt(q), _pt(r)\n    return _cross_sq(_sub(q, p), _sub(r, p)) / 4")],
    'triangle_from_sides': [("    a2 = s * (s - a) * (s - b) * (s - c)\n    area = Surd.sqrt(a2)", "    a2 = s * (s - a) * (s - b)\n    area = Surd.sqrt(a2)"),
                            ("    if min(a, b, c) <= 0 or a + b <= c or a + c <= b or b + c <= a:", "    if min(a, b, c) <= 0 or a + b < c or a + c <= b or b + c <= a:")],
    'triangle_from_sides_sq': [("(a2 * a2 + b2 * b2 + c2 * c2)\n    if k <= 0:", "(a2 * a2 + b2 * b2)\n    if k <= 0:"),
                               ("    if k <= 0:\n        raise BadInput('the sides do not form a triangle')\n    area_sq = k / 16", "    if k < 0:\n        raise BadInput('the sides do not form a triangle')\n    area_sq = k / 16")],
    'circumradius_sq': [("    k = 4 * _cross_sq(_sub(q, p), _sub(r, p))", "    k = 2 * _cross_sq(_sub(q, p), _sub(r, p))"),
                        ("    return dist2(q, r) * dist2(p, r) * dist2(p, q) / k", "    return dist2(q, r) * dist2(p, r) / k")],
    'inradius': [("+ Surd.sqrt(dist2(p, r)) + Surd.sqrt(dist2(p, q))\n    return Surd.sqrt(k) / perim", "+ Surd.sqrt(dist2(p, r))\n    return Surd.sqrt(k) / perim"),
                 ("    return Surd.sqrt(k) / perim ", "    return Surd.sqrt(k) / (perim * 2) ")],
    'circumradius': [("    return Surd.sqrt(circumradius_sq(p, q, r))", "    return Surd(circumradius_sq(p, q, r))"),
                     ("    return Surd.sqrt(circumradius_sq(p, q, r))", "    return Surd.sqrt(circumradius_sq(p, q, r)) * 2")],
    'inradius_numeric': [("        return _round(_dec(k).sqrt() / per, digits)", "        return _round(_dec(k).sqrt() / per / 2, digits)"),
                         ("        per = sum((_dec(d2).sqrt() for d2 in (dist2(q, r), dist2(p, r), dist2(p, q))), Decimal(0))", "        per = sum((_dec(d2).sqrt() for d2 in (dist2(q, r), dist2(p, r))), Decimal(0))")],
    'incenter_numeric': [("        tot = a + b + c\n        return tuple(_round(", "        tot = a + b\n        return tuple(_round("),
                         ("        a, b, c = _dec(dist2(q, r)).sqrt(), _dec(dist2(p, r)).sqrt(), _dec(dist2(p, q)).sqrt()", "        a, b, c = _dec(dist2(p, q)).sqrt(), _dec(dist2(p, r)).sqrt(), _dec(dist2(q, r)).sqrt()")],
    'centroid': [("    return tuple(sum(p[i] for p in ps) / n for i in range(len(ps[0])))", "    return tuple(sum(p[i] for p in ps) // n for i in range(len(ps[0])))"),
                 ("    n = len(ps)\n    return tuple(sum(p[i] for p in ps)", "    n = len(set(ps))\n    return tuple(sum(p[i] for p in ps)")],
    'circumcenter': [("    return circle_through3(p, q, r)[0]", "    return circle_through3(p, q, r)[1]"),
                     ("    return circle_through3(p, q, r)[0]", "    return centroid(p, q, r)")],
    'orthocenter': [("return tuple(p[i] + q[i] + r[i] - 2 * o[i] for i in range(2))", "return tuple(p[i] + q[i] + r[i] - o[i] for i in range(2))"),
                    ("return tuple(p[i] + q[i] + r[i] - 2 * o[i] for i in range(2))", "return tuple(p[i] + q[i] - 2 * o[i] for i in range(2))")],
    'incenter': [("    return tuple((a * p[i] + b * q[i] + c * r[i]) / tot for i in range(2))", "    return tuple((a * q[i] + b * p[i] + c * r[i]) / tot for i in range(2))"),
                 ("    tot = a + b + c\n    return tuple((a * p[i]", "    tot = a + b\n    return tuple((a * p[i]")],
    'angle_cos': [("    return Surd(_dot(u, v)) / Surd.sqrt(uu * vv)", "    return Surd(_dot(u, v)) / Surd.sqrt(uu + vv)"),
                  ("    if uu == 0 or vv == 0:\n        raise BadInput('a zero vector has no direction')\n    return Surd(", "    if uu == 0 and vv == 0:\n        raise BadInput('a zero vector has no direction')\n    return Surd(")],
    'angle_degrees_exact': [("(Surd(0, Fraction(-1, 2), 3), 150)", "(Surd(0, Fraction(-1, 2), 3), 120)"), ("(Surd(0, Fraction(1, 2), 2), 45)", "(Surd(0, Fraction(1, 2), 2), 135)")],
    'angle_degrees_numeric': [("* 180 / _pi(ctx.prec + 5)\n        return _round(ang, digits)", "* 90 / _pi(ctx.prec + 5)\n        return _round(ang, digits)"),
                              ("ang = _atan2(cr, _dec(_dot(u, v)))", "ang = _atan2(cr, abs(_dec(_dot(u, v))))")],
    'triangle_angles_numeric': [("angle_degrees_numeric(_sub(p, r), _sub(q, r), digits))", "angle_degrees_numeric(_sub(p, r), _sub(q, p), digits))"),
                                ("angle_degrees_numeric(_sub(p, q), _sub(r, q), digits),", "angle_degrees_numeric(_sub(p, q), _sub(r, p), digits),")],
    'convex_hull': [("            while len(h) >= 2 and _cross2(_sub(h[-1], h[-2]), _sub(p, h[-2])) <= 0:", "            while len(h) >= 2 and _cross2(_sub(h[-1], h[-2]), _sub(p, h[-2])) < 0:"),
                    ("    return lower[:-1] + upper[:-1]", "    return lower + upper")],
    'polygon_signed_area': [("    return _signed2(_pts(poly, 3, dim=2)) / 2", "    return _signed2(_pts(poly, 3, dim=2)) // 2"),
                            ("    return _signed2(_pts(poly, 3, dim=2)) / 2", "    return _signed2(_pts(poly, 3, 4999, dim=2)) / 2")],
    'polygon_area': [("    return abs(_signed2(_pts(poly, 3, dim=2))) / 2", "    return _signed2(_pts(poly, 3, dim=2)) / 2"),
                     ("    return abs(_signed2(_pts(poly, 3, dim=2))) / 2", "    return abs(_signed2(_pts(poly, 3, 4999, dim=2))) / 2")],
    'polygon_perimeter_terms': [("terms[key] = terms.get(key, Fraction(0)) + (s.q if s.q != 0 else s.p)", "terms[key] = terms.get(key, Fraction(0)) + (s.q if s.q != 0 else 0)"),
                                ("    return sorted(_side_terms(poly).items())", "    return sorted(_side_terms(poly).items(), reverse=True)")],
    'polygon_perimeter': [("    if len(irr) > 1:\n        raise NotCovered('several radicals')", "    if len(irr) > 2:\n        raise NotCovered('several radicals')"),
                          ("    rat = next((c for d, c in terms if d == 1), Fraction(0))", "    rat = next((c for d, c in terms if d == 1), Fraction(1))")],
    'polygon_perimeter_numeric': [("(Decimal(d).sqrt() if d != 1 else Decimal(1))", "(Decimal(d).sqrt() if d > 2 else Decimal(1))"),
                                  ("    digits = _digits(digits)\n    terms = polygon_perimeter_terms(poly)", "    terms = polygon_perimeter_terms(poly)")],
    'polygon_centroid': [("    return (cx / (3 * a2), cy / (3 * a2))", "    return (cx / (2 * a2), cy / (2 * a2))"),
                         ("        cy += (a[1] + b[1]) * w", "        cy += (a[0] + b[1]) * w")],
    'is_simple_polygon': [("            adjacent = (j == i + 1) or (i == 0 and j == n - 1)", "            adjacent = (j == i + 1)"),
                          ("    if len(set(poly)) != n:\n        return False\n    for i in range(n):", "    for i in range(n):")],
    'is_convex': [("    if not _is_simple(poly):\n        return False\n    n = len(poly)\n    signs", "    n = len(poly)\n    signs"),
                  ("            signs.add(c > 0)", "            signs.add(c >= 0)\n        else:\n            signs.add(True)")],
    'point_in_polygon': [("            x = a[0] + (pt[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])", "            x = a[0] - (pt[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])"),
                         ("        if _on_segment(pt, poly[i], poly[(i + 1) % n]):\n            return 'boundary'", "        if False:\n            return 'boundary'")],
    'pick_area': [("    return Fraction(interior) + Fraction(boundary, 2) - 1", "    return Fraction(interior) + Fraction(boundary, 2) + 1"),
                  ("or interior < 0 or boundary < 3:", "or interior < 0 or boundary < 4:")],
    'lattice_points_on_segment': [("    return gcd(abs(a[0] - b[0]), abs(a[1] - b[1])) + 1", "    return gcd(abs(a[0] - b[0]), abs(a[1] - b[1]))"),
                                  ("    return gcd(abs(a[0] - b[0]), abs(a[1] - b[1])) + 1", "    return gcd(abs(a[0] - b[0]), abs(a[1] - b[1]) + 1) + 1")],
    'lattice_polygon_stats': [("    i_pts = (a2 - b + 2) // 2\n    return {'area2'", "    i_pts = (a2 - b) // 2\n    return {'area2'"),
                              ("    if not _is_simple(poly):\n        raise BadInput('not a simple polygon (self-crossing", "    if False:\n        raise BadInput('not a simple polygon (self-crossing")],
    'lattice_points_in_convex_polygon': [("        closed += max(0, math.floor(hi) - math.ceil(lo) + 1)", "        closed += max(0, math.floor(hi) - math.floor(lo) + 1)"),
                                         ("        if ymin < k < ymax:", "        if ymin <= k <= ymax:")],
    'lattice_points_in_triangle': [("        i_pts = (a2 - b + 2) // 2\n        return {'total': i_pts + b", "        i_pts = (a2 - b) // 2\n        return {'total': i_pts + b"),
                                   ("abs(ip[i][1] - ip[(i + 1) % 3][1])) for i in range(3))", "abs(ip[i][1] - ip[(i + 1) % 3][1])) for i in range(2))")],
    'lattice_points_in_circle': [("    fl = math.floor(r2) if not strict else math.ceil(r2) - 1", "    fl = math.floor(r2) if not strict else math.floor(r2) - 1"),
                                 ("    if m > 2 * 10 ** 6:", "    if m > 3 * 10 ** 6:")],
    'reflect_point_line': [("    return (2 * foot[0] - p[0], 2 * foot[1] - p[1])", "    return (foot[0] - p[0], foot[1] - p[1])"),
                           ("    t = _dot(_sub(p, a), d) / _dot(d, d)\n    foot", "    t = _dot(_sub(p, b), d) / _dot(d, d)\n    foot")],
    'reflect_points_line': [("    return [reflect_point_line(p, a, b) for p in _pts(points, 1, dim=2)]", "    return [reflect_point_line(p, a, b) for p in _pts(points, 1, dim=2)][::-1]"),
                            ("    return [reflect_point_line(p, a, b) for p in _pts(points, 1, dim=2)]", "    return [reflect_point_line(p, a, b) for p in _pts(points, 1, 4999, dim=2)]")],
    'reflect_point_point': [("    return tuple(2 * y - x for x, y in zip(p, c))", "    return tuple(y - x for x, y in zip(p, c))"),
                            ("    _same_dim(p, c)\n    return tuple(2 * y - x", "    return tuple(2 * y - x")],
    'rotate_point_90': [("    for _ in range(quarter_turns % 4):", "    for _ in range(abs(quarter_turns) % 4):"),
                        ("        d = (-d[1], d[0])", "        d = (d[1], -d[0])")],
    'rotate_points_90': [("    return [rotate_point_90(p, center, quarter_turns) for p in _pts(points, 1, dim=2)]", "    return [rotate_point_90(p, center, -quarter_turns) for p in _pts(points, 1, dim=2)]"),
                         ("    return [rotate_point_90(p, center, quarter_turns) for p in _pts(points, 1, dim=2)]", "    return [rotate_point_90(p, center, quarter_turns) for p in _pts(points, 1, 4999, dim=2)]")],
    'rotate_point_cs': [("    return (c[0] + co * d[0] - si * d[1], c[1] + si * d[0] + co * d[1])", "    return (c[0] + co * d[0] + si * d[1], c[1] + si * d[0] + co * d[1])"),
                        ("    if co * co + si * si != 1:", "    if co * co + si * si > 1:")],
    'rotate_point_cos': [("    si = Surd.sqrt(1 - co * co) * direction", "    si = Surd.sqrt(1 - co * co)"),
                         ("    if abs(co) > 1:", "    if abs(co) >= 1:")],
    'tetrahedron_volume': [("    return abs(_dot(u, _cross3(v, w))) / 6", "    return abs(_dot(u, _cross3(v, w))) / 3"),
                           ("    return abs(_dot(u, _cross3(v, w))) / 6", "    return _dot(u, _cross3(v, w)) / 6")],
    'plane_through': [("    return _primitive(list(n) + [_dot(n, p)])", "    return _primitive(list(n) + [_dot(n, (0, 0, 0))])"),
                      ("    if first < 0:\n        ints = [-v for v in ints]", "    if first > 0:\n        ints = [-v for v in ints]")],
    'dist_point_plane2': [("    return _dot(n, _sub(pt, p)) ** 2 / _dot(n, n)", "    return _dot(n, _sub(pt, p)) ** 2 // _dot(n, n)"),
                          ("    return _dot(n, _sub(pt, p)) ** 2 / _dot(n, n)", "    return _dot(n, pt) ** 2 / _dot(n, n)")],
    'line_plane_intersection': [("        return 'in plane' if num == 0 else 'parallel'", "        return 'parallel' if num == 0 else 'in plane'"),
                                ("    t = num / den\n    return tuple(a[i] + t * d[i] for i in range(3))", "    t = -num / den\n    return tuple(a[i] + t * d[i] for i in range(3))")],
    'skew_lines_dist2': [("        return dist_point_line2(p3, p1, p2)", "        return Fraction(0)"),
                         ("    return _dot(_sub(p3, p1), n) ** 2 / _dot(n, n)\n\n\n@op('geometry.sphere_through4'", "    return _dot(_sub(p3, p1), n) ** 2 / _dot(d1, d1)\n\n\n@op('geometry.sphere_through4'")],
    'sphere_through4': [("    rhs = [_dot(t, t) - _dot(p, p) for t in (q, r, s)]", "    rhs = [_dot(t, t) + _dot(p, p) for t in (q, r, s)]"),
                        ("    return (c, dist2(c, p))", "    return (c, dist2(c, (0, 0, 0)))")],
    'convex_polygon_intersection': [("    return _hull(_clip(_ccw(P), _ccw(Q)))", "    return _clip(_ccw(P), _ccw(Q))"),
                                    ("            if st >= 0:\n                if ss < 0:", "            if st > 0:\n                if ss < 0:")],
    'polygons_union_area': [("    return _slab_area(polys, 'union')", "    return _slab_area(polys, 'inter')"),
                            ("    if len(all_edges) > 300:", "    if len(all_edges) > 400:")],
    'polygons_intersection_area': [("    return _slab_area(polys, 'inter')", "    return _slab_area(polys, 'union')"),
                                   ("    if len(polys) < 2:\n        raise BadInput('at least two polygons are needed')", "    if len(polys) < 1:\n        raise BadInput('at least two polygons are needed')")],
    'regular_ngon_basic': [("'diagonals': n * (n - 3) // 2", "'diagonals': n * (n - 2) // 2"),
                           ("'diameters': n // 2 if n % 2 == 0 else 0", "'diameters': n // 2")],
    'regular_ngon_chords_by_direction': [("(n // 2 - 1 if s % 2 == 0 else n // 2)", "(n // 2 if s % 2 == 0 else n // 2 - 1)"),
                                         ("[(s, s + n // 2) for s in range(n // 2)] if n % 2 == 0 else []", "[(s, s + n // 2) for s in range(n // 2 + 1)] if n % 2 == 0 else []")],
    'regular_ngon_chords_by_length': [("for k in range(1, n // 2 + 1)}", "for k in range(1, n // 2)}"),
                                      ("{k: (n // 2 if 2 * k == n else n) for", "{k: (n // 2 if 2 * k == n else n - 1) for")],
    'regular_ngon_triangles': [("            if 2 * m < n:\n                acute += 1", "            if 2 * m <= n:\n                acute += 1"),
                               ("            isosc = a == b or b == c or a == c", "            isosc = a == b or b == c")],
    'regular_ngon_quadrilaterals': [("kite += (a == b and c == d) or (b == c and d == a)", "kite += (a == b and c == d)"),
                                    ("p1, p2 = a == c, b == d", "p1, p2 = a == c, b == c")],
    'regular_ngon_vertex_rectangles': [("comb(n // 2, 2) if n % 2 == 0 else 0", "comb(n // 2, 2)"),
                                       ("n // 4 if n % 4 == 0 else 0", "n // 4")],
    'regular_ngon_angle': [("steps = n - d2 if d1 < d2 else d2", "steps = d2 if d1 < d2 else n - d2"),
                           ("    return Fraction(180 * steps, n)", "    return Fraction(90 * steps, n)")],
    'regular_ngon_chord_angle': [("diff = Fraction(180 * (a + b - c - d), n) % 180", "diff = Fraction(180 * (a + b + c + d), n) % 180"),
                                 ("    return min(diff, 180 - diff)", "    return diff")],
    'regular_ngon_vertices_numeric': [("out.append((_round(r * c, digits), _round(r * s, digits)))", "out.append((_round(r * s, digits), _round(r * c, digits)))"),
                                      ("    n = _int(n, 3, 10 ** 4)", "    n = _int(n, 3, 10 ** 5)")],
    'regular_ngon_area_exact': [("8: Surd(0, Fraction(1, 2), 2), 12: Surd(Fraction(1, 2))}", "8: Surd(0, Fraction(1, 2), 2), 12: Surd(Fraction(1, 3))}"),
                                ("    return sin[n] * (Fraction(n, 2) * r2)", "    return sin[n] * (Fraction(n, 3) * r2)")],
    'regular_ngon_side_numeric': [("s, _c = _sincos(_pi(ctx.prec + 5) / n)", "s, _c = _sincos(_pi(ctx.prec + 5) / (n + 1))"),
                                  ("        return _round(2 * _dec(rad) * s, digits)", "        return _round(_dec(rad) * s, digits)")],
    'regular_ngon_diagonal_intersections': [("regions = 1 + diagonals + sum((m - 1) * c for m, c in by_m.items())", "regions = 1 + diagonals + sum(m * c for m, c in by_m.items())"),
                                            ("        by_m[m] = by_m.get(m, 0) + 1\n    diagonals", "        by_m[m] = by_m.get(m, 0) + (2 if s > 3 else 1)\n    diagonals")],
    'regular_ngon_rectangles_on_lines': [("                rect += c * (c - 1) // 2", "                rect += c * (c - 1)"),
                                         ("    if n % 2:\n        return {'rectangles': 0, 'squares': 0}", "    if n % 4:\n        return {'rectangles': 0, 'squares': 0}")],
}


def apply(src, old, new):
    n = src.count(old)
    if n != 1:
        return None, n
    return src.replace(old, new), 1


DRIVER = r'''
import json, sys, types, importlib
sys.path.insert(0, {root!r})
import tools.igc.mathbackend as mb
name = 'tools.igc.mathbackend.geometry'
mod = types.ModuleType(name)
mod.__package__ = 'tools.igc.mathbackend'
mod.__file__ = {file!r}
sys.modules[name] = mod
src = open({mfile!r}, encoding='utf-8').read()
exec(compile(src, 'geometry_mutant.py', 'exec'), mod.__dict__)
sim = importlib.import_module('tools.igc.mathbackend.sim_geometry')
rep = sim.run({seed}, {n}, write=False, only={{{op!r}}})
o = rep['ops'][{op!r}]
print('RESULT ' + json.dumps(dict(generated=o['generated'], answered=o['answered'], agree=o['agree'], refused=o['refused'], refused_expected=o['refused_expected'], disagree=o['disagree'])))
'''


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--ops', default='')
    ap.add_argument('--timeout', type=int, default=900)
    ap.add_argument('--tmp', default=str(ROOT / 'build-mathbackend' / 'mutants'))
    ap.add_argument('--out', default=str(ROOT / 'build-mathbackend' / 'geometry_mutation.json'))
    args = ap.parse_args(argv)
    src = SRC.read_text(encoding='utf-8')
    tmp = Path(args.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    want = [o if o.startswith('geometry.') else 'geometry.' + o for o in args.ops.split(',') if o]
    sys.path.insert(0, str(ROOT))
    from tools.igc.mathbackend import registry
    reg = sorted(k for k in registry() if k.startswith('geometry.'))
    missing = [k for k in reg if k.split('.', 1)[1] not in MUT]
    print(f'{len(reg)} registered operations, {len(MUT)} with mutations; without mutations: {missing}')
    rows, survivors = [], []
    for op in reg:
        short = op.split('.', 1)[1]
        if want and op not in want:
            continue
        for idx, (old, new) in enumerate(MUT.get(short, [])):
            mutated, cnt = apply(src, old, new)
            if mutated is None:
                rows.append((op, idx, 'BAD-MUTATION', f'old text occurs {cnt} times'))
                survivors.append((op, idx))
                continue
            mfile = tmp / f'{short}_{idx}.py'
            mfile.write_text(mutated, encoding='utf-8')
            drv = tmp / f'driver_{short}_{idx}.py'
            drv.write_text(DRIVER.format(root=str(ROOT), file=str(SRC), mfile=str(mfile), seed=args.seed, n=args.n, op=op), encoding='utf-8')
            t0 = time.time()
            flags = 0x08000000 if sys.platform == 'win32' else 0     # CREATE_NO_WINDOW
            try:
                p = subprocess.run([sys.executable, '-W', 'ignore', str(drv)], capture_output=True, text=True, timeout=args.timeout, creationflags=flags)
                line = next((l for l in p.stdout.splitlines() if l.startswith('RESULT ')), None)
                if line is None:
                    status, detail = 'CRASH', (p.stderr or p.stdout)[-300:].replace('\n', ' | ')
                    killed = False
                else:
                    r = json.loads(line[7:])
                    unexpected = r['refused'] - r['refused_expected']
                    killed = r['disagree'] > 0
                    status = 'killed' if killed else ('killed-by-refusal-only' if unexpected > 0 else 'SURVIVED')
                    detail = f"disagree {r['disagree']}, unexpected refusals {unexpected}, of {r['generated']} cases"
            except subprocess.TimeoutExpired:
                status, detail, killed = 'timeout', f'>{args.timeout}s', True
            rows.append((op, idx, status, detail + f' [{time.time() - t0:.0f}s]'))
            if not killed:
                survivors.append((op, idx))
            print(f'{op:46s} #{idx}  {status:22s} {detail}', flush=True)
    Path(args.out).write_text(json.dumps(rows, indent=1), encoding='utf-8')
    print(f'\nmutants: {len(rows)}, caught with a disagreement: {sum(1 for r in rows if r[2] in ("killed", "timeout"))}, survivors / not killed by a disagreement: {survivors}')
    return 1 if survivors else 0


if __name__ == '__main__':
    sys.exit(main())
