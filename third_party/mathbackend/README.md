# Code copied into tools/igc/mathbackend

Strata is MIT. Code is copied here only from projects whose licence allows it (MIT, BSD, Apache-2.0, ISC, BSL-1.0, zlib, PSF, public domain), by the developer's decision of 2026-11-05;
copyleft code (GPL, LGPL, AGPL, MPL, EPL) is never copied. Each project's licence text is kept unchanged in `third_party/mathbackend/<project>/LICENSE`, and each copied piece carries a
comment block in its backend file: `Adapted from <project> <version>, <path>, <licence>, Copyright (c) <holder>.`

| project | version | licence | files taken | backend function | changes |
|---|---|---|---|---|---|
| (none) | - | - | - | - | - |

All six backend modules (`counting.py`, `numbers.py`, `algebra.py`, `geometry.py`, `search.py`, `symbolic.py`) are implemented directly from textbook algorithms using Python's standard library (`fractions.Fraction`, `math`, `itertools`, `functools`). SymPy (BSD-3), mpmath (BSD-3), and NetworkX (BSD-3) are used strictly as independent reference oracles in `sim_*.py` and unit tests (`tools/test_mathbackend_*.py`), never imported by the runtime backend modules.
