#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thesis Figure 3.1: powered-crosswind geometry for the centred reference case.

Identical to frame 2 of 'python wind_window.py --axes --phirep' (same camera,
scene, kite position, e_r and e_t), with the apparent wind label renamed from
V_app,v to V_app,ship. Nothing else is changed; no physics is touched.

Run from the examples folder:  python figure_3_1_powered_crosswind_geometry.py
Outputs (folder wind_window_axes_phirep, existing files are not touched):
  figure_3_1_powered_crosswind_geometry.svg / .pdf / .png
"""

import sys
from pathlib import Path

FRAME     = 2                          # animation frame of wind_window.py --axes --phirep
PNG_SCALE = 2.5                        # 4000 x 3000 px

# Shared scene of wind_window.py with the --axes --phirep camera; only the V label text is renamed
sys.argv = [sys.argv[0], '--axes', '--phirep']
_src = Path('wind_window.py').read_text(encoding='utf-8').split('# Static figures')[0]
_src = _src.replace(r"r'$\boldsymbol{V}_{\mathrm{app,v}}$'", r"r'$\boldsymbol{V}_{\mathrm{app,ship}}$'")
assert _src.count(r"r'$\boldsymbol{V}_{\mathrm{app,ship}}$'") == 1    # (wind_window.py itself now uses app,ship)
# representative-azimuth label with its value (the label of the --phirep static figure / frames)
_old = r"label(r'$\phi_{\mathrm{QSM,rep}} = \mathit{\Delta}\phi$', Pproj, PHIREP_LBL"
assert _src.count(_old) == 1
_src = _src.replace(_old, r"label(r'$\phi_{\mathrm{QSM,rep}} = \mathit{\Delta}\phi = 11.5^\circ$', Pproj, PHIREP_LBL")
_src = _src.replace("PHIREP_LBL = (2.3, -0.7)", "PHIREP_LBL = (3.6, -0.7)")   # longer label: keep its left end in place
exec(compile(_src, 'wind_window.py', 'exec'))

from thesis_export import svg_to_pdf, svg_to_png

assert OUT_DIR.name == 'wind_window_axes_phirep'
STEM = OUT_DIR / 'figure_3_1_powered_crosswind_geometry'

# Same drawing sequence as one animation frame of wind_window.py
static_objs, Prep = draw_static()
tk = 2*np.pi*FRAME/N_FRAMES
kite_objs, knot = draw_kite(tk)
teth_objs = draw_tether(knot)
axes_objs = draw_kite_axes(tk)

save_svg(str(STEM))
svg = STEM.with_suffix('.svg')
Path(str(STEM) + '_tex.svg').replace(svg)
svg_to_png(svg, STEM.with_suffix('.png'), W*PNG_SCALE, H*PNG_SCALE)
assert abs(np.degrees(DPHI) - 11.5) < 1e-9          # value in the phi_QSM,rep label
svg_to_pdf(svg, STEM.with_suffix('.pdf'), W, H, check_png=Path(_scratch) / 'f31_pdfcheck.png' if (_scratch := __import__('os').environ.get('PDFCHECK_DIR')) else None)
print(f'> frame {FRAME} -> {STEM}.svg/.pdf/.png')
