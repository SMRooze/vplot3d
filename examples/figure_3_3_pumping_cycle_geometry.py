#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thesis Figure 3.3: one pumping cycle (overview of the four phases, side-view 3D).

Same scene as the end of step 4 of 'python pumping_cycle.py --view side3d'
(camera, closed cycle, figure-eight during powered reel-out, single kite at the
end of the RI->RO transition, faint tether and ground grid); the pumping-cycle
model is not changed. Additions only:
  - apparent wind label V_app,v -> V_app,ship (bold italic vector, as in Fig. 3.1)
  - reel-speed conditions v_r = v_out > 0 and v_r = v_in < 0 under the RO and RI phase names
  - apparent wind arrow black (as in Figs. 3.1, 3.2)
  - canvas cropped to the drawing
Phase-end fractions, beta_RORI,end, beta_trac and phi_c are defined in the text.

Run from the examples folder:  python figure_3_3_pumping_cycle_geometry.py
Outputs (folder pumping_cycle_view_side3d, existing files are not touched):
  figure_3_3_pumping_cycle_geometry.svg / .pdf / .png
"""

import sys
from pathlib import Path

# --detailed: step-6 variant with the phase-end fractions, beta_trac (black: fixed, not an optimisation
# variable), beta_RORI,end with its construction line O -> P3, and phi_c; own output name
DETAILED = '--detailed' in sys.argv

# Shared scene of pumping_cycle.py, side3d camera (everything before its output section)
sys.argv = [sys.argv[0], '--view', 'side3d']
_src = Path('pumping_cycle.py').read_text(encoding='utf-8').split('# Output\n')[0]
_old = r"label(r'$V_{\mathrm{app{,}v}}$'"
assert _src.count(_old) == 1
_src = _src.replace(_old, r"label(r'$\boldsymbol{V}_{\mathrm{app{,}ship}}$'")
_old = "linewidth=2.6, zorder=Z_TOP, color=WIND)"       # apparent wind arrow: black, as in Figs. 3.1, 3.2
assert _src.count(_old) == 1
_src = _src.replace(_old, "linewidth=2.6, zorder=Z_TOP, color='k')")
exec(compile(_src, 'pumping_cycle.py', 'exec'))

from thesis_export import crop_svg, svg_to_pdf, svg_to_png

PNG_SCALE = 2.5
STEM      = OUT_DIR / ('figure_3_3_pumping_cycle_geometry' + ('_detailed' if DETAILED else ''))
assert OUT_DIR.name == 'pumping_cycle_view_side3d'

W, H = CAMERA['width'], CAMERA['height']
draw_background()                      # cycle, arrows, phase labels, V_app,ship, ground grid
if DETAILED:
    # step 6 of pumping_cycle.py: beta_trac (black label, grey dotted arc), f_retr,end, f_trac,end,
    # v_out, v_in, beta_RORI,end, phi_c; construction line O -> end of RO->RI transition (P3)
    annotations(STEP_KEYS[6])
    assert np.allclose(P3, sph(F_TRAC_END, PHI_C, BETA_RORI))
    Line(PO, P3, linewidth=0.9, color='#C4C4C4', zorder=Z_GRID + 1)
    draw_kite('RO', 0.0)               # single kite, as in the step-6 still
else:
    def below_phase(text, P, off, lines=1.3):
        '''Blue label centred one line below a phase name anchored at P with offset off (offsets are
        in units of each label's own font size: phase names FS_PATH, math labels FS).'''
        o = (np.array(off)*FS_PATH - [0, lines*FS_PATH])/FS
        return label(text, P, tuple(o), color=CYAN)

    # reel-speed conditions under the phase names (replace the separate v_out / v_in labels)
    below_phase(r'$v_r = v_{\mathrm{out}} > 0$', Pm_out(), LBL['ro'])
    below_phase(r'$v_r = v_{\mathrm{in}} < 0$', phase_pos('RI', 0.35), LBL['ri'])
    draw_kite('RIRO', 1.0)             # single kite: end of step 4 (end of RI->RO, start of RO)

save_svg(str(STEM))
svg = STEM.with_suffix('.svg')
Path(str(STEM) + '_tex.svg').replace(svg)
w, h = crop_svg(svg, margin=24)
svg_to_png(svg, STEM.with_suffix('.png'), w*PNG_SCALE, h*PNG_SCALE)
_chk = __import__('os').environ.get('PDFCHECK_DIR')
svg_to_pdf(svg, STEM.with_suffix('.pdf'), w, h, check_png=Path(_chk) / 'f33_pdfcheck.png' if _chk else None)
print(f'> -> {STEM}.svg/.pdf/.png ({w:g} x {h:g} px)')
