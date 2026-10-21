#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Presentation animation of one pumping cycle, one clip per phase.

Scene of the simple thesis Figure 3.3 (figure_3_3_pumping_cycle_geometry.py):
side3d camera of pumping_cycle.py, closed cycle with direction arrows, the four
phase names, v_r = v_out > 0 and v_r = v_in < 0 under the RO and RI names,
black V_app,ship, plus the pattern-centre direction phi_c on the ground. No
phase-end fractions or elevation parameters. The kite and its tether move
through the phases; the cycle model (phase_pos) of pumping_cycle.py is unchanged.

Kite attitude: as pumping_cycle.py, except that the heading also blends at the
start of powered reel-out (coming out of the RI->RO descent), so the full cycle
loops without a jump in the kite's orientation (visual only).

Run from the examples folder:  python presentation_pumping_cycle.py [--stills]
Outputs (new folder pumping_cycle_presentation, nothing else is touched):
  phase1_RO.mp4, phase2_RO-RI.mp4, phase3_RI.mp4, phase4_RI-RO.mp4   one clip per phase
  full_cycle_loop.mp4                 the four phases back to back, loops seamlessly
  phaseN_start.png, phaseN_end.png    first / last frame of every clip (end of N = start of N+1)
  frames/phaseN/frame_XXX.png         frame sequences

Representative state: during powered reel-out only, one dark-red marker at the
representative QSM state (rep_state of pumping_cycle.py: phi_QSM,rep, beta_trac,
at the current tether length), as in step 1 of pumping_cycle.py; no label.

Re-render selected phases only:  python presentation_pumping_cycle.py --only RO
"""

import sys
import shutil
import subprocess
from pathlib import Path

_STILLS = '--stills' in sys.argv      # pumping_cycle.py defines its own STILLS_ONLY: restore ours after exec
# --only RO,RI ...: re-render only these phases (the loop is always rebuilt from all frame folders)
_ONLY = sys.argv[sys.argv.index('--only') + 1].split(',') if '--only' in sys.argv else None

# Shared scene of pumping_cycle.py, side3d camera (everything before its output section)
sys.argv = [sys.argv[0], '--view', 'side3d']
_src = Path('pumping_cycle.py').read_text(encoding='utf-8').split('# Output\n')[0]
for _old, _new in ((r"label(r'$V_{\mathrm{app{,}v}}$'", r"label(r'$\boldsymbol{V}_{\mathrm{app{,}ship}}$'"),
                   ("linewidth=2.6, zorder=Z_TOP, color=WIND)", "linewidth=2.6, zorder=Z_TOP, color='k')")):
    assert _src.count(_old) == 1, _old
    _src = _src.replace(_old, _new)
exec(compile(_src, 'pumping_cycle.py', 'exec'))
STILLS_ONLY = _STILLS

OUT_DIR = Path('pumping_cycle_presentation')
CLIPS = [('phase1_RO', 'RO'), ('phase2_RO-RI', 'RORI'), ('phase3_RI', 'RI'), ('phase4_RI-RO', 'RIRO')]
RO_BLEND = 0.05                       # fraction of reel-out over which the heading turns into the eight
# reel-in name + v_in condition left of the reel-in arc: the kite (canopy up-right of the bridle knot)
# and the tether never pass there, whereas the static figure's position is crossed by the kite
LBL['ri'] = (-9.9, 1.1)
LBL['riro'] = (-9.0, 3.12)            # RI->RO name slightly left: clear of the kite at the start of the descent

###############################################################################
# Kite attitude with a heading blend at the start of reel-out as well
###############################################################################
_kite_frame_cycle = kite_frame

def kite_frame(phase, s):
    if phase != 'RO' or s >= RO_BLEND:
        return _kite_frame_cycle(phase, s)
    h, er = tangential_heading('RO', s)
    hp, _ = tangential_heading('RIRO', 1.0)
    hp = hp - hp.dot(er)*er
    w = smooth(s/RO_BLEND)
    h = (1 - w)*hp/np.linalg.norm(hp) + w*h
    ex = h/np.linalg.norm(h)
    return ex, np.cross(er, ex), er

# seamless phase boundaries: same kite pose at the end of one clip and the start of the next
for (_, a), (_, b) in zip(CLIPS, CLIPS[1:] + CLIPS[:1]):
    assert np.allclose(phase_pos(a, 1.0), phase_pos(b, 0.0))
    assert np.allclose(kite_frame(a, 1.0)[0], kite_frame(b, 0.0)[0], atol=1e-3), (a, b)

###############################################################################
# Static scene
###############################################################################
def below_phase(text, P, off, lines=1.3):
    '''Blue label centred one line below a phase name (offsets in each label's own font size).'''
    o = (np.array(off)*FS_PATH - [0, lines*FS_PATH])/FS
    return label(text, P, tuple(o), color=CYAN)

draw_background()                     # cycle, arrows, phase names, V_app,ship, ground grid
below_phase(r'$v_r = v_{\mathrm{out}} > 0$', Pm_out(), LBL['ro'])
below_phase(r'$v_r = v_{\mathrm{in}} < 0$', phase_pos('RI', 0.35), LBL['ri'])
annotations(('phi_c',))               # pattern-centre direction on the ground

###############################################################################
# Output
###############################################################################
W, H = CAMERA['width'], CAMERA['height']

def write_retry(path, data, tries=20):
    import time
    for i in range(tries):
        try:
            return path.write_bytes(data)
        except OSError:
            time.sleep(0.25*(i + 1))
    path.write_bytes(data)

def save_png(png):
    import resvg_py
    png = Path(png)
    png.parent.mkdir(parents=True, exist_ok=True)
    stem = png.with_suffix('')
    save_svg(str(stem))
    svg = Path(str(stem) + '_tex.svg')
    write_retry(png, bytes(resvg_py.svg_to_bytes(svg_path=str(svg), width=W, height=H, background='#ffffff')))
    svg.unlink()

def encode(frames_dir, mp4):
    import imageio_ffmpeg
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-framerate', str(FPS),
                    '-i', str(Path(frames_dir) / 'frame_%03d.png'),
                    '-pix_fmt', 'yuv420p', '-c:v', 'libx264', '-crf', '18', str(mp4)], check=True)

OUT_DIR.mkdir(exist_ok=True)
for name, phase in CLIPS:
    if _ONLY and phase not in _ONLY:
        continue
    n = FRAMES[phase]
    fdir = OUT_DIR / 'frames' / name
    shutil.rmtree(fdir, ignore_errors=True)
    fdir.mkdir(parents=True)
    for k in ([0, n - 1] if STILLS_ONLY else range(n)):
        kite = draw_kite(phase, k/(n - 1), rep=True)     # rep: red representative state, reel-out only
        save_png(fdir / f'frame_{k:03d}.png')
        remove(kite)
        print(f'> {name} frame {k+1}/{n}')
    write_retry(OUT_DIR / f'{name}_start.png', (fdir / 'frame_000.png').read_bytes())
    write_retry(OUT_DIR / f'{name}_end.png', (fdir / f'frame_{n-1:03d}.png').read_bytes())
    if not STILLS_ONLY:
        encode(fdir, OUT_DIR / f'{name}.mp4')

if not STILLS_ONLY:
    # loop: phases back to back, last frame of each phase dropped (it equals the next phase's first)
    cdir = OUT_DIR / 'frames' / 'full_cycle_loop'
    shutil.rmtree(cdir, ignore_errors=True)
    cdir.mkdir(parents=True)
    k = 0
    for name, _ in CLIPS:
        for f in sorted((OUT_DIR / 'frames' / name).glob('frame_*.png'))[:-1]:
            write_retry(cdir / f'frame_{k:03d}.png', f.read_bytes())
            k += 1
    encode(cdir, OUT_DIR / 'full_cycle_loop.mp4')

plt.close()
print('> done:', OUT_DIR.resolve())
