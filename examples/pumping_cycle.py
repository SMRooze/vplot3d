#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pumping cycle of a ship-towing kite in 3D: four phases, one clip per slide.

Rebuilds the 2D pumping-cycle sequence (thesis Fig. 2.2, slides 11-16) in 3D:
powered reel-out RO (figure-eight at beta_trac while the tether pays out),
RO->RI transition (climb at constant tether length), reel-in RI (tether
shortens, elevation evolves), RI->RO transition (descent at constant tether
length). The closed cycle is drawn as a static background; kite, tether and
the step-specific control annotations change per step.

Geometry follows the thesis: the crosswind pattern is centred at the pattern
centre azimuth phi_c with lobe offset Delta_phi (Eq. A.4); during reel-in the
elevation evolves from beta_RORI,end to BETA_RI_END (Fig. 2.2 "beta_k evolves").

Outputs (folder OUT_DIR):
  step1.mp4 ... step6.mp4      one clip per slide (steps 5, 6 static)
  full_cycle.mp4               steps 1-4 back to back
  stepN_start.png, stepN_end.png   first/last frame of every step (poster frames)
  frames/stepN/frame_XXX.png   frame sequences

Run from the examples folder:  python pumping_cycle.py [--stills]
  --stills  only the start/end stills of every step (quick layout check)
"""

import sys
import shutil
import subprocess
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

data_path = Path.cwd().parent / 'data'
sys.path.append(str(data_path))

import vplot3d.vplot3d as v3d
from vplot3d.vplot3d import init_view, Line, Vector, Point, ArcMeasure, Annotation3D, save_svg
from kiteV3 import KiteV3

###############################################################################
# PARAMETERS
###############################################################################
# Camera (fixed for all steps so the clips line up between slides)
CAMERA = dict(width=1920, height=1080, elev=22.0, azim=-145.0, zoom=1.56,
              xmin=-0.1, xmax=1.0, ymin=-0.5, ymax=0.5, zmin=-0.07, zmax=0.88)

# Cycle geometry (tether length normalised with l_max = 1)
F_RETR_END   = 0.50                 # tether length fraction at start of reel-out (P1, P4)
F_TRAC_END   = 0.95                 # tether length fraction at end of reel-out (P2, P3)
BETA_TRAC    = np.deg2rad(30.0)     # reel-out elevation
BETA_RORI    = np.deg2rad(55.0)     # beta_RORI,end: elevation at which reel-in begins (P3)
BETA_RI_END  = np.deg2rad(78.0)     # elevation at the end of reel-in (P4); beta_k evolves during RI
PHI_C        = np.deg2rad(0.0)      # pattern centre azimuth
DPHI         = np.deg2rad(11.5)     # lobe offset Delta_phi
B_ELEV       = np.deg2rad(7.0)      # elevation amplitude of the figure-eight (before PATTERN_SCALE)
PATTERN_SCALE = 0.55                # schematic size of the drawn figure-eight relative to the cycle
N_LOOPS      = 3                    # complete figure-eights during reel-out
T_EXIT       = 1.25*np.pi           # reel-out ends here in the next eight: top of the left lobe

# Timing (30 fps)
FPS          = 30
FRAMES       = dict(RO=90, RORI=60, RI=75, RIRO=60)   # 3 s, 2 s, 2.5 s, 2 s
STATIC_SEC   = 2.0                  # length of the static clips (steps 5, 6)

# Kite attitude: no rigid-body pitch, so the bridle always stays aligned with the tether
# (depowering changes the wing relative to the bridle, which the mesh model does not resolve)
PITCH = dict(RO=(0, 0), RORI=(0, 0), RI=(0, 0), RIRO=(0, 0))       # deg at phase start/end
KITE_SCALE   = 0.016                # mesh units (m) -> l_max units
HEADING_BLEND = 0.25                # fraction of a phase over which the heading turns into the new path

# Styling
CYAN   = '#00A6D6'                  # optimisation variables only (TU Delft cyan)
WIND   = '#E0201B'                  # apparent wind arrow
RED    = '#9B1B1B'                  # tether end at P1 (step 1)
TETHER = '#A0A0A0'
GRID   = '#E4E4E4'
FS     = 34                         # label font size (px)
FS_PATH = 30                        # path label font size
LW_PATH = 2.0

# Alternative camera views: python pumping_cycle.py --view NAME  (own output folder each)
VIEWS = {
    'side':      dict(elev=8.0,  azim=-92.0),    # like the 2D figure: x to the right, eights seen edge-on
    'side3d':    dict(elev=15.0, azim=-112.0),   # mostly from the side, eights just opening up
    'oblique':   dict(elev=22.0, azim=-128.0),   # between side and default
    'upwind':    dict(elev=24.0, azim=-160.0),   # more from upwind, symmetric eights
}
VIEW = sys.argv[sys.argv.index('--view') + 1] if '--view' in sys.argv else None
if VIEW:
    CAMERA.update(VIEWS[VIEW])
CHECK = '--check' in sys.argv       # only report text overlaps, render nothing

# Label offsets (font units, x right / y up) and positions of the reel-out and wind arrows.
# Defaults are tuned for the default camera; LAYOUT[view] overrides individual entries.
LBL = dict(ro=(5.0, 1.0), rori=(3.0, 0.0), ri=(2.9, 0.3), riro=(-4.6, -1.5), vapp=(0.0, -1.1),
           beta_trac=(1.9, -0.8), beta_rori=(-2.5, -0.3), f_retr=(1.9, -1.0), f_trac=(2.4, -0.5),
           v_out=(1.3, -0.4), v_in=(-1.2, 0.0), phi_c=(1.0, -0.1))
RO_ARROW = dict(r=0.70, dphi=-10.0, dbeta=0.0)     # beside the pattern; dphi beyond the lobe tip (deg)
WIND_POS = (-0.05, 0.36, 0.50)
LAYOUT = {   # per-view arrow placement (label offsets come from pumping_cycle_layouts.json)
    'side':   dict(ro_arrow=dict(r=0.66, dphi=12.65, dbeta=-9.0), wind=(-0.12, 0.0, 0.30)),
    'side3d': dict(ro_arrow=dict(r=0.66, dphi=12.65, dbeta=-9.0), wind=(-0.12, 0.1, 0.30)),
}
RO_ARROW.update(LAYOUT.get(VIEW, {}).get('ro_arrow', {}))
WIND_POS = LAYOUT.get(VIEW, {}).get('wind', WIND_POS)
LABEL_STEPS = dict(ro=range(1, 7), rori=range(1, 7), ri=range(1, 7), riro=range(1, 7), vapp=range(1, 7),
                   f_retr=(1, 5, 6), f_trac=(1, 5, 6), v_out=(1, 5, 6), beta_trac=(1, 5, 6),
                   v_in=(3, 5, 6), beta_rori=(2, 6), phi_c=(1, 5, 6))
if '--layout' not in sys.argv and Path('pumping_cycle_layouts.json').exists():
    import json
    LBL.update({k: tuple(v) for k, v in
                json.loads(Path('pumping_cycle_layouts.json').read_text()).get(VIEW or 'default', {}).items()})

OUT_DIR = Path('pumping_cycle' + (f'_view_{VIEW}' if VIEW else ''))
STILLS_ONLY = '--stills' in sys.argv

###############################################################################
# Geometry
###############################################################################
EX, EY, EZ = np.eye(3)
PO = np.zeros(3)

def sph(r, phi, beta):
    return r*np.array([np.cos(beta)*np.cos(phi), np.cos(beta)*np.sin(phi), np.sin(beta)])

def smooth(s):
    return 0.5 - 0.5*np.cos(np.pi*s)

def pattern_angles(t):
    '''Figure-eight centred at (phi_c, beta_trac), lobe centres near phi_c -+ Delta_phi (scaled).'''
    return (PHI_C + PATTERN_SCALE*2*DPHI*np.sin(t),
            BETA_TRAC + PATTERN_SCALE*B_ELEV*np.sin(2*t))

def phase_pos(phase, s):
    '''Kite (bridle knot) position in phase at progress s in [0, 1].'''
    if phase == 'RO':
        t = (2*np.pi*N_LOOPS + T_EXIT)*s
        r = F_RETR_END + (F_TRAC_END - F_RETR_END)*s
        return sph(r, *pattern_angles(t))
    if phase == 'RORI':
        phi2, beta2 = pattern_angles(T_EXIT)
        return sph(F_TRAC_END, phi2 + (PHI_C - phi2)*smooth(s), beta2 + (BETA_RORI - beta2)*smooth(s))
    if phase == 'RI':
        r = F_TRAC_END + (F_RETR_END - F_TRAC_END)*s
        return sph(r, PHI_C, BETA_RORI + (BETA_RI_END - BETA_RORI)*s)
    if phase == 'RIRO':
        return sph(F_RETR_END, PHI_C, BETA_RI_END + (BETA_TRAC - BETA_RI_END)*smooth(s))

def tangential_heading(phase, s):
    '''Unit tangential part (perpendicular to the tether) of the path direction.'''
    ds = 1e-4
    s0, s1 = max(s - ds, 0), min(s + ds, 1)
    d  = phase_pos(phase, s1) - phase_pos(phase, s0)
    er = phase_pos(phase, s)/np.linalg.norm(phase_pos(phase, s))
    h  = d - d.dot(er)*er
    return h/np.linalg.norm(h), er

PHASES = ['RO', 'RORI', 'RI', 'RIRO']

def kite_frame(phase, s):
    '''Body frame (e_x heading, e_y span, e_z along tether) incl. heading blend and depower pitch.'''
    h, er = tangential_heading(phase, s)
    i = PHASES.index(phase)
    prev = PHASES[i - 1]
    if phase != 'RO' and s < HEADING_BLEND:
        hp, _ = tangential_heading(prev, 1.0)
        hp = hp - hp.dot(er)*er
        w = smooth(s/HEADING_BLEND)
        h = (1 - w)*hp/np.linalg.norm(hp) + w*h
        h = h/np.linalg.norm(h)
    ex, ez = h, er
    ey = np.cross(ez, ex)
    p0, p1 = PITCH[phase]
    p = np.deg2rad(p0 + (p1 - p0)*s)
    return np.cos(p)*ex - np.sin(p)*ez, ey, np.sin(p)*ex + np.cos(p)*ez

P1, P2 = phase_pos('RO', 0), phase_pos('RO', 1)
P3, P4 = phase_pos('RI', 0), phase_pos('RI', 1)

###############################################################################
# Scene
###############################################################################
fig = plt.figure()
ax  = fig.add_subplot(projection='3d', proj_type='ortho', computed_zorder=False)
ax.set_axis_off()
init_view(**CAMERA)
v3d.RAW_MATH = False
mpl.rcParams['svg.fonttype']     = 'path'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family']      = 'STIXGeneral'
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

Z_GRID, Z_PATH, Z_ANN, Z_TETHER, Z_KITE, Z_TOP = 1, 3, 4, 5, 6, 10

def label(text, P, xytext, fontsize=FS, color='k', name=None, **kw):
    a = Annotation3D(text, P, xytext=xytext, fontsize=fontsize, color=color,
                     ha=kw.pop('ha', 'center'), va=kw.pop('va', 'center'), **kw)
    a.check_name = name
    return a

def path_arrow(phase, s, length=0.05):
    '''Small arrowhead on the path at progress s, pointing along the motion.'''
    d = phase_pos(phase, min(s + 1e-3, 1)) - phase_pos(phase, max(s - 1e-3, 0))
    e = d/np.linalg.norm(d)
    P = phase_pos(phase, s)
    return Vector(P - 0.5*length*e, length*e, linewidth=LW_PATH, zorder=Z_PATH, color='k')

def draw_background():
    objs = []
    # faint ground grid
    for x in np.arange(0.0, 1.01, 0.1):
        objs += ax.plot([x, x], [-0.2, 0.2], [0, 0], color=GRID, linewidth=0.8, zorder=Z_GRID)
    for y in np.arange(-0.2, 0.21, 0.1):
        objs += ax.plot([0.0, 1.0], [y, y], [0, 0], color=GRID, linewidth=0.8, zorder=Z_GRID)
    objs.append(Point(PO, scale=0.9, shape='Point1M', zorder=Z_TOP, color='k'))

    # closed cycle: four paths, arrowheads, corner points
    for ph in PHASES:
        n = 1200 if ph == 'RO' else 200
        Pp = np.array([phase_pos(ph, s) for s in np.linspace(0, 1, n)]).T
        objs += ax.plot(*Pp, color='k', linewidth=LW_PATH if ph != 'RO' else 1.4, zorder=Z_PATH)
    for ph, s in (('RORI', 0.55), ('RI', 0.5), ('RIRO', 0.5)):
        objs.append(path_arrow(ph, s))
    for P in (P1, P2, P3, P4):
        objs.append(Point(P, scale=0.9, shape='Point1M', zorder=Z_PATH+1, color='k', bgcolor='k'))

    # reel-out direction arrow beside the RO pattern (on the negative-phi side, right in this view)
    Pm = Pm_out()
    objs.append(Vector(Pm, 0.1*sph(1, PHI_C, BETA_TRAC), linewidth=LW_PATH, zorder=Z_PATH, color='k'))

    # path labels
    objs.append(label('powered reel-out, RO', Pm, LBL['ro'], FS_PATH, name='ro'))
    objs.append(label(r'RO$\,\rightarrow\,$RI' '\n' 'transition', phase_pos('RORI', 0.5), LBL['rori'], FS_PATH, name='rori'))
    objs.append(label('reel-in, RI', phase_pos('RI', 0.35), LBL['ri'], FS_PATH, name='ri'))
    objs.append(label(r'RI$\,\rightarrow\,$RO' '\n' 'transition', phase_pos('RIRO', 0.5), LBL['riro'], FS_PATH, name='riro'))

    # apparent wind
    Pw = np.array(WIND_POS)
    objs.append(Vector(Pw, 0.18*EX, linewidth=2.6, zorder=Z_TOP, color=WIND))
    objs.append(label(r'$V_{\mathrm{app{,}v}}$', Pw + 0.08*EX, LBL['vapp'], FS + 4, name='vapp'))
    return objs

def Pm_out():
    '''Start of the reel-out direction arrow (see draw_background).'''
    return sph(RO_ARROW['r'], PHI_C - PATTERN_SCALE*2*DPHI + np.deg2rad(RO_ARROW['dphi']),
               BETA_TRAC + np.deg2rad(RO_ARROW['dbeta']))

def tick(P, text, xytext, name):
    '''Cyan tick across the reel-out path at P with a label.'''
    er = P/np.linalg.norm(P)
    eb = np.array([-np.sin(BETA_TRAC)*np.cos(PHI_C), -np.sin(BETA_TRAC)*np.sin(PHI_C), np.cos(BETA_TRAC)])
    return [Line(P - 0.035*eb, 0.07*eb, linewidth=2.4, color=CYAN, zorder=Z_ANN),
            label(text, P, xytext, color=CYAN, name=name)]

def annotations(keys):
    '''Step-specific control annotations.'''
    objs = []
    ground = sph(1, PHI_C, 0)
    if 'beta_trac' in keys:
        R = 0.22
        objs.append(ArcMeasure(PO, ground, sph(1, PHI_C, BETA_TRAC), radius=R, linewidth=1.2,
                               color='#707070', dashes=(2, 2), zorder=Z_ANN))
        objs.append(label(r'$\beta_{\mathrm{trac}} = 30^\circ$', sph(R, PHI_C, 0), LBL['beta_trac'], name='beta_trac'))
    if 'beta_rori' in keys:
        R = 0.4
        objs.append(ArcMeasure(PO, ground, sph(1, PHI_C, BETA_RORI), radius=R, linewidth=1.8,
                               color=CYAN, dashes=(6, 2, 1, 2), zorder=Z_ANN))
        objs.append(label(r'$\beta_{\mathrm{RORI{,}end}}$', sph(R, PHI_C, BETA_RORI*0.5), LBL['beta_rori'], color=CYAN, name='beta_rori'))
    if 'f' in keys:
        objs += tick(P1, r'$f_{\mathrm{retr{,}end}}$', LBL['f_retr'], 'f_retr')
        objs += tick(P2, r'$f_{\mathrm{trac{,}end}}$', LBL['f_trac'], 'f_trac')
    if 'v_out' in keys:
        objs.append(label(r'$v_{\mathrm{out}}$', Pm_out(), LBL['v_out'], color=CYAN, name='v_out'))
    if 'v_in' in keys:
        objs.append(label(r'$v_{\mathrm{in}}$', phase_pos('RI', 0.45), LBL['v_in'], color=CYAN, name='v_in'))
    if 'phi_c' in keys:
        # pattern centre direction in the e_x-e_y plane (phi_c = 0: along e_x)
        Pg = sph(F_TRAC_END*np.cos(BETA_TRAC), PHI_C, 0)
        objs.append(Line(PO, Pg, linewidth=1.8, color=CYAN, dashes=(6, 3), zorder=Z_ANN))
        objs.append(label(r'$\phi_c$', Pg, LBL['phi_c'], color=CYAN, name='phi_c'))
    return objs

def rep_state(s):
    '''Representative QSM state during reel-out: (phi_QSM,rep, beta_trac) at the current tether
    length. phi_QSM,rep = arccos(cos phi_c cos Delta_phi); drawn at the schematic lobe centre.'''
    phi_rep = np.arccos(np.cos(PHI_C)*np.cos(DPHI))
    r = F_RETR_END + (F_TRAC_END - F_RETR_END)*s
    return sph(r, PHI_C + PATTERN_SCALE*phi_rep, BETA_TRAC)

STEP_KEYS = {
    1: ('beta_trac', 'f', 'v_out', 'rep', 'phi_c'),
    2: ('beta_rori',),
    3: ('v_in',),
    4: (),
    5: ('beta_trac', 'f', 'v_out', 'v_in', 'phi_c'),
    6: ('beta_trac', 'f', 'v_out', 'v_in', 'beta_rori', 'phi_c'),
}
STEP_PHASE = {1: 'RO', 2: 'RORI', 3: 'RI', 4: 'RIRO', 5: None, 6: None}

def draw_kite(phase, s, rep=False):
    P = phase_pos(phase, s)
    ex, ey, ez = kite_frame(phase, s)
    kv = KiteV3.rotated(PO, e1=ex, e2=ey, e3=ez, voff=P, scale=KITE_SCALE)
    kv.lines.set_zorder(Z_KITE)
    kv.mesh.set_zorder(Z_KITE+1)
    teth = Line(PO, P, linewidth=1.4, color=TETHER, zorder=Z_TETHER)
    objs = [kv, teth]
    if rep and phase == 'RO':
        objs.append(Point(rep_state(s), scale=1.4, shape='Point1M', zorder=Z_TOP, color=RED, bgcolor=RED))
    return objs

def remove(objs):
    for o in objs:
        o.remove()

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

def copy_retry(src, dst):
    '''File copy that survives a sync client (OneDrive) briefly locking the target.'''
    write_retry(Path(dst), Path(src).read_bytes())

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

def check_overlaps(n_samples=13, pad=3):
    '''Report every label that overlaps another label, a drawn line (paths, arcs, arrows,
    ticks, tether), a point marker, the kite (sampled along each phase) or the canvas edge.'''
    from mpl_toolkits.mplot3d import proj3d
    ax.set_box_aspect([1, 1, 1], zoom=v3d.PLOT_ZOOM)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    def to_disp(X):
        x, y, _ = proj3d.proj_transform(*np.asarray(X, float).T, ax.get_proj())
        return ax.transData.transform(np.column_stack([x, y]))

    def densify(P, step=2.0):
        out = [P[:1]]
        for a, b in zip(P[:-1], P[1:]):
            n = max(int(np.hypot(*(b - a))/step), 1)
            out.append(a + (b - a)*np.linspace(0, 1, n + 1)[1:, None])
        return np.vstack(out)

    def inside(pts, bb):
        return np.any((pts[:, 0] > bb.x0 + pad) & (pts[:, 0] < bb.x1 - pad) &
                      (pts[:, 1] > bb.y0 + pad) & (pts[:, 1] < bb.y1 - pad))

    found = set()
    draw_background()
    for step in range(1, 7):
        ann = annotations(STEP_KEYS[step])
        phase = STEP_PHASE[step]
        samples = [('RO', 0.0)] if phase is None else [(phase, s) for s in np.linspace(0, 1, n_samples)]
        for ph, s in samples:
            kite = draw_kite(ph, s, rep='rep' in STEP_KEYS[step])
            fig.canvas.draw()
            r = fig.canvas.get_renderer()
            labels = [c for c in ax.get_children() if isinstance(c, Annotation3D)]
            boxes = {(c.check_name or c.get_text()): c.get_window_extent(r) for c in labels}
            fb = fig.bbox
            for n, b in boxes.items():
                if b.x0 < 0 or b.y0 < 0 or b.x1 > fb.width or b.y1 > fb.height:
                    found.add((step, n, 'canvas edge'))
            names = list(boxes)
            for i, a in enumerate(names):
                for b_ in names[i+1:]:
                    if boxes[a].overlaps(boxes[b_]):
                        found.add((step, a, 'label ' + b_))
            for ln in ax.lines:
                if ln.get_color() == GRID or not ln.get_visible():
                    continue
                pts = densify(to_disp(np.column_stack(ln.get_data_3d())))
                what = 'tether' if ln.get_color() == TETHER else 'line ' + str(ln.get_gid() or '')
                for n, b in boxes.items():
                    if inside(pts, b):
                        found.add((step, n, what))
            for pt in v3d.points:
                q = to_disp([pt.p])[0]
                for n, b in boxes.items():
                    if b.x0 - 6 < q[0] < b.x1 + 6 and b.y0 - 6 < q[1] < b.y1 + 6:
                        found.add((step, n, 'point marker'))
            kv = kite[0]
            kp = to_disp(np.vstack([kv.mesh_nodes, kv.polyline_nodes]))
            for n, b in boxes.items():
                if inside(kp, b):
                    found.add((step, n, 'kite'))
            remove(kite)
        remove(ann)
    print(f'> overlap check, view = {VIEW or "default"}: {len(found)} issue(s)')
    for f in sorted(found):
        print('   step %d  %-10s x %s' % f)
    return found

LAYOUT_FILE = Path('pumping_cycle_layouts.json')

def auto_layout(n_samples=13, pad=3):
    '''Place every label at the nearest offset (spiral around its anchor) that is clear of
    all lines, point markers, other labels, the kite and the red dot over the whole cycle and
    the canvas edge; tether crossings are penalised. Result is stored per view in LAYOUT_FILE.'''
    import json
    from mpl_toolkits.mplot3d import proj3d
    ax.set_box_aspect([1, 1, 1], zoom=v3d.PLOT_ZOOM)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    def to_disp(X):
        x, y, _ = proj3d.proj_transform(*np.asarray(X, float).T, ax.get_proj())
        return ax.transData.transform(np.column_stack([x, y]))

    def densify(P, step=2.0):
        out = [P[:1]]
        for a, b in zip(P[:-1], P[1:]):
            n = max(int(np.hypot(*(b - a))/step), 1)
            out.append(a + (b - a)*np.linspace(0, 1, n + 1)[1:, None])
        return np.vstack(out)

    def hits(pts, bb, grow=0):
        return int(np.count_nonzero((pts[:, 0] > bb.x0 - grow + pad) & (pts[:, 0] < bb.x1 + grow - pad) &
                                    (pts[:, 1] > bb.y0 - grow + pad) & (pts[:, 1] < bb.y1 + grow - pad)))

    # obstacles: everything that can be on screen at some point (superset of all steps)
    bg = draw_background()
    ann = annotations(STEP_KEYS[6] + ('rep',))
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    lines = np.vstack([densify(to_disp(np.column_stack(ln.get_data_3d())))
                       for ln in ax.lines if ln.get_color() not in (GRID, TETHER)])
    points = to_disp([pt.p for pt in v3d.points] + [rep_state(s) for s in np.linspace(0, 1, 40)])
    kite_pts, tether_pts = {}, {}
    for step in range(1, 7):
        ph = STEP_PHASE[step] or 'RO'
        ss = [0.0] if STEP_PHASE[step] is None else np.linspace(0, 1, n_samples)
        kp, tp = [], []
        for s in ss:
            P = phase_pos(ph, s)
            ex, ey, ez = kite_frame(ph, s)
            kv = KiteV3.rotated(PO, e1=ex, e2=ey, e3=ez, voff=P, scale=KITE_SCALE)
            kp.append(to_disp(np.vstack([kv.mesh_nodes, kv.polyline_nodes])))
            kv.remove()
            tp.append(densify(to_disp([PO, P])))
        kite_pts[step], tether_pts[step] = np.vstack(kp), np.vstack(tp)

    labels = [c for c in ax.get_children() if isinstance(c, Annotation3D)]
    steps_of = LABEL_STEPS
    order = ['vapp', 'ro', 'rori', 'ri', 'riro', 'f_retr', 'f_trac', 'v_out', 'v_in',
             'beta_trac', 'beta_rori', 'phi_c']
    labels.sort(key=lambda c: order.index(c.check_name))
    fb = fig.bbox
    placed, result = [], {}
    cand = [(0.0, 0.0)] + [(rho*np.cos(t), rho*np.sin(t))
                          for rho in np.arange(0.8, 9.01, 0.35) for t in np.linspace(0, 2*np.pi, 32, endpoint=False)]
    for c in labels:
        n = c.check_name
        base = np.array(LBL[n])
        best = None
        for off in [base] + [np.array(v) for v in cand[1:]]:     # current offset first, then spiral
            c.xyann = tuple(off)
            b = c.get_window_extent(r)
            if b.x0 < 5 or b.y0 < 5 or b.x1 > fb.width - 5 or b.y1 > fb.height - 5:
                continue
            if hits(lines, b) or hits(points, b, grow=8) or any(b.overlaps(p) for p in placed):
                continue
            if any(hits(kite_pts[st], b) for st in steps_of[n]):
                continue
            teth = sum(hits(tether_pts[st], b) for st in steps_of[n])
            cost = (teth > 0, np.hypot(*off), np.hypot(*(off - base)))
            if best is None or cost < best[0]:
                best = (cost, tuple(off), b)
            if teth == 0 and best[1] == tuple(off):
                break                                   # nearest clean position found
        if best is None:
            print(f'> no clear position for {n}, keeping default')
            c.xyann = tuple(base)
            placed.append(c.get_window_extent(r))
            result[n] = tuple(base)
        else:
            c.xyann = best[1]
            placed.append(best[2])
            result[n] = tuple(round(v, 2) for v in best[1])
    remove(ann)
    remove(bg)
    data = json.loads(LAYOUT_FILE.read_text()) if LAYOUT_FILE.exists() else {}
    data[VIEW or 'default'] = result
    LAYOUT_FILE.write_text(json.dumps(data, indent=1))
    print(f'> layout for view {VIEW or "default"} written to {LAYOUT_FILE}')
    return result

if '--layout' in sys.argv:
    auto_layout()
    sys.exit(0)

if CHECK:
    check_overlaps()
    sys.exit(0)

OUT_DIR.mkdir(exist_ok=True)
draw_background()

for step in range(1, 7):
    ann = annotations(STEP_KEYS[step])
    phase = STEP_PHASE[step]
    fdir = OUT_DIR / 'frames' / f'step{step}'
    shutil.rmtree(fdir, ignore_errors=True)
    if phase is None:
        # static steps: kite at P1 in the reel-out pose
        kite = draw_kite('RO', 0.0)
        save_png(OUT_DIR / f'step{step}_start.png')
        copy_retry(OUT_DIR / f'step{step}_start.png', OUT_DIR / f'step{step}_end.png')
        if not STILLS_ONLY:
            fdir.mkdir(parents=True)
            for k in range(int(STATIC_SEC*FPS)):
                copy_retry(OUT_DIR / f'step{step}_start.png', fdir / f'frame_{k:03d}.png')
            encode(fdir, OUT_DIR / f'step{step}.mp4')
        remove(kite)
    else:
        n = FRAMES[phase]
        ks = [0, n - 1] if STILLS_ONLY else range(n)
        if not STILLS_ONLY:
            fdir.mkdir(parents=True)
        for k in ks:
            kite = draw_kite(phase, k/(n - 1), rep='rep' in STEP_KEYS[step])
            if STILLS_ONLY:
                save_png(OUT_DIR / f'step{step}_{"start" if k == 0 else "end"}.png')
            else:
                save_png(fdir / f'frame_{k:03d}.png')
                if k == 0:
                    copy_retry(fdir / 'frame_000.png', OUT_DIR / f'step{step}_start.png')
                if k == n - 1:
                    copy_retry(fdir / f'frame_{k:03d}.png', OUT_DIR / f'step{step}_end.png')
            remove(kite)
            print(f'> step {step} frame {k+1}/{n}')
        if not STILLS_ONLY:
            encode(fdir, OUT_DIR / f'step{step}.mp4')
    remove(ann)

if not STILLS_ONLY:
    # continuous version: steps 1-4 back to back
    cdir = OUT_DIR / 'frames' / 'full_cycle'
    shutil.rmtree(cdir, ignore_errors=True)
    cdir.mkdir(parents=True)
    k = 0
    for step in (1, 2, 3, 4):
        for f in sorted((OUT_DIR / 'frames' / f'step{step}').glob('frame_*.png')):
            copy_retry(f, cdir / f'frame_{k:03d}.png')
            k += 1
    encode(cdir, OUT_DIR / 'full_cycle.mp4')

plt.close()
print('> CAMERA', CAMERA)
