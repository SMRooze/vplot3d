#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wind window: effect of the pattern centre azimuth phi_c on the representative
QSM azimuth phi_QSM,rep (thesis Eq. 2.6 / A.8).

The complete figure-eight is shifted in the apparent wind aligned wind window
by phi_c, and the representative state is recomputed in every frame:

    phi_QSM,rep = sign(phi_c) * arccos(cos(phi_c) * cos(Delta_phi))

Eq. 2.6 is unsigned (arccos >= 0); the sign of phi_c selects the side of the
wind window (phi_c = 0 keeps the right lobe, as in the static figure). The lobe
centres phi_c -+ Delta_phi (Eq. A.4) are marked for comparison.

Camera, sphere, styling and path parametrisation are taken unchanged from
wind_window.py, so frames line up with the other figures. Output goes to its
own folder and nothing of wind_window.py's output is touched.

Run from the examples folder:  python wind_window_phic.py
Outputs (folder wind_window_phic):
  frames/frame_000.png ...   animation frames
  loop.mp4                   looping video
  fig_phic_<deg>.png/.svg    stills at a few phi_c values (+ @2x)
"""

import sys
import shutil
import subprocess
from pathlib import Path
import numpy as np

# Shared scene: parameters, camera, geometry, styling and helpers of wind_window.py
# (everything before its output section; nothing is drawn or saved by this part)
PHIREP_SWEEP = '--phirep' in sys.argv          # variant: phi_QSM,rep as azimuth of the projected point
sys.argv = [sys.argv[0]]                       # default camera of wind_window.py
_src = Path('wind_window.py').read_text(encoding='utf-8').split('# Static figures')[0]
exec(compile(_src, 'wind_window.py', 'exec'))

# Own camera for this sweep: from slightly downwind, so the whole wind window
# (phi = -90..90 deg) is seen face-on and phi increases to the right.
CAMERA_PHIC = dict(CAMERA, elev=22.0, azim=12.0, zoom=1.95,
                   xmin=-0.1, xmax=1.0, ymin=-1.05, ymax=1.05, zmin=-0.095, zmax=1.105)
init_view(**CAMERA_PHIC)
print('> CAMERA_PHIC', CAMERA_PHIC)

###############################################################################
# PARAMETERS
###############################################################################
OUT_DIR    = Path('wind_window_phic_phirep' if PHIREP_SWEEP else 'wind_window_phic')
PHIC_AMP   = np.deg2rad(40.0)       # sweep amplitude of phi_c (thesis bound: 60 deg)
N_FRAMES   = 72                     # frames per loop
FPS        = 12                     # 72 / 12 = 6 s per loop
STILLS_DEG = (-40, -20, 0, 20, 40)       # extra stills at these phi_c values
GHOST_COLOR = '#A9CDEB'             # reference pattern at phi_c = 0
LOBE_COLOR = '#555555'              # lobe centre markers phi_c -+ Delta_phi
REP_ARC_R  = 0.72                   # radius of the ground arc for phi_QSM,rep (R units)
PHIREP_SWEEP_LBL = (1.1, 0.9)       # --phirep: offset of the phi_QSM,rep label from its mid-arc point (font units; x mirrored with the sign of phi)

def phi_qsm_rep(phi_c):
    '''Thesis Eq. 2.6, signed with the side of the pattern centre.'''
    s = 1.0 if phi_c >= 0 else -1.0
    return s*np.arccos(np.cos(phi_c)*np.cos(DPHI))

###############################################################################
# Scene parts
###############################################################################
def draw_frame_static():
    '''Wind window, unit vectors, apparent wind: identical in every frame.'''
    objs = []
    g = np.linspace(-np.pi/2, np.pi/2, 181)
    objs += ax.plot(R*np.cos(g), R*np.sin(g), 0*g, color='k', zorder=Z_SPHERE, **dashed())
    objs += ax.plot([0, 0], [-R, R], [0, 0], color='k', zorder=Z_SPHERE, **dashed())
    m = np.linspace(0, np.pi, 181)
    objs += ax.plot(0*m, -R*np.cos(m), R*np.sin(m), color='k', zorder=Z_SPHERE, **dashed())
    m = np.linspace(0, np.pi/2, 91)
    objs += ax.plot(R*np.cos(m), 0*m, R*np.sin(m), color='k', zorder=Z_SPHERE, **dashed())

    # reference pattern at phi_c = 0 (faint)
    t = np.linspace(0, 2*np.pi, 721)
    P = np.array([sph(2*DPHI*np.sin(ti), BETA_TRAC + B*np.sin(2*ti)) for ti in t]).T
    objs += ax.plot(*P, color=GHOST_COLOR, linewidth=2, dashes=(3, 3), zorder=Z_PATH-1)

    ul = 0.45
    objs.append(Vector(PO, ul*EX, linewidth=LW_VEC, zorder=Z_TOP, color='k'))
    objs.append(Vector(PO, ul*EY, linewidth=LW_VEC, zorder=Z_TOP, color='k'))
    objs.append(Vector(PO, 0.55*EZ, linewidth=LW_VEC, zorder=Z_TOP, color='k'))
    objs.append(Point(PO, scale=1.6, shape='Point1M', zorder=Z_TOP+1, color='k'))
    objs.append(label(r'$\mathbf{O}$', PO, (0.7, 0.55)))
    objs.append(label(r'$\boldsymbol{e}_x$', ul*EX, (-0.9, 0.1), fontsize=FONTSIZE*FONT_UNIT))
    objs.append(label(r'$\boldsymbol{e}_y$', ul*EY, (0.2, -0.9), fontsize=FONTSIZE*FONT_UNIT))
    objs.append(label(r'$\boldsymbol{e}_z$', 0.55*EZ, (0.9, -0.2), fontsize=FONTSIZE*FONT_UNIT))
    # V_app,v points along e_x, i.e. almost towards the viewer: stated as text instead
    objs.append(fig.text(0.97, 0.955, r'$\boldsymbol{e}_x \parallel \boldsymbol{V}_{\mathrm{app,v}}$',
                         fontsize=0.8*FONTSIZE, ha='right', va='top'))
    return objs

def draw_frame_dynamic(phi_c):
    '''Shifted pattern, lobe centres, representative state and arc measures.'''
    global PHI_C
    PHI_C = phi_c                                   # used by path_angles()
    objs = []
    t = np.linspace(0, 2*np.pi, 721)
    P = np.array([path_point(ti) for ti in t]).T
    objs += ax.plot(*P, color=PATH_COLOR, linewidth=PATH_LW, solid_capstyle='round',
                    solid_joinstyle='round', zorder=Z_PATH)
    objs.append(Point(path_point(0), scale=1.3, shape='Point1M', zorder=Z_PATH+1,
                      edgecolor='#5B9BD5', facecolor='#5B9BD5', bgcolor='w'))

    # lobe centres phi_c -+ Delta_phi (Eq. A.4)
    for s in (-1, 1):
        objs.append(Point(sph(phi_c + s*DPHI, BETA_TRAC), scale=1.1, shape='Point1M',
                          zorder=Z_PATH+2, edgecolor=LOBE_COLOR, facecolor=LOBE_COLOR, bgcolor='w'))

    # pattern centre direction: meridian from the ground to the crossing point
    m = np.linspace(0, BETA_TRAC, 60)
    Pm = np.array([sph(phi_c, b) for b in m]).T
    objs += ax.plot(*Pm, color=ANGLE_COLOR, linewidth=1.6, dashes=(6, 3), zorder=Z_ARC)

    # representative state
    phi_rep = phi_qsm_rep(phi_c)
    Prep = sph(phi_rep, BETA_TRAC)
    objs.append(Point(Prep, scale=2.2, shape='Point1M', zorder=Z_TOP, color=RED, bgcolor=RED))
    objs.append(ArcMeasure(PO, sph(phi_rep, 0), Prep, radius=R, linewidth=2.4, zorder=Z_ARC,
                           color=RED, dashes=(2, 2.5)))

    # ground arc measures from the apparent wind direction (phi = 0); skipped when too short
    if abs(np.degrees(phi_c)) > 2.5:
        objs.append(ArcMeasure(PO, EX, sph(phi_c, 0), vn=EZ, radius=R, linewidth=2.4,
                               zorder=Z_ARC, color=ANGLE_COLOR))
    if abs(np.degrees(phi_c)) > 8:
        objs.append(label(r'$\phi_c$', sph(phi_c, 0), (0.0, -1.0), color=ANGLE_COLOR))
    if PHIREP_SWEEP:
        # phi_QSM,rep: azimuth from e_x to the projection of the representative point onto the e_x-e_y plane
        Pproj = np.array([Prep[0], Prep[1], 0.0])
        rproj = np.linalg.norm(Pproj)                               # = R cos(beta_trac)
        objs.append(Line(Pproj, Prep - Pproj, linewidth=1.6, color=RED, dashes=(4, 3), zorder=Z_ARC))
        objs.append(Line(PO, Pproj, linewidth=1.6, color=RED, dashes=(4, 3), zorder=Z_ARC))
        objs.append(Line(PO, R*EX, linewidth=LW_THIN, color='k', dashes=DASHES, zorder=Z_ARC))
        objs.append(Point(Pproj, scale=1.1, shape='Point1M', zorder=Z_ARC+1,
                          edgecolor=RED, facecolor=RED, bgcolor='w'))
        objs.append(ArcMeasure(PO, EX, Pproj, vn=EZ, radius=rproj, linewidth=2.4, zorder=Z_ARC, color=RED))
        objs.append(label(r'$\phi_{\mathrm{QSM,rep}}$', rproj*sph(phi_rep/2, 0),
                          (np.sign(phi_rep)*PHIREP_SWEEP_LBL[0], PHIREP_SWEEP_LBL[1]),
                          color=RED, fontsize=0.8*FONTSIZE))
        return objs + readout(phi_c, phi_rep)
    if abs(np.degrees(phi_rep)) > 2.5:
        objs.append(ArcMeasure(PO, EX, sph(phi_rep, 0), vn=EZ, radius=REP_ARC_R, linewidth=2.4,
                               zorder=Z_ARC, color=RED))
    if abs(np.degrees(phi_rep)) > 8:
        # mid-arc, pushed sideways away from the e_x label
        objs.append(label(r'$\phi_{\mathrm{QSM,rep}}$', REP_ARC_R*sph(phi_rep/2, 0), (0.9*np.sign(phi_rep), 0.9),
                          color=RED, fontsize=0.8*FONTSIZE))
    return objs + readout(phi_c, phi_rep)

def readout(phi_c, phi_rep):
    '''Numeric readout (screen fixed, top left and right).'''
    objs = []
    dq = np.degrees
    objs.append(fig.text(0.03, 0.955, rf'$\phi_c = {dq(phi_c):+.1f}^\circ$',
                         color=ANGLE_COLOR, fontsize=0.8*FONTSIZE, ha='left', va='top'))
    objs.append(fig.text(0.03, 0.885,
                         rf'$\phi_{{\mathrm{{QSM,rep}}}} = \cos^{{-1}}(\cos\phi_c\,\cos\Delta\phi) = {dq(phi_rep):+.1f}^\circ$',
                         color=RED, fontsize=0.62*FONTSIZE, ha='left', va='top'))
    objs.append(fig.text(0.03, 0.825,
                         rf'lobe centres $\phi_c \mp \Delta\phi = {dq(phi_c - DPHI):+.1f}^\circ,\ {dq(phi_c + DPHI):+.1f}^\circ$',
                         color=LOBE_COLOR, fontsize=0.5*FONTSIZE, ha='left', va='top'))
    objs.append(fig.text(0.97, 0.885, rf'$\Delta\phi = {dq(DPHI):.1f}^\circ$',
                         color=LOBE_COLOR, fontsize=0.6*FONTSIZE, ha='right', va='top'))
    return objs

###############################################################################
# Output
###############################################################################
OUT_DIR.mkdir(exist_ok=True)
frames_dir = OUT_DIR / 'frames'
shutil.rmtree(frames_dir, ignore_errors=True)
frames_dir.mkdir()

draw_frame_static()

for deg in STILLS_DEG:
    objs = draw_frame_dynamic(np.deg2rad(deg))
    save(OUT_DIR / f'fig_phic_{deg:+d}'.replace('+', 'p').replace('-', 'm'))
    remove(objs)

for k in range(N_FRAMES):
    phi_c = PHIC_AMP*np.sin(2*np.pi*k/N_FRAMES)     # 0 -> +amp -> 0 -> -amp -> 0, seamless loop
    objs = draw_frame_dynamic(phi_c)
    save(frames_dir / f'frame_{k:03d}', png_scales=(1,))
    (frames_dir / f'frame_{k:03d}.svg').unlink()
    remove(objs)
    print(f'> frame {k+1}/{N_FRAMES}  phi_c = {np.degrees(phi_c):+6.1f}  '
          f'phi_QSM,rep = {np.degrees(phi_qsm_rep(phi_c)):+6.1f} deg')

import imageio_ffmpeg
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-framerate', str(FPS),
                '-i', str(frames_dir / 'frame_%03d.png'),
                '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', '-pix_fmt', 'yuv420p',
                '-c:v', 'libx264', str(OUT_DIR / 'loop.mp4')], check=True)
plt.close()
print(f'> Delta_phi = {np.degrees(DPHI):.1f} deg, amplitude = {np.degrees(PHIC_AMP):.0f} deg, N = {N_FRAMES}')
