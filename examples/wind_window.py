#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wind window with figure-eight flight path: static figures and looping animation.

One scene, one camera. The static figures and every animation frame are drawn
by the same functions with the same CAMERA block, so the kite, the tether and
the flight path line up by construction.

Outputs (in folder OUT_DIR):
  fig_windwindow_full.svg/.png/@2x.png    reference reproduction
  fig_windwindow_nokite.svg/.png/@2x.png  background plate (no kite, tether, L, D, alpha)
  fig_windwindow_rep.svg/.png/@2x.png     tether to the representative QSM state
  frames/frame_000.png ... (+ .svg)       animation frames, kite at r(t_k)
  sprites/kite_only_000.png ...           kite + tether stub, transparent background
  bridle_knot_px.csv                      bridle knot pixel coordinates per frame
  check_overlay.png                       bridle knots of all frames on the plate
  loop.mp4, windwindow.fbf.svg            animation

Labels are rendered with matplotlib mathtext as vector paths (no xelatex
needed); PNGs are rasterized from the SVGs with resvg, the video is encoded
with the ffmpeg binary shipped by imageio-ffmpeg.

Run from the examples folder:  python wind_window.py [--no-anim] [--axes]
  --axes  variant with the kite's e_r (reeling) and e_tau (tangential) unit
          vectors in every frame; own camera, output folder wind_window_axes

Convention (as in flight_path.py):
    - Absolute points start with P, vectors with v
    - Names of vplot3d objects end with _obj
"""

import sys
import csv
import shutil
import subprocess
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d

data_path = Path.cwd().parent / 'data'
sys.path.append(str(data_path))

import vplot3d.vplot3d as v3d
from vplot3d.vplot3d import init_view, Line, Vector, Point, Arc, ArcMeasure, Annotation3D, save_svg
from kiteV3 import KiteV3

###############################################################################
# CAMERA (never changes between outputs)
###############################################################################
KITE_AXES  = '--axes' in sys.argv  # variant: e_r (reeling) and e_tau (tangential) at the kite

CAMERA = dict(width=1600, height=1200,       # canvas in px (4:3)
              elev=6.0, azim=58.0,           # deg, orthographic
              zoom=2.50,
              xmin=-0.32, xmax=1.03, ymin=-1.019, ymax=0.981, zmin=-0.14, zmax=1.11)
if KITE_AXES:
    # zoomed out and shifted right: room for e_r and its label at the left lobe
    CAMERA.update(zoom=2.26, xmin=-0.235, xmax=1.115, ymin=-1.072, ymax=0.928)

###############################################################################
# PARAMETERS
###############################################################################
R          = 1.0                  # tether length = wind window radius
PHI_C      = np.deg2rad(0.0)      # centre azimuth of figure-eight
DPHI       = np.deg2rad(11.5)     # half azimuth sweep (lobe centres at PHI_C +- DPHI)
BETA_TRAC  = np.deg2rad(30.0)     # mean elevation angle, traction phase
B          = np.deg2rad(8.0)      # elevation amplitude of figure-eight
T_STATIC   = 0.20*np.pi           # path parameter of kite in static figure (upper part of right lobe)
HEADING_SWAP = False              # True: kite leads with the other chord direction
N_FRAMES   = 48                   # animation frames per lap
FPS        = 8                    # 48 frames / 8 fps = 6 s per lap
PHI_ARC_FROM = np.deg2rad(0.0)    # start of the ground arc measure labelled phi_c (reference: phi = 0)

# Kite model and kinematics (only used for L, D and alpha)
KITE_SCALE = 0.025                # mesh units (m) -> R units
AC_BODY    = np.array([0.1, 0.0, 9.9])   # aerodynamic centre in mesh coords relative to bridle knot
E_LD       = 5.0                  # lift-to-drag ratio
F_REEL     = 0.37                 # reeling factor
L_LEN      = 0.26                 # drawn length of lift vector (R units)
D_LEN      = 0.14                 # drawn length of drag vector (direction physical, length not to scale)
AOA_LEN    = 0.30                 # drawn length of chord / apparent wind lines
KITE_AXES_LEN = 0.20              # drawn length of e_r, e_tau (R units)
KITE_AXES_LBL = 40                # label distance beyond the arrow tip (px)
KITE_AXES_LBL_MIN = 95            # minimum projected distance of the arrow tip region from the kite (px)
KITE_AXES_COLOR = 'k'

# Styling
FONTSIZE   = 54                   # labels sized for slides (~1.6x the thesis figure)
FONT_UNIT  = 1.3                  # extra factor for unit vector and wind labels
LW_THIN    = 1.0
LW_VEC     = 3.4                  # vectors; arrowheads scale with line width
DASHES     = (5, 4)
PATH_COLOR = '#A9CDEB'
PATH_LW    = 8
RED        = '#9B1B1B'
GREEN      = '#1FA37A'
ANGLE_COLOR = '#0079A6'               # beta_trac and phi_c arc measures + labels (slide blue)

OUT_DIR    = Path('wind_window_axes' if KITE_AXES else 'wind_window')
HAVE_ANIM  = '--no-anim' not in sys.argv

###############################################################################
# Geometry
###############################################################################
EX, EY, EZ = np.eye(3)
PO = np.zeros(3)

def sph(phi, beta, r=R):
    '''Cartesian position on sphere of radius r.'''
    return r*np.array([np.cos(beta)*np.cos(phi), np.cos(beta)*np.sin(phi), np.sin(beta)])

def spherical_vector_base(phi, beta):
    cp, sp, cb, sb = np.cos(phi), np.sin(phi), np.cos(beta), np.sin(beta)
    return np.array([cb*cp, cb*sp, sb]), np.array([-sp, cp, 0]), np.array([-sb*cp, -sb*sp, cb])

def path_angles(t):
    return PHI_C + 2*DPHI*np.sin(t), BETA_TRAC + B*np.sin(2*t)

def path_point(t):
    return sph(*path_angles(t))

def path_tangent(t):
    phi, beta = path_angles(t)
    dphi, dbeta = 2*DPHI*np.cos(t), 2*B*np.cos(2*t)
    _, ephi, ebeta = spherical_vector_base(phi, beta)
    return R*(np.cos(beta)*dphi*ephi + dbeta*ebeta)

def body_frame(t):
    '''Kite body frame at path parameter t: e_x,k heading, e_z,k along tether.'''
    r  = path_point(t)
    ez = r/np.linalg.norm(r)
    dr = path_tangent(t)
    ex = dr - dr.dot(ez)*ez
    ex = ex/np.linalg.norm(ex)
    if HEADING_SWAP:
        ex = -ex
    ey = np.cross(ez, ex)
    return ex, ey, ez

def tangential_velocity_factor(phi, beta, chi, E, f):
    '''Massless kite, Schmehl et al. (2013) Eqs. (2.20)-(2.22), elevation formulation.'''
    a = -np.sin(phi)*np.sin(chi) - np.cos(phi)*np.sin(beta)*np.cos(chi)
    b =  np.cos(phi)*np.cos(beta)
    return a + np.sqrt(a*a + b*b - 1 + E*E*(b - f)**2)

def aero_state(t):
    '''Apparent wind direction, lift and drag directions (unit) at path parameter t.'''
    phi, beta = path_angles(t)
    er, ephi, ebeta = spherical_vector_base(phi, beta)
    ex, _, _ = body_frame(t)
    chi = np.arctan2(ex.dot(ephi), ex.dot(ebeta))          # course angle
    lam = tangential_velocity_factor(phi, beta, chi, E_LD, F_REEL)
    va  = EX - (F_REEL*er + lam*ex)                        # apparent wind (vw = 1)
    ea  = va/np.linalg.norm(va)
    D   = er.dot(ea)*ea                                     # resultant aero force along tether
    L   = er - D
    return ea, L/np.linalg.norm(L), D/np.linalg.norm(D)

###############################################################################
# Scene
###############################################################################
fig = plt.figure()
ax  = fig.add_subplot(projection='3d', proj_type='ortho', computed_zorder=False)
ax.set_axis_off()
init_view(**CAMERA)
print('> CAMERA', CAMERA)

# Label rendering: mathtext converted to paths (no Latex post-processing)
v3d.RAW_MATH = False
mpl.rcParams['svg.fonttype']     = 'path'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family']      = 'STIXGeneral'
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

def label(text, P, xytext, **kw):
    return Annotation3D(text, P, xytext=xytext, fontsize=kw.pop('fontsize', FONTSIZE),
                        ha='center', va='center', **kw)

def dashed(**kw):
    return dict(linewidth=LW_THIN, dashes=DASHES, **kw)

Z_SPHERE, Z_PATH, Z_ARC, Z_TETHER, Z_KITE, Z_FORCE, Z_TOP = 1, 2, 4, 5, 6, 8, 10

def draw_static():
    '''Everything that is identical in all outputs.'''
    objs = []
    # Wind window: ground semicircle (downwind), base diameter, meridians
    g = np.linspace(-np.pi/2, np.pi/2, 181)
    objs += ax.plot(R*np.cos(g), R*np.sin(g), 0*g, color='k', zorder=Z_SPHERE, **dashed())
    objs += ax.plot([0, 0], [-R, R], [0, 0], color='k', zorder=Z_SPHERE, **dashed())
    m = np.linspace(0, np.pi, 181)                                    # phi = -90 .. +90 meridian
    objs += ax.plot(0*m, -R*np.cos(m), R*np.sin(m), color='k', zorder=Z_SPHERE, **dashed())
    m = np.linspace(0, np.pi/2, 91)                                   # phi = 0 meridian
    objs += ax.plot(R*np.cos(m), 0*m, R*np.sin(m), color='k', zorder=Z_SPHERE, **dashed())

    # Figure-eight
    t = np.linspace(0, 2*np.pi, 721)
    P = np.array([path_point(ti) for ti in t]).T
    objs += ax.plot(*P, color=PATH_COLOR, linewidth=PATH_LW, solid_capstyle='round',
                    solid_joinstyle='round', zorder=Z_PATH)
    objs.append(Point(path_point(0), scale=1.3, shape='Point1M', zorder=Z_PATH+1,
                      edgecolor='#5B9BD5', facecolor='#5B9BD5', bgcolor='w'))

    # Unit vectors, origin
    ul = 0.45
    objs.append(Vector(PO, ul*EX, linewidth=LW_VEC, zorder=Z_TOP, color='k'))
    objs.append(Vector(PO, ul*EY, linewidth=LW_VEC, zorder=Z_TOP, color='k'))
    objs.append(Vector(PO, 0.55*EZ, linewidth=LW_VEC, zorder=Z_TOP, color='k'))
    objs.append(Point(PO, scale=1.6, shape='Point1M', zorder=Z_TOP+1, color='k'))
    objs.append(label(r'$\mathbf{O}$', PO, (0.15, -1.3)))
    objs.append(label(r'$\boldsymbol{e}_x$', ul*EX, (0.2, 1.0), fontsize=FONTSIZE*FONT_UNIT))
    objs.append(label(r'$\boldsymbol{e}_y$', ul*EY, (0.1, -0.9), fontsize=FONTSIZE*FONT_UNIT))
    objs.append(label(r'$\boldsymbol{e}_z$', 0.55*EZ, (1.4, -0.2), fontsize=FONTSIZE*FONT_UNIT))

    # Apparent wind (free stream) arrow, right of centre
    Pv = np.array([0.0, -0.135, 0.32])
    objs.append(Vector(Pv, 0.24*EX, linewidth=LW_VEC, zorder=Z_TOP, color='k'))
    objs.append(label(r'$\boldsymbol{V}_{\mathrm{app,v}}$', Pv + 0.12*EX, (0.3, 1.2), fontsize=FONTSIZE*FONT_UNIT))

    # Representative state
    phi_rep = np.arccos(np.cos(PHI_C)*np.cos(DPHI))
    Prep = sph(phi_rep, BETA_TRAC)
    objs.append(Point(Prep, scale=2.2, shape='Point1M', zorder=Z_TOP, color=RED, bgcolor=RED))
    # large labels do not fit inside the lobe: place below it
    objs.append(label(r'$\phi_{\mathrm{QSM,rep}} = \mathit{\Delta}\phi$', Prep,
                      (1.9, -3.6)))

    # Arc measures: beta_trac along the meridian of the red point, phi_c on the ground
    objs.append(ArcMeasure(PO, sph(phi_rep, 0), Prep, radius=R, linewidth=2.4, zorder=Z_ARC,
                           color=ANGLE_COLOR, dashes=(2, 2.5)))
    objs.append(label(r'$\beta_{\mathrm{trac}}$', sph(phi_rep, np.deg2rad(8)), (0.9, 0), color=ANGLE_COLOR))
    objs.append(ArcMeasure(PO, sph(PHI_ARC_FROM, 0), sph(phi_rep, 0), vn=EZ, radius=R,
                           linewidth=2.4, zorder=Z_ARC, color=ANGLE_COLOR))
    objs.append(label(r'$\phi_c = 0^\circ$', sph(PHI_ARC_FROM, 0), (-1.0, -1.1), color=ANGLE_COLOR))
    return objs, Prep

def draw_tether(P, stub=None):
    v = P if stub is None else stub*P/np.linalg.norm(P)
    Ps = P - v
    return [Line(Ps, v, linewidth=1.8, zorder=Z_TETHER, color='k')]

def draw_kite(t):
    '''Kite with bridle knot exactly at r(t). Returns (objects, knot position).'''
    P = path_point(t)
    ex, ey, ez = body_frame(t)
    assert np.linalg.det(np.array([ex, ey, ez])) > 0.999      # proper rotation, never mirrored
    kv = KiteV3.rotated(PO, e1=ex, e3=ez, voff=P, scale=KITE_SCALE)
    kv.lines.set_zorder(Z_KITE)
    kv.mesh.set_zorder(Z_KITE+1)
    knot = kv.polyline_nodes[np.argmin(kv.polyline_nodes @ ez)]
    assert np.linalg.norm(knot - P) < 1e-9, 'bridle knot is not on the path'
    return [kv], knot

def draw_forces(t):
    P = path_point(t)
    ex, ey, ez = body_frame(t)
    Pac = P + KITE_SCALE*(AC_BODY[0]*ex + AC_BODY[1]*ey + AC_BODY[2]*ez)
    ea, eL, D = aero_state(t)
    ea_sym = ea - ea.dot(ey)*ey                    # apparent wind in kite symmetry plane
    ea_sym = ea_sym/np.linalg.norm(ea_sym)
    objs = []
    objs.append(Vector(Pac, L_LEN*eL, linewidth=LW_VEC, zorder=Z_FORCE, color=GREEN))
    objs.append(Vector(Pac, D_LEN*D,  linewidth=LW_VEC, zorder=Z_FORCE, color=GREEN))
    objs.append(label(r'$\mathbf{L}$', Pac + L_LEN*eL, (-0.2, 0.9)))
    objs.append(label(r'$\mathbf{D}$', Pac + D_LEN*D, (-0.8, -0.3)))
    # Angle of attack: chord line (black) vs incoming apparent wind (green), arc measure
    objs.append(Line(Pac - 0.5*AOA_LEN*ex, 1.6*AOA_LEN*ex, zorder=Z_FORCE, color='k', **dashed()))
    objs.append(Line(Pac, -AOA_LEN*ea_sym, zorder=Z_FORCE, color=GREEN, linewidth=1.8, dashes=DASHES))
    objs.append(ArcMeasure(Pac, -ea_sym, ex, radius=0.8*AOA_LEN, linewidth=2.0, zorder=Z_FORCE, color='k'))
    alpha = np.degrees(np.arccos(np.clip(-ea_sym.dot(ex), -1, 1)))
    objs.append(label(r'$\alpha$', Pac + 0.85*AOA_LEN*(ex - ea_sym)/np.linalg.norm(ex - ea_sym), (0.9, 0)))
    return objs, alpha

def draw_kite_axes(t):
    '''Unit vectors of the kite position: e_r (radial, reeling direction) and
    e_tau (tangential, flight direction), attached at the aerodynamic centre.'''
    P = path_point(t)
    ex, ey, ez = body_frame(t)
    Pac = P + KITE_SCALE*(AC_BODY[0]*ex + AC_BODY[1]*ey + AC_BODY[2]*ez)
    er, etau = ez, ex                               # e_r along tether, e_tau along course
    fs  = FONTSIZE*FONT_UNIT
    # Label offsets in screen space: beyond the arrow tip, pushed apart sideways
    # when both vectors project in similar directions (kite seen edge-on)
    pa  = np.array(to_px(Pac))
    d   = [np.array(to_px(Pac + KITE_AXES_LEN*e)) - pa for e in (er, etau)]
    ln  = [np.linalg.norm(v) for v in d]
    d   = [v/max(l, 1e-9) for v, l in zip(d, ln)]
    # foreshortened arrows: keep the label at least KITE_AXES_LBL_MIN px from the kite
    off = [(KITE_AXES_LBL + max(0, KITE_AXES_LBL_MIN - l))*v for v, l in zip(d, ln)]
    if d[0] @ d[1] > np.cos(np.radians(50)):
        bis = d[0] + d[1]
        n   = np.array([-bis[1], bis[0]])/np.linalg.norm(bis)
        s   = 1 if n @ (d[0] - d[1]) >= 0 else -1
        off = [off[0] + s*0.55*fs*n, off[1] - s*0.55*fs*n]
    objs = []
    for e, o, lbl in ((er, off[0], r'$\boldsymbol{e}_r$'), (etau, off[1], r'$\boldsymbol{e}_\tau$')):
        objs.append(Vector(Pac, KITE_AXES_LEN*e, linewidth=LW_VEC, zorder=Z_FORCE, color=KITE_AXES_COLOR))
        objs.append(label(lbl, Pac + KITE_AXES_LEN*e, (o[0]/fs, -o[1]/fs),   # px (y down) -> font units (y up)
                          fontsize=fs, color=KITE_AXES_COLOR))
    return objs

def artist(o):
    '''matplotlib artist behind a vplot3d object (or the artist itself).'''
    for a in ('line', 'arc'):
        if hasattr(o, a):
            return getattr(o, a)
    return o

def remove(objs):
    for o in objs:
        o.remove()

###############################################################################
# Output helpers
###############################################################################
W, H = CAMERA['width'], CAMERA['height']

def to_px(P):
    '''3D point -> pixel coordinates (origin top left) of the saved SVG/PNG.'''
    xs, ys, _ = proj3d.proj_transform(*P, ax.get_proj())
    x, y = ax.transData.transform((xs, ys))
    s = 72/fig.dpi
    return x*s, H - y*s

def write_retry(path, data, tries=20):
    '''Write bytes, retrying while a sync client (OneDrive) holds the file.'''
    import time
    for i in range(tries):
        try:
            return path.write_bytes(data)
        except OSError:
            time.sleep(0.25*(i+1))
    path.write_bytes(data)

def save(stem, png_scales=(1, 2), background='#ffffff'):
    import resvg_py
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    save_svg(str(stem))
    svg = stem.with_suffix('.svg')
    Path(str(stem) + '_tex.svg').replace(svg)
    for s in png_scales:
        png = stem.with_suffix('.png') if s == 1 else stem.parent / (stem.name + f'@{s}x.png')
        kw = dict(svg_path=str(svg), width=W*s, height=H*s)
        if background:
            kw['background'] = background
        write_retry(png, bytes(resvg_py.svg_to_bytes(**kw)))

###############################################################################
# Static figures
###############################################################################
OUT_DIR.mkdir(exist_ok=True)
static_objs, Prep = draw_static()

kite_objs, knot = draw_kite(T_STATIC)
teth_objs = draw_tether(knot)
force_objs, alpha = draw_forces(T_STATIC)
print(f'> static kite at t = {T_STATIC:.3f}, alpha = {alpha:.1f} deg')
save(OUT_DIR / 'fig_windwindow_full')
remove(force_objs); remove(teth_objs); remove(kite_objs)

save(OUT_DIR / 'fig_windwindow_nokite')

teth_objs = draw_tether(Prep)
save(OUT_DIR / 'fig_windwindow_rep')
remove(teth_objs)

###############################################################################
# Animation
###############################################################################
if HAVE_ANIM:
    frames_dir  = OUT_DIR / 'frames'
    fbf_dir     = OUT_DIR / 'frames_svg'
    sprites_dir = OUT_DIR / 'sprites'
    for d in (frames_dir, fbf_dir, sprites_dir):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir()

    rows, prev_ex = [], None
    for k in range(N_FRAMES):
        tk = 2*np.pi*k/N_FRAMES
        kite_objs, knot = draw_kite(tk)
        teth_objs = draw_tether(knot)
        axes_objs = draw_kite_axes(tk) if KITE_AXES else []
        save(frames_dir / f'frame_{k:03d}', png_scales=(1,))
        (frames_dir / f'frame_{k:03d}.svg').replace(fbf_dir / f'frame_{k:03d}.svg')
        x, y = to_px(knot)
        rows.append((k, round(x, 2), round(y, 2)))

        # continuity check: heading changes smoothly between frames
        ex = body_frame(tk)[0]
        if prev_ex is not None:
            assert np.degrees(np.arccos(np.clip(ex.dot(prev_ex), -1, 1))) < 45, 'heading jump'
        prev_ex = ex

        # sprite: kite + tether stub only, transparent background
        remove(teth_objs)
        teth_objs = draw_tether(knot, stub=0.12*R)
        for a in static_objs:
            artist(a).set_visible(False)
        # hidden artists must not be post-processed as markers by save_svg
        registries = (v3d.points, v3d.vectors, v3d.arcmeasures)
        hidden = [[o for o in reg if o in static_objs] for reg in registries]
        for reg, hid in zip(registries, hidden):
            for o in hid:
                reg.remove(o)
        save(sprites_dir / f'kite_only_{k:03d}', png_scales=(1,), background=None)
        for reg, hid in zip(registries, hidden):
            reg[:0] = hid
        for a in static_objs:
            artist(a).set_visible(True)
        (sprites_dir / f'kite_only_{k:03d}.svg').unlink()
        remove(axes_objs); remove(teth_objs); remove(kite_objs)
        print(f'> frame {k+1}/{N_FRAMES}')

    with open(OUT_DIR / 'bridle_knot_px.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame', 'x_px', 'y_px'])
        w.writerows(rows)

    # Overlay check: all bridle knots on the no-kite plate
    from PIL import Image, ImageDraw
    im = Image.open(OUT_DIR / 'fig_windwindow_nokite.png').convert('RGB')
    dr = ImageDraw.Draw(im)
    for _, x, y in rows:
        dr.ellipse([x-3, y-3, x+3, y+3], fill=(220, 0, 0))
    im.save(OUT_DIR / 'check_overlay.png')

    # Video (6 s per lap)
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg, '-y', '-loglevel', 'error', '-framerate', str(FPS),
                    '-i', str(frames_dir / 'frame_%03d.png'),
                    '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', '-pix_fmt', 'yuv420p',
                    '-c:v', 'libx264', str(OUT_DIR / 'loop.mp4')], check=True)

    # Frame-by-frame SVG animation (svg2fbf lives next to the interpreter in a venv)
    import os
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH']
    if shutil.which('svg2fbf'):
        try:
            v3d.save_svg2fbf(file=str(OUT_DIR / 'windwindow.fbf.svg'), path=str(fbf_dir),
                             width=W, height=H, atype='loop', fps=FPS)
        except SystemExit as e:
            print('> svg2fbf failed:', e)
    else:
        print('> svg2fbf not found, skipping fbf.svg')

plt.close()
print(f'> R = {R}, B = {np.degrees(B):.1f} deg, N = {N_FRAMES}')
