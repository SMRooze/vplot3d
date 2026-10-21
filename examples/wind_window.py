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
  --axes  variant with the kite's e_r (reeling) and e_t (tangential) unit
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
KITE_AXES  = '--axes' in sys.argv  # variant: e_r (reeling) and e_t (tangential) at the kite
PHIREP_ELEV = 16.0                  # --phirep variant: camera elevation (deg)
PHIREP_DZ  = -0.06                  # --phirep variant: vertical shift of the view box (R units)
PHIREP_ZOOM = 2.33                  # --phirep variant: zoom

CAMERA = dict(width=1600, height=1200,       # canvas in px (4:3)
              elev=6.0, azim=58.0,           # deg, orthographic
              zoom=2.50,
              xmin=-0.32, xmax=1.03, ymin=-1.019, ymax=0.981, zmin=-0.14, zmax=1.11)
if KITE_AXES:
    # zoomed out and shifted right: room for e_r and its label at the left lobe
    CAMERA.update(zoom=2.26, xmin=-0.235, xmax=1.115, ymin=-1.072, ymax=0.928)
PHIREP     = '--phirep' in sys.argv  # variant: phi_QSM,rep drawn as azimuth of the projected point
SWEEP      = sys.argv[sys.argv.index('--sweep') + 1] if '--sweep' in sys.argv else None
                                    # animation: '--sweep phic' or '--sweep beta' (with --phirep); same first frame
if PHIREP:
    # higher viewpoint: angles in the e_x-e_y plane become readable
    CAMERA.update(elev=PHIREP_ELEV, zoom=PHIREP_ZOOM, zmin=CAMERA['zmin'] + PHIREP_DZ, zmax=CAMERA['zmax'] + PHIREP_DZ)

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
KITE_AXES_LEN = 0.20              # drawn length of e_r, e_t (R units)
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

# sweeps: constant label offsets (font units) from their anchors; defaults, overridden by the result
# of 'python wind_window.py --axes --phirep --sweep phic --sweep-layout' (clear in all frames of both sweeps)
SWEEP_LBL = dict(phi_rep=(1.6, -1.1), phi_c=(-1.3, -0.8), beta=(2.0, 0.3))
SWEEP_LAYOUT_FILE = Path('wind_window_sweep_layout.json')
if SWEEP_LAYOUT_FILE.exists() and '--sweep-layout' not in sys.argv:
    import json
    SWEEP_LBL.update({k: tuple(v) for k, v in json.loads(SWEEP_LAYOUT_FILE.read_text()).items()})
PHIREP_LBL = (2.3, -0.7)            # offset of the phi_QSM,rep label from the projected point (font units)

OUT_DIR    = Path('wind_window' + ('_axes' if KITE_AXES else '') + ('_phirep' if PHIREP else '')
                  + (f'_sweep_{SWEEP}' if SWEEP else ''))
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
    Pv = np.array([0.0, 0.30, 0.78]) if SWEEP else np.array([0.0, -0.135, 0.32])   # sweep: out of the pattern's way
    objs.append(Vector(Pv, 0.24*EX, linewidth=LW_VEC, zorder=Z_TOP, color='k'))
    objs.append(label(r'$\boldsymbol{V}_{\mathrm{app,ship}}$', Pv + 0.12*EX, (0.3, 1.2), fontsize=FONTSIZE*FONT_UNIT))

    # Representative state
    # Eq. 2.6 is unsigned; the sign of phi_c selects the side of the wind window
    phi_rep = (1 if PHI_C >= 0 else -1)*np.arccos(np.cos(PHI_C)*np.cos(DPHI))
    Prep = sph(phi_rep, BETA_TRAC)
    objs.append(Point(Prep, scale=2.2, shape='Point1M', zorder=Z_TOP, color=RED, bgcolor=RED))
    if not PHIREP:
        # large labels do not fit inside the lobe: place below it
        objs.append(label(r'$\phi_{\mathrm{QSM,rep}} = \mathit{\Delta}\phi$', Prep,
                          (1.9, -3.6)))

    # Arc measures: beta_trac along the meridian of the red point, phi_c on the ground
    objs.append(ArcMeasure(PO, sph(phi_rep, 0), Prep, radius=R, linewidth=2.4, zorder=Z_ARC,
                           color=ANGLE_COLOR, dashes=(2, 2.5)))
    if SWEEP:
        # fixed offset from the arc midpoint: the label glides with the arc (no re-placement jitter)
        objs.append(label(r'$\beta_{\mathrm{trac}}$', sph(phi_rep, BETA_TRAC/2), SWEEP_LBL['beta'], color=ANGLE_COLOR))
        objs[-1].sweep_name = 'beta'
    else:
        objs.append(label(r'$\beta_{\mathrm{trac}}$', sph(phi_rep, np.deg2rad(8)), (0.9, 0), color=ANGLE_COLOR))
    if not PHIREP:
        objs.append(ArcMeasure(PO, sph(PHI_ARC_FROM, 0), sph(phi_rep, 0), vn=EZ, radius=R,
                               linewidth=2.4, zorder=Z_ARC, color=ANGLE_COLOR))
        objs.append(label(r'$\phi_c = 0^\circ$', sph(PHI_ARC_FROM, 0), (-1.0, -1.1), color=ANGLE_COLOR))
        return objs, Prep

    # phi_c: pattern centre direction, highlighted meridian from the ground to the crossing point
    b = np.linspace(0, BETA_TRAC, 60)
    Pc = np.array([sph(PHI_C, bi) for bi in b]).T
    objs += ax.plot(*Pc, color=ANGLE_COLOR, linewidth=2.0, dashes=(6, 3), zorder=Z_ARC)
    if not SWEEP:
        objs.append(label(rf'$\phi_c = {np.degrees(PHI_C):.0f}^\circ$', sph(PHI_C, 0), (-1.0, -1.1),
                          color=ANGLE_COLOR))
    # extend the dashed e_x line (blue) out to the foot of the phi_c meridian on the ground circle
    ec = sph(PHI_C, 0)
    objs.append(Line(R*np.cos(BETA_TRAC)*ec, R*(1 - np.cos(BETA_TRAC))*ec, linewidth=1.6,
                     color=ANGLE_COLOR, dashes=(6, 3), zorder=Z_ARC))
    if SWEEP:
        # phi_c as an azimuth on the ground circle, measured from the dashed e_x line
        objs.append(Line(rproj_c := R*np.cos(BETA_TRAC)*EX, R*EX - rproj_c, linewidth=LW_THIN,
                         color='k', dashes=DASHES, zorder=Z_ARC))
        if abs(np.degrees(PHI_C)) > 2.5:
            objs.append(ArcMeasure(PO, EX, ec, vn=EZ, radius=R, linewidth=2.4, zorder=Z_ARC,
                                   color=ANGLE_COLOR))

    # phi_QSM,rep: azimuth from e_x to the projection of the representative point onto the e_x-e_y plane
    Pproj = np.array([Prep[0], Prep[1], 0.0])
    rproj = np.linalg.norm(Pproj)                                   # = R cos(beta_trac)
    objs.append(Line(Pproj, Prep - Pproj, linewidth=1.6, color=RED, dashes=(4, 3), zorder=Z_ARC))
    objs.append(Line(PO, Pproj, linewidth=1.6, color=RED, dashes=(4, 3), zorder=Z_ARC))
    objs.append(Line(PO, rproj*EX, linewidth=LW_THIN, color='k', dashes=DASHES, zorder=Z_ARC))
    objs.append(Point(Pproj, scale=1.1, shape='Point1M', zorder=Z_ARC+1,
                      edgecolor=RED, facecolor=RED, bgcolor='w'))
    objs.append(ArcMeasure(PO, EX, Pproj, vn=EZ, radius=rproj, linewidth=2.4, zorder=Z_ARC, color=RED))
    if SWEEP:
        # symbols only (changing numbers are distracting); constant offsets from their anchors, chosen
        # by the --sweep-layout pass to be clear in every frame of both sweeps -> smooth, rigid motion
        objs.append(label(r'$\phi_{\mathrm{QSM,rep}}$', Pproj, SWEEP_LBL['phi_rep'], color=RED))
        objs[-1].sweep_name = 'phi_rep'
        objs.append(label(r'$\phi_c$', ec, SWEEP_LBL['phi_c'], color=ANGLE_COLOR))
        objs[-1].sweep_name = 'phi_c'
    else:
        objs.append(label(r'$\phi_{\mathrm{QSM,rep}} = \mathit{\Delta}\phi$', Pproj, PHIREP_LBL, color=RED))
        objs[-1].phirep_label = True          # animation frames show phi_QSM,rep only (see Animation)
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

def CL_V3(a):
    '''Lift coefficient of the TU Delft V3 surrogate, thesis Eq. C.1 (alpha in rad).'''
    return 0.17 + 5.69*a - 10.78*a**2

def CD_V3(a):
    '''Drag coefficient of the TU Delft V3 surrogate, thesis Eq. C.2 (alpha in rad).'''
    return 0.14 - 0.18*a + 1.79*a**2

ALPHA_REF = np.deg2rad(7.5)       # alpha_pow: L_LEN and D_LEN are the drawn lengths at this angle
FORCE_SCALE = 1.0                 # common scale of drawn L and D (alpha close-up uses < 1)
AOA_SCALE   = 1.0                 # scale of the chord / inflow lines and alpha arc

def rotate(v, k, ang):
    '''Rodrigues rotation of v about unit axis k by ang.'''
    return v*np.cos(ang) + np.cross(k, v)*np.sin(ang) + k*k.dot(v)*(1 - np.cos(ang))

def draw_forces(t, alpha=None):
    '''L, D, chord line, apparent wind line and alpha at the kite. With alpha [rad] given, the chord
    is set at alpha from the inflow and L, D scale with CL(alpha), CD(alpha) at fixed apparent wind.
    Returns (objects, alpha [deg], rotation axis, wing pitch change, aerodynamic centre).'''
    P = path_point(t)
    ex, ey, ez = body_frame(t)
    Pac = P + KITE_SCALE*(AC_BODY[0]*ex + AC_BODY[1]*ey + AC_BODY[2]*ez)
    ea, eL, D = aero_state(t)
    ea_sym = ea - ea.dot(ey)*ey                    # apparent wind in kite symmetry plane
    ea_sym = ea_sym/np.linalg.norm(ea_sym)
    a_in = -ea_sym                                 # direction the flow comes from
    alpha0 = np.arccos(np.clip(a_in.dot(ex), -1, 1))
    k = np.cross(a_in, ex)
    k = k/np.linalg.norm(k)
    AOA = AOA_LEN*AOA_SCALE
    if alpha is None:
        chord, l_len, d_len = ex, L_LEN*FORCE_SCALE, D_LEN*FORCE_SCALE
    else:
        chord = rotate(a_in, k, alpha)
        l_len = FORCE_SCALE*L_LEN*CL_V3(alpha)/CL_V3(ALPHA_REF)
        d_len = FORCE_SCALE*D_LEN*CD_V3(alpha)/CD_V3(ALPHA_REF)
    objs = []
    objs.append(Vector(Pac, l_len*eL, linewidth=LW_VEC, zorder=Z_FORCE, color=GREEN))
    objs.append(Vector(Pac, d_len*D,  linewidth=LW_VEC, zorder=Z_FORCE, color=GREEN))
    objs.append(label(r'$\mathbf{L}$', Pac + l_len*eL, (-0.2, 0.9)))
    objs[-1].sweep_name = 'L'
    objs.append(label(r'$\mathbf{D}$', Pac + d_len*D, (-0.8, -0.3)))
    objs[-1].sweep_name = 'D'
    # Angle of attack: chord line (black) vs incoming apparent wind (green), arc measure
    objs.append(Line(Pac - 0.5*AOA*chord, 1.6*AOA*chord, zorder=Z_FORCE, color='k', **dashed()))
    objs.append(Line(Pac, AOA*a_in, zorder=Z_FORCE, color=GREEN, linewidth=1.8, dashes=DASHES))
    if alpha is None:
        objs.append(ArcMeasure(Pac, a_in, chord, radius=0.8*AOA, linewidth=2.0, zorder=Z_FORCE, color='k'))
    else:   # small angles leave no room for an arrowhead: plain arc, same look in every frame
        objs.append(Arc(Pac, a_in, chord, radius=0.8*AOA, linewidth=2.0, zorder=Z_FORCE, color='k'))
    alpha_deg = np.degrees(np.arccos(np.clip(a_in.dot(chord), -1, 1)))
    objs.append(label(r'$\alpha$', Pac + 0.85*AOA*(chord + a_in)/np.linalg.norm(chord + a_in), (0.9, 0)))
    objs[-1].sweep_name = 'alpha'
    return objs, alpha_deg, k, (0.0 if alpha is None else alpha - alpha0), Pac

def pitch_wing(kite, axis, ang, Pac, knot):
    '''Pitch the wing mesh about the aerodynamic centre (depower-like); the bridle lines and the
    KCU near the bridle knot stay aligned with the tether.'''
    kv = kite[0]
    X = np.array(kv.mesh_nodes, float)
    ez = (Pac - knot)/np.linalg.norm(Pac - knot)
    wing = (X - knot) @ ez > 5.0*KITE_SCALE          # KCU hangs within ~2 m of the knot
    X[wing] = Pac + np.array([rotate(x - Pac, axis, ang) for x in X[wing]])
    kv.mesh.set_verts([X[f] for f in kv.mesh_faces])
    kv.mesh_nodes = X

def draw_kite_axes(t):
    '''Unit vectors of the kite position: e_r (radial, reeling direction) and
    e_t (tangential, flight direction; thesis notation), attached at the aerodynamic centre.'''
    P = path_point(t)
    ex, ey, ez = body_frame(t)
    Pac = P + KITE_SCALE*(AC_BODY[0]*ex + AC_BODY[1]*ey + AC_BODY[2]*ez)
    er, etau = ez, ex                               # e_r along tether, e_t along course
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
    for e, o, lbl in ((er, off[0], r'$\boldsymbol{e}_r$'), (etau, off[1], r'$\boldsymbol{e}_t$')):
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

if SWEEP:
    # Sweep of phi_c or beta_trac while the kite flies the figure-eight; pattern, phi_c construction,
    # phi_QSM,rep (Eq. 2.6) and beta_trac arc are redrawn in every frame. Both sweeps start from the
    # same state (phi_c = 0, beta_trac = 30 deg, kite at T_STATIC), so their first frames are identical.
    SWEEP_FRAMES = 72
    SWEEP_FPS    = 12                     # 6 s per loop
    KITE_LOOPS   = 2                      # figure-eights flown by the kite per loop
    PHIC_AMP     = np.deg2rad(30.0)       # phic:  phi_c = PHIC_AMP sin(2 pi k/N)  (0 -> + -> 0 -> - -> 0)
    BETA_0, BETA_MAX = BETA_TRAC, np.deg2rad(50.0)   # beta: 30 -> 50 -> 30 deg (30 deg is the lower bound)

    ALPHA_MIN, ALPHA_MAX = np.deg2rad(3.0), np.deg2rad(12.0)   # alpha: 3 -> 12 -> 3 deg
    ALPHA_VIEW_HALF = 0.14                # close-up: half size of the view box around the kite (R units)
    ALPHA_VIEW_YAW  = 12.0                # close-up: view turned off the span axis (deg)
    ALPHA_VIEW_TILT = 10.0                # close-up: view tilted towards the tether direction (deg)

    if SWEEP == 'alpha':
        # Close-up camera looking along the kite span axis e_y,k, so inflow, chord, alpha, L and D
        # lie in the picture plane (airfoil-like view); flow drawn from left to right.
        ex_k, ey_k, ez_k = body_frame(T_STATIC)
        P_k = path_point(T_STATIC)
        Pac_k = P_k + KITE_SCALE*(AC_BODY[0]*ex_k + AC_BODY[1]*ey_k + AC_BODY[2]*ez_k)
        a_in_k = -aero_state(T_STATIC)[0]
        v = ey_k
        az = np.arctan2(v[1], v[0])
        if a_in_k @ np.array([-np.sin(az), np.cos(az), 0]) > 0:     # inflow must come from the left
            v = -v
        # slightly oblique so the canopy reads as a 3D wing; the alpha plane stays nearly face-on
        v = rotate(v, ez_k, np.deg2rad(ALPHA_VIEW_YAW))
        v = v + np.tan(np.deg2rad(ALPHA_VIEW_TILT))*ez_k
        v = v/np.linalg.norm(v)
        cen = Pac_k - 0.01*ez_k
        FORCE_SCALE = 0.42                # shorter L, D so they fit the close-up (ratios unchanged)
        AOA_SCALE = 0.55                  # chord / inflow lines reach well beyond the wing
        h = ALPHA_VIEW_HALF
        init_view(width=CAMERA['width'], height=CAMERA['height'],
                  xmin=cen[0]-h, xmax=cen[0]+h, ymin=cen[1]-h, ymax=cen[1]+h, zmin=cen[2]-h, zmax=cen[2]+h,
                  zoom=1.0, elev=np.degrees(np.arcsin(v[2])), azim=np.degrees(np.arctan2(v[1], v[0])))
        print('> alpha close-up camera: elev %.1f, azim %.1f' % (np.degrees(np.arcsin(v[2])),
                                                                  np.degrees(np.arctan2(v[1], v[0]))))

    def sweep_state(k):
        '''(phi_c, beta_trac, kite path parameter, alpha or None) of frame k.'''
        w = 2*np.pi*k/SWEEP_FRAMES
        t = T_STATIC + KITE_LOOPS*w
        if SWEEP == 'phic':
            return PHIC_AMP*np.sin(w), BETA_0, t, None
        if SWEEP == 'alpha':
            return 0.0, BETA_0, T_STATIC, ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN)*0.5*(1 - np.cos(w))
        return 0.0, BETA_0 + (BETA_MAX - BETA_0)*0.5*(1 - np.cos(w)), t, None

    ax.set_autoscale_on(False)            # new geometry must not widen the axes limits (camera drift)
    print('> sweep limits', ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d())
    SWEEP_STILLS = {'phic': [(np.deg2rad(d), BETA_0, None, f'phic_{"p" if d >= 0 else "m"}{abs(d)}') for d in (-30, -15, 0, 15, 30)],
                    'beta': [(0.0, np.deg2rad(d), None, f'beta_{d}') for d in (30, 40, 50)],
                    'alpha': [(0.0, BETA_0, np.deg2rad(d), f'alpha_{d}') for d in (3, 7, 12)]}[SWEEP]
    ax.set_box_aspect([1, 1, 1], zoom=v3d.PLOT_ZOOM)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    from mpl_toolkits.mplot3d import proj3d
    # phi_rep, phi_c, beta are placed deterministically (smooth motion); only these may be re-placed:
    MOVABLE = ('e_r', 'e_t', 'L', 'D', 'alpha')

    def _disp(X):
        x, y, _ = proj3d.proj_transform(*np.asarray(X, float).T, ax.get_proj())
        return ax.transData.transform(np.column_stack([x, y]))

    def _dense(P, step=2.0):
        out = [P[:1]]
        for a, b in zip(P[:-1], P[1:]):
            n = max(int(np.hypot(*(b - a))/step), 1)
            out.append(a + (b - a)*np.linspace(0, 1, n + 1)[1:, None])
        return np.vstack(out)

    def autoplace(kite_nodes, pad=3):
        '''Move the MOVABLE labels to the nearest offset clear of lines, points, the kite,
        other labels and the canvas edge (spiral around the current offset).'''
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        anns = [c for c in ax.get_children() if isinstance(c, Annotation3D)]
        movable = sorted([a for a in anns if getattr(a, 'sweep_name', None) in MOVABLE],
                         key=lambda a: MOVABLE.index(a.sweep_name))
        placed = [a.get_window_extent(r) for a in anns if a not in movable]
        pts = [_dense(_disp(np.column_stack(ln.get_data_3d()))) for ln in ax.lines if ln.get_visible()]
        pts = np.vstack(pts + ([_disp(kite_nodes)] if len(kite_nodes) else []))
        fb = fig.bbox
        def clear(b):
            if b.x0 < 5 or b.y0 < 5 or b.x1 > fb.width - 5 or b.y1 > fb.height - 5:
                return False
            if np.any((pts[:, 0] > b.x0 + pad) & (pts[:, 0] < b.x1 - pad) &
                      (pts[:, 1] > b.y0 + pad) & (pts[:, 1] < b.y1 - pad)):
                return False
            g = b.expanded(1, 1).padded(20)                   # keep a clear gap between labels
            return not any(g.overlaps(p) for p in placed)
        for a in movable:
            base = np.array(LAST_OFFSET.get(a.sweep_name, a.xyann), float)   # stay put while clear
            cands = [base] + [base + rho*np.array([np.cos(t), np.sin(t)])
                              for rho in np.arange(0.4, 6.01, 0.3) for t in np.linspace(0, 2*np.pi, 24, endpoint=False)]
            for c in cands:
                a.xyann = tuple(c)
                b = a.get_window_extent(r)
                if clear(b):
                    break
            else:
                a.xyann = tuple(base)
                b = a.get_window_extent(r)
            placed.append(b)
            LAST_OFFSET[a.sweep_name] = tuple(a.xyann)

    LAST_OFFSET = {}                      # label offsets of the previous frame (temporal coherence)

    def sweep_frame(phi_c, beta, t, alpha, stem, scales):
        global PHI_C, BETA_TRAC
        PHI_C, BETA_TRAC = phi_c, beta
        fig.canvas.draw()          # lay out first: screen-space label offsets must not depend on history
        objs = [] if alpha is not None else draw_static()[0]      # alpha close-up: kite and forces only
        if alpha is None:
            # phi_c / beta sweep: no kite, tether or kite axes; only the pattern and the representative state
            autoplace(np.empty((0, 3)))
        else:
            # alpha sweep: kite with pitched wing, tether, L, D, alpha
            kite, knot = draw_kite(t)
            forces, _, axis, dpitch, Pac = draw_forces(t, alpha)
            pitch_wing(kite, axis, dpitch, Pac, knot)
            objs += draw_tether(knot) + forces + kite
            kv = kite[0]
            autoplace(np.vstack([kv.mesh_nodes, kv.polyline_nodes]))
        save(stem, png_scales=scales)
        remove(objs)
        if '--debug-limits' in sys.argv:
            print('> leftovers', stem, len(ax.lines), len(ax.collections), len(ax.texts),
                  len([c for c in ax.get_children() if isinstance(c, Annotation3D)]),
                  len(v3d.points), len(v3d.vectors), len(v3d.arcmeasures), len(v3d.lines), len(v3d.markers))

    if '--sweep-layout' in sys.argv:
        # One constant offset per label (phi_rep, phi_c, beta) that is clear in EVERY frame of BOTH the
        # phi_c and the beta sweep: the labels then move rigidly with their elements (no jitter) and the
        # first frames of both sweeps stay identical. Clear = inside the canvas, not on any drawn line
        # or marker, not on another label (with a gap).
        import json
        names = ('phi_rep', 'phi_c', 'beta')
        px = fig.dpi/72*FONTSIZE                                   # 1 font unit in display px
        frames = []
        for mode in ('phic', 'beta'):
            for k in range(SWEEP_FRAMES):
                w = 2*np.pi*k/SWEEP_FRAMES
                if mode == 'phic':
                    PHI_C, BETA_TRAC = PHIC_AMP*np.sin(w), BETA_0
                else:
                    PHI_C, BETA_TRAC = 0.0, BETA_0 + (BETA_MAX - BETA_0)*0.5*(1 - np.cos(w))
                fig.canvas.draw()
                objs, _ = draw_static()
                fig.canvas.draw()
                r = fig.canvas.get_renderer()
                anns = [c for c in ax.get_children() if isinstance(c, Annotation3D)]
                mine = {a.sweep_name: a for a in anns if getattr(a, 'sweep_name', None) in names}
                info = {n: (ax.transData.transform(mine[n].xy), mine[n].get_window_extent(r).size) for n in names}
                fixed = [a.get_window_extent(r) for a in anns if a not in mine.values()]
                # obstacles: meaningful elements only; thin black dashed construction lines of the wind
                # window (sphere, ground circle, e_x leg) may pass behind text, as in the static figures
                pts = np.vstack([_dense(_disp(np.column_stack(ln.get_data_3d()))) for ln in ax.lines
                                 if ln.get_visible() and not (mpl.colors.to_hex(ln.get_color()) == '#000000'
                                                              and ln.get_linewidth() <= LW_THIN + 1e-6)])
                frames.append((info, fixed, pts))
                remove(objs)
        fb = fig.bbox
        cand = [(0.0, 0.0)] + [(rho*np.cos(t), rho*np.sin(t)) for rho in np.arange(0.3, 7.01, 0.25)
                                for t in np.linspace(0, 2*np.pi, 48, endpoint=False)]
        chosen, result = {}, {}
        for n in names:
            pref = np.array(SWEEP_LBL[n])
            best = None
            for c in sorted(cand, key=lambda c: np.hypot(*(np.array(c)))):
                off = pref + np.array(c)
                bad = 0
                for i, (info, fixed, pts) in enumerate(frames):
                    anc, (wd, ht) = info[n]
                    cx, cy = anc + off*px
                    x0, x1, y0, y1 = cx - wd/2, cx + wd/2, cy - ht/2, cy + ht/2
                    if x0 < 5 or y0 < 5 or x1 > fb.width - 5 or y1 > fb.height - 5:
                        bad += 1; continue
                    g = 12                                         # gap to other labels (px)
                    if any(x0 - g < b.x1 and b.x0 < x1 + g and y0 - g < b.y1 and b.y0 < y1 + g
                           for b in fixed + [chosen[m][i] for m in chosen]):
                        bad += 1; continue
                    if np.any((pts[:, 0] > x0 + 2) & (pts[:, 0] < x1 - 2) & (pts[:, 1] > y0 + 2) & (pts[:, 1] < y1 - 2)):
                        bad += 1
                if best is None or bad < best[0]:
                    best = (bad, off)
                if bad == 0:
                    break
            off = best[1]
            result[n] = [round(float(v), 3) for v in off]
            from matplotlib.transforms import Bbox
            chosen[n] = [Bbox.from_extents(*(info[n][0] + off*px - info[n][1]/2), *(info[n][0] + off*px + info[n][1]/2))
                         for info, _, _ in frames]
            print(f'> {n}: offset {result[n]}  (frames with a conflict: {best[0]} of {len(frames)})')
        SWEEP_LAYOUT_FILE.write_text(json.dumps(result, indent=1))
        print(f'> written {SWEEP_LAYOUT_FILE}')
        sys.exit(0)

    for phi_c, beta, alpha, name in SWEEP_STILLS:
        LAST_OFFSET.clear()
        sweep_frame(phi_c, beta, T_STATIC, alpha, OUT_DIR / f'fig_{name}', (1, 2))
    LAST_OFFSET.clear()                   # animation starts fresh: identical first frame in both sweeps
    if '--sweep-stills' not in sys.argv:
        fdir = OUT_DIR / 'frames'
        shutil.rmtree(fdir, ignore_errors=True)
        fdir.mkdir()
        for k in range(SWEEP_FRAMES):
            sweep_frame(*sweep_state(k), fdir / f'frame_{k:03d}', (1,))
            (fdir / f'frame_{k:03d}.svg').unlink()
            print(f'> sweep frame {k+1}/{SWEEP_FRAMES}')
        import imageio_ffmpeg
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-framerate', str(SWEEP_FPS),
                        '-i', str(fdir / 'frame_%03d.png'), '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                        '-pix_fmt', 'yuv420p', '-c:v', 'libx264', str(OUT_DIR / 'loop.mp4')], check=True)
    plt.close()
    sys.exit(0)

static_objs, Prep = draw_static()

kite_objs, knot = draw_kite(T_STATIC)
teth_objs = draw_tether(knot)
force_objs, alpha = draw_forces(T_STATIC)[:2]
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

    # animation: representative azimuth labelled phi_QSM,rep only (static figures keep '= Delta phi');
    # the shorter label keeps its left end beside the arc
    for a in static_objs:
        if getattr(a, 'phirep_label', False):
            a.set_text(r'$\phi_{\mathrm{QSM,rep}}$')
            a.xyann = (PHIREP_LBL[0] - 1.1, PHIREP_LBL[1])

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
