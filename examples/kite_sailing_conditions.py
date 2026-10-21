#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conceptual three-panel schematic: useful kite flight region and representative
flight-path type versus sailing condition for a ship-towing kite.

Qualitative only. Identical camera, vessel and wind-window hemisphere in all
panels; only the apparent wind direction, the highlighted useful region and the
representative trajectory change.

The scene is projected with a fixed orthographic camera onto plain 2D
matplotlib axes, so all panels share the same projection exactly and the output
is clean vector graphics (PDF/SVG) with Arial text.

Outputs (folder kite_sailing_conditions/):
  kite_sailing_conditions.pdf / .svg / .png

Run from the examples folder:  python kite_sailing_conditions.py

Frame: x along ship course, y to port, z up. Tether attachment (bow) at origin.
"""

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch

###############################################################################
# CAMERA (identical for all panels)
###############################################################################
AZIM = np.deg2rad(165.0)   # camera azimuth: behind the vessel, slightly to port
ELEV = np.deg2rad(30.0)    # camera elevation: slightly from above

C_DIR = np.array([np.cos(ELEV)*np.cos(AZIM), np.cos(ELEV)*np.sin(AZIM), np.sin(ELEV)])  # towards camera
E_RIGHT = np.array([-np.sin(AZIM), np.cos(AZIM), 0.0])
E_UP = np.cross(C_DIR, E_RIGHT)

def proj(P):
    '''Orthographic projection of (..., 3) points to screen (..., 2).'''
    P = np.asarray(P, float)
    return np.stack([P @ E_RIGHT, P @ E_UP], axis=-1)

def depth(P):
    return np.asarray(P, float) @ C_DIR

###############################################################################
# PARAMETERS
###############################################################################
R = 1.0                     # wind window radius (tether length)
C_USEFUL = 0.30             # useful region: tether direction has forward component >= C_USEFUL
BETA_MIN = np.deg2rad(4.0)  # keep useful region clear of the base plane

# Apparent wind: angle (from the bow, wind from port) -> direction wind blows towards.
# Values only set the qualitative layout; no angles are shown in the figure.
CASES = [
    dict(title='More upwind',               wind_from=np.deg2rad(45.0),
         path=dict(ac=-59.0, bc=24.0, A=13.0, Bq=3.2, tilt=90.0), t_kite=0.25, arrows=False),
    dict(title='Reaching',                  wind_from=np.deg2rad(95.0),
         path=dict(ac=-36.0, bc=28.0, A=20.0, Bq=6.0, tilt=50.0), t_kite=0.75, arrows=True),
    dict(title='Broad reaching / downwind', wind_from=np.deg2rad(145.0),
         path=dict(ac=-14.0, bc=24.0, A=40.0, Bq=7.5, tilt=0.0),  t_kite=0.75, arrows=True),
]

# Style
mpl.rcParams.update({
    'font.family': 'Arial',
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial',
    'mathtext.it': 'Arial:italic',
    'mathtext.bf': 'Arial:bold',
    'pdf.fonttype': 42,
    'svg.fonttype': 'path',
})
FS_TITLE = 8.5
FS_LABEL = 7.5
FS_NOTE = 7.0
WIND_POS = np.array([-0.98, 1.06])   # screen position of the apparent wind arrow centre
WIND_LEN = 0.42
BLACK = '#1a1a1a'
GREY = '#7a7a7a'
GRID = '#c4c4c4'
ACCENT = '#00A6D6'          # TU Delft cyan, trajectories only
REGION_FC = '#8c8c8c'
LW_GEOM = 0.6
LW_GRID = 0.35
LW_PATH = 1.4

FIG_W_CM = 16.0
FIG_H_CM = 5.4
OUT_DIR = Path('kite_sailing_conditions')

###############################################################################
# Geometry
###############################################################################
EX, EY, EZ = np.eye(3)
O = np.zeros(3)

def sph(a, b, r=R):
    '''Point at azimuth a (from +x towards port) and elevation b (rad).'''
    a, b = np.asarray(a), np.asarray(b)
    return r*np.stack([np.cos(b)*np.cos(a), np.cos(b)*np.sin(a), np.sin(b)], axis=-1)

def wind_to(case):
    th = case['wind_from']
    return -np.array([np.cos(th), np.sin(th), 0.0])

def useful_region(w, n=400):
    '''Closed outline (on the sphere) of {u.w >= 0, u.x >= C_USEFUL, beta >= BETA_MIN}.'''
    a = np.linspace(-np.pi, np.pi, 1441)
    lo, hi = [], []
    for b in np.linspace(BETA_MIN, np.pi/2, n):
        u = sph(a, np.full_like(a, b))
        ok = (u @ w >= 0) & (u[:, 0] >= C_USEFUL)
        if not ok.any():
            break
        idx = np.flatnonzero(ok)
        lo.append(sph(a[idx[0]], b))
        hi.append(sph(a[idx[-1]], b))
    return np.array(lo + hi[::-1])

def figure_eight(ac, bc, A, Bq, tilt, n=600):
    '''Figure-eight in the local tangent plane at (ac, bc), major axis rotated by tilt (deg).'''
    t = np.linspace(0, 2*np.pi, n)
    s, q = np.deg2rad(A)*np.sin(t), np.deg2rad(Bq)*np.sin(2*t)
    ps = np.deg2rad(tilt)
    s, q = np.cos(ps)*s - np.sin(ps)*q, np.sin(ps)*s + np.cos(ps)*q
    bc = np.deg2rad(bc)
    return t, sph(np.deg2rad(ac) + s/np.cos(bc + q), bc + q)

def hull_faces():
    '''Simplified vessel: extruded deck outline plus a deckhouse. Bow at origin.'''
    L, Bm, T = 0.78, 0.10, 0.07
    xs = [0.0, -0.08, -0.20, -L+0.03, -L]
    deck = [(xs[0], 0)] + [(x, Bm*(0.55 if i == 1 else 1)) for i, x in enumerate(xs[1:], 1)]
    deck = deck + [(x, -y) for x, y in deck[::-1] if y != 0]
    deck = np.array([(x, y, 0.0) for x, y in deck])
    keel = deck*[1.0, 0.85, 0.0] + [0, 0, -T]
    faces = []
    nd = len(deck)
    for i in range(nd):
        j = (i + 1) % nd
        faces.append(np.array([deck[i], deck[j], keel[j], keel[i]]))
    faces.append(deck)
    # deckhouse near the stern
    x0, x1, yh, h = -L + 0.07, -L + 0.20, 0.08, 0.09
    box = np.array([[x0, -yh, 0], [x1, -yh, 0], [x1, yh, 0], [x0, yh, 0]])
    top = box + [0, 0, h]
    for i in range(4):
        j = (i + 1) % 4
        faces.append(np.array([box[i], box[j], top[j], top[i]]))
    faces.append(top)
    return faces

###############################################################################
# Drawing
###############################################################################
def line3(ax, P, **kw):
    xy = proj(P)
    ax.plot(xy[:, 0], xy[:, 1], solid_capstyle='round', **kw)

def arrow3(ax, P0, P1, color=BLACK, lw=0.9, ms=7, z=6):
    a, b = proj(P0), proj(P1)
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=ms, lw=lw,
                                 color=color, shrinkA=0, shrinkB=0, zorder=z))

def draw_hemisphere(ax):
    a = np.linspace(0, 2*np.pi, 361)
    # base circle
    line3(ax, sph(a, 0*a), color=GREY, lw=LW_GEOM, zorder=1)
    # silhouette: great circle perpendicular to the view direction, upper half
    u = np.cross(C_DIR, EZ); u /= np.linalg.norm(u)
    v = np.cross(C_DIR, u)
    sil = R*(np.outer(np.cos(a), u) + np.outer(np.sin(a), v))
    sil = sil[sil[:, 2] >= -1e-9]
    sil = sil[np.argsort(np.arctan2(sil @ v, sil @ u))]
    line3(ax, sil, color=GREY, lw=LW_GEOM, zorder=1)
    # faint grid: parallels and meridians
    for b in np.deg2rad([30, 60]):
        line3(ax, sph(a, np.full_like(a, b)), color=GRID, lw=LW_GRID, zorder=1)
    bb = np.linspace(0, np.pi/2, 91)
    for am in np.deg2rad(np.arange(0, 180, 30)):
        line3(ax, sph(np.full_like(bb, am), bb), color=GRID, lw=LW_GRID, zorder=1)
        line3(ax, sph(np.full_like(bb, am + np.pi), bb), color=GRID, lw=LW_GRID, zorder=1)

def draw_vessel(ax):
    faces = hull_faces()
    faces.sort(key=lambda f: depth(f.mean(axis=0)))
    for f in faces:
        n = np.cross(f[1] - f[0], f[2] - f[0])
        shade = '#ffffff' if abs(n[2]) > 0.5*np.linalg.norm(n) else '#e6e6e6'
        ax.add_patch(Polygon(proj(f), closed=True, fc=shade, ec=BLACK, lw=0.5,
                             joinstyle='round', zorder=4))
    ax.plot(*proj(O), 'o', ms=2.2, color=BLACK, zorder=5)    # tether attachment

def draw_case(ax, case, first):
    w = wind_to(case)

    # wind window edge (plane perpendicular to the apparent wind), dashed
    a = np.linspace(0, np.pi, 181)
    n_perp = np.cross(EZ, w); n_perp /= np.linalg.norm(n_perp)
    edge = R*(np.outer(np.cos(a), n_perp) + np.outer(np.sin(a), EZ))
    line3(ax, edge, color=GREY, lw=LW_GEOM, ls=(0, (3, 2)), zorder=2)

    # useful region
    reg = useful_region(w)
    ax.add_patch(Polygon(proj(reg), closed=True, fc=REGION_FC, alpha=0.22, ec='none', zorder=2))
    ax.add_patch(Polygon(proj(reg), closed=True, fc='none', ec=GREY, lw=0.5, zorder=2))

    draw_vessel(ax)

    # trajectory, tether and kite
    t, P = figure_eight(**case['path'])
    k = np.argmin(abs(t - case['t_kite']*2*np.pi))
    line3(ax, np.array([O, P[k]]), color=GREY, lw=0.5, zorder=5)
    line3(ax, P, color=ACCENT, lw=LW_PATH, zorder=6)
    ax.plot(*proj(P[k]), 'o', ms=3.4, mfc=ACCENT, mec='white', mew=0.6, zorder=7)
    if case['arrows']:
        for f in (0.12, 0.62):
            i = int(f*len(t))
            arrow3(ax, P[i], P[i + 6], color=ACCENT, lw=0.01, ms=7, z=7)

    # apparent wind arrow: projected 3D direction, same screen position in every panel
    d = proj(w)*WIND_LEN
    ax.add_patch(FancyArrowPatch(WIND_POS - d/2, WIND_POS + d/2, arrowstyle='-|>', mutation_scale=8,
                                 lw=0.9, color=BLACK, shrinkA=0, shrinkB=0, zorder=6))
    ax.text(WIND_POS[0], WIND_POS[1] - abs(d[1])/2 - 0.06, r'$V_{\mathrm{app},v}$',
            fontsize=FS_LABEL, ha='center', va='top', color=BLACK)
    return w

def setup_axes(ax):
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.set_xlim(-1.30, 1.30)
    ax.set_ylim(-0.62, 1.22)

###############################################################################
# Figure
###############################################################################
cm = 1/2.54
fig, axs = plt.subplots(1, 3, figsize=(FIG_W_CM*cm, FIG_H_CM*cm))
fig.subplots_adjust(left=0.005, right=0.995, top=0.91, bottom=0.11, wspace=0.02)

for i, (ax, case) in enumerate(zip(axs, CASES)):
    setup_axes(ax)
    draw_hemisphere(ax)
    w = draw_case(ax, case, i == 0)
    ax.set_title(case['title'], fontsize=FS_TITLE, color=BLACK, pad=2)

    # ship course arrow, alongside the vessel on the port side
    yc = 0.30
    arrow3(ax, np.array([-0.62, yc, 0]), np.array([-0.05, yc, 0]), color=BLACK, lw=0.9, ms=7)
    ax.annotate('Ship course', proj(np.array([-0.62, yc, 0])), xytext=(-2, -1), textcoords='offset points',
                fontsize=FS_LABEL - 0.5, ha='right', va='top', color=BLACK, annotation_clip=False)

    if i == 0:
        reg = useful_region(w)
        anchor = proj(reg.mean(axis=0))
        ax.annotate('useful wind-\nwindow region', anchor, xytext=(1.0, 1.02), textcoords='data',
                    fontsize=FS_LABEL - 1, color=GREY, ha='center', va='bottom', linespacing=1.0,
                    arrowprops=dict(arrowstyle='-', lw=0.4, color=GREY, shrinkA=1, shrinkB=0))

fig.text(0.5, 0.025, 'Representative trend from literature; favourable trajectory depends on sailing condition.',
         ha='center', va='bottom', fontsize=FS_NOTE, color=GREY)

OUT_DIR.mkdir(exist_ok=True)
for ext in ('pdf', 'svg'):
    fig.savefig(OUT_DIR / f'kite_sailing_conditions.{ext}', facecolor='white')
fig.savefig(OUT_DIR / 'kite_sailing_conditions.png', dpi=600, facecolor='white')
print('> saved to', OUT_DIR.resolve())
