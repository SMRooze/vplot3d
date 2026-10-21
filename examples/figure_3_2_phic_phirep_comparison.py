#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thesis Figure 3.2: pattern-centre azimuth phi_c, lobe centres and representative
QSM azimuth phi_QSM,rep, for the centred pattern and a shifted pattern.

Top view (along -e_z) of the apparent-wind-aligned wind window: azimuths are
seen as true angles in the e_x-e_y plane, the prescribed figure-eight is drawn
as its projection. The same path parametrisation and parameters as
wind_window.py / wind_window_phic.py:

    phi(t)  = phi_c + 2 Delta_phi sin t
    beta(t) = beta_trac + B sin 2t
    lobe centres          phi_L,R = phi_c -+ Delta_phi                (Eq. A.4)
    representative state  phi_QSM,rep = arccos(cos phi_c cos Delta_phi) (Eq. 2.6)

phi_QSM,rep is taken on the positive (reference) lobe in both panels.

Run from the examples folder:  python figure_3_2_phic_phirep_comparison.py
Outputs (folder wind_window_phic_phirep, existing files are not touched):
  figure_3_2_phic_phirep_comparison.pdf / .png / .svg
"""

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch

###############################################################################
# PARAMETERS (as in wind_window.py)
###############################################################################
R         = 1.0
DPHI      = np.deg2rad(11.5)      # fixed lobe offset Delta_phi
BETA_TRAC = np.deg2rad(30.0)      # representative powered elevation
B         = np.deg2rad(8.0)       # elevation amplitude of the figure-eight
PANELS    = [('(a) Centred pattern', np.deg2rad(0.0)),
             ('(b) Shifted pattern', np.deg2rad(25.0))]

OUT_DIR = Path('wind_window_phic_phirep')
STEM    = OUT_DIR / 'figure_3_2_phic_phirep_comparison'
DPI_PNG = 600

# Style (same colours as wind_window.py / Figure 3.1)
PATH_COLOR  = '#A9CDEB'           # prescribed figure-eight (light blue)
ANGLE_COLOR = '#0079A6'           # pattern centre / elevation geometry (blue)
RED         = '#9B1B1B'           # representative QSM state
LOBE_COLOR  = '#555555'           # lobe centres
GREY        = '#9a9a9a'
FS          = 9.0                 # labels (pt), figure printed at full text width
FS_TITLE    = 9.0
LW_THIN     = 0.6

mpl.rcParams.update({
    'font.family': 'Arial',
    'mathtext.fontset': 'stix',   # math as in Figure 3.1 (vplot3d, STIX)
    'pdf.fonttype': 42,
    'svg.fonttype': 'path',
})

###############################################################################
# Geometry
###############################################################################
def phi_qsm_rep(phi_c):
    '''Thesis Eq. 2.6 on the positive (reference) lobe.'''
    return np.arccos(np.cos(phi_c)*np.cos(DPHI))

def top(phi, beta, r=R):
    '''Top-view coordinates (screen x = e_x, screen y = e_y) of a point on the sphere.'''
    return r*np.cos(beta)*np.cos(phi), r*np.cos(beta)*np.sin(phi)

def ray(phi, r):
    return np.array([0, r*np.cos(phi)]), np.array([0, r*np.sin(phi)])

def deg(a):
    return np.degrees(a)

def fmt(a, sign=False):
    v = deg(a)
    s = f'{v:+.1f}' if sign else f'{v:.1f}'
    return s.replace('-', '\N{MINUS SIGN}')

###############################################################################
# Drawing
###############################################################################
R_RAY   = 1.10                    # length of the azimuth rays
R_BETA  = R*np.cos(BETA_TRAC)     # radius of the beta_trac circle in top view
R_PHIC  = 0.30                    # arc radius phi_c
R_REP   = 0.46                    # arc radius phi_QSM,rep
R_DPHI  = 1.02                    # arc radius Delta_phi (outside the pattern)

def angle_arc(ax, r, a0, a1, color, lw=1.1, arrow=True, z=4):
    '''Arc from azimuth a0 to a1 (rad) at radius r, with an arrowhead at a1.'''
    if abs(a1 - a0) < 1e-9:
        return
    lo, hi = sorted((deg(a0), deg(a1)))
    ax.add_patch(Arc((0, 0), 2*r, 2*r, theta1=lo, theta2=hi, color=color, lw=lw, zorder=z))
    if arrow:
        s = np.sign(a1 - a0)
        e = np.array([r*np.cos(a1), r*np.sin(a1)])
        t = s*np.array([-np.sin(a1), np.cos(a1)])
        ax.add_patch(FancyArrowPatch(e - 0.02*t, e, arrowstyle='-|>', mutation_scale=7, lw=0,
                                     color=color, shrinkA=0, shrinkB=0, zorder=z))

def draw_panel(ax, title, phi_c):
    phi_L, phi_R = phi_c - DPHI, phi_c + DPHI
    phi_rep = phi_qsm_rep(phi_c)

    # wind window footprint: ground semicircle (downwind half), base diameter
    g = np.linspace(-np.pi/2, np.pi/2, 361)
    ax.plot(R*np.cos(g), R*np.sin(g), color='k', lw=LW_THIN, dashes=(4, 3), zorder=1)
    ax.plot([0, 0], [-R, R], color='k', lw=LW_THIN, dashes=(4, 3), zorder=1)

    # e_x reference direction (phi = 0) and unit vectors
    ax.plot(*ray(0, R_RAY), color='k', lw=LW_THIN, dashes=(4, 3), zorder=2)
    for v, lbl, off in (((0.28, 0), r'$\boldsymbol{e}_x$', (0.02, -0.075)),
                        ((0, 0.28), r'$\boldsymbol{e}_y$', (-0.075, 0.0))):
        ax.add_patch(FancyArrowPatch((0, 0), v, arrowstyle='-|>', mutation_scale=9, lw=1.1,
                                     color='k', shrinkA=0, shrinkB=0, zorder=6))
        ax.text(v[0] + off[0], v[1] + off[1], lbl, fontsize=FS + 1, ha='center', va='center', zorder=6)
    ax.plot(0, 0, 'o', ms=4, mfc='w', mec='k', mew=1.1, zorder=7)
    ax.text(-0.05, -0.07, r'$\mathbf{O}$', fontsize=FS, ha='right', va='top')

    # reference direction: apparent wind relative to the ship along e_x (compact, as text)
    ax.text(0.05, 0.86, r'$\boldsymbol{e}_x \parallel \boldsymbol{V}_{\mathrm{app,ship}}$', fontsize=FS - 0.5,
            ha='left', va='center', zorder=6)

    # prescribed figure-eight (top-view projection)
    t = np.linspace(0, 2*np.pi, 721)
    x, y = top(phi_c + 2*DPHI*np.sin(t), BETA_TRAC + B*np.sin(2*t))
    ax.plot(x, y, color=PATH_COLOR, lw=2.4, solid_capstyle='round', zorder=3)

    # lobe centres: thin dashed rays and markers at beta_trac
    for p in (phi_L, phi_R):
        ax.plot(*ray(p, R_RAY), color=LOBE_COLOR, lw=0.7, dashes=(2.5, 2), zorder=2)
        ax.plot(*top(p, BETA_TRAC), 'o', ms=3.6, mfc='w', mec=LOBE_COLOR, mew=1.0, zorder=5)

    # pattern centre direction phi_c
    ax.plot(*ray(phi_c, R_RAY), color=ANGLE_COLOR, lw=1.3, dashes=(5, 2.5), zorder=3)
    ax.plot(*top(phi_c, BETA_TRAC), 'o', ms=3.6, mfc='w', mec=ANGLE_COLOR, mew=1.0, zorder=5)
    angle_arc(ax, R_PHIC, 0.0, phi_c, ANGLE_COLOR)

    # Delta_phi: offset from the pattern centre phi_c to the lobe centre phi_R (not phi_L -> phi_R = 2 Delta_phi)
    angle_arc(ax, R_DPHI, phi_c, phi_R, LOBE_COLOR, lw=0.8)
    assert np.isclose(phi_R - phi_c, DPHI)

    # representative QSM state on the positive (reference) lobe
    ax.plot(*ray(phi_rep, R_BETA), color=RED, lw=1.3, dashes=(3, 2), zorder=4)   # ends at the state
    angle_arc(ax, R_REP, 0.0, phi_rep, RED)
    ax.plot(*top(phi_rep, BETA_TRAC), 'o', ms=6.5, color=RED, zorder=8)

    ax.set_title(title, fontsize=FS_TITLE, loc='left', pad=4)
    return phi_L, phi_R, phi_rep

def ray_label(ax, phi, text, color, r=R_RAY + 0.04, dy=0.0, dx=0.0, **kw):
    x, y = r*np.cos(phi) + dx, r*np.sin(phi) + dy
    ax.text(x, y, text, color=color, fontsize=FS, ha='left', va='center', zorder=9, **kw)

def setup(ax):
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.set_xlim(-0.16, 1.80)
    ax.set_ylim(-0.52, 0.97)

###############################################################################
# Figure
###############################################################################
cm = 1/2.54
fig, axs = plt.subplots(1, 2, figsize=(16*cm, 6.9*cm))
fig.subplots_adjust(left=0.01, right=0.99, top=0.89, bottom=0.02, wspace=0.04)

values = []
for ax, (title, phi_c) in zip(axs, PANELS):
    setup(ax)
    phi_L, phi_R, phi_rep = draw_panel(ax, title, phi_c)
    values.append((phi_c, phi_L, phi_R, phi_rep))

    # checks (see task list): lobe centres, Eq. 2.6, same reference lobe
    assert np.isclose(phi_L, phi_c - DPHI) and np.isclose(phi_R, phi_c + DPHI)
    assert np.isclose(phi_rep, np.arccos(np.cos(phi_c)*np.cos(DPHI)))
    assert phi_c <= phi_rep <= phi_R + 1e-12, 'representative state must stay on the positive lobe'

    sgn = phi_c == 0
    # ray labels outside the window (values next to the rays)
    ray_label(ax, phi_R, rf'$\phi_R = {fmt(phi_R, sgn)}^\circ$', LOBE_COLOR, dy=0.03 if sgn else 0.0)
    ray_label(ax, phi_L, rf'$\phi_L = {fmt(phi_L, sgn)}^\circ$', LOBE_COLOR)
    ray_label(ax, phi_c, rf'$\phi_c = {deg(phi_c):.0f}^\circ$', ANGLE_COLOR, dy=-0.04 if not sgn else 0.0)
    # Delta_phi label at the outer arc on the positive side
    a = phi_c + 0.5*DPHI
    ax.text(1.085*np.cos(a), 1.085*np.sin(a), r'$\Delta\phi$', color=LOBE_COLOR, fontsize=FS - 1,
            ha='center', va='center', zorder=9)
    # phi_QSM,rep: compact label inside the window, next to its arc
    ax.text(0.10, -0.36, rf'$\phi_{{\mathrm{{QSM,rep}}}} = {fmt(phi_rep)}^\circ$', color=RED,
            fontsize=FS, ha='left', va='center', zorder=9)
    ax.annotate('', xy=(R_REP*np.cos(phi_rep/2), R_REP*np.sin(phi_rep/2)), xytext=(0.33, -0.30),
                arrowprops=dict(arrowstyle='-', color=RED, lw=0.5, shrinkA=0, shrinkB=1))
    # compact value box: the fixed offset, identical in both panels
    ax.text(1.78, -0.50, rf'$\Delta\phi = {fmt(DPHI)}^\circ$', color=LOBE_COLOR, fontsize=FS,
            ha='right', va='bottom')

OUT_DIR.mkdir(exist_ok=True)
fig.savefig(STEM.with_suffix('.pdf'), facecolor='white')
fig.savefig(STEM.with_suffix('.svg'), facecolor='white')
fig.savefig(STEM.with_suffix('.png'), dpi=DPI_PNG, facecolor='white')
for phi_c, phi_L, phi_R, phi_rep in values:
    print(f'> phi_c = {deg(phi_c):5.1f}  phi_L = {deg(phi_L):5.1f}  phi_R = {deg(phi_R):5.1f}  '
          f'phi_QSM,rep = {deg(phi_rep):.4f} deg')
print('> saved', STEM)
