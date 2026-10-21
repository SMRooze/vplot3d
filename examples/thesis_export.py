# -*- coding: utf-8 -*-
"""
Export helpers for the thesis figures: SVG -> high-resolution PNG (resvg) and
SVG -> vector PDF (headless Microsoft Edge / Chrome print, page size = SVG size).
"""

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

BROWSERS = [r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
            r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
            r'C:\Program Files\Google\Chrome\Application\chrome.exe']

def svg_to_png(svg, png, width, height, background='#ffffff'):
    '''Rasterize with resvg on a pure white background.'''
    import resvg_py
    data = bytes(resvg_py.svg_to_bytes(svg_path=str(svg), width=int(round(width)),
                                       height=int(round(height)), background=background))
    write_retry(Path(png), data)

def write_retry(path, data, tries=20):
    '''Write bytes, retrying while a sync client (OneDrive) or viewer holds the file.'''
    import time
    for i in range(tries):
        try:
            return path.write_bytes(data)
        except OSError:
            time.sleep(0.25*(i + 1))
    path.write_bytes(data)

def crop_svg(svg, margin=20):
    '''Crop the SVG canvas to its drawn content plus margin (px): rewrites the root width,
    height and viewBox in place. Returns the new (width, height) in px.'''
    import io
    import resvg_py
    from PIL import Image, ImageChops
    svg = Path(svg)
    text = svg.read_text(encoding='utf-8')
    m = re.search(r'<svg\b[^>]*>', text)
    root = m.group(0)
    w = float(re.search(r'\swidth="([\d.]+)', root).group(1))
    h = float(re.search(r'\sheight="([\d.]+)', root).group(1))
    im = Image.open(io.BytesIO(bytes(resvg_py.svg_to_bytes(svg_path=str(svg), width=int(w), height=int(h),
                                                           background='#ffffff')))).convert('RGB')
    x0, y0, x1, y1 = ImageChops.difference(im, Image.new('RGB', im.size, 'white')).getbbox()
    x0, y0 = max(x0 - margin, 0), max(y0 - margin, 0)
    x1, y1 = min(x1 + margin, w), min(y1 + margin, h)
    cw, ch = x1 - x0, y1 - y0
    new = re.sub(r'\swidth="[\d.]+"', f' width="{cw:g}"', root, count=1)
    new = re.sub(r'\sheight="[\d.]+"', f' height="{ch:g}"', new, count=1)
    new = re.sub(r'\sviewBox="[^"]*"', f' viewBox="{x0:g} {y0:g} {cw:g} {ch:g}"', new, count=1)
    svg.write_text(text.replace(root, new, 1), encoding='utf-8')
    return cw, ch

def _browser():
    exe = next((b for b in BROWSERS if os.path.exists(b)), None) or shutil.which('msedge') or shutil.which('chrome')
    if exe is None:
        raise RuntimeError('no Edge/Chrome found for the SVG -> PDF conversion')
    return exe

def _html(svg, width_px, height_px):
    body = Path(svg).read_text(encoding='utf-8')
    body = body[body.index('<svg'):]                       # drop the XML prolog / doctype
    # The HTML parser only knows xlink:href; vplot3d writes the XLink namespace under another
    # prefix (e.g. ns4:href), which would drop every <use> glyph (all labels). Use plain href.
    body = re.sub(r'\s[A-Za-z_][\w.-]*:href=', ' href=', body)
    return ('<!doctype html><html><head><meta charset="utf-8"><style>'
            f'@page {{ size: {width_px}px {height_px}px; margin: 0 }}'
            'html, body { margin: 0; padding: 0; background: #fff }'
            f'svg {{ display: block; width: {width_px}px; height: {height_px}px }}'
            '</style></head><body>' + body + '</body></html>')

def svg_to_pdf(svg, pdf, width_px, height_px, check_png=None):
    '''Print the SVG to a single-page vector PDF of the same size (1 px = 1/96 in).
    check_png: also save a browser screenshot of the printed page (to verify labels etc.).'''
    exe = _browser()
    with tempfile.TemporaryDirectory() as d:
        page = Path(d) / 'page.html'
        page.write_text(_html(svg, width_px, height_px), encoding='utf-8')
        out = Path(d) / 'out.pdf'
        subprocess.run([exe, '--headless', '--disable-gpu', '--no-pdf-header-footer',
                        f'--user-data-dir={d}', f'--print-to-pdf={out}', page.as_uri()],
                       check=True, capture_output=True, timeout=120)
        write_retry(Path(pdf), out.read_bytes())
        if check_png:
            shot = Path(d) / 'shot.png'
            subprocess.run([exe, '--headless', '--disable-gpu', '--hide-scrollbars', f'--user-data-dir={d}',
                            f'--window-size={width_px},{height_px}', f'--screenshot={shot}', page.as_uri()],
                           check=True, capture_output=True, timeout=120)
            write_retry(Path(check_png), shot.read_bytes())
