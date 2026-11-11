"""Build the project webpage from main.ipynb and output/.

Write-ups and key functions are read from the notebook, so rerun this after editing them:
    .venv/bin/python make_web.py                 # local preview: web/index.html + web/images/
    .venv/bin/python make_web.py --site DIR      # into the personal site: DIR/cs180-proj4.html + DIR/images/cs180-proj4/
"""
import argparse
import html
import json
import re
import shutil
from pathlib import Path

import mistune
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).parent
args = argparse.ArgumentParser()
args.add_argument('--site', type=Path, help='personal site directory to build into')
args = args.parse_args()
if args.site:
    WEB, PAGE, IMG_URL = args.site, 'cs180-proj4.html', 'images/cs180-proj4'
else:
    WEB, PAGE, IMG_URL = ROOT / 'web', 'index.html', 'images'
IMG = WEB / IMG_URL
OUT = ROOT / 'output'
NAME = 'Puthyrath Hy'
MAX_SIDE = 1800

nb = {c['id']: ''.join(c['source']) for c in json.loads((ROOT / 'main.ipynb').read_text())['cells']}
nbB_cells = {c['id']: c for c in json.loads((ROOT / 'partB.ipynb').read_text())['cells']}
nbB = {k: ''.join(c['source']) for k, c in nbB_cells.items()}


def stream_lines(cell_id):
    """Printed stdout lines of a Part B cell."""
    out = ''.join(''.join(o['text']) for o in nbB_cells[cell_id].get('outputs', []) if o.get('output_type') == 'stream')
    return [l for l in out.splitlines() if l.strip()]


# ---------- markdown / code ----------

def md(text):
    """Markdown -> HTML, leaving $...$ and $$...$$ untouched for MathJax."""
    text = re.sub(r'^\*\*A\.\d write-up:\*\*\s*', '', text.strip())
    math = []

    def stash(m):
        math.append(m.group(0))
        return f'MATHPLACEHOLDER{len(math) - 1}X'
    text = re.sub(r'\$\$.+?\$\$|\$[^$\n]+?\$', stash, text, flags=re.S)
    out = mistune.html(text)
    return re.sub(r'MATHPLACEHOLDER(\d+)X', lambda m: html.escape(math[int(m.group(1))], quote=False), out)


def split_md(text, *headers):
    """Split a write-up at bold headers like '**Comparison**'; returns the pieces in order."""
    parts, rest = [], text
    for h in headers:
        i = rest.index(h)
        parts.append(rest[:i])
        rest = rest[i:]
    return parts + [rest]


def code(cell_id, title, src_nb=None):
    src = (src_nb or nb)[cell_id].strip()
    return (f'<details class="code"><summary>{title}</summary>'
            f'<pre><code class="language-python">{html.escape(src)}</code></pre></details>')


# ---------- images ----------

def export(src, name, max_side=MAX_SIDE):
    im = Image.open(src).convert('RGB')
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    im.save(IMG / name, quality=85)
    return f'{IMG_URL}/{name}'


def rect_input(name):
    """Source photo cropped around the rectangle, with the 4 corners outlined."""
    d = json.loads((ROOT / 'points' / f'{name}.json').read_text())
    im = ImageOps.exif_transpose(Image.open(ROOT / d['images'][0])).convert('RGB')
    s = im.width / d['shapes'][0][1]
    pts = [(x * s, y * s) for x, y in d['points'][0]]
    draw = ImageDraw.Draw(im)
    lw = max(3, im.width // 300)
    draw.line(pts + [pts[0]], fill=(255, 60, 60), width=lw)
    for x, y in pts:
        draw.ellipse([x - 3 * lw, y - 3 * lw, x + 3 * lw, y + 3 * lw], outline=(255, 60, 60), width=lw)
    xs, ys = zip(*pts)
    pad = 0.6 * max(max(xs) - min(xs), max(ys) - min(ys))
    im = im.crop((max(min(xs) - pad, 0), max(min(ys) - pad, 0),
                  min(max(xs) + pad, im.width), min(max(ys) + pad, im.height)))
    im.thumbnail((1000, 1000), Image.LANCZOS)
    im.save(IMG / f'{name}_input.jpg', quality=85)
    return f'{IMG_URL}/{name}_input.jpg'


def fig(src, caption, cls=''):
    return f'<figure class="{cls}"><img src="{src}" alt="{html.escape(caption)}" loading="lazy"><figcaption>{caption}</figcaption></figure>'


def homography_text(name, i):
    """Recovered H for a pair, recomputed from its points exactly as the notebook does."""
    import numpy as np
    d = json.loads((ROOT / 'points' / f'{name}_{i}_{i + 1}.json').read_text())
    p1, p2 = (np.array(p, dtype=float) for p in d['points'])
    A, b = [], []
    for (x, y), (xp, yp) in zip(p1, p2):
        A += [[x, y, 1, 0, 0, 0, -x * xp, -y * xp], [0, 0, 0, x, y, 1, -x * yp, -y * yp]]
        b += [xp, yp]
    h = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]
    H = np.append(h, 1).reshape(3, 3)
    rows = [' '.join(f'{v:>11.4f}' if abs(v) >= 1e-3 or v == 0 else f'{v:>11.2e}' for v in r) for r in H]
    return '\n'.join(rows)


# ---------- build ----------

if IMG.exists():
    shutil.rmtree(IMG)
IMG.mkdir(parents=True)

MOSAICS = [('wall', 'Wall'), ('door', 'Door'), ('chimney', 'Chimney')]
RECTS = [('chessboard', 'Chessboard'), ('double_door', 'Double door')]

a3_impl, a3_rect, a3_cmp = split_md(nb['cell-29'], '**Rectification**', '**Comparison**')
a4_proc, a4_results = split_md(nb['cell-36'], '**Mosaic 1')
a4_each = re.split(r'(?=\*\*Mosaic \d)', a4_results.strip())
a4_each = [p for p in a4_each if p.strip()]

sec = ['<div class="part" id="partA"><p class="eyebrow">Part A</p><h2>Image Warping and Mosaicing</h2></div>']

# A.1
sec.append('<section id="a1"><h2>A.1 &nbsp;Shooting the Pictures</h2>' + md(nb['cell-08']))
for key, title in MOSAICS:
    sec.append(fig(export(OUT / f'{key}_sources.jpg', f'{key}_sources.jpg'), f'{title} set (left, middle, right)', 'wide'))
sec.append('<div class="row">' + ''.join(
    fig(export(ROOT / json.loads((ROOT / 'points' / f'{k}.json').read_text())['images'][0], f'{k}_photo.jpg', 1000),
        f'{t} (for rectification)') for k, t in RECTS) + '</div></section>')

# A.2
sec.append('<section id="a2"><h2>A.2 &nbsp;Recovering Homographies</h2>')
sec.append(code('cell-11', 'Implementation: <code>build_homography_system</code>, <code>computeH</code>'))
sec.append('<h3>Correspondences and recovered H</h3>')
for key, title in MOSAICS:
    for i in range(2):
        sec.append(fig(export(OUT / f'{key}_{i}_{i + 1}_points.jpg', f'{key}_{i}_{i + 1}_points.jpg'),
                       f'{title}: images {i} and {i + 1}', 'wide'))
        sec.append(f'<pre class="matrix"><span>H ({title.lower()} {i} &rarr; {i + 1})</span>'
                   f'{homography_text(key, i)}</pre>')
sec.append('<h3>Derivation</h3>' + md(nb['cell-18']) + '</section>')

# A.3
sec.append('<section id="a3"><h2>A.3 &nbsp;Warping the Images</h2>')
sec.append(md(a3_impl))
sec.append(code('cell-21', 'Implementation: <code>warpImageNearestNeighbor</code>, <code>warpImageBilinear</code>'))
sec.append(md(a3_rect))
for key, title in RECTS:
    sec.append(f'<h3>{title}</h3><div class="row">'
               + fig(rect_input(key), 'Input, with the rectangle corners marked')
               + fig(export(OUT / f'{key}_rectified_nn.jpg', f'{key}_rectified_nn.jpg', 1000), 'Rectified (nearest neighbor)')
               + fig(export(OUT / f'{key}_rectified_bilinear.jpg', f'{key}_rectified.jpg', 1000), 'Rectified (bilinear)')
               + '</div>')
sec.append(md(a3_cmp))
for key, title in RECTS:
    sec.append(fig(export(OUT / f'{key}_interp_zoom.jpg', f'{key}_interp_zoom.jpg'),
                   f'{title}, zoomed in: nearest neighbor vs. bilinear', 'wide'))
sec.append('</section>')

# A.4
sec.append('<section id="a4"><h2>A.4 &nbsp;Blending the Images into a Mosaic</h2>')
sec.append(md(a4_proc))
sec.append(code('cell-32', 'Implementation: <code>make_alpha_mask</code>, <code>build_mosaic</code>'))
sec.append(code('cell-34', 'Implementation: chaining homographies to the reference image'))
for (key, title), text in zip(MOSAICS, a4_each):
    sec.append(f'<h3>{title}</h3>'
               + fig(export(OUT / f'{key}_sources.jpg', f'{key}_sources.jpg'), 'Source images', 'wide')
               + fig(export(OUT / f'{key}.jpg', f'{key}_mosaic.jpg'), 'Mosaic', 'wide mosaic')
               + md(text))
sec.append('</section>')

# ---------- Part B ----------

PAIRS = [(k, t, i) for k, t in MOSAICS for i in range(2)]
sec.append('<div class="part" id="partB"><p class="eyebrow">Part B</p><h2>Feature Matching for Autostitching</h2></div>')

# B.1
sec.append('<section id="b1"><h2>B.1 &nbsp;Harris Corner Detection</h2>' + md(nbB['R_x3Dgpv0JEN']))
sec.append(code('612fff34', 'Implementation: <code>detect_corners</code>', nbB))
sec.append('<h3>Adaptive Non-Maximal Suppression</h3>' + md(nbB['Ib77FTH90RC_']))
sec.append(code('b3c0423e', 'Implementation: <code>anms</code>', nbB))
sec.append(md(nbB['ySm0rXaR1oBI']))
for key, title in MOSAICS:
    sec.append(fig(export(OUT / f'b_{key}_corners.jpg', f'b_{key}_corners.jpg'),
                   f'{title} (middle image): all Harris corners vs. after ANMS', 'wide'))
sec.append(md(nbB['760bac09']))
for key, title in MOSAICS:
    sec.append(fig(export(OUT / f'b_{key}_top_vs_anms.jpg', f'b_{key}_top_vs_anms.jpg'),
                   f'{title}: 500 strongest corners vs. 500 corners from ANMS', 'wide'))
sec.append('</section>')

# B.2
sec.append('<section id="b2"><h2>B.2 &nbsp;Feature Descriptor Extraction</h2>' + md(nbB['ErWsuoKoOqJ6']))
sec.append(code('a90725c2', 'Implementation: <code>extract_descriptors</code>', nbB))
sec.append(fig(export(OUT / 'b_descriptors.jpg', 'b_descriptors.jpg'),
               'Sampled features: location, raw 40x40 window, and normalized 8x8 descriptor', 'wide'))
sec.append('</section>')

# B.3
sec.append('<section id="b3"><h2>B.3 &nbsp;Feature Matching</h2>' + md(nbB['4KLUwol5smq-']))
sec.append(code('90008eb6', 'Implementation: <code>match_features</code>', nbB))
sec.append(md(nbB['ps4c0L-rvFlr']))
for key, title, i in PAIRS:
    sec.append(fig(export(OUT / f'b_{key}_{i}_{i + 1}_matches.jpg', f'b_{key}_{i}_{i + 1}_matches.jpg'),
                   f'{title}: images {i} and {i + 1}, matches passing the ratio test', 'wide'))
sec.append(md(nbB['8bf8ef93']))
sec.append(fig(export(OUT / 'b_ratio_hist.jpg', 'b_ratio_hist.jpg', 1000),
               'Distribution of the 1-NN / 2-NN ratio over all features', 'narrow'))
sec.append('</section>')

# B.4
sec.append('<section id="b4"><h2>B.4 &nbsp;RANSAC and Automatic Mosaics</h2>' + md(nbB['RJ6sJk6n0MES']))
sec.append(code('9cedc239', 'Implementation: <code>ransac_homography</code>', nbB))
for key, title, i in PAIRS:
    sec.append(fig(export(OUT / f'b_{key}_{i}_{i + 1}_ransac.jpg', f'b_{key}_{i}_{i + 1}_ransac.jpg'),
                   f'{title}: images {i} and {i + 1}, RANSAC inliers (green) and outliers (red)', 'wide'))
sec.append('<h3>Manual vs. automatic</h3>' + md(nbB['qWeYTb9B5Ifm']))
rows = [re.match(r'(\w+) (\d->\d): error on manual points: manual H ([\d.]+) px, automatic H ([\d.]+) px', l)
        for l in stream_lines('f46557dd')]
sec.append('<table><tr><th>Pair</th><th>Manual H</th><th>Automatic H</th></tr>' + ''.join(
    f'<tr><td>{m[1]} {m[2].replace("->", " &rarr; ")}</td><td>{m[3]} px</td><td>{m[4]} px</td></tr>'
    for m in rows if m) + '</table><p class="note">Mean reprojection error on the Part A correspondence points.</p>')
for key, title in MOSAICS:
    sec.append(f'<h3>{title}</h3>' + fig(export(OUT / f'b_{key}_manual_vs_auto.jpg', f'b_{key}_manual_vs_auto.jpg'),
                                          'Manual (left) vs. automatic (right)', 'wide mosaic'))
sec.append('<h3>Experiment: sloppier clicks</h3>' + md(nbB['0e9a3341']))
sec.append(fig(export(OUT / 'b_click_noise_curve.jpg', 'b_click_noise_curve.jpg', 1000),
               'Error vs. simulated click noise', 'narrow'))
for key, title in MOSAICS:
    sec.append(fig(export(OUT / f'b_{key}_sloppy_vs_auto.jpg', f'b_{key}_sloppy_vs_auto.jpg'),
                   f'{title}: careful clicks, simulated sloppy clicks, automatic', 'wide mosaic'))
sec.append(md(nbB['c9b900ef']))
for key, title in MOSAICS:
    sec.append(fig(export(OUT / f'b_{key}_sloppy_zoom.jpg', f'b_{key}_sloppy_zoom.jpg'),
                   f'{title}: zoom on the left seam', 'wide'))
sec.append('</section>')

PROJECT_NAV = ('<p class="projects"><a href="cs180.html">Project 0</a><a href="cs180-proj1.html">Project 1</a>'
               '<a href="cs180-proj2.html">Project 2</a><a href="cs180-proj3.html">Project 3</a><b>Project 4</b></p>'
               if args.site else '')

ACK = ("This project was super fun! I'd like to thank Claude for assistance with skeleton functions, helper "
       "functions, conceptual help, general formatting, and polishing my writing.")
sec.append(f'<section id="ack"><h2>Acknowledgements</h2><p>{html.escape(ACK)}</p></section>')

# Styled after International House: cream stucco, dark walnut doors, terracotta roof tiles, arches, brass plaques.
page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CS180 Project 4: Image Warping, Mosaicing &amp; Autostitching</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,600;0,700;1,500&family=EB+Garamond:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono&display=swap">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script>
  window.MathJax = {{ tex: {{ inlineMath: [['$', '$']], displayMath: [['$$', '$$']] }} }};
  document.addEventListener('DOMContentLoaded', () => hljs.highlightAll());
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js"></script>
<style>
  :root {{
    --stucco: #f2e8d5; --stucco-2: #e9dcc1; --stone: #cdb894;
    --walnut: #3b2417; --walnut-2: #5b3a27; --text: #2e1e14; --muted: #7a6150;
    --terracotta: #a5442a; --clay: #c66b45; --brass: #b8913f;
  }}
  * {{ box-sizing: border-box; }}
  html {{ scroll-behavior: smooth; }}
  body {{ margin: 0; color: var(--text); font: 19px/1.7 'EB Garamond', Georgia, serif;
         background: var(--stucco) radial-gradient(ellipse at top, #f7f0e2, var(--stucco) 70%) fixed; }}

  /* walnut entrance with a stucco archway */
  header {{ padding: 56px 16px 0; text-align: center;
           background: repeating-linear-gradient(90deg, var(--walnut) 0 7px, #42291b 7px 9px, var(--walnut) 9px 23px); }}
  .arch {{ max-width: 760px; margin: 0 auto; padding: 84px 32px 40px; background: var(--stucco);
          border: 10px solid var(--stone); border-bottom: none; border-radius: 400px 400px 0 0; }}
  .projects {{ margin: -48px 0 36px; display: flex; gap: 6px 18px; justify-content: center; flex-wrap: wrap;
              font: 600 15px/1.4 'Cormorant Garamond', serif; }}
  .projects a {{ color: var(--muted); text-decoration: none; border-bottom: 1px solid transparent; }}
  .projects a:hover {{ color: var(--terracotta); border-bottom-color: var(--terracotta); }}
  .projects b {{ color: var(--walnut); border-bottom: 1px solid var(--walnut); }}
  .eyebrow {{ margin: 0; font: 600 14px/1 'Cormorant Garamond', serif; letter-spacing: 0.32em;
             text-transform: uppercase; color: var(--terracotta); }}
  header h1 {{ margin: 14px 0 6px; font: 700 clamp(30px, 4.6vw, 50px)/1.05 'Cormorant Garamond', serif; color: var(--walnut); }}
  .byline {{ margin: 0; font-style: italic; color: var(--muted); }}
  nav {{ margin-top: 26px; display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; }}
  nav a {{ padding: 10px 16px 6px; border-radius: 22px 22px 3px 3px; background: var(--walnut);
          border: 1px solid var(--brass); color: var(--stucco); text-decoration: none;
          font: 600 15px/1 'Cormorant Garamond', serif; letter-spacing: 0.04em; transition: background 0.2s; }}
  nav a:hover {{ background: var(--terracotta); }}
  nav.partnav {{ align-items: center; margin-top: 14px; }}
  nav.partnav span {{ font: 600 13px/1 'Cormorant Garamond', serif; letter-spacing: 0.24em; text-transform: uppercase;
                     color: var(--terracotta); min-width: 64px; }}
  .part {{ margin-top: 72px; padding: 28px 16px 8px; text-align: center; border-top: 3px double var(--terracotta); }}
  .part h2 {{ font-size: clamp(30px, 5vw, 42px); margin-top: 10px; justify-content: center; }}
  .part h2::before, .part h2::after {{ display: none; }}
  figure.narrow {{ max-width: 680px; margin-left: auto; margin-right: auto; }}
  p.note {{ text-align: center; color: var(--muted); font-style: italic; font-size: 16px; margin-top: -6px; }}

  /* clay roof tiles */
  .tiles {{ height: 24px; background:
           radial-gradient(circle at 50% 0, var(--clay) 0 9px, var(--terracotta) 10px 12px, transparent 13px) 0 6px / 26px 20px repeat-x,
           linear-gradient(var(--terracotta), var(--terracotta)) top / 100% 7px no-repeat; }}

  main {{ max-width: 980px; margin: 0 auto; padding: 0 16px 72px; }}
  section {{ padding-top: 56px; }}
  h2 {{ display: flex; align-items: center; gap: 16px; margin: 0 0 24px; color: var(--walnut);
       font: 700 clamp(26px, 4vw, 34px)/1.2 'Cormorant Garamond', serif; text-align: center; }}
  h2::before, h2::after {{ content: ''; flex: 1; height: 2px;
       background: linear-gradient(90deg, transparent, var(--terracotta) 30%, var(--terracotta) 70%, transparent); }}
  h3 {{ margin: 40px 0 12px; font: 700 25px/1.2 'Cormorant Garamond', serif; color: var(--terracotta); }}
  strong {{ color: var(--walnut); }}
  a {{ color: var(--terracotta); }}

  figure {{ margin: 22px 0; }}
  figure img {{ display: block; width: 100%; height: auto; border: 6px solid var(--walnut-2);
               outline: 1px solid var(--brass); outline-offset: -3px; border-radius: 3px;
               box-shadow: 0 6px 18px rgba(59, 36, 23, 0.18); }}
  figure.mosaic img {{ background: #000; }}
  figcaption {{ margin-top: 8px; text-align: center; color: var(--muted); font-style: italic; font-size: 16px; }}
  .row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 22px; align-items: start; }}
  .row figure {{ margin: 8px 0; }}

  code {{ font-family: 'JetBrains Mono', monospace; font-size: 0.78em; }}
  p code, li code {{ background: var(--stucco-2); padding: 1px 5px; border-radius: 3px; }}
  details.code {{ margin: 22px 0; border: 1px solid var(--stone); border-radius: 4px; overflow: hidden; background: #fbf7ef; }}
  details.code summary {{ cursor: pointer; padding: 10px 16px; background: var(--walnut); color: var(--stucco);
                         font: 600 17px/1.4 'Cormorant Garamond', serif; letter-spacing: 0.02em; }}
  details.code summary code {{ color: #f3d9a6; }}
  details.code pre {{ margin: 0; overflow-x: auto; }}
  details.code pre code.hljs {{ background: #fbf7ef; padding: 16px; font-size: 13px; line-height: 1.55; }}
  pre.matrix {{ margin: -6px 0 30px; padding: 12px 16px; background: var(--stucco-2); border-left: 4px solid var(--brass);
               font: 13px/1.5 'JetBrains Mono', monospace; overflow-x: auto; }}
  pre.matrix span {{ display: block; color: var(--muted); }}
  table {{ border-collapse: collapse; margin: 18px auto; font-size: 17px; }}
  th, td {{ padding: 6px 18px; border-bottom: 1px solid var(--stone); text-align: left; }}
  th {{ border-bottom: 2px solid var(--terracotta); font-family: 'Cormorant Garamond', serif; font-size: 19px; }}
  mjx-container[display="true"] {{ overflow-x: auto; overflow-y: hidden; }}

  /* brass plaque, like the one by the patio door */
  #ack p {{ max-width: 640px; margin: 0 auto; padding: 22px 28px; text-align: center; font-style: italic;
           color: #3a2a12; background: linear-gradient(135deg, #d9b968, var(--brass) 50%, #a37d32);
           border: 2px solid #8a6a2a; border-radius: 4px; box-shadow: inset 0 0 0 4px rgba(255, 240, 200, 0.35), 0 6px 18px rgba(59, 36, 23, 0.2); }}

  footer {{ background: var(--walnut); color: var(--stone); text-align: center; font-style: italic; font-size: 16px; }}
  footer .tiles {{ transform: scaleY(-1); }}
  footer p {{ margin: 0; padding: 22px 16px 28px; }}
  @media print {{
    h2, h3, details.code {{ break-after: avoid; }}
    figure, pre.matrix, table, #ack p {{ break-inside: avoid; }}
    figure img {{ box-shadow: none; }}
  }}
</style>
</head>
<body>
<header>
  <div class="arch">
    {PROJECT_NAV}
    <p class="eyebrow">CS180 &middot; Project 4</p>
    <h1>Image Warping, Mosaicing &amp; Autostitching</h1>
    <p class="byline">{html.escape(NAME)}</p>
    <nav class="partnav"><span>Part A</span><a href="#a1">Pictures</a><a href="#a2">Homographies</a><a href="#a3">Warping</a><a href="#a4">Mosaics</a></nav>
    <nav class="partnav"><span>Part B</span><a href="#b1">Corners</a><a href="#b2">Descriptors</a><a href="#b3">Matching</a><a href="#b4">RANSAC</a></nav>
  </div>
</header>
<div class="tiles"></div>
<main>
{chr(10).join(sec)}
</main>
<footer><div class="tiles"></div><p>Photographed at International House, Berkeley</p></footer>
</body>
</html>
"""
(WEB / PAGE).write_text(page)
print('wrote', WEB / PAGE, f'({sum(f.stat().st_size for f in IMG.iterdir()) / 1e6:.1f} MB of images)')
