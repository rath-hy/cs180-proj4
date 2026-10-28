"""Click correspondence points on one or more images and save them to JSON.

Usage:
    python pick_points.py OUT.json IMG1 [IMG2 ...] [--max-dim 1600]

Controls:
    left click   add a point to the image you clicked on
    right click  undo the last point on that image
    close window save and exit

For a pair of images, click the same feature in each image in the same order
(e.g. left, right, left, right, ...). Point i in every image must be the same
physical feature. Points are (x, y) = (column, row) in the image as loaded with
the same --max-dim the notebook uses.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps


def load_image(path, max_dim=1600):
    im = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    if max_dim and max(im.size) > max_dim:
        s = max_dim / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    return np.asarray(im)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out')
    parser.add_argument('images', nargs='+')
    parser.add_argument('--max-dim', type=int, default=1600)
    args = parser.parse_args()

    ims = [load_image(p, args.max_dim) for p in args.images]
    out = Path(args.out)
    pts = [[] for _ in ims]
    if out.exists():
        prev = json.loads(out.read_text())
        if prev.get('images') == [str(p) for p in args.images]:
            pts = prev['points']
            print(f'Resuming with {[len(p) for p in pts]} existing points.')

    fig, axes = plt.subplots(1, len(ims), figsize=(8 * len(ims), 8), squeeze=False)
    axes = list(axes[0])
    artists = [[] for _ in ims]
    colors = plt.cm.tab10.colors

    def redraw(k):
        for a in artists[k]:
            a.remove()
        artists[k] = []
        for i, (x, y) in enumerate(pts[k]):
            c = colors[i % len(colors)]
            artists[k].append(axes[k].plot(x, y, '+', color=c, ms=14, mew=2)[0])
            artists[k].append(axes[k].text(x + 6, y - 6, str(i), color=c, fontsize=12,
                                           weight='bold'))
        axes[k].set_title(f'{Path(args.images[k]).name}  ({len(pts[k])} pts)')
        fig.canvas.draw_idle()

    def on_click(event):
        # ignore clicks outside the images or while the zoom/pan tool is active
        if event.inaxes not in axes or getattr(fig.canvas.toolbar, 'mode', ''):
            return
        k = axes.index(event.inaxes)
        if event.button == 1:
            pts[k].append([float(event.xdata), float(event.ydata)])
        elif event.button == 3 and pts[k]:
            pts[k].pop()
        redraw(k)

    for k, (ax, im) in enumerate(zip(axes, ims)):
        ax.imshow(im)
        ax.axis('off')
        redraw(k)
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.suptitle('Left click: add   Right click: undo   Close window: save')
    plt.tight_layout()
    plt.show()

    counts = [len(p) for p in pts]
    if len(set(counts)) != 1:
        print(f'WARNING: point counts differ across images: {counts}')
    out.write_text(json.dumps({
        'images': [str(p) for p in args.images],
        'shapes': [list(im.shape[:2]) for im in ims],
        'points': pts,
    }, indent=2))
    print(f'Saved {counts} points to {out}')


if __name__ == '__main__':
    main()
