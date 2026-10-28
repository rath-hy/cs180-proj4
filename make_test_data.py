"""Generate synthetic test sets with known homographies.

Each "scene" is a photo treated as a plane in front of a pinhole camera. Views are rendered by
rotating the camera about its center, so view i -> view j is exactly a homography. Correspondences
are taken at strong corners in the overlap and jittered by ~0.7px to mimic hand clicking.

Writes data/test/*.jpg, points/provided/test_*.json (reference clicks; copy into points/ to use them) and data/test/ground_truth.json.
Usage: python make_test_data.py
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter, maximum_filter

ROOT = Path(__file__).parent
CS180 = ROOT.parent
OUT_IMG, OUT_PTS = ROOT / 'data' / 'test', ROOT / 'points' / 'provided'
OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_PTS.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(180)
CLICK_NOISE = 0.7


def load(path, max_dim=None):
    im = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    if max_dim and max(im.size) > max_dim:
        s = max_dim / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    return im


def K(f, cx, cy):
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])


def rot(yaw=0.0, pitch=0.0, roll=0.0):
    y, p, r = np.radians([yaw, pitch, roll])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def apply(H, pts):
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


def render(scene, H_view_to_scene, size):
    """Render a view: each view pixel samples the scene at H_view_to_scene @ pixel."""
    H = H_view_to_scene / H_view_to_scene[2, 2]
    corners = apply(H, np.array([[0, 0], [size[0] - 1, 0], [size[0] - 1, size[1] - 1], [0, size[1] - 1]], float))
    assert (corners >= 0).all() and (corners[:, 0] < scene.width).all() and (corners[:, 1] < scene.height).all(), \
        f'view leaves the scene: {corners.round()}'
    # PIL samples at pixel corners, our convention is pixel centers: shift by half a pixel both ways
    S = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]])
    Hp = S @ H @ np.linalg.inv(S)
    return scene.transform(size, Image.PERSPECTIVE, (Hp / Hp[2, 2]).ravel()[:8], Image.BICUBIC)


def corners_in(im, n, mask=None):
    """Strong, well-spread Harris corners (x, y) of a PIL image."""
    g = np.asarray(im.convert('L'), float) / 255
    gy, gx = np.gradient(gaussian_filter(g, 1))
    a, b, c = (gaussian_filter(v, 2) for v in (gx * gx, gx * gy, gy * gy))
    R = a * c - b * b - 0.05 * (a + c) ** 2
    R[:20], R[-20:], R[:, :20], R[:, -20:] = 0, 0, 0, 0
    if mask is not None:
        R[~mask] = 0
    peaks = (R == maximum_filter(R, 25)) & (R > 0)
    ys, xs = np.nonzero(peaks)
    order = np.argsort(-R[ys, xs])
    pts = []
    for i in order:  # greedy: strongest first, keep them spread out
        p = np.array([xs[i], ys[i]], float)
        if all(np.hypot(*(p - q)) > 0.08 * max(im.size) for q in pts):
            pts.append(p)
        if len(pts) == n:
            break
    return np.array(pts)


def save_points(name, images, shapes, pts_list):
    noisy = [p + rng.normal(0, CLICK_NOISE, p.shape) for p in pts_list]
    (OUT_PTS / f'{name}.json').write_text(json.dumps({
        'images': [str(p) for p in images], 'shapes': shapes,
        'points': [np.round(p, 2).tolist() for p in noisy]}, indent=2))


ground_truth = {}


def make_mosaic(name, scene, f, angles, view_size, n_pts=12):
    """angles: list of (yaw, pitch) per view, in degrees."""
    W, Hh = view_size
    Ks, Kv = K(f, scene.width / 2, scene.height / 2), K(f, W / 2, Hh / 2)
    Hs = [Ks @ rot(*a) @ np.linalg.inv(Kv) for a in angles]  # view i -> scene
    views, paths = [], []
    for i, H in enumerate(Hs):
        v = render(scene, H, view_size)
        p = OUT_IMG / f'{name}_{i}.jpg'
        v.save(p, quality=95)
        views.append(v)
        paths.append(p.relative_to(ROOT))
    for i in range(len(Hs) - 1):
        H_ij = np.linalg.inv(Hs[i + 1]) @ Hs[i]
        H_ij /= H_ij[2, 2]
        # corners of view i that are also inside view i+1
        yy, xx = np.mgrid[0:Hh, 0:W]
        q = apply(H_ij, np.c_[xx.ravel(), yy.ravel()]).reshape(Hh, W, 2)
        mask = (q[..., 0] > 20) & (q[..., 0] < W - 20) & (q[..., 1] > 20) & (q[..., 1] < Hh - 20)
        p_i = corners_in(views[i], n_pts, mask)
        save_points(f'{name}_{i}_{i + 1}', paths[i:i + 2], [[Hh, W]] * 2, [p_i, apply(H_ij, p_i)])
        ground_truth[f'{name}: H {i}->{i + 1}'] = H_ij.round(8).tolist()
    print(f'{name}: {len(views)} views of {view_size}, overlap points saved')
    return [str(p) for p in paths]


def make_rectify(name, poster, poster_px, view_size, f, pitch, yaw, roll=0, extent=8):
    """Paste a poster on a tiled plane and view it obliquely. Saves its 4 corners (TL, TR, BR, BL)."""
    pw, ph = poster_px
    poster = poster.resize((pw, ph), Image.LANCZOS)
    tile = 60
    sw, sh = extent * pw, extent * ph
    yy, xx = np.mgrid[0:sh, 0:sw]
    checker = ((xx // tile + yy // tile) % 2)[..., None] * np.array([40, 40, 30]) + np.array([170, 160, 140])
    checker = checker + rng.normal(0, 6, checker.shape)
    plane = Image.fromarray(np.clip(checker, 0, 255).astype(np.uint8))
    x0, y0 = (sw - pw) // 2, (sh - ph) // 2
    plane.paste(poster, (x0, y0))
    W, Hh = view_size
    H = K(f, sw / 2, sh / 2) @ rot(yaw, pitch, roll) @ np.linalg.inv(K(f, W / 2, Hh / 2))  # view -> plane
    # shift the principal point so the poster lands in the middle of the view
    c = apply(np.linalg.inv(H), np.array([[sw / 2, sh / 2]]))[0]
    H = H @ np.linalg.inv(np.array([[1, 0, W / 2 - c[0]], [0, 1, Hh / 2 - c[1]], [0, 0, 1]]))
    v = render(plane, H, view_size)
    path = OUT_IMG / f'{name}.jpg'
    v.save(path, quality=95)
    # poster corners are at pixel-edge positions on the plane
    plane_corners = np.array([[x0 - 0.5, y0 - 0.5], [x0 + pw - 0.5, y0 - 0.5],
                              [x0 + pw - 0.5, y0 + ph - 0.5], [x0 - 0.5, y0 + ph - 0.5]])
    view_corners = apply(np.linalg.inv(H), plane_corners)
    assert ((view_corners > 0) & (view_corners < [W, Hh])).all(), f'{name}: poster not fully in view'
    save_points(name, [path.relative_to(ROOT)], [[Hh, W]], [view_corners])
    print(f'{name}: poster {pw}x{ph} (aspect {pw / ph:.3f})')
    return str(path.relative_to(ROOT)), (pw, ph)


if __name__ == '__main__':
    campanile = load(CS180 / 'proj3/images/campanile.jpg', 3600)
    campanile = campanile.crop((0, 1500, campanile.width, 1500 + 1250))
    venice = load(CS180 / 'proj2/venice.jpg').resize((1800, 1200), Image.LANCZOS)
    bayon = load(CS180 / 'proj2/bayon.jpg').resize((1400, 1597), Image.LANCZOS)

    config = {'mosaics': {}, 'rectify': {}}
    config['mosaics']['test_mosaic1'] = make_mosaic('test_mosaic1', campanile, 900, [(-22, 0), (0, 0), (22, 0)], (800, 600))
    config['mosaics']['test_mosaic2'] = make_mosaic('test_mosaic2', venice, 1000, [(-12, 2), (12, -2)], (800, 600))
    config['mosaics']['test_mosaic3'] = make_mosaic('test_mosaic3', bayon, 800, [(0, -12), (3, 12)], (600, 800))

    config['rectify']['test_rect1'] = make_rectify(
        'test_rect1', load(CS180 / 'proj2/cameraman.png').convert('RGB'), (300, 300), (900, 675), 800, 30, 10, 5)
    config['rectify']['test_rect2'] = make_rectify(
        'test_rect2', load(CS180 / 'proj2/venice.jpg'), (360, 240), (900, 675), 700, -10, 35, extent=10)

    (OUT_IMG / 'ground_truth.json').write_text(json.dumps(ground_truth, indent=2))
    print(json.dumps(config, indent=2))
