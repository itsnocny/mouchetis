"""Speech bubbles set into a photo.

A bubble pasted on a photo gives itself away fast: too white, a colour the
light of the room doesn't have, sharper than the plane it floats in, no grain,
no shadow, placed anywhere. Each pass below fixes one of these, and the
console prints what it measured to do so.

    mouchetis bubbles photo.jpg --say "I'm not posing."
    mouchetis bubbles photo.jpg --say "I'm not posing." --opacity 0.5
    mouchetis bubbles photo.jpg --say "hand-l:That's mine. 🔒" --style manga
    mouchetis bubbles photo.jpg --think "none:It's cold." --behind
    mouchetis bubbles photo_dots.png --detect-on photo.jpg --say "..."
    mouchetis bubbles photo.jpg --machine --variants --say "..."

The anchors (`mouth`, `hand-l`, `shoulders`...) are the same as for the dot
screen; `mouchetis photo.jpg --anchors` lists them.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .stipple import (ANCHOR_BOXES, ANCHORS, HEAD_IDX, INK, PAPER, Detection,
                      anchor_box, anchor_conf, anchor_point, detect, draw_rich,
                      face, load_image, luminance, rich_width, save_image,
                      structureless)

SS = 3          # supersampling for drawing outlines

# A bubble's white is never 255: that's the first thing that gives it away.
# Each style starts from a colour, the photo's ambient light does the rest.
STYLES = {
    # paper: the same ink and paper as the dot screen, thin stroke
    "paper": dict(fill=PAPER, ink=INK, stroke=0.0011, stroke_alpha=0.8,
                  fonts=(), shadow=1.0, opacity=1.0),
    # manga: bold stroke, near-white fill
    "manga": dict(fill=(252, 251, 248), ink=(16, 16, 16), stroke=0.0026,
                  stroke_alpha=1.0, shadow=0.8, opacity=1.0,
                  fonts=("/System/Library/Fonts/Supplemental/Futura.ttc",
                         "/System/Library/Fonts/Avenir Next.ttc")),
    # glass: the bubble is made of the scene itself, frosted and lightened
    "glass": dict(fill=None, ink=INK, stroke=0.0014, stroke_alpha=0.55,
                  fonts=(), shadow=0.45, opacity=1.0),
    # chalk: hand-drawn stroke, a washed fill that lets the scene through
    "chalk": dict(fill=PAPER, ink=(40, 36, 34), stroke=0.0022, stroke_alpha=0.85,
                  shadow=0.6, opacity=0.84,
                  fonts=("/System/Library/Fonts/Noteworthy.ttc",
                         "/System/Library/Fonts/Supplemental/Chalkboard.ttc")),
}

# Correction for the grain estimator: the MAD of a 3x3 median residual
# underestimates the noise standard deviation. Measured on a flat patch plus
# known Gaussian noise (sigma 4 to 12): ratio 0.86, within 2 %; 0.90 at
# sigma 2 (quantization).
GRAIN_CAL = 1.0 / 0.86


# reading the scene

@dataclass
class Scene:
    busy: np.ndarray        # local busyness, ~0..1.5
    subject: np.ndarray     # outline 0..1
    face: np.ndarray        # forbidden area (faces, dilated), bool
    occupied: np.ndarray    # bubbles already placed, bool
    L: np.ndarray           # luminance 0..1
    grad2: np.ndarray
    lap2: np.ndarray
    light: np.ndarray       # unit vector TOWARDS the light, image coordinates
    white: float            # luminance of this photo's white
    med_l: float
    sharp_ref: float        # reference sharpness (the subject, assumed sharp)
    edge_thr: float
    busy_sorted: np.ndarray  # to rank a bubble against the whole image
    head_c: np.ndarray | None  # centre of the speaking head, px
    head_h: float            # height of that head, px: the unit for placement
    facing: float            # -1 looks to the image's left, +1 to the right, 0 straight on


# Anchors that mean the head: a bubble attached to them is placed relative to
# the whole head, not the exact point.
HEAD_LABELS = {"mouth", "nose", "face", "head", "eye-l", "eye-r", "ear-l", "ear-r"}

# face mesh: tip of the nose, outermost cheeks
M_NOSE, M_LEFT, M_RIGHT = 1, 234, 454


def face_zone(det: Detection, shape: tuple[int, int]) -> np.ndarray:
    """Faces plus a margin. Neither the bubble nor the tail tip goes in."""
    h, w = shape
    m = np.zeros((h, w), np.uint8)
    for f in det.faces:
        cv2.fillConvexPoly(m, cv2.convexHull(f.astype(np.int32)), 1)
    if not det.faces and det.pose is not None:
        idx = [i for i in HEAD_IDX if det.vis is None or det.vis[i] > 0.3]
        if len(idx) >= 3:
            pts = det.pose[idx]
            ext = float(np.ptp(pts, axis=0).max())
            cv2.fillConvexPoly(m, cv2.convexHull(pts.astype(np.int32)), 1)
            # the pose landmarks fit inside the face: grow towards the head
            k = max(3, int(ext * 0.6) | 1)
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    ys = np.where(m.any(1))[0]
    if ys.size:
        k = max(3, int((ys.max() - ys.min()) * 0.15) | 1)
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return m.astype(bool)


def gaze(det: Detection) -> tuple[np.ndarray | None, float, float]:
    """Centre of the head, its height, and the side it looks towards.

    The gaze shows in the offset of the nose from the middle of the cheeks
    (face mesh) or of the ears (pose): zero straight on, and in three-quarter
    view it pulls towards the side the person looks at. A bubble on that side
    reads as their speech; behind the back of their neck it rings false.
    """
    if det.faces:
        if det.pose is not None:        # the subject's face, not a reflection
            f = min(det.faces, key=lambda f: float(np.linalg.norm(f.mean(0) - det.pose[0])))
        else:
            f = max(det.faces, key=lambda f: float(np.ptp(f[:, 0]) * np.ptp(f[:, 1])))
        l, r = f[M_LEFT], f[M_RIGHT]
        off = (f[M_NOSE][0] - (l[0] + r[0]) / 2) / (abs(r[0] - l[0]) + 1e-6)
        # the face mesh spans forehead to chin; the head with hair is ~1.3x
        return f.mean(0), float(np.ptp(f[:, 1])) * 1.3, float(np.clip(off / 0.18, -1, 1))
    if det.pose is None:
        return None, 0.0, 0.0
    vis = det.vis if det.vis is not None else np.ones(33, np.float32)
    idx = [i for i in HEAD_IDX if vis[i] > 0.3]
    if len(idx) < 3:
        return None, 0.0, 0.0
    pts = det.pose[idx]
    nose = det.pose[0]
    el, er = det.pose[7], det.pose[8]
    if vis[7] > 0.3 and vis[8] > 0.3:
        off = (nose[0] - (el[0] + er[0]) / 2) / (abs(el[0] - er[0]) + 1e-6)
        facing = float(np.clip(off / 0.25, -1, 1))
    elif vis[7] > 0.3 or vis[8] > 0.3:  # profile: only one ear seen
        ear = el if vis[7] > 0.3 else er
        facing = float(np.sign(nose[0] - ear[0]))
    else:
        facing = 0.0
    # the pose landmarks (eyes, ears, mouth) cover ~60 % of the head
    return pts.mean(0), float(np.ptp(pts, axis=0).max()) * 1.7, facing


def analyse(img: np.ndarray, det: Detection) -> Scene:
    h, w = img.shape[:2]
    diag = math.hypot(h, w)
    L = luminance(img).astype(np.float32)
    # Edges and sharpness are measured after a median filter: on raw
    # luminance, sensor grain passes for sharp edges and a blurred area is
    # judged sharp (measured: 1.04 x the subject on a blurred flat patch plus
    # noise of sigma 5).
    Lf = cv2.medianBlur((L * 255).astype(np.uint8), 5).astype(np.float32) / 255
    gx = cv2.Sobel(Lf, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Lf, cv2.CV_32F, 0, 1, ksize=3)
    grad2 = gx * gx + gy * gy
    lap = cv2.Laplacian(Lf, cv2.CV_32F, ksize=3)
    lap2 = lap * lap

    busy = cv2.GaussianBlur(np.sqrt(grad2), (0, 0), 0.006 * diag)
    busy = np.clip(busy / (np.percentile(busy, 95) + 1e-6), 0, 1.5)

    subject = (det.body.astype(np.float32) if det.body is not None
               else np.zeros((h, w), np.float32))

    # Light direction: mean slope of the luminance at a very large scale.
    # Rough but robust, and a shadow cast the wrong way shows more than an
    # approximate one.
    sw = 64
    small = cv2.resize(L, (sw, max(8, int(sw * h / w))), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 4)
    sgy, sgx = np.gradient(small)
    v = np.array([sgx.mean(), sgy.mean()], np.float32)
    n = float(np.linalg.norm(v))
    light = v / n if n > 2e-4 else np.array([-0.45, -0.89], np.float32)

    # Reference sharpness: Laplacian / gradient ratio on the subject's sharp
    # edges. The ratio doesn't depend on contrast, only on edge width. A sharp
    # plain wall and a blurred plain wall can't be told apart, but they don't
    # need to be.
    # an edge is at least a step of ~4 % in luminance
    edge_thr = max(float(np.percentile(grad2, 80)), 0.15 ** 2)
    sel = grad2 > edge_thr
    if (subject > 0.5).sum() > 500:
        sel &= subject > 0.5
    sharp_ref = float(lap2[sel].mean() / (grad2[sel].mean() + 1e-9)) if sel.any() else 1.0

    head_c, head_h, facing = gaze(det)
    return Scene(busy, subject, face_zone(det, (h, w)), np.zeros((h, w), bool),
                 L, grad2, lap2, light, float(np.percentile(L, 99.5)),
                 float(np.median(L)), sharp_ref, edge_thr,
                 np.sort(busy[::4, ::4], axis=None), head_c, head_h, facing)


# what the bubble says, and who to

@dataclass
class Spec:
    text: str
    target: np.ndarray | None       # where the tail points, px; None = no tail
    label: str
    form: str


def parse_spec(raw: str, form: str, det: Detection, shape: tuple[int, int]) -> Spec:
    """'[anchor:]text'. Anchor = name, area, normalized 'x,y', or 'none'.

    A ':' inside the text itself is fine: if what comes before it isn't a
    known anchor, it's all taken as text, and the tail goes to the mouth.
    """
    h, w = shape
    head, sep, rest = raw.partition(":")
    key = head.strip().lower()
    text = rest.strip().replace("\\n", "\n")
    if sep and key == "none":
        return Spec(text, None, "none", form)
    if sep and key in ANCHORS:
        if det.pose is None:
            sys.exit(f"'{key}': no pose detected. Give 'x,y:text' or 'none:text'.")
        if anchor_conf(key, det) < 0.25:
            print(f"  /!\\ anchor '{key}' is unreliable: the tail may point off.",
                  file=sys.stderr)
        return Spec(text, anchor_point(key, det), key, form)
    if sep and key in ANCHOR_BOXES:
        box = anchor_box(key, det) if det.pose is not None or key == "body" else None
        if box is None:
            sys.exit(f"area '{key}' can't be used on this photo.")
        return Spec(text, np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
                                   np.float32), key, form)
    if sep:
        parts = head.split(",")
        if len(parts) == 2:
            try:
                v = [float(x) for x in parts]
            except ValueError:
                v = None
            if v is not None:
                px = np.array([v[0] * w if v[0] <= 1 else v[0],
                               v[1] * h if v[1] <= 1 else v[1]], np.float32)
                return Spec(text, px, f"{v[0]:g},{v[1]:g}", form)
    if det.pose is None:
        sys.exit("no pose detected: give 'x,y:text' or 'none:text'.")
    if anchor_conf("mouth", det) < 0.25:
        print("  /!\\ mouth barely visible: the tail may point off.", file=sys.stderr)
    return Spec(raw.strip().replace("\\n", "\n"), anchor_point("mouth", det),
                "mouth", form)


def style_font(style: str, px: int) -> ImageFont.FreeTypeFont:
    for path in STYLES[style]["fonts"]:
        try:
            return ImageFont.truetype(path, px)
        except OSError:
            continue
    return face(px)


def layout(text: str, fo, fs: int, max_w: float, aspect: float = 1.7):
    """Breaks the text so the block tends towards `aspect` (width/height).

    The bubble fits the text, never the other way round.
    """
    lead = int(fs * 1.28)
    total = rich_width(" ".join(text.split()), fo, fs)
    target = min(max_w, max(fs * 5.0, math.sqrt(total * lead * aspect)))
    lines: list[str] = []
    for para in text.split("\n"):
        line = ""
        for word in para.split():
            trial = (line + " " + word).strip()
            if line and rich_width(trial, fo, fs) > target:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
    tw = max(rich_width(l, fo, fs) for l in lines) if lines else fs
    th = lead * (len(lines) - 1) + fs * 1.15
    return lines, tw, th, lead


# shape

# how far the outline reaches beyond the base ellipse, per form
REACH = {"oval": 1.0, "round": 1.0, "thought": 1.13, "shout": 1.32}


def outline(form: str, rx: float, ry: float, wobble: float,
            rng: np.random.Generator, n: int = 520) -> np.ndarray:
    """Outline centred on 0. Never a perfect ellipse: an exact ellipse reads
    as clip art, a slight irregularity as a drawn line."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(th), np.sin(th)
    if form == "round":                      # superellipse n=4: a soft rectangle
        x = rx * np.sign(c) * np.abs(c) ** 0.5
        y = ry * np.sign(s) * np.abs(s) ** 0.5
    else:
        x, y = rx * c, ry * s
    r = np.ones_like(th)
    for k in (2, 3, 5):
        r += wobble * rng.uniform(0.4, 1.0) / k * np.sin(k * th + rng.uniform(0, 2 * np.pi))
    if form == "thought":                    # round lobes, sharp dips: a cloud
        per = 2 * np.pi * math.sqrt((rx * rx + ry * ry) / 2)
        nb = max(7, int(round(per / (0.62 * min(rx, ry)))))
        r *= 1 + 0.12 * np.abs(np.sin(nb * th / 2 + rng.uniform(0, np.pi))) ** 0.8
    elif form == "shout":                    # irregular spikes
        nb = int(rng.integers(13, 19))
        amp = rng.uniform(0.12, 0.3, nb)
        t = th * nb / (2 * np.pi)
        r *= 1 + amp[t.astype(int) % nb] * (1 - np.abs(2 * (t % 1.0) - 1))
    return np.stack([x * r, y * r], 1)


def rotate(pts: np.ndarray, deg: float) -> np.ndarray:
    """Counter-clockwise rotation ON SCREEN (y pointing down), like PIL.rotate."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.stack([c * pts[:, 0] + s * pts[:, 1], -s * pts[:, 0] + c * pts[:, 1]], 1)


def _radius(u: np.ndarray, rx: float, ry: float, tilt: float) -> float:
    """Radius of the base ellipse in direction u (image coordinates)."""
    ub = rotate(u[None], -tilt)[0]
    return 1.0 / math.sqrt((ub[0] / rx) ** 2 + (ub[1] / ry) ** 2)


def tail_tip(target: np.ndarray, center: np.ndarray, scene: Scene, gap: float) -> np.ndarray:
    """The tip heads for the anchor but stops at the edge of the face."""
    h, w = scene.face.shape
    u = center - target
    dist = float(np.linalg.norm(u))
    u = u / max(dist, 1e-6)
    t = target.astype(np.float32).copy()
    walked = 0.0
    while walked < dist:
        x, y = int(t[0]), int(t[1])
        if not (0 <= x < w and 0 <= y < h) or not scene.face[y, x]:
            break
        t += u * 2.0
        walked += 2.0
    return t + u * gap


def tail_polys(form: str, center, tip, rx, ry, tilt, curve) -> list[np.ndarray]:
    u = tip - center
    dist = float(np.linalg.norm(u))
    u = u / max(dist, 1e-6)
    re = _radius(u, rx, ry, tilt)
    if dist <= re * REACH[form] * 1.05:
        return []                            # the anchor is under the bubble
    perp = np.array([-u[1], u[0]], np.float32)
    if form == "thought":                    # a string of shrinking bubbles
        edge = center + u * re * 1.12
        out = []
        th = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        for t, k in ((0.22, 0.20), (0.56, 0.13), (0.88, 0.075)):
            p = edge + (tip - edge) * t
            rr = k * min(rx, ry)
            out.append(np.stack([p[0] + rr * 1.25 * np.cos(th), p[1] + rr * np.sin(th)], 1))
        return out
    half = 0.2 * min(rx, ry)
    base = center + u * re * 0.6
    b1, b2 = base + perp * half, base - perp * half
    bend = perp * curve * 0.18 * (dist - re * 0.6)
    t = np.linspace(0, 1, 32)[:, None]

    def bez(p0, p1, p2):
        return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2

    s1 = bez(b1, (b1 + tip) / 2 + bend, tip)
    s2 = bez(tip, (b2 + tip) / 2 + bend, b2)
    return [np.vstack([s1, s2[1:]])]


def raster(polys, origin, size_ss) -> np.ndarray:
    im = Image.new("L", size_ss, 0)
    d = ImageDraw.Draw(im)
    for p in polys:
        q = (np.asarray(p, np.float64) - origin) * SS
        d.polygon([tuple(v) for v in q], fill=255)
    return np.asarray(im, np.float32) / 255.0


# placement

def place(scene: Scene, bw, bh, tw, th, spec: Spec, cfg, reach: float | None):
    """Finds the cheapest centre for the bubble.

    You know who speaks because the bubble almost touches their head, so
    closeness is a CONSTRAINT (a window of 0.3 to 2.4 head heights, widened by
    `reach` if nothing fits), and a calm background only breaks ties within
    that window. Inside the window: beside rather than above, at mouth height,
    on the side the person looks at.

    Forbidden: faces, other bubbles, edges, text over the subject. Computed
    with a box filter, so for every centre at once.
    """
    h, w = scene.busy.shape
    diag = math.hypot(h, w)
    q = min(1.0, 480 / max(h, w))
    sz = (max(8, int(w * q)), max(8, int(h * q)))

    def red(a):
        return cv2.resize(a.astype(np.float32), sz, interpolation=cv2.INTER_AREA)

    def mean(a, k):
        return cv2.boxFilter(a, -1, (max(1, int(k[0] * q)), max(1, int(k[1] * q))),
                             normalize=True, borderType=cv2.BORDER_REPLICATE)

    busy, subj = red(scene.busy), red(scene.subject)
    forbid = red(scene.face | scene.occupied)
    # The bubble may overlap the hair or a shoulder rather than move far away;
    # the text stays off the subject. With --behind it slides in further.
    w_b, w_t = (0.2, 4.0) if cfg.behind else (0.6, 4.0)
    cost = mean(busy, (bw, bh)) + w_b * mean(subj, (bw, bh)) + w_t * mean(subj, (tw, th))
    cost[mean(forbid, (bw, bh)) > 1e-3] = np.inf

    ys, xs = np.mgrid[0:sz[1], 0:sz[0]].astype(np.float32)
    cx, cy = (xs + 0.5) / q, (ys + 0.5) / q
    m = 0.025 * min(h, w)
    cost[(cx - bw / 2 < m) | (cx + bw / 2 > w - m) |
         (cy - bh / 2 < m) | (cy + bh / 2 > h - m)] = np.inf
    t = spec.target
    if t is not None:
        H = scene.head_h if scene.head_h > 0 else 0.08 * diag
        head = spec.label in HEAD_LABELS and scene.head_c is not None
        ref = scene.head_c if head else t
        # distance from the anchor to the bubble's EDGE, not its centre: a big
        # bubble needn't sit further away than a small one
        dx, dy = cx - ref[0], cy - ref[1]
        d = np.hypot(dx, dy) + 1e-6
        r_dir = 1.0 / np.sqrt((dx / d / (bw / 2)) ** 2 + (dy / d / (bh / 2)) ** 2)
        gap = d - r_dir
        pref, lo, hi = (0.8, 0.3, 2.4) if head else (0.4, 0.1, 2.0)
        if reach is not None:
            cost[(gap < lo * H) | (gap > hi * H * reach)] = np.inf
        cost += 0.8 * ((gap - pref * H) / H) ** 2
        # beside, at mouth height (centre slightly above)
        cost += 0.5 * ((cy - (t[1] - 0.2 * H)) / H) ** 2
        if head and abs(scene.facing) >= 0.3:
            cost += 1.5 * abs(scene.facing) * (dx * np.sign(scene.facing) < 0)
    if not np.isfinite(cost).any():
        return None
    i = int(np.argmin(cost))
    return np.array([cx.flat[i], cy.flat[i]], np.float32)


# compositing

def rel_lum(rgb) -> float:
    c = np.asarray(rgb, np.float64) / 255.0
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def contrast(a, b) -> float:
    la, lb = rel_lum(a), rel_lum(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def worst_contrast(ink, pixels: np.ndarray, dark_ink: bool) -> float:
    """Contrast against the least favourable background pixel (1st
    percentile, so the grain doesn't dictate the measure)."""
    c = np.clip(pixels, 0, 255) / 255.0
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    lum = 0.2126 * c[:, 0] + 0.7152 * c[:, 1] + 0.0722 * c[:, 2]
    bg = float(np.percentile(lum, 1 if dark_ink else 99))
    li = rel_lum(ink)
    return (max(li, bg) + 0.05) / (min(li, bg) + 0.05)


def grain_sigma(src: np.ndarray, ring: np.ndarray, grad2: np.ndarray) -> np.ndarray:
    """Standard deviation of the photo's noise around the bubble, per channel,
    measured on the flattest pixels of the ring (otherwise it measures
    texture)."""
    med = cv2.medianBlur(np.clip(src, 0, 255).astype(np.uint8), 3).astype(np.float32)
    hp = src - med
    sel = ring.copy()
    if ring.sum() > 200:
        sel &= grad2 < np.percentile(grad2[ring], 35)
    if sel.sum() < 40:
        sel = ring if ring.sum() >= 40 else np.ones_like(ring)
    sig = 1.4826 * np.median(np.abs(hp[sel]), axis=0) * GRAIN_CAL
    return np.clip(sig, 0.3, 14.0).astype(np.float32)


def focus_blur(scene: Scene, sl, ring: np.ndarray, fs: int, cfg, diag) -> tuple[float, str]:
    """Blur to apply so the bubble isn't sharper than its plane."""
    g2, l2 = scene.grad2[sl], scene.lap2[sl]
    sel = ring & (g2 > scene.edge_thr)
    if sel.sum() < max(30, 0.01 * ring.sum()):
        return 0.0, "no measurable edge around (flat area): no blur"
    ratio = float(l2[sel].mean() / (g2[sel].mean() + 1e-9)) / (scene.sharp_ref + 1e-9)
    if ratio >= 0.7:
        return 0.0, f"sharp plane ({ratio:.2f} x the subject)"
    sig = cfg.blur * (0.7 - ratio) / 0.7 * 0.004 * diag
    sig = min(sig, 0.045 * fs)               # the text must stay readable
    return sig, f"blurred plane ({ratio:.2f} x the subject) -> blur {sig:.1f} px"


@dataclass
class Fit:
    fs: int
    fs0: int
    fo: object
    lines: list
    tw: float
    th: float
    lead: int
    rx: float
    ry: float
    hx: float               # half extent, tilt and outline included
    hy: float
    center: np.ndarray
    window: float | None


def fit(scene: Scene, spec: Spec, font_fn, size: float, wobble: float, tilt: float,
        sw: float, behind: bool = False, extra_h: float = 0.0) -> Fit | None:
    """Text size and bubble position.

    A bubble far from the head no longer reads as that person's speech, so
    first try smaller text (down to 60 %), then a wider window, and only as a
    last resort anywhere. `extra_h` reserves extra height under the text, in
    multiples of the type size (the speaker line).
    """
    w = scene.busy.shape[1]
    reach = REACH[spec.form] * (1 + wobble * 0.6)
    pc = argparse.Namespace(behind=behind)
    fs0 = max(12, int(size * w))
    for window in (1.0, 1.6, None):
        fs = fs0
        floor = 10 if window is None else max(10, int(fs0 * 0.6))
        while fs >= floor:
            fo = font_fn(fs)
            lines, tw, th, lead = layout(spec.text, fo, fs, 0.38 * w)
            th_box = th + extra_h * fs
            pad = 0.55 * fs
            k = 1.19 if spec.form == "round" else 1.42
            rx = (tw / 2 + pad) * k * (1 + wobble)
            ry = (th_box / 2 + pad) * k * (1 + wobble)
            a = math.radians(tilt)
            hx = math.sqrt((rx * math.cos(a)) ** 2 + (ry * math.sin(a)) ** 2) * reach + sw
            hy = math.sqrt((rx * math.sin(a)) ** 2 + (ry * math.cos(a)) ** 2) * reach + sw
            center = place(scene, 2 * hx, 2 * hy, tw, th_box, spec, pc, window)
            if center is not None:
                return Fit(fs, fs0, fo, lines, tw, th, lead, rx, ry, hx, hy, center, window)
            fs = int(fs * 0.86)
    return None


def draw_bubble(out: np.ndarray, src: np.ndarray, scene: Scene, spec: Spec,
                style: str, rng: np.random.Generator, cfg) -> dict | None:
    h, w = src.shape[:2]
    diag = math.hypot(h, w)
    st = STYLES[style]
    sw = max(1.0, st["stroke"] * diag)
    form = spec.form
    tilt = float(rng.uniform(-1, 1) * cfg.tilt)

    ft = fit(scene, spec, lambda px: style_font(style, px), cfg.size, cfg.wobble,
             tilt, sw, cfg.behind)
    if ft is None:
        print(f"  [{spec.label}] no free space, even small: bubble skipped",
              file=sys.stderr)
        return None
    fs, fs0, fo, lines, th, lead = ft.fs, ft.fs0, ft.fo, ft.lines, ft.th, ft.lead
    rx, ry, hx, hy, center, window = ft.rx, ft.ry, ft.hx, ft.hy, ft.center, ft.window

    # --- outline + tail, on a local canvas
    body = rotate(outline(form, rx, ry, cfg.wobble, rng), tilt) + center
    polys = [body]
    tip = None
    if spec.target is not None:
        tip = tail_tip(spec.target, center, scene, cfg.gap * diag)
        polys += tail_polys(form, center, tip, rx, ry, tilt, rng.choice((-1, 1)) *
                            rng.uniform(0.5, 1.0))
    allp = np.vstack(polys)
    shadow_reach = 0.01 * diag * cfg.shadow + 0.025 * diag
    x0 = int(max(0, allp[:, 0].min() - shadow_reach))
    y0 = int(max(0, allp[:, 1].min() - shadow_reach))
    x1 = int(min(w, allp[:, 0].max() + shadow_reach + 1))
    y1 = int(min(h, allp[:, 1].max() + shadow_reach + 1))
    wl, hl = x1 - x0, y1 - y0
    origin = np.array([x0, y0], np.float32)
    sl = (slice(y0, y1), slice(x0, x1))
    m_ss = raster(polys, origin, (wl * SS, hl * SS))

    # stroke centred on the outline: half outside, half inside
    sw_ss = max(2, int(round(sw * SS)))
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sw_ss // 2 * 2 + 1,) * 2)
    ki = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ((sw_ss - sw_ss // 2) * 2 + 1,) * 2)
    outer, inner = cv2.dilate(m_ss, ko), cv2.erode(m_ss, ki)

    def down(a):
        return cv2.resize(a, (wl, hl), interpolation=cv2.INTER_AREA)

    fa = down(inner)
    sa = np.clip(down(outer - inner), 0, 1) * st["stroke_alpha"]
    is_glass = st["fill"] is None
    if is_glass:
        kr = max(3, int(2.5 * sw * SS) | 1)
        band = np.clip(down(inner - cv2.erode(inner, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kr, kr)))), 0, 1)
        band = cv2.GaussianBlur(band, (0, 0), max(0.6, sw * 0.6))
    if style == "chalk":
        # hand-drawn stroke: uneven pressure and a second, offset pass
        nz = cv2.GaussianBlur(rng.standard_normal((hl, wl)).astype(np.float32), (0, 0), 3 + sw)
        nz = (nz - nz.mean()) / (nz.std() + 1e-6)
        sa = sa * np.clip(0.7 + 0.35 * nz, 0.25, 1.0)
        dx, dy = rng.uniform(-1, 1, 2) * sw * 0.8
        sa = np.maximum(sa, cv2.warpAffine(sa, np.float32([[1, 0, dx], [0, 1, dy]]),
                                           (wl, hl)) * 0.45)
    ua = np.maximum(down(outer), 0)
    src_loc = src[sl].astype(np.float32)

    # --- ambient: colour and level of the light around the bubble
    solid = ua > 0.02
    rr = max(3, int(0.02 * diag) | 1)
    ring = cv2.dilate(solid.astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rr, rr))).astype(bool) & ~solid
    if ring.sum() < 50:
        ring = ~solid if (~solid).sum() >= 50 else np.ones_like(solid)
    amb = np.median(src_loc[ring], axis=0)
    loc_l = float(np.median(scene.L[sl][ring]))
    chroma = np.clip(amb / (amb.mean() + 1e-3), 0.8, 1.25)
    # the bubble's white never exceeds the photo's white, and drops a little
    # in a dark area
    lum = (1 - cfg.light) + cfg.light * float(np.clip(
        scene.white * (0.82 + 0.18 * np.clip(loc_l / (scene.med_l + 1e-3), 0, 1.6)),
        0.55, 1.0))

    # --- shading: lit side / shadow side, along the light direction
    yy, xx = np.mgrid[0:hl, 0:wl].astype(np.float32)
    cl = center - origin
    nx, ny = (xx - cl[0]) / rx, (yy - cl[1]) / ry
    p = nx * scene.light[0] + ny * scene.light[1]
    shade = np.clip(1 + 0.5 * cfg.shading * p, 1 - cfg.shading, 1 + cfg.shading)

    smoke = False
    if is_glass:
        # Frosted glass: the scene itself, blurred until no detail competes
        # with the letters, then lightened. Translucent without being
        # transparent: a plain opacity leaves the scene sharp under the text.
        # Faces are kept out of the blur (normalized convolution) so their
        # colour doesn't tint the bubble.
        keep = (~scene.face[sl]).astype(np.float32)
        sg = 0.018 * diag
        frost = (cv2.GaussianBlur(src_loc * keep[..., None], (0, 0), sg) /
                 np.maximum(cv2.GaussianBlur(keep, (0, 0), sg), 1e-3)[..., None])
        white = 255.0 * scene.white
        # in a dark area SMOKED glass (darkened, light text) fits in better than
        # a light plate that would glow in the dark
        smoke = cfg.glass == "smoked" or (cfg.glass == "auto" and loc_l < 0.3)
        tone = (np.full(3, 0.10 * white) if smoke else np.full(3, white)) * chroma ** cfg.ambient
        ink = np.array(PAPER if smoke else INK, np.float32)
        stroke_col = np.clip(np.full(3, white) * chroma, 0, 255)
    else:
        base = np.array(st["fill"], np.float32) * chroma ** cfg.ambient * lum
        fill = np.clip(np.broadcast_to(base, (hl, wl, 3)) * shade[..., None], 0, 255)
        ink = np.array(st["ink"], np.float32) * chroma ** (cfg.ambient * 0.5)
        stroke_col = ink
        # readability: text / background contrast of at least 4.5 (WCAG AA)
        ratio = contrast(ink, base)
        for _ in range(6):
            if ratio >= 4.5:
                break
            ink = ink * 0.7 if rel_lum(base) > 0.18 else ink + (255 - ink) * 0.3
            ratio = contrast(ink, base)

    # --- text
    tl = Image.new("RGBA", (wl, hl), (0, 0, 0, 0))
    td = ImageDraw.Draw(tl)
    y = cl[1] - th / 2 - fs * 0.08
    fcol = tuple(int(v) for v in np.clip(ink, 0, 255)) + (238,)
    for line in lines:
        lw = rich_width(line, fo, fs)
        draw_rich(tl, td, (cl[0] - lw / 2, y), line, fo, fcol, fs)
        y += lead
    if abs(tilt) > 0.05:
        tl = tl.rotate(tilt, resample=Image.BICUBIC, center=(float(cl[0]), float(cl[1])))
    tarr = np.asarray(tl, np.float32)
    ta, trgb = tarr[..., 3] / 255.0 * fa, tarr[..., :3]

    opac = None
    if is_glass:
        # Readability guaranteed UNDER EVERY LETTER, not on average: one dark
        # patch of the scene behind a single word is enough to lose it. Take the
        # lowest opacity that holds 4.5:1 everywhere, the most translucent
        # bubble possible without giving up the text.
        kt = max(3, int(fs * 0.3) | 1)
        under = cv2.dilate((ta > 0.05).astype(np.uint8),
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kt, kt))).astype(bool)
        if not under.any():
            under = fa > 0.5
        steps = [cfg.opacity] if cfg.opacity is not None else list(np.arange(0.35, 0.96, 0.05))
        for opac in steps:
            fill = np.clip((frost + (tone - frost) * opac) * shade[..., None], 0, 255)
            ratio = worst_contrast(ink, fill[under], dark_ink=not smoke)
            if ratio >= 4.5:
                break
        if cfg.opacity is not None and ratio < 4.5:
            print(f"  /!\\ --opacity {cfg.opacity:g}: contrast {ratio:.1f}:1 under the "
                  f"least readable letter (4.5 recommended)", file=sys.stderr)

    # --- bubble layer
    if is_glass:
        # rim: the edge facing the light brightens, the other darkens. That's
        # what gives the glass volume instead of a blurred flat patch.
        lit = np.clip((nx * scene.light[0] + ny * scene.light[1]) /
                      np.maximum(np.hypot(nx, ny), 1e-3), -1, 1)
        fill = fill + band[..., None] * np.where(lit[..., None] > 0,
                                                 lit[..., None] * 0.55 * (255 - fill),
                                                 lit[..., None] * 0.30 * fill)
        sa = sa * (0.35 + 0.65 * np.clip(0.5 + 0.5 * lit, 0, 1))
    col = fill * (1 - sa[..., None]) + stroke_col * sa[..., None]
    col = col * (1 - ta[..., None]) + trgb * ta[..., None]
    alpha = np.maximum(np.maximum(fa * st["opacity"], sa), ta)

    sig = grain_sigma(src_loc, ring, scene.grad2[sl])
    if cfg.grain > 0:
        nl = rng.standard_normal((hl, wl)).astype(np.float32)
        nc = rng.standard_normal((hl, wl, 3)).astype(np.float32)
        col = col + cfg.grain * (nl[..., None] * sig.mean() * 0.85 + nc * sig * 0.5)

    blur, blur_msg = focus_blur(scene, sl, ring, fs, cfg, diag)
    if blur > 0.3:
        pm = cv2.GaussianBlur(col * alpha[..., None], (0, 0), blur)
        alpha = cv2.GaussianBlur(alpha, (0, 0), blur)
        col = pm / np.maximum(alpha[..., None], 1e-4)

    subj = scene.subject[sl]
    hidden = 0.0
    if cfg.behind:
        a0 = alpha.sum()
        alpha = alpha * (1 - np.clip(subj, 0, 1))
        hidden = 1 - alpha.sum() / max(a0, 1e-6)

    # --- cast shadow, away from the light, then compositing
    off = 0.010 * diag * cfg.shadow
    M = np.float32([[1, 0, -scene.light[0] * off], [0, 1, -scene.light[1] * off]])
    sh = cv2.GaussianBlur(cv2.warpAffine(ua, M, (wl, hl)), (0, 0), 0.007 * diag) * 0.30
    sh = np.maximum(sh, cv2.GaussianBlur(ua, (0, 0), 0.0015 * diag) * 0.10)
    sh *= cfg.shadow * st["shadow"]
    if cfg.behind:
        sh *= 1 - np.clip(subj, 0, 1)
    loc = out[sl]
    loc *= (1 - sh[..., None])
    loc[:] = loc * (1 - alpha[..., None]) + np.clip(col, 0, 255) * alpha[..., None]

    scene.occupied[sl] |= cv2.dilate((ua > 0.3).astype(np.uint8),
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rr, rr))
                                     ).astype(bool)

    side, gap_h = "", None
    if spec.target is not None and scene.head_h > 0:
        head = spec.label in HEAD_LABELS and scene.head_c is not None
        ref = scene.head_c if head else spec.target
        v = center - ref
        dd = float(np.hypot(*v)) + 1e-6
        gap_h = (dd - 1.0 / math.sqrt((v[0] / dd / hx) ** 2 + (v[1] / dd / hy) ** 2)) / scene.head_h
        if head and abs(scene.facing) >= 0.3:
            side = "on the side they look at" if v[0] * scene.facing > 0 \
                else "BEHIND their gaze (no room in front)"
    return dict(
        label=spec.label, form=form, fs=fs, fs0=fs0, center=center, tip=tip,
        window=window, side=side, gap_h=gap_h, opac=opac, smoke=smoke,
        box=(x0, y0, x1, y1),
        calm=1 - np.searchsorted(scene.busy_sorted,
                                 float(scene.busy[sl][solid].mean())) / scene.busy_sorted.size,
        on_face=int(((alpha > 0.03) & scene.face[sl]).sum()),
        contrast=ratio, grain=sig, blur=blur_msg, hidden=hidden,
        tint=[round(float(c), 2) for c in chroma], lum=lum,
    )


# the machine speaking: a stippled bubble, for mouchetis --say
#
# Three things, and the text is none of them: it stays sharp, in solid ink, at
# every stage.
#   1. the bubble is made of the dot screen: its stroke and its blank interior
#      go through the same dots as the body, and drift towards formless with
#      --stage;
#   2. the tail is the machine's ATTRIBUTION: a construction line to the
#      mouth, which drifts with --stage until it points at no one;
#   3. a label line under the text says who speaks, according to the machine.
#
# All of a bubble's randomness comes from a fixed draw (seed + index): from one
# stage to the next these are the SAME fields, only the thresholds move. So the
# variants read as a sequence, not as eight different bubbles.

@dataclass
class MachinePlan:
    spec: Spec
    ft: Fit
    tilt: float
    base: np.ndarray        # outline at rest, centred on 0, not tilted
    harm: np.ndarray        # (k, amplitude, phase) of the formless distortions
    crop: tuple             # x0, y0, x1, y1: the same for every stage
    drift: np.ndarray       # displacement of the target at stage 1, px
    conf: float             # real confidence of the anchor, at rest
    seed: int
    lw: float               # stroke width, at least one screen cell
    H: float                # head height, px
    face: np.ndarray        # the scene's face area (shared)
    head_c: np.ndarray | None


def plan_machine(img: np.ndarray, det: Detection, ident: np.ndarray, raws: list[str],
                 cfg) -> list[MachinePlan]:
    """Places the bubbles ONCE, for every stage.

    Placement doesn't see the face: the identity area is filled with the
    image's median colour before any measurement. So the bubble's position
    carries no information taken from the face pixels. The gaze side comes
    from the geometry of the anchors (head orientation), not from pixels: the
    outline already gives that away.
    """
    h, w = img.shape[:2]
    diag = math.hypot(h, w)
    unit = max(1.0, min(h, w) * cfg.grain / 1200.0)
    zone = ident > 0.02
    blind = img.copy()
    if zone.any():
        blind[zone] = np.median(img[~zone], axis=0).astype(img.dtype)
    scene = analyse(blind, det)
    H = scene.head_h if scene.head_h > 0 else 0.08 * diag
    k = max(3, int(0.12 * H) | 1)
    scene.face |= cv2.dilate(zone.astype(np.uint8),
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)
    lw = max(1.8 * unit, 0.0011 * diag)

    plans = []
    for i, raw in enumerate(raws):
        spec = parse_spec(raw, "oval", det, (h, w))
        seed = cfg.seed + 101 + i
        rng = np.random.default_rng(seed)
        tilt = float(rng.uniform(-1, 1) * 2.0)
        ft = fit(scene, spec, face, cfg.say_size, 0.025, tilt, lw,
                 extra_h=1.4 if cfg.speaker else 0.0)
        if ft is None:
            print(f"  --say [{spec.label}]: no free space, bubble skipped",
                  file=sys.stderr)
            continue
        base = outline("oval", ft.rx, ft.ry, 0.025, rng)
        ks = np.array([2, 3, 4, 5, 7, 9, 13], np.float32)
        harm = np.stack([ks, rng.uniform(0.5, 1.0, ks.size) / np.sqrt(ks),
                         rng.uniform(0, 2 * np.pi, ks.size)], 1)
        drift = np.zeros(2, np.float32)
        conf = 1.0
        pts = [ft.center + 1.7 * np.array([ft.hx, ft.hy]), ft.center - 1.7 * np.array([ft.hx, ft.hy])]
        if spec.target is not None:
            v = spec.target - ft.center
            v = v / (np.linalg.norm(v) + 1e-6)
            perp = np.array([-v[1], v[0]], np.float32) * rng.choice((-1, 1))
            drift = (perp * 0.85 + v * rng.uniform(-0.3, 0.3)) * 1.6 * H * cfg.say_drift
            if spec.label in ANCHORS or spec.label in ANCHOR_BOXES:
                conf = anchor_conf(spec.label, det)
            m = 0.6 * ft.fs + 4 * lw
            pts += [spec.target - m, spec.target + m,
                    spec.target + drift - m, spec.target + drift + m]
        pts = np.array(pts)
        crop = (int(max(0, pts[:, 0].min())), int(max(0, pts[:, 1].min())),
                int(min(w, pts[:, 0].max() + 1)), int(min(h, pts[:, 1].max() + 1)))
        plans.append(MachinePlan(spec, ft, tilt, base, harm, crop, drift.astype(np.float32),
                                 float(conf), seed, lw, H, scene.face, scene.head_c))
        # the next bubbles avoid this one
        oc = np.zeros((h, w), np.uint8)
        cv2.ellipse(oc, (int(ft.center[0]), int(ft.center[1])),
                    (int(ft.hx * 1.15), int(ft.hy * 1.15)), 0, 0, 360, 1, -1)
        scene.occupied |= oc.astype(bool)

        where = ""
        if spec.target is not None and scene.head_c is not None and spec.label in HEAD_LABELS:
            side = ft.center[0] - scene.head_c[0]
            where = (" facing the camera" if abs(scene.facing) < 0.3 else
                     " on the side they look at" if side * scene.facing > 0 else
                     " BEHIND their gaze")
        far = "  /!\\ placed far from the head (no room)" if ft.window is None else ""
        print(f"  --say [{spec.label}]: {ft.fs} px"
              f"{' (shrunk from %d)' % ft.fs0 if ft.fs < ft.fs0 else ''},{where}{far}",
              file=sys.stderr)
    return plans


def _inside(mask: np.ndarray, p: np.ndarray) -> bool:
    x, y = int(p[0]), int(p[1])
    return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and bool(mask[y, x])


def speaker(pl: MachinePlan, s: float) -> tuple[str, float, np.ndarray | None]:
    """What the machine thinks it knows about who speaks, at stage s.

    The starting confidence is the anchor's REAL confidence; it decreases with
    the stage, and as soon as the target has left the head the speaker is
    unknown.
    """
    if pl.spec.target is None:
        return "speaker: unassigned", 0.0, None
    # Pace measured over 8 stages: sure up to ~0.35, doubtful ("?") from ~0.45
    # to ~0.6, unknown afterwards. Judged on the destroyed area alone, it went
    # from "confidence 0.76" to "unknown" with no doubtful step.
    conf = pl.conf * (1 - s)
    tgt = pl.spec.target + pl.drift * s ** 2
    # head anchor: the target must stay on the head; another anchor or a point
    # given by hand: near its original position
    head = pl.spec.label in HEAD_LABELS and pl.head_c is not None
    ref = pl.head_c if head else pl.spec.target
    on_target = float(np.linalg.norm(tgt - ref)) < 0.7 * pl.H
    if conf < 0.35 or not on_target:
        return "speaker: unknown", conf, tgt
    q = " ?" if conf < 0.6 else ""
    name = pl.spec.label if (pl.spec.label in ANCHORS or pl.spec.label in ANCHOR_BOXES) \
        else "given point"
    return f"speaker: {name}{q} · confidence {conf:.2f}", conf, tgt


def _text_layer(pl: MachinePlan, s: float, ink, show_speaker: bool) -> np.ndarray:
    """Text plus speaker line, as an RGBA layer the size of the bubble's crop."""
    x0, y0, x1, y1 = pl.crop
    ft = pl.ft
    cl = ft.center - np.array([x0, y0], np.float32)
    tl = Image.new("RGBA", (x1 - x0, y1 - y0), (0, 0, 0, 0))
    d = ImageDraw.Draw(tl)
    extra = 1.4 * ft.fs if show_speaker else 0.0
    y = cl[1] - (ft.th + extra) / 2 - ft.fs * 0.08
    col = tuple(int(v) for v in ink)
    for line in ft.lines:
        lw = rich_width(line, ft.fo, ft.fs)
        draw_rich(tl, d, (cl[0] - lw / 2, y), line, ft.fo, col + (255,), ft.fs)
        y += ft.lead
    if show_speaker:
        cs = max(9, int(ft.fs * 0.55))
        cf = face(cs)
        txt = speaker(pl, s)[0]
        d.text((cl[0] - cf.getlength(txt) / 2, y - ft.lead + ft.fs * 1.15 + 0.45 * ft.fs),
               txt, font=cf, fill=col + (170,))      # same grey as the caption
    if abs(pl.tilt) > 0.05:
        tl = tl.rotate(pl.tilt, resample=Image.BICUBIC, center=(float(cl[0]), float(cl[1])))
    return np.asarray(tl, np.float32)


@dataclass
class MachineFields:
    """The bubble at stage s, as local fields (the `crop` frame), before rendering."""
    fill: np.ndarray        # interior (left blank)
    outer: np.ndarray       # extent, stroke included
    stroke: np.ndarray      # outline + attribution, already holed
    mix: np.ndarray         # share of formless texture
    blot: np.ndarray        # that texture
    halo: np.ndarray        # blank area around the letters
    att: np.ndarray         # the attribution line alone, already holed


def machine_fields(pl: MachinePlan, s: float, unit: float, ink, show_speaker: bool) -> MachineFields:
    """Distorted outline, holed stroke, formless texture and halo of the letters.

    Shared by both renders: in the dot screen (mouchetis --say) and on the
    untouched photo (mouchetis bubbles --machine).
    """
    s = float(np.clip(s, 0, 1))
    x0, y0, x1, y1 = pl.crop
    wl, hl = x1 - x0, y1 - y0
    origin = np.array([x0, y0], np.float32)
    ft = pl.ft
    rng = np.random.default_rng(pl.seed + 7)       # same fields at every stage

    gaps = cv2.GaussianBlur(rng.random((hl, wl)).astype(np.float32), (0, 0), 2.5 * unit)
    gaps = (np.argsort(np.argsort(gaps, axis=None)).reshape(hl, wl) /
            float(gaps.size)).astype(np.float32)   # rank -> uniform 0..1
    blot = structureless((hl, wl), 0.42, 3.0 * unit, rng, 0.22)

    # outline: the radial distortion grows with the stage, and the high
    # harmonics only appear with it
    th = np.linspace(0, 2 * np.pi, pl.base.shape[0], endpoint=False)
    r = np.ones_like(th)
    for k, a, ph in pl.harm:
        r += 0.28 * s ** 1.8 * a * (1.0 if k < 7 else s) * np.sin(k * th + ph)
    # the formless spills outwards; inwards it is bounded, otherwise the
    # outline crosses the text (76 px of distortion at 0.95)
    r = np.maximum(r, 0.9)
    body = rotate(pl.base * r[:, None], pl.tilt) + ft.center
    m_ss = raster([body], origin, (wl * SS, hl * SS))
    lw_ss = max(2, int(round(pl.lw * SS)))
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (lw_ss // 2 * 2 + 1,) * 2)
    ki = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ((lw_ss - lw_ss // 2) * 2 + 1,) * 2)
    outer, inner = cv2.dilate(m_ss, ko), cv2.erode(m_ss, ki)

    # attribution: a construction line, not a comic-book tail. It stops at the
    # edge of the destroyed area; once the target has drifted off the head, it
    # ends in empty space.
    line = np.zeros_like(m_ss)
    _, conf, tgt = speaker(pl, s)
    if tgt is not None:
        u = tgt - ft.center
        dist = float(np.linalg.norm(u))
        u = u / max(dist, 1e-6)
        start = ft.center + u * _radius(u, ft.rx, ft.ry, pl.tilt) * 1.02
        tip = tgt.astype(np.float32).copy()
        walked = 0.0
        while _inside(pl.face, tip) and walked < dist:
            tip -= u * 2.0
            walked += 2.0
        if float(np.dot(tip - start, u)) > 2 * pl.lw:
            th_ss = max(1, int(round(pl.lw * SS)))
            a = tuple(int(v) for v in (start - origin) * SS)
            b = tuple(int(v) for v in (tip - origin) * SS)
            cv2.line(line, a, b, 1.0, th_ss, cv2.LINE_AA)
            # a perpendicular tick, shorter the less sure the machine is
            # (like --lines)
            pv = np.array([-u[1], u[0]]) * (0.15 + 0.85 * max(conf, 0)) * 0.5 * ft.fs
            c1 = tuple(int(v) for v in (tip + pv - origin) * SS)
            c2 = tuple(int(v) for v in (tip - pv - origin) * SS)
            cv2.line(line, c1, c2, 1.0, th_ss, cv2.LINE_AA)

    def down(a):
        return cv2.resize(a, (wl, hl), interpolation=cv2.INTER_AREA)

    fill = down(inner)
    stroke = np.clip(np.maximum(down(outer - inner), down(line)), 0, 1)
    # the stroke gets holes: the share of holes follows the stage
    holes = np.clip((gaps - 0.62 * s ** 1.4) / 0.08 + 0.5, 0, 1)
    stroke *= holes
    att = np.clip(down(line), 0, 1) * holes

    # blank halo around the letters: the text stays readable at every stage
    ta = _text_layer(pl, s, ink, show_speaker)[..., 3] / 255.0
    kh = max(3, int(0.22 * ft.fs) | 1)
    halo = cv2.dilate((ta > 0.08).astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kh, kh)))
    halo = np.clip(cv2.GaussianBlur(halo.astype(np.float32), (0, 0), kh * 0.35) * 1.5, 0, 1)

    region = np.clip(cv2.GaussianBlur(np.maximum(down(outer), stroke), (0, 0), 2 * unit)
                     * 1.5, 0, 1)
    mix = s * 0.85 * region * (1 - halo)
    return MachineFields(fill, down(outer), stroke, mix, blot, halo, att)


def machine_density(d: np.ndarray, ident: np.ndarray, plans: list[MachinePlan],
                    s: float, unit: float, ink, show_speaker: bool) -> np.ndarray:
    """Writes the bubbles into the density field, BEFORE stippling.

    Blank inside, an ink stroke at the edge, an attribution line towards the
    mouth. With the stage: the outline distorts, the stroke gets holes, the
    formless texture invades the bubble, except for a halo around the letters
    that keeps the text readable. Nothing is written in the identity area.
    """
    d = d.copy()
    for pl in plans:
        x0, y0, x1, y1 = pl.crop
        f = machine_fields(pl, s, unit, ink, show_speaker)
        loc = d[y0:y1, x0:x1]
        new = np.maximum(loc * (1 - f.fill), f.stroke)
        new = new * (1 - f.mix) + f.blot * f.mix
        new *= 1 - f.halo        # paper around the letters, even outside the bubble
        keep = ident[y0:y1, x0:x1] > 0.02
        d[y0:y1, x0:x1] = np.where(keep, loc, new)
    return d


def machine_text(out: np.ndarray, plans: list[MachinePlan], s: float, ink,
                 show_speaker: bool) -> np.ndarray:
    """The text, AFTER stippling: solid ink, never stippled."""
    out = out.copy()
    for pl in plans:
        x0, y0, x1, y1 = pl.crop
        lay = _text_layer(pl, float(np.clip(s, 0, 1)), ink, show_speaker)
        a = lay[..., 3:4] / 255.0
        out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - a) + lay[..., :3] * a
    return out


def machine_on_photo(img: np.ndarray, plans: list[MachinePlan], s: float, scene: Scene,
                     unit: float, style: str, cfg) -> np.ndarray:
    """The machine bubble on the UNTOUCHED photo.

    The photo itself is left alone: only the bubble carries the machine's
    texture. Style `stipple`: the inside is the scene as the machine sees it,
    what lies behind the bubble redrawn as ink dots on the screen's paper,
    density following the light, like a window of dot screen cut into the
    normal photo. Styles `glass` and `paper`: a fill matched to the photo's
    light. In every case the outline, attribution line and blots are ink dots
    drawn on a fixed grid: from one stage to the next the same dots stay and
    new ones appear. The text, laid last, is sharp at every stage.
    """
    h, w = img.shape[:2]
    diag = math.hypot(h, w)
    out = img.astype(np.float32).copy()
    s = float(np.clip(s, 0, 1))
    for pl in plans:
        x0, y0, x1, y1 = pl.crop
        sl = (slice(y0, y1), slice(x0, x1))
        hl, wl = y1 - y0, x1 - x0
        src = img[sl].astype(np.float32)
        rng = np.random.default_rng(pl.seed + 13)       # same draws at every stage

        # extent of the fill: the interior, plus the letters' halo even where
        # the distorted outline cuts inwards
        f0 = machine_fields(pl, s, unit, INK, cfg.speaker)
        cover = np.maximum(f0.fill, f0.halo)
        solid = f0.outer > 0.02
        rr = max(3, int(0.02 * diag) | 1)
        ring = cv2.dilate(solid.astype(np.uint8), cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (rr, rr))).astype(bool) & ~solid
        if ring.sum() < 50:
            ring = ~solid if (~solid).sum() >= 50 else np.ones_like(solid)
        amb = np.median(src[ring], axis=0)
        loc_l = float(np.median(scene.L[sl][ring]))
        chroma = np.clip(amb / (amb.mean() + 1e-3), 0.8, 1.25)
        white = 255.0 * scene.white

        smoke = False
        d_in = None
        if style == "stipple":
            # the scene behind the bubble as ink demand: (1 - luminance), set
            # on the bubble alone (as the screen is set on the body), then capped
            # so the text keeps reading. A slight smoothing below the dot size
            # avoids sensor aliasing.
            L = cv2.GaussianBlur(scene.L[sl], (0, 0), 1.5 * unit)
            raw = np.clip(1.0 - L, 0, 1)
            inside_px = f0.fill > 0.5
            if inside_px.sum() > 64:
                lo, hi = np.percentile(raw[inside_px], [2, 98])
                raw = np.clip((raw - lo) / max(hi - lo, 1e-3), 0, 1)
            d_in = raw * cfg.density
        elif style == "glass":
            keep = (~scene.face[sl]).astype(np.float32)
            sg = 0.018 * diag
            frost = (cv2.GaussianBlur(src * keep[..., None], (0, 0), sg) /
                     np.maximum(cv2.GaussianBlur(keep, (0, 0), sg), 1e-3)[..., None])
            smoke = cfg.glass == "smoked" or (cfg.glass == "auto" and loc_l < 0.3)
            tone = (np.full(3, 0.10 * white) if smoke else np.full(3, white)) * chroma ** cfg.ambient
        ink = (np.array(PAPER if smoke else INK, np.float32) *
               chroma ** (cfg.ambient * 0.5))
        lay = _text_layer(pl, s, ink, cfg.speaker)
        ta = lay[..., 3] / 255.0

        if style == "stipple":
            fill = np.broadcast_to(np.array(PAPER, np.float32), (hl, wl, 3)).copy()
            ink = np.array(INK, np.float32)
            lay = _text_layer(pl, s, ink, cfg.speaker)
            ta = lay[..., 3] / 255.0
        elif style == "glass":
            kt = max(3, int(pl.ft.fs * 0.3) | 1)
            under = cv2.dilate((ta > 0.05).astype(np.uint8),
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kt, kt))).astype(bool)
            steps = [cfg.opacity] if cfg.opacity is not None else list(np.arange(0.35, 0.96, 0.05))
            for opac in steps:
                fill = np.clip(frost + (tone - frost) * opac, 0, 255)
                if worst_contrast(ink, fill[under], dark_ink=not smoke) >= 4.5:
                    break
        else:
            lum = (1 - cfg.light) + cfg.light * float(np.clip(
                scene.white * (0.82 + 0.18 * np.clip(loc_l / (scene.med_l + 1e-3), 0, 1.6)),
                0.55, 1.0))
            fill = np.broadcast_to(np.array(PAPER, np.float32) * chroma ** cfg.ambient * lum,
                                   (hl, wl, 3)).copy()

        # the photo's grain in the fill, so it isn't smoother than the photo,
        # except with stipple, where the paper has the screen's grain
        sig = grain_sigma(src, ring, scene.grad2[sl])
        if style == "stipple":
            fill = fill + rng.normal(0, 1.5, (hl, wl, 1)).astype(np.float32)
        elif cfg.grain > 0:
            fill = fill + cfg.grain * (rng.standard_normal((hl, wl, 1)).astype(np.float32)
                                       * sig.mean() * 0.85 +
                                       rng.standard_normal((hl, wl, 3)).astype(np.float32)
                                       * sig * 0.5)

        # light cast shadow, away from the light
        off = 0.008 * diag * cfg.shadow
        M = np.float32([[1, 0, -scene.light[0] * off], [0, 1, -scene.light[1] * off]])
        sh = cv2.GaussianBlur(cv2.warpAffine(cover, M, (wl, hl)), (0, 0), 0.007 * diag)
        sh *= 0.22 * cfg.shadow * (0.45 if style == "glass" else 1.0)
        # the shadow's tail touched 1 grey level far from the bubble; below 1 %
        # it doesn't show, so leave the photo alone there
        sh[sh < 0.01] = 0
        loc = out[sl]
        loc *= 1 - (sh * (1 - cover))[..., None]
        loc[:] = loc * (1 - cover[..., None]) + np.clip(fill, 0, 255) * cover[..., None]

        # ink dots: outline, attribution, blots, on a FIXED grid
        if d_in is not None:      # same recipe as machine_density
            d_ink = np.maximum(d_in * f0.fill, f0.stroke)
            d_ink = (d_ink * (1 - f0.mix) + f0.blot * f0.mix) * (1 - f0.halo)
        else:
            d_ink = (f0.stroke * (1 - f0.mix) + f0.blot * f0.mix) * (1 - f0.halo)
        gh, gw = max(1, int(hl / unit)), max(1, int(wl / unit))
        cell = cv2.resize(d_ink, (gw, gh), interpolation=cv2.INTER_AREA)
        hit = (rng.random((gh, gw)) < cell).astype(np.float32)
        hit = cv2.resize(hit, (wl, hl), interpolation=cv2.INTER_NEAREST)
        a = hit * (0.86 + 0.14 * rng.random((hl, wl)).astype(np.float32))
        loc[:] = loc * (1 - a[..., None]) + ink * a[..., None]

        # text, sharp
        loc[:] = loc * (1 - ta[..., None]) + lay[..., :3] * ta[..., None]
    return np.clip(out, 0, 255)


def crumble_on_photo(img: np.ndarray, plans: list[MachinePlan], s: float, scene: Scene,
                     unit: float, cfg, probe: list | None = None) -> np.ndarray:
    """Crumbling glass: the machine bubble that holds up on its own on a photo.

    The fill is the photo's frosted glass, which is what blends it in. The
    machine shows at the EDGE: it breaks into shards that keep the glass's
    colour at that spot (so the photo's), never ink. With the stage the
    crumbling moves towards the centre, never reaching the text halo, and
    shards scatter around. One fixed draw per shard: from one stage to the
    next the same shards fall, more and more of them.
    """
    h, w = img.shape[:2]
    diag = math.hypot(h, w)
    out = img.astype(np.float32).copy()
    s = float(np.clip(s, 0, 1))
    c = max(3, int(round(2.4 * unit)))                  # shard size, px
    for pl in plans:
        x0, y0, x1, y1 = pl.crop
        sl = (slice(y0, y1), slice(x0, x1))
        hl, wl = y1 - y0, x1 - x0
        src = img[sl].astype(np.float32)
        f0 = machine_fields(pl, s, unit, INK, cfg.speaker)

        # --- glass: ambient light, frosting, ink, an opacity that keeps the text readable
        solid = f0.fill > 0.5
        rr = max(3, int(0.02 * diag) | 1)
        ring = cv2.dilate(solid.astype(np.uint8), cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (rr, rr))).astype(bool) & ~solid
        if ring.sum() < 50:
            ring = ~solid if (~solid).sum() >= 50 else np.ones_like(solid)
        amb = np.median(src[ring], axis=0)
        loc_l = float(np.median(scene.L[sl][ring]))
        chroma = np.clip(amb / (amb.mean() + 1e-3), 0.8, 1.25)
        white = 255.0 * scene.white
        keep = (~scene.face[sl]).astype(np.float32)
        sg = 0.018 * diag
        frost = (cv2.GaussianBlur(src * keep[..., None], (0, 0), sg) /
                 np.maximum(cv2.GaussianBlur(keep, (0, 0), sg), 1e-3)[..., None])
        smoke = cfg.glass == "smoked" or (cfg.glass == "auto" and loc_l < 0.3)
        tone = (np.full(3, 0.10 * white) if smoke else np.full(3, white)) * chroma ** cfg.ambient
        ink = np.array(PAPER if smoke else INK, np.float32) * chroma ** (cfg.ambient * 0.5)
        lay = _text_layer(pl, s, ink, cfg.speaker)
        ta = lay[..., 3] / 255.0
        kt = max(3, int(pl.ft.fs * 0.3) | 1)
        under = cv2.dilate((ta > 0.05).astype(np.uint8),
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kt, kt))).astype(bool)
        steps = [cfg.opacity] if cfg.opacity is not None else list(np.arange(0.35, 0.96, 0.05))
        for opac in steps:
            glass = np.clip(frost + (tone - frost) * opac, 0, 255)
            # 5.0 rather than 4.5: the grain, added afterwards, costs ~0.2 (measured 4.3-4.4)
            if worst_contrast(ink, glass[under], dark_ink=not smoke) >= 5.0:
                break

        # --- crumbling
        din = cv2.distanceTransform((f0.fill > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
        dout = cv2.distanceTransform((f0.fill <= 0.5).astype(np.uint8), cv2.DIST_L2, 5)
        # The crumbling moves freely towards the centre; it's a GUARD around the
        # letters that stays glass. Capping it at the distance from the text
        # corner nearest the edge stopped it at 2-13 px: the text almost fills
        # the bubble, and nothing crumbled.
        reach = 0.55 * min(pl.ft.rx, pl.ft.ry)
        depth = 0.6 * c + s ** 1.2 * reach
        core = np.clip((din - depth) / 1.5, 0, 1)        # intact glass, smooth edge
        kg = max(3, int(0.5 * pl.ft.fs) | 1)
        guard = cv2.dilate((ta > 0.05).astype(np.uint8),
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kg, kg))).astype(np.float32)

        gh, gw = max(1, int(np.ceil(hl / c))), max(1, int(np.ceil(wl / c)))
        rng = np.random.default_rng(pl.seed + 29)       # one fixed draw per shard
        r_cell = rng.random((gh, gw)).astype(np.float32)
        tint = 1 + 0.06 * rng.standard_normal((gh, gw)).astype(np.float32)

        def cells(a):
            pad = np.zeros((gh * c, gw * c), np.float32)
            pad[:hl, :wl] = a
            return pad.reshape(gh, c, gw, c).mean((1, 3))

        din_c, dout_c = cells(din), cells(dout)
        p_in = np.where(din_c > 0, np.clip(din_c / max(depth, 1e-3), 0, 1) ** 1.5, 0)
        p_in = np.maximum(p_in, (cells(guard) > 0.3).astype(np.float32))   # guard
        spread = s ** 1.2 * 0.5 * min(pl.ft.rx, pl.ft.ry)
        p_out = np.where((din_c <= 0) & (spread > 1),
                         0.35 * s * np.clip(1 - dout_c / max(spread, 1e-3), 0, 1), 0)
        kept = (r_cell < np.maximum(p_in, p_out)).astype(np.float32)

        def up(a):
            return np.kron(a, np.ones((c, c), np.float32))[:hl, :wl]

        shards = cv2.GaussianBlur(up(kept), (0, 0), 0.3 * c)
        shards = np.clip((shards - 0.25) * 2.2, 0, 1)           # crisp shards
        shards *= 1 - core
        alpha = np.maximum(np.maximum(core, shards), np.clip(f0.halo * 1.5, 0, 1))
        alpha = np.maximum(alpha, ta)
        if probe is not None:                   # for measurements
            probe.append(dict(alpha=alpha, halo=f0.halo, depth=depth, reach=reach))

        # each shard keeps the glass's colour, barely varied
        fill = glass * np.where(core[..., None] > 0.5, 1.0, up(tint)[..., None])

        # rim on the intact edge, on the light side
        cl = pl.ft.center - np.array([x0, y0], np.float32)
        yy, xx = np.mgrid[0:hl, 0:wl].astype(np.float32)
        nx, ny = (xx - cl[0]) / pl.ft.rx, (yy - cl[1]) / pl.ft.ry
        lit = np.clip((nx * scene.light[0] + ny * scene.light[1]) /
                      np.maximum(np.hypot(nx, ny), 1e-3), -1, 1)
        band = np.clip(1 - np.abs(din - depth - 2.0) / 2.5, 0, 1) * core
        fill = fill + band[..., None] * np.where(lit[..., None] > 0,
                                                 lit[..., None] * 0.5 * (255 - fill),
                                                 lit[..., None] * 0.25 * fill)

        sig = grain_sigma(src, ring, scene.grad2[sl])
        if cfg.grain > 0:
            fill = fill + cfg.grain * (rng.standard_normal((hl, wl, 1)).astype(np.float32)
                                       * sig.mean() * 0.85 +
                                       rng.standard_normal((hl, wl, 3)).astype(np.float32)
                                       * sig * 0.5)

        # light cast shadow, shards included
        off = 0.008 * diag * cfg.shadow
        M = np.float32([[1, 0, -scene.light[0] * off], [0, 1, -scene.light[1] * off]])
        sh = cv2.GaussianBlur(cv2.warpAffine(alpha, M, (wl, hl)), (0, 0), 0.006 * diag)
        sh *= 0.12 * cfg.shadow
        sh[sh < 0.01] = 0

        loc = out[sl]
        loc *= 1 - (sh * (1 - alpha))[..., None]
        loc[:] = loc * (1 - alpha[..., None]) + np.clip(fill, 0, 255) * alpha[..., None]
        # attribution line, thin, in the text's ink
        a = f0.att * 0.85
        loc[:] = loc * (1 - a[..., None]) + ink * a[..., None]
        loc[:] = loc * (1 - ta[..., None]) + lay[..., :3] * ta[..., None]
    return np.clip(out, 0, 255)


# input / output

def detect_for(img: np.ndarray, ref_path: Path | None) -> Detection:
    """Detection on the image itself, or on the original it comes from.

    On a dot-screen render the pose can't be detected any more (that's the
    point), so detect on the original photo and scale the points to the
    render. The caption margin is at the bottom, so the width is enough.
    """
    if ref_path is None:
        return detect(img)
    ref = load_image(ref_path)
    det = detect(ref)
    h, w = img.shape[:2]
    k = w / ref.shape[1]
    body = None
    if det.body is not None:
        b = cv2.resize(det.body.astype(np.float32), (w, int(round(ref.shape[0] * k))))
        body = np.zeros((h, w), np.float32)
        n = min(h, b.shape[0])
        body[:n] = b[:n]
    bbox = tuple(int(v * k) for v in det.bbox) if det.bbox else None
    out = Detection(None if det.pose is None else det.pose * k, det.vis,
                    [f * k for f in det.faces], body, det.size * k, bbox)
    out.source = det.source
    return out


class Say(argparse.Action):
    """--say / --think / --shout, in the order they are given."""
    def __call__(self, parser, ns, values, option_string=None):
        items = list(getattr(ns, self.dest) or [])
        items.append((self.const, values))
        setattr(ns, self.dest, items)


def write_map(path: Path, scene: Scene, reports: list) -> None:
    """Map of the placement cost. Faces are a flat patch on it, otherwise the
    busyness contours would trace the profile of the face."""
    m = np.clip(scene.busy / 1.5 + 0.6 * scene.subject, 0, 1)
    m = (m * 200).astype(np.uint8)
    m[scene.face] = 255
    m = cv2.cvtColor(m, cv2.COLOR_GRAY2RGB)
    for r in reports:
        x0, y0, x1, y1 = r["box"]
        cv2.rectangle(m, (x0, y0), (x1, y1), (230, 60, 40), 2)
        if r["tip"] is not None:
            cv2.line(m, tuple(int(v) for v in r["center"]), tuple(int(v) for v in r["tip"]),
                     (230, 60, 40), 2)
    cv2.imwrite(str(path), cv2.cvtColor(m, cv2.COLOR_RGB2BGR))


def machine_main(img: np.ndarray, det: Detection, scene: Scene, cfg) -> None:
    h, w = img.shape[:2]
    raws = [raw for kind, raw in cfg.bubbles]
    if any(kind != "speech" for kind, _ in cfg.bubbles):
        print("  --machine: --think / --shout are drawn as machine bubbles",
              file=sys.stderr)
    style = cfg.style or "crumble"
    if style not in ("crumble", "stipple", "glass", "paper"):
        print(f"  --machine: style '{style}' not supported, using paper", file=sys.stderr)
        style = "paper"
    mcfg = argparse.Namespace(grain=cfg.dots, seed=cfg.seed, say_size=cfg.size,
                              say_drift=cfg.say_drift, speaker=cfg.speaker)
    plans = plan_machine(img, det, np.zeros((h, w), np.float32), raws, mcfg)
    if not plans:
        sys.exit("no bubble found room")
    unit = max(1.0, min(h, w) * cfg.dots / 1200.0)
    if cfg.variants:
        n = 8 if cfg.variants < 0 else cfg.variants
        folder = cfg.photo.with_name(cfg.photo.stem + "_machine")
        folder.mkdir(exist_ok=True)
        runs = [(float(st), folder / f"stage_{st:.2f}.png") for st in np.linspace(0.05, 0.95, n)]
    else:
        runs = [(cfg.stage, cfg.out or cfg.photo.with_name(cfg.photo.stem + "_machine.png"))]
    for st, path in runs:
        out = (crumble_on_photo(img, plans, st, scene, unit, cfg) if style == "crumble"
               else machine_on_photo(img, plans, st, scene, unit, style, cfg))
        save_image(out.astype(np.uint8), path, cfg.long_side)
        who = "  |  ".join(speaker(pl, st)[0] for pl in plans) if cfg.speaker else ""
        print(f"  {path.name}   {who}")
    print("\nthe photo is not anonymized: the face is still visible")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Sets speech bubbles into a real photo: placed in calm areas, "
                    "matched to the photo's light, grain, sharpness and shadows.")
    p.add_argument("photo", type=Path)
    p.add_argument("--say", action=Say, const="speech", dest="bubbles", metavar="[ANCHOR:]TEXT",
                   help="speech bubble. ANCHOR = name (mouth, hand-l...), area, "
                        "normalized 'x,y', or 'none' (no tail). Default: mouth. "
                        "'\\n' forces a line break. Repeatable.")
    p.add_argument("--think", action=Say, const="thought", dest="bubbles",
                   metavar="[ANCHOR:]TEXT", help="thought bubble (cloud, trail of dots)")
    p.add_argument("--shout", action=Say, const="shout", dest="bubbles",
                   metavar="[ANCHOR:]TEXT", help="shout bubble (spikes)")
    p.add_argument("--style", choices=tuple(STYLES) + ("crumble", "stipple"), default=None,
                   help="glass (default, frosted), paper, manga, chalk; with --machine: "
                        "crumble (default: glass whose edge breaks into shards), "
                        "stipple, glass, paper")
    p.add_argument("--density", type=float, default=0.45,
                   help="--machine --style stipple: maximum ink of the scene redrawn "
                        "in the bubble, 0..1 (default 0.45; high = scene more present, "
                        "text stands out less)")
    p.add_argument("--opacity", type=float, default=None, metavar="0..1",
                   help="glass: from nearly frost (0.35) to nearly paper (0.95). "
                        "Default: the lowest that keeps the text readable everywhere")
    p.add_argument("--glass", choices=("auto", "clear", "smoked"), default="auto",
                   help="clear glass, or smoked (dark, light text). auto: smoked "
                        "in a dark area")
    p.add_argument("--shape", choices=("oval", "round"), default="oval",
                   help="shape of the speech bubbles")
    p.add_argument("--size", type=float, default=0.026,
                   help="text size as a fraction of the width (default 0.026)")
    p.add_argument("--ambient", type=float, default=0.4,
                   help="how much the bubble takes on the local light's tint, 0..1")
    p.add_argument("--light", type=float, default=0.6,
                   help="how much the bubble takes on the photo's white level, 0..1")
    p.add_argument("--shading", type=float, default=0.08,
                   help="lighting gradient across the bubble, along the light direction")
    p.add_argument("--shadow", type=float, default=1.0, help="cast shadow strength, 0 = none")
    p.add_argument("--grain", type=float, default=1.0,
                   help="multiple of the grain measured on the photo, 0 = smooth bubble")
    p.add_argument("--blur", type=float, default=1.0,
                   help="follows the sharpness of the plane, 0 = always sharp")
    p.add_argument("--behind", action="store_true",
                   help="the bubble goes BEHIND the subject (the outline covers it)")
    p.add_argument("--tilt", type=float, default=3.0,
                   help="max tilt in degrees, drawn at random (default 3)")
    p.add_argument("--wobble", type=float, default=0.025,
                   help="irregularity of the outline (0 = perfect ellipse)")
    p.add_argument("--gap", type=float, default=0.012,
                   help="space between the tail tip and the anchor, fraction of the diagonal")
    p.add_argument("--detect-on", dest="detect_on", type=Path, default=None,
                   help="detect the pose on this photo (the original of a dot-screen render)")
    p.add_argument("--variants", nargs="?", type=int, const=-1, default=0, metavar="N",
                   help="a FOLDER of N renders: styles in turn (default 4); with "
                        "--machine, one render per stage from 0.05 to 0.95 (default 8)")
    p.add_argument("--machine", action="store_true",
                   help="the MACHINE's bubbles on the untouched photo: your text sharp, "
                        "the bubble falls apart with --stage, its attribution line "
                        "drifts from the mouth, a line says who speaks according to "
                        "the machine")
    p.add_argument("--stage", type=float, default=0.05,
                   help="--machine: state of the bubble, 0 = sharp, 1 = formless (default 0.05)")
    p.add_argument("--dots", type=float, default=1.8,
                   help="--machine: size of the bubble's ink dots (like --grain of the "
                        "dot screen, default 1.8)")
    p.add_argument("--say-drift", dest="say_drift", type=float, default=1.0,
                   help="--machine: drift of the attribution line at stage 1 (0 = none)")
    p.add_argument("--no-speaker", dest="speaker", action="store_false",
                   help="--machine: remove the 'speaker: ...' line")
    p.add_argument("--map", action="store_true",
                   help="also write the placement map (faces as a flat patch)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=Path, help="default: <photo>_bubbles.png")
    p.add_argument("--long-side", dest="long_side", type=int, default=None,
                   help="shrink the long side on output")
    cfg = p.parse_args()

    if not cfg.bubbles:
        p.error("nothing to say: add at least one --say, --think or --shout")

    img = load_image(cfg.photo)
    h, w = img.shape[:2]
    det = detect_for(img, cfg.detect_on)
    print(f"{cfg.photo.name}: {w}x{h}  pose {'yes' if det.pose is not None else 'NO'}"
          f"  faces {len(det.faces)}  outline {'yes' if det.body is not None else 'no'}")
    specs = [parse_spec(raw, cfg.shape if kind == "speech" else kind, det, (h, w))
             for kind, raw in cfg.bubbles]
    base = analyse(img, det)
    if base.head_c is not None:
        f = base.facing
        reg = ("straight on" if abs(f) < 0.3 else
               f"towards the image's {'right' if f > 0 else 'left'} ({abs(f):.1f})")
        print(f"head: {base.head_h:.0f} px tall, looking {reg}")
    ang = math.degrees(math.atan2(-base.light[1], base.light[0]))
    print(f"estimated light: comes from {ang:.0f} deg (0 = right, 90 = top)  "
          f"photo white {base.white:.2f}")

    if cfg.machine:
        machine_main(img, det, base, cfg)
        return
    if cfg.style in ("stipple", "crumble"):
        p.error(f"--style {cfg.style} only exists with --machine")

    if cfg.variants:
        if cfg.variants < 0:
            cfg.variants = 4
        folder = cfg.photo.with_name(cfg.photo.stem + "_bubbles")
        folder.mkdir(exist_ok=True)
        styles = [cfg.style] if cfg.style else list(STYLES)
        runs = [(styles[i % len(styles)], cfg.seed + i,
                 folder / f"{i + 1:02d}_{styles[i % len(styles)]}.png")
                for i in range(cfg.variants)]
    else:
        runs = [(cfg.style or "glass", cfg.seed,
                 cfg.out or cfg.photo.with_name(cfg.photo.stem + "_bubbles.png"))]

    for style, seed, path in runs:
        scene = Scene(**{**base.__dict__, "occupied": np.zeros((h, w), bool)})
        rng = np.random.default_rng(seed)
        out = img.astype(np.float32).copy()
        reports = []
        print(f"\n-> {path.name}  ({style})")
        for spec in specs:
            r = draw_bubble(out, img, scene, spec, style, rng, cfg)
            if r is None:
                continue
            reports.append(r)
            where = ""
            if r["gap_h"] is not None:
                where = f", {r['gap_h']:.1f} heads from the anchor"
                if r["side"]:
                    where += f", {r['side']}"
                if r["window"] is None:
                    where += "  /!\\ too cramped near the head: placed further away"
                elif r["window"] > 1:
                    where += " (wider window)"
            small = f" (shrunk from {r['fs0']})" if r["fs"] < r["fs0"] else ""
            print(f"  [{r['label']}] {r['form']}, {r['fs']} px{small}{where}")
            if r["opac"] is not None:
                print(f"     {'smoked' if r['smoke'] else 'clear'} glass, opacity {r['opac']:.2f}")
            print(f"     calmer spot than {100 * r['calm']:.0f} % of the image | "
                  f"contrast {r['contrast']:.1f}:1"
                  f"{' (worst letter)' if r['opac'] is not None else ''} | "
                  f"grain {r['grain'].mean():.1f}")
            print(f"     tint x{r['tint']} lum x{r['lum']:.2f} | "
                  f"{r['blur']}" + (f" | hidden {100 * r['hidden']:.0f} %" if cfg.behind else ""))
            if r["on_face"]:
                print(f"     /!\\ {r['on_face']} px of the bubble over a face", file=sys.stderr)
        save_image(np.clip(out, 0, 255).astype(np.uint8), path, cfg.long_side)
        if cfg.map:
            write_map(path.with_name(path.stem + "_map.png"), scene, reports)
    print("\nmetadata: none (written from the array, like the dot screen)")


if __name__ == "__main__":
    main()
