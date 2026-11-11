"""The body rebuilt as a dot screen, the face erased.

The density of the dots follows the light of the photo, and no original pixel
is kept. In the face area the dots come from the texture around it, so nothing
of the face is left.

The tracking coordinates go in a caption under the image.

Everything runs locally. No metadata in the output, and no image at all if
the head could not be located.

    mouchetis photo.jpg
    mouchetis photo.jpg --variants
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

# MediaPipe imports matplotlib.pyplot for its drawing utilities (and with it
# all of dateutil), which we don't use: plt is only called inside a function,
# never at import time. So the module is stubbed out before the import, which
# saves several seconds on a cold start.
if "matplotlib.pyplot" not in sys.modules:
    import types as _types
    sys.modules.setdefault("matplotlib", _types.ModuleType("matplotlib"))
    sys.modules["matplotlib.pyplot"] = _types.ModuleType("matplotlib.pyplot")

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from . import models

# The whole pipeline works in RGB; BGR only when writing files.

# A humanist face for the caption, not a terminal font.
EMOJI_FONT = "/System/Library/Fonts/Apple Color Emoji.ttc"
EMOJI_STRIKES = (20, 32, 48, 64, 96, 160)   # the only sizes the font contains
EMOJI_RE = re.compile(
    "((?:[\U0001F300-\U0001FAFF☀-➿⬀-⯿]"
    "[️‍⃣]*)+)")

CAPTION_FONTS = ("/System/Library/Fonts/Optima.ttc",
                 "/System/Library/Fonts/Supplemental/Futura.ttc",
                 "/System/Library/Fonts/Palatino.ttc")

PAPER = (236, 232, 224)
INK = (24, 22, 21)

# MediaPipe pose indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 2, 5, 7, 8
L_SHO, R_SHO, L_HIP, R_HIP = 11, 12, 23, 24
HEAD_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


# input / output

def load_image(path: Path, max_side: int = 2000) -> np.ndarray:
    """Loads as RGB, applies the EXIF orientation, then drops the metadata."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        arr = np.asarray(im).copy()          # copy(): no link to the file left
    h, w = arr.shape[:2]
    if max(h, w) > max_side:
        k = max_side / max(h, w)
        arr = cv2.resize(arr, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    return arr


def save_image(arr: np.ndarray, path: Path, long_side: int | None) -> None:
    """Final downscale (kills leftover high frequencies), written without metadata."""
    if long_side:
        h, w = arr.shape[:2]
        if max(h, w) > long_side:
            k = long_side / max(h, w)
            arr = cv2.resize(arr, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


# detection

@dataclass
class Detection:
    pose: np.ndarray | None          # (33, 2) px
    vis: np.ndarray | None           # (33,) visibility
    faces: list                      # every face found, (478, 2) px each
    body: np.ndarray | None          # (H, W) float 0..1
    size: float                      # subject scale in px (shoulders -> hips)
    bbox: tuple[int, int, int, int] | None
    source: str | None = None        # where the face location comes from


def segment(mp_img, shape: tuple[int, int], pose: np.ndarray | None) -> np.ndarray | None:
    """The person's outline, as a uint8 category mask.

    Confidence masks (float32) are avoided on purpose: reading one aborts the
    process in 0.10.35.
    """
    h, w = shape
    opts = vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=str(models.path("selfie_segmenter.tflite"))),
        running_mode=vision.RunningMode.IMAGE,
        output_category_mask=True,
        output_confidence_masks=False,
    )
    with vision.ImageSegmenter.create_from_options(opts) as seg:
        res = seg.segment(mp_img)
    cm = getattr(res, "category_mask", None)
    if cm is None:
        return None
    cat = np.asarray(cm.numpy_view())
    if cat.ndim == 3:
        cat = cat[..., 0]
    if cat.shape != (h, w):
        cat = cv2.resize(cat, (w, h), interpolation=cv2.INTER_NEAREST)
    return pick_polarity((cat > 0).astype(np.float32), pose)


def pick_polarity(m: np.ndarray, pose: np.ndarray | None) -> np.ndarray | None:
    """Picks which of the two classes is the person, and rejects implausible masks.

    The model doesn't guarantee the class order, so the keypoints decide. An
    empty mask must NOT be inverted into 'the whole image is the body': that
    would silently disable --bg black. When in doubt, return None and
    body_mask() falls back to the convex hull of the pose.
    """
    h, w = m.shape
    cands = (m, 1.0 - m)
    if pose is not None and len(pose):
        px = np.clip(pose[:, 0].astype(int), 0, w - 1)
        py = np.clip(pose[:, 1].astype(int), 0, h - 1)
        at_kp = [float(c[py, px].mean()) for c in cands]
        best = int(np.argmax(at_kp))
        if at_kp[best] < 0.5:
            return None                     # doesn't cover the subject: unusable
    else:
        best = int(np.argmin([float(c.mean()) for c in cands]))
    chosen = cands[best]
    cover = float(chosen.mean())
    if not 0.01 < cover < 0.9:              # neither empty nor the whole image
        return None
    return chosen


def _retry_face_on_crop(img: np.ndarray, pose: np.ndarray, vis: np.ndarray,
                        fopts) -> list:
    """Runs the face mesh again on an enlarged crop of the head."""
    h, w = img.shape[:2]
    pts = pose[HEAD_IDX]
    if vis is not None:
        keep = pts[vis[HEAD_IDX] > 0.2]
        pts = keep if len(keep) >= 3 else pts
    c = pts.mean(axis=0)
    rad = max(float(np.linalg.norm(pts - c, axis=1).max()) * 3.2, 0.04 * max(h, w))
    x0, y0 = int(max(0, c[0] - rad)), int(max(0, c[1] - rad))
    x1, y1 = int(min(w, c[0] + rad)), int(min(h, c[1] + rad))
    if x1 - x0 < 24 or y1 - y0 < 24:
        return []
    crop = img[y0:y1, x0:x1]
    k = min(4.0, max(1.0, 640.0 / max(crop.shape[:2])))
    if k > 1.01:
        crop = cv2.resize(crop, (int(crop.shape[1] * k), int(crop.shape[0] * k)),
                          interpolation=cv2.INTER_CUBIC)
    ch, cw = crop.shape[:2]
    with vision.FaceLandmarker.create_from_options(fopts) as det:
        rr = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=np.ascontiguousarray(crop)))
    out = []
    for lm in (rr.face_landmarks or []):
        f = np.array([[l.x * cw / k + x0, l.y * ch / k + y0] for l in lm], np.float32)
        out.append(f)
    return out


def detect(img: np.ndarray, num_faces: int = 3) -> Detection:
    h, w = img.shape[:2]
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img))

    pose = vis = body = bbox = None
    opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(models.path("pose_landmarker_heavy.task"))),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        output_segmentation_masks=False,
    )
    with vision.PoseLandmarker.create_from_options(opts) as det:
        r = det.detect(mp_img)
    if r.pose_landmarks:
        lms = r.pose_landmarks[0]
        pose = np.array([[l.x * w, l.y * h] for l in lms], np.float32)
        vis = np.array([getattr(l, "visibility", 1.0) or 0.0 for l in lms], np.float32)

    # Several faces: a photo taken in a mirror shows the subject a second time,
    # and that reflection must be erased like the original.
    fopts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(models.path("face_landmarker.task"))),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=max(1, num_faces),
        min_face_detection_confidence=0.25,
        min_face_presence_confidence=0.25,
    )
    with vision.FaceLandmarker.create_from_options(fopts) as det:
        r = det.detect(mp_img)
    faces = [np.array([[l.x * w, l.y * h] for l in lm], np.float32)
             for lm in (r.face_landmarks or [])]

    # A face that is small in the frame (full-length shot, subject in a
    # mirror) often goes undetected. Try again on an enlarged crop around the
    # pose's head landmarks.
    src_label = "face mesh" if faces else None
    if not faces and pose is not None:
        faces = _retry_face_on_crop(img, pose, vis, fopts)
        if faces:
            src_label = "face mesh on an enlarged crop"
    if not faces:
        src_label = "pose only (no face detected)"

    body = segment(mp_img, (h, w), pose)

    # subject scale
    size = 0.25 * min(h, w)
    if pose is not None:
        sh = 0.5 * (pose[L_SHO] + pose[R_SHO])
        hp = 0.5 * (pose[L_HIP] + pose[R_HIP])
        d = float(np.linalg.norm(sh - hp))
        if d > 10:
            size = d
    if body is not None:
        ys, xs = np.where(body > 0.5)
        if xs.size:
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    if bbox is None and pose is not None:
        bbox = (int(pose[:, 0].min()), int(pose[:, 1].min()),
                int(pose[:, 0].max()), int(pose[:, 1].max()))
    d = Detection(pose, vis, faces, body, size, bbox)
    d.source = src_label
    return d


# masks

def structureless(shape: tuple[int, int], level: float, scale: float,
                  rng: np.random.Generator, spread: float = 0.16) -> np.ndarray:
    """Smooth density field drawn at random, with NO link to the photo.

    It has the right texture and the right average level, but carries no
    information about the subject.
    """
    h, w = shape
    sc = max(2.0, scale)
    n = rng.random((max(2, int(h / sc)), max(2, int(w / sc)))).astype(np.float32)
    n = cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
    n = cv2.GaussianBlur(n, (0, 0), sc * 0.6)
    sd = float(n.std()) or 1.0
    return np.clip((n - float(n.mean())) / sd * spread + level, 0.0, 1.0)


def levels(x: np.ndarray, mask: np.ndarray, lo_p: float = 2.0,
           hi_p: float = 98.0) -> np.ndarray:
    """Stretches the tonal range over the SUBJECT, not the whole image.

    Otherwise pale skin and a mid-grey background ask for the same amount of
    ink: the body loses all its shading and the background stands out more
    than the body.
    """
    v = x[mask > 0.5]
    if v.size < 64:
        return x
    lo, hi = np.percentile(v, [lo_p, hi_p])
    if hi - lo < 1e-3:
        return x
    return np.clip((x - lo) / (hi - lo), 0, 1)


def _soften(mask: np.ndarray, px: float) -> np.ndarray:
    """Softens the edge WITHOUT ever eating into the hard area.

    A plain feather erodes the edge of the core, which is then no longer
    fully destroyed (22 pixels sensitive to the original in the test).
    """
    return np.maximum(mask, _feather(mask, px))


def _feather(mask: np.ndarray, px: int) -> np.ndarray:
    k = max(3, int(px) | 1)
    return np.clip(cv2.GaussianBlur(mask, (k, k), 0), 0, 1)


# face mesh: top of the forehead, chin, and the two side extremes
F_TOP, F_CHIN, F_LEFT, F_RIGHT = 10, 152, 234, 454


def _head_zone(f: np.ndarray, det: Detection, shape: tuple[int, int],
               grow: float, seal: bool = False, mode: str = "face") -> np.ndarray:
    """Identity area of ONE face, cut out of the person's OUTLINE.

    The shape of the head, hair included, is already in the segmentation
    mask, so it is used as is, cut just below the chin, rather than
    approximated by a disc or an ellipse. Only the connected component of the
    face is kept, so a raised arm passing above the chin isn't swallowed.

    The face itself stays covered in any case, usable outline or not.
    """
    h, w = shape
    hull = cv2.convexHull(f.astype(np.int32))
    covered = np.zeros((h, w), np.float32)
    cv2.fillConvexPoly(covered, hull, 1.0)
    r = float(np.linalg.norm(f - f.mean(axis=0), axis=1).max())
    # The face mesh already spans from the hairline to the chin: a large
    # margin only bites into the hair.
    pad = max(3, int(r * (0.07 if mode == "face" else 0.14) * grow)) | 1
    covered = cv2.dilate(covered, np.ones((pad, pad), np.uint8))

    if mode == "face":
        # tight around the measured face, without extending to the skull: the
        # hull of the face mesh, widened just enough
        return covered
    if len(f) <= max(F_TOP, F_CHIN):
        return covered

    top, chin = f[F_TOP], f[F_CHIN]
    up = top - chin
    n = float(np.linalg.norm(up))
    if n <= 1.0:
        return covered
    up /= n
    half = n / 2.0
    side = np.array([-up[1], up[0]], np.float32)
    across = float(np.abs((f - chin) @ side).max())

    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    cut = chin - up * (0.14 * half * grow)
    above = ((gx - cut[0]) * up[0] + (gy - cut[1]) * up[1]) > 0
    lat = np.abs((gx - chin[0]) * side[0] + (gy - chin[1]) * side[1])
    band = above & (lat < across * 2.1 * grow)

    if det.body is None:
        # No geometric fallback: a stand-in ellipse or disc is exactly the
        # shape we don't want. Stick to the face.
        return covered
    zone = (band & (det.body > 0.5)).astype(np.uint8)

    # keep only the head: the connected component that contains the face
    cnt, lab = cv2.connectedComponents(zone, connectivity=8)
    c = f.mean(axis=0).astype(int)
    cy, cx = int(np.clip(c[1], 0, h - 1)), int(np.clip(c[0], 0, w - 1))
    l = int(lab[cy, cx])
    if l == 0 and cnt > 1:                     # centroid outside: largest part
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        l = int(sizes.argmax())
    if l > 0:
        zone = (lab == l).astype(np.float32)
    else:
        zone = zone.astype(np.float32)

    if seal:
        zone = _seal(zone, 2.0 * half)
    return np.clip(zone + covered, 0, 1)


def _seal(zone: np.ndarray, d: float, k_rel: float = 0.20) -> np.ndarray:
    """Closes the identity area over objects that hide the face.

    A phone held in front of the face isn't part of the segmented outline: the
    cut leaves a notch or a hole there, and the object (with the hand holding
    it, its case, a reflection in the screen) would stay intact in the middle
    of the erased area. So notches are closed (morphological closing), then
    enclosed holes are filled (flood fill from the outside).
    """
    z = (zone > 0.5).astype(np.uint8)
    k = max(3, int(k_rel * d)) | 1
    z = cv2.morphologyEx(z, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    pad = cv2.copyMakeBorder(z, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff = pad.copy()
    cv2.floodFill(ff, np.zeros((pad.shape[0] + 2, pad.shape[1] + 2), np.uint8),
                  (0, 0), 1)
    holes = (1 - ff)[1:-1, 1:-1]
    return ((z | holes) > 0).astype(np.float32)


def _cut_from_silhouette(body: np.ndarray, up: np.ndarray, cut: np.ndarray,
                         ref: np.ndarray, across: float, anchor: np.ndarray,
                         lat_k: float, shape: tuple[int, int]) -> np.ndarray:
    """Cuts the head out of the outline: half-plane plus a side band, then the
    connected component of the head."""
    h, w = shape
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    side = np.array([-up[1], up[0]], np.float32)
    above = ((gx - cut[0]) * up[0] + (gy - cut[1]) * up[1]) > 0
    lat = np.abs((gx - ref[0]) * side[0] + (gy - ref[1]) * side[1])
    zone = (above & (lat < across * lat_k) & (body > 0.5)).astype(np.uint8)

    cnt, lab = cv2.connectedComponents(zone, connectivity=8)
    ay = int(np.clip(anchor[1], 0, h - 1))
    ax = int(np.clip(anchor[0], 0, w - 1))
    l = int(lab[ay, ax])
    if l == 0 and cnt > 1:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        l = int(sizes.argmax())
    return (lab == l).astype(np.float32) if l > 0 else zone.astype(np.float32)


def _face_zone_from_pose(det: Detection, shape: tuple[int, int],
                         grow: float) -> np.ndarray | None:
    """An area tight around the FACE, not the whole head.

    Destroying the whole head takes everything at face height with it, phone
    and hand included. Here the area is built on the estimated head
    landmarks, then clipped by the outline so its edges stay organic where it
    meets the contour of the head.

    An object in front of the face still gets partly caught (that's
    geometry), but it sticks out all around instead of being swallowed.
    """
    h, w = shape
    if det.pose is None:
        return None
    pts = det.pose[HEAD_IDX]
    if det.vis is not None:
        keep = pts[det.vis[HEAD_IDX] > 0.2]
        pts = keep if len(keep) >= 3 else pts
    if len(pts) < 3:
        return None

    nose = det.pose[NOSE]
    sho = 0.5 * (det.pose[L_SHO] + det.pose[R_SHO])
    up = nose - sho
    n = float(np.linalg.norm(up))
    if n < 8.0:
        return None
    up /= n
    side = np.array([-up[1], up[0]], np.float32)

    c = pts.mean(axis=0)
    along = float(np.abs((pts - c) @ up).max())
    across = float(np.abs((pts - c) @ side).max())
    # margin to cover the forehead and chin, out of reach of the pose landmarks
    axis_up = max(along * 2.0, 0.13 * n) * grow
    axis_side = max(across * 1.12, 0.10 * n) * grow
    ang = float(np.degrees(np.arctan2(up[1], up[0])))

    def ell():
        z = np.zeros((h, w), np.uint8)
        cv2.ellipse(z, (int(c[0]), int(c[1])), (int(axis_up), int(axis_side)),
                    ang, 0, 360, 1, -1)
        return z

    zone = ell()
    if det.body is not None:
        clipped = zone & (det.body > 0.5).astype(np.uint8)
        # if the outline cuts off too much (an unsegmented object in front of
        # the face), keep the full area rather than leave the face uncovered
        if clipped.sum() > 0.45 * zone.sum():
            zone = clipped
    return zone.astype(np.float32)


def _head_zone_from_pose(det: Detection, shape: tuple[int, int],
                         grow: float, seal: bool = False) -> np.ndarray | None:
    """Identity area without a face mesh, cut out of the outline.

    Used when the face mesh fails (full-length shot, subject in a mirror). The
    head is cut out of the outline, along the pose's shoulders-to-nose axis.
    """
    if det.pose is None or det.body is None:
        return None
    pose, vis = det.pose, det.vis
    nose = pose[NOSE]
    sho = 0.5 * (pose[L_SHO] + pose[R_SHO])
    up = nose - sho
    d = float(np.linalg.norm(up))
    if d < 8.0:
        return None
    up /= d
    side = np.array([-up[1], up[0]], np.float32)

    pts = pose[HEAD_IDX]
    if vis is not None:
        keep = pts[vis[HEAD_IDX] > 0.2]
        pts = keep if len(keep) >= 3 else pts
    across = float(np.abs((pts - nose) @ side).max())
    across = max(across, 0.16 * d)

    # cut between the shoulders and the nose; grow lowers it to enlarge the area
    cut = sho + up * (0.45 / max(0.3, grow)) * d
    # pose landmarks don't cover the hair, hence a wide band
    zone = _cut_from_silhouette(det.body, up, cut, nose, across, nose,
                                2.9 * grow, shape)
    zone = _seal(zone, d) if seal else zone
    # Feathering pushes the outer edge below the full-destruction threshold,
    # and without a face mesh no safety mask covers it: dilate so the area
    # reaches the outline of the head.
    k = max(3, int(0.028 * d)) | 1
    return cv2.dilate(zone, np.ones((k, k), np.uint8))


def _with_fade(m: np.ndarray, scale: float, fade: float) -> np.ndarray:
    """Core kept at exactly 1, then a fade outwards.

    A simply feathered mask goes from full destruction to nothing over a
    narrow band, and the edge shows. Here the core stays at exactly 1 (the
    anonymity guarantee is about the core) and only the outside gets a ramp.
    """
    if fade <= 0:
        return m
    sig = max(2.0, fade * max(scale, 8.0))
    return np.maximum(m, cv2.GaussianBlur(m, (0, 0), sig))


def face_pose_gap(det: Detection) -> list[float]:
    """Distance between each detected face and the head located by the pose.

    A face found far from the estimated head is suspicious: a reflection in a
    phone screen, a pattern taken for a face, a second person. It gets
    reported rather than dropped, since a reflection in a mirror is a real
    face that must be erased.
    """
    if det.pose is None or not det.faces:
        return []
    head = det.pose[HEAD_IDX].mean(axis=0)
    scale = max(1.0, float(np.linalg.norm(det.pose[NOSE]
                - 0.5 * (det.pose[L_SHO] + det.pose[R_SHO]))))
    return [float(np.linalg.norm(f.mean(axis=0) - head)) / scale for f in det.faces]


def head_mask(det: Detection, shape: tuple[int, int], grow: float = 1.0,
              seal: bool = False, mode: str = "face", fade: float = 0.0):
    """Returns (union, areas). The per-face areas let each patch match its own
    tone: a reflection in a mirror never has the subject's lighting.

    Returns (None, []) if no head can be located.
    """
    h, w = shape
    m = np.zeros((h, w), np.float32)
    zones = []

    for f in det.faces:
        if len(f) > 10:
            z = _head_zone(f, det, shape, grow, seal, mode)
            fh = float(np.linalg.norm(f[F_TOP] - f[F_CHIN])) if len(f) > F_CHIN else det.size
            soft = max(3.0, 0.07 * max(fh, 8.0))
            zones.append(_with_fade(_soften(z, soft), fh, fade))
            m = np.clip(m + z, 0, 1)

    if m.max() < 0.5 and mode == "face":
        z = _face_zone_from_pose(det, shape, grow)
        if z is not None and z.max() > 0.5:
            soft = max(2.5, 0.022 * float(np.linalg.norm(
                det.pose[NOSE] - 0.5 * (det.pose[L_SHO] + det.pose[R_SHO]))))
            sc = float(np.linalg.norm(det.pose[NOSE]
                                      - 0.5 * (det.pose[L_SHO] + det.pose[R_SHO]))) * 0.3
            zones.append(_with_fade(_soften(z, soft), sc, fade))
            return _with_fade(_soften(np.clip(m + z, 0, 1), soft), sc, fade), zones

    if m.max() < 0.5:
        z = _head_zone_from_pose(det, shape, grow, seal)
        if z is not None and z.max() > 0.5:
            soft = max(2.5, 0.022 * float(np.linalg.norm(
                det.pose[NOSE] - 0.5 * (det.pose[L_SHO] + det.pose[R_SHO]))))
            zones.append(_feather(z, soft))
            m = np.clip(m + z, 0, 1)
            return _feather(m, soft), zones

    if m.max() < 0.5:
        return None, []
    fh = np.mean([np.linalg.norm(f[F_TOP] - f[F_CHIN])
                  for f in det.faces if len(f) > F_CHIN]) if det.faces else det.size
    m = _with_fade(_soften(m, max(3.0, 0.07 * max(float(fh), 8.0))), float(fh), fade)
    return m, (zones or [m])


def body_mask(det: Detection, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    if det.body is not None and det.body.shape == (h, w):
        return _feather(det.body.astype(np.float32), 0.04 * det.size)
    m = np.zeros((h, w), np.float32)
    if det.pose is not None:
        hull = cv2.convexHull(det.pose.astype(np.int32))
        cv2.fillConvexPoly(m, hull, 1.0)
        m = cv2.dilate(m, np.ones((max(5, int(0.12 * det.size)),) * 2, np.uint8))
    return _feather(m, 0.06 * det.size)


def region_mask(specs: list[str], shape: tuple[int, int], what: str) -> np.ndarray:
    """Areas given by hand: 'x,y,r' (circle) or 'x0,y0,x1,y1' (rectangle), in
    normalized 0..1 coordinates."""
    h, w = shape
    m = np.zeros((h, w), np.float32)
    for spec in specs:
        try:
            v = [float(x) for x in spec.split(",")]
        except ValueError:
            v = []
        if len(v) == 3:
            cv2.circle(m, (int(v[0] * w), int(v[1] * h)), max(2, int(v[2] * w)), 1.0, -1)
        elif len(v) == 4:
            cv2.rectangle(m, (int(v[0] * w), int(v[1] * h)),
                          (int(v[2] * w), int(v[3] * h)), 1.0, -1)
        else:
            sys.exit(f"invalid {what}: {spec!r}. Expected 'x,y,r' (circle) or "
                     f"'x0,y0,x1,y1' (rectangle), normalized between 0 and 1")
    return _feather(m, 0.012 * min(h, w)) if specs else m


# background

def kill_background(img: np.ndarray, keep: np.ndarray, strength: float,
                    ident: np.ndarray | None = None) -> np.ndarray:
    """Destroys the background's information (room, mirror, window) without
    blacking it out.

    Heavy downscale THEN blur: a plain Gaussian blur leaves the overall
    structure recoverable, a downscale to 5 % removes it for good.

    The identity area is removed BEFORE the downscale. Otherwise the
    background holds a smeared trace of the face, and that trace becomes a
    path for face pixels to affect the render where the soft edge of the body
    mixes in a little background.
    """
    h, w = img.shape[:2]
    src = img
    if ident is not None:
        m = ident > 0.5
        if m.any() and (~m).any():
            src = np.where(m[..., None], np.median(img[~m], axis=0), img)
    sw, sh = max(1, int(w * 0.05)), max(1, int(h * 0.05))
    back = cv2.resize(cv2.resize(src, (sw, sh), interpolation=cv2.INTER_AREA),
                      (w, h), interpolation=cv2.INTER_CUBIC)
    k = max(3, int(min(h, w) * 0.025 * strength) | 1)
    back = cv2.GaussianBlur(back, (k, k), 0)
    a = keep[..., None]
    return img * a + back * (1 - a)


# dot screen

def luminance(img: np.ndarray) -> np.ndarray:
    return (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]) / 255.0


def bleed_field(ident: np.ndarray, body: np.ndarray, size: float,
                amount: float, rng: np.random.Generator) -> np.ndarray:
    """Lets the face destruction spread into the body.

    A closed identity area reads as a shape placed on the head (a round
    blob), and no work on its edge changes that, since the outline of the
    face itself is oval. Letting the destruction run down in irregular
    tongues over the neck and shoulders leaves no closed outline to read.
    """
    if amount <= 0:
        return np.zeros_like(ident)
    far = (ident <= 0.5).astype(np.uint8)
    d = cv2.distanceTransform(far, cv2.DIST_L2, 5)
    span = max(4.0, size * amount)
    g = np.clip(1.0 - d / span, 0, 1) ** 1.6
    n = structureless(ident.shape, 0.5, span * 0.32, rng)
    n01 = np.clip((n - 0.5) / 0.32 + 0.5, 0, 1)
    return np.clip(g * (0.25 + 1.5 * n01), 0, 1) * body


def continued_texture(ink: np.ndarray, zone: np.ndarray,
                      size: float) -> np.ndarray | None:
    """Fills the identity area by EXTENDING the texture around it.

    A texture taken from elsewhere on the body has its own patterns, which
    match nothing at the edge of the area: the patch reads as sewn on, even
    with a soft edge. Inpainting instead carries the neighbouring values
    inwards, so the texture continues.

    Useful side effect: the content comes ONLY from pixels outside the area.
    Nothing of the face gets in, by construction of the algorithm.
    """
    m = (zone > 0.5).astype(np.uint8)
    if not m.any() or m.all():
        return None
    src = np.clip(ink * 255.0, 0, 255).astype(np.uint8)
    r = int(np.clip(size * 0.06, 3, 24))
    out = cv2.inpaint(src, m, r, cv2.INPAINT_TELEA)
    return out.astype(np.float32) / 255.0


def borrowed_texture(ink: np.ndarray, zone: np.ndarray, body: np.ndarray
                     ) -> np.ndarray | None:
    """Fills the identity area with BODY TEXTURE taken from elsewhere.

    A structureless field gives a flat patch, and a flat patch reads as a
    solid blob whatever the shape of its outline. Borrowing the density of
    another part of the body gives the area a real texture. It still holds no
    information about the face: these are body pixels, taken outside the area.
    """
    h, w = zone.shape
    ys, xs = np.nonzero(zone > 0.5)
    if not len(ys):
        return None
    span = int(ys.max() - ys.min()) + 1
    best = None
    for dy in (int(span * 1.35), int(span * 1.9), int(span * 2.5), int(span * 3.1)):
        for dx in (0, int(span * 0.5), -int(span * 0.5)):
            sy, sx = ys + dy, xs + dx
            if sy.min() < 0 or sy.max() >= h or sx.min() < 0 or sx.max() >= w:
                continue
            if (zone[sy, sx] > 0.15).any():          # never borrow from itself
                continue
            cov = float((body[sy, sx] > 0.5).mean())
            if best is None or cov > best[0]:
                best = (cov, dy, dx)
    if best is None or best[0] < 0.55:               # not enough body to borrow
        return None
    _, dy, dx = best
    out = np.zeros_like(ink)
    out[ys, xs] = ink[ys + dy, xs + dx]
    return out


L_EYE_IN, L_EYE_OUT, R_EYE_IN, R_EYE_OUT = 1, 3, 4, 6
MOUTH_L, MOUTH_R = 9, 10


def face_lines(pose: np.ndarray, vis: np.ndarray | None, zone: np.ndarray,
               weight: float) -> np.ndarray:
    """The lines the MACHINE thinks it sees where the face is.

    Estimated positions of the eyes, the bridge of the nose and the mouth,
    taken from the POSE landmarks, which exist even when an object hides the
    face and the face mesh is out. These are not the subject's features but
    the model's guess: each stroke is drawn fainter the less sure the model is
    of its point.

    Accepted trade-off: this brings back a rough face geometry (eye spacing,
    eyes-to-mouth distance). --lines 0 removes it.
    """
    h, w = zone.shape
    f = np.zeros((h, w), np.float32)
    if pose is None or weight <= 0 or len(pose) <= MOUTH_R:
        return f

    le, re = pose[L_EYE], pose[R_EYE]
    eye_d = float(np.linalg.norm(re - le))
    if eye_d < 4.0:
        return f
    # Sized to the HEAD, not to the eye spacing alone: strokes based on the
    # eye spacing covered only 5 % of the area and got lost in the dots.
    # Thin strokes: sized for small marks, they made a big cross once they
    # became long axes.
    lw = max(2, int(round(eye_d * 0.09)))

    def conf(*idx):
        if vis is None:
            return 1.0
        return float(np.clip(np.mean([vis[i] for i in idx]), 0.0, 1.0))

    def stroke(a, b, c):
        cv2.line(f, tuple(np.asarray(a, np.int32)), tuple(np.asarray(b, np.int32)),
                 float(c), lw, cv2.LINE_AA)

    axis = (re - le) / eye_d
    perp = np.array([-axis[1], axis[0]], np.float32)

    # CIRCLES at the eyes and a mouth stroke make a smiley: the drawing takes
    # over the point. So only construction marks are drawn (axes and ticks),
    # which read as a measurement.
    span = eye_d * 1.55
    # eye axis, across the head
    mid = 0.5 * (le + re)
    stroke(mid - axis * span, mid + axis * span, conf(L_EYE, R_EYE))
    # centre line, from the forehead to the estimated chin
    nose = pose[NOSE]
    mouth = 0.5 * (pose[MOUTH_L] + pose[MOUTH_R])
    stroke(mid - perp * eye_d * 0.75, mouth + (mouth - nose) * 0.9, conf(NOSE))

    # a tick across each estimated point, shorter the less sure the model is
    tick = eye_d * 0.30
    for i in (L_EYE, R_EYE, NOSE, MOUTH_L, MOUTH_R, L_EAR, R_EAR):
        if i >= len(pose):
            continue
        c = conf(i)
        p0 = pose[i]
        v = perp if i in (L_EYE, R_EYE, L_EAR, R_EAR) else axis
        stroke(p0 - v * tick * c, p0 + v * tick * c, c)

    # stroke coverage x model confidence; the opacity is applied when mixing,
    # so the line IMPOSES its density instead of adding to it. Added to an
    # already medium density, it didn't stand out.
    return np.clip(f, 0, 1) * (zone > 0.5)


def density_field(img: np.ndarray, body: np.ndarray, ident: np.ndarray,
                  zones: list, stage: float, size: float, cfg, rng,
                  pose=None, vis=None) -> np.ndarray:
    """Ink demand per pixel, and its drift towards formless."""
    subject = np.clip(body + ident, 0, 1)
    raw = np.clip((1.0 - luminance(img)) ** cfg.gamma, 0, 1)

    # Levels are set on the BODY ALONE, identity area excluded: the dark
    # features of the face otherwise crushed the whole tonal range and the
    # body came out at the same density as the background (figure and ground
    # swapped).
    body_only = np.clip(body - ident, 0, 1)
    ink_true = levels(raw, body_only > 0.5)

    # The background is COMPRESSED, not stretched. A percentile stretch
    # amplifies tone steps: in a photo taken in a mirror, the bright glass and
    # the dark room share the background, and stretching turned their border
    # into a hard break following the edge of the mirror.
    out = (1.0 - subject) > 0.5
    med = float(np.median(raw[out])) if out.any() else float(np.median(raw))
    back = np.clip(0.5 + (raw - med) * cfg.bg_contrast, 0, 1)
    d = ink_true * subject + back * cfg.bg_density * (1.0 - subject)

    # The body drifts towards formless with --stage; the identity area has a
    # floor at --face-floor (1.0 by default = no face structure left).
    # ident is exactly 1 on the guaranteed core (see _with_fade), so no
    # saturation is needed, and it would crush the fade.
    core = np.clip(ident, 0, 1)
    floor = max(stage, cfg.face_floor)
    mix = np.maximum(np.clip(stage, 0, 1) * body, core * floor)
    mix = np.maximum(mix, bleed_field(ident, body, size, cfg.bleed, rng) * floor)
    # The patch that replaces the face borrows NOTHING from the face. Matched
    # to the face's average grey, it gave a pale round blob with a rim: once
    # levels are set on the body, facial skin (paler than the rest, it catches
    # the light) clips to zero ink. So the patch is matched to the BODY
    # TEXTURE AROUND it, mean and spread, so it blends in. Useful side effect:
    # not a single number comes from the face any more.
    ref_all = body_only > 0.5

    def stats(zone):
        if cfg.face_tone is not None:
            return float(np.clip(cfg.face_tone, 0, 1)), 0.16
        hard = (zone > 0.5).astype(np.uint8)
        k = max(3, int(size * 0.45)) | 1
        ring = cv2.dilate(hard, np.ones((k, k), np.uint8)) - hard
        sel = (ring > 0) & ref_all
        if sel.sum() < 256:
            sel = ref_all
        if sel.sum() < 256:
            sel = np.ones(zone.shape, bool)
        v = ink_true[sel]
        return float(v.mean()), float(np.clip(v.std(), 0.10, 0.34))

    sc = size * cfg.blotch
    lv, sp = stats(np.zeros(img.shape[:2], np.float32))
    flat = structureless(img.shape[:2], lv, sc, rng, sp)
    for z in zones:
        fill = continued_texture(ink_true, z, size)
        if fill is None:                        # empty area, or covering everything
            fill = borrowed_texture(ink_true, z, ref_all.astype(np.float32))
        if fill is None:
            lv, sp = stats(z)
            fill = structureless(img.shape[:2], lv, sc, rng, sp)
        flat = np.where(z > 0.5, fill, flat)

    # The guessed lines are made of the same dots, not an overlay. They must
    # CONTRAST with their surroundings: drawn in ink over an already dense
    # area, they drowned. So they are drawn as paper on a dark area and as
    # ink on a light one.
    L = face_lines(pose, vis, np.clip(ident, 0, 1), cfg.lines)
    if cfg.lines > 0 and L.max() > 0:
        near = L > 0.05
        local = float(flat[near].mean()) if near.any() else 0.5
        target = 0.0 if local > 0.45 else 1.0
        # A HARD value: a blend left 19 % of the texture in the stroke, which
        # chopped it up until it was unreadable (measured contrast 0.197,
        # invisible to the eye). So the stroke imposes its value, and --lines
        # only sets the overall opacity.
        hard = (L > 0.35).astype(np.float32) * float(np.clip(cfg.lines, 0, 1))
        flat = np.clip(flat * (1 - hard) + target * hard, 0, 1)

    return np.clip(d * (1 - mix) + flat * mix, 0, 1)


def _marks(d: np.ndarray, gate: np.ndarray, cfg, rng: np.random.Generator,
           grain_mul: float = 1.0) -> np.ndarray:
    """Stochastic screen: one dot per cell, drawn with p = local tone.

    DENSITY carries the tone, not dot size: dots of different sizes made
    clumps instead of a graded screen.
    """
    h, w = d.shape
    u = max(1.0, min(h, w) * cfg.grain * grain_mul / 1200.0)
    gh, gw = max(1, int(h / u)), max(1, int(w / u))
    cell = cv2.resize(d, (gw, gh), interpolation=cv2.INTER_AREA)
    hit = rng.random((gh, gw)) < np.clip(cell * cfg.density, 0, 1)
    hit = cv2.resize(hit.astype(np.uint8), (w, h),
                     interpolation=cv2.INTER_NEAREST).astype(bool)
    return hit & gate


def stipple(d: np.ndarray, ident: np.ndarray, cfg, rng: np.random.Generator) -> np.ndarray:
    """Two sealed passes.

    Without a seal, a dot born OUTSIDE the identity area can spill into it and
    bring neighbouring pixels in. So the identity area is stippled on its own.
    Its dots may spill outwards (harmless, and it avoids a paper rim at the
    seam) but nothing gets in. Cells straddling the border only average values
    from their own side, d being 0 across the border.
    """
    h, w = d.shape
    u = max(1.0, min(h, w) * cfg.grain / 1200.0)
    hard = ident > 0.53                       # where density_field saturates mix to 1
    r = max(1, int(round(u)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    spill = cv2.dilate(hard.astype(np.uint8), k).astype(bool)
    # With --face-grain above 1 the identity area is stippled COARSER than the
    # body: dot size becomes a measure of what the machine knows. It has almost
    # no information about the face, so it renders it at low resolution. No
    # solid patch, no pictogram: the same texture, less defined.
    return (_marks(d * hard, spill, cfg, rng, cfg.face_grain)
            | _marks(d * ~hard, ~hard, cfg, rng))


def compose(ink: np.ndarray, cfg, rng: np.random.Generator) -> np.ndarray:
    paper, inkc = np.array(PAPER, np.float32), np.array(INK, np.float32)
    if cfg.polarity == "light":
        paper, inkc = inkc, paper
    a = ink[..., None].astype(np.float32) * (0.86 + 0.14 * rng.random(ink.shape)[..., None])
    out = paper * (1 - a) + inkc * a
    return np.clip(out + rng.normal(0, 1.5, (*ink.shape, 1)).astype(np.float32), 0, 255)


# caption

def _emoji_font(px: int):
    """Apple Color Emoji only exists at a few fixed sizes: take the first one
    above the wanted size, the glyph is scaled down afterwards."""
    for st in EMOJI_STRIKES:
        if st >= px:
            try:
                return ImageFont.truetype(EMOJI_FONT, st), st
            except OSError:
                continue
    return None, 0


def _runs(text: str):
    """Splits into alternating text / emoji runs, keeping ZWJ sequences whole."""
    return [(part, bool(i % 2)) for i, part in enumerate(EMOJI_RE.split(text)) if part]


def rich_width(text: str, font, px: int) -> float:
    w = 0.0
    for part, is_emoji in _runs(text):
        w += px * 1.18 * len(EMOJI_RE.findall(part) or [part]) if is_emoji \
            else font.getlength(part)
    return w


def draw_rich(img: Image.Image, d: ImageDraw.ImageDraw, xy, text: str,
              font, fill, px: int) -> float:
    """Writes a line mixing Optima and colour emoji."""
    x, y = xy
    ef, strike = _emoji_font(px)
    for part, is_emoji in _runs(text):
        if not is_emoji or ef is None:
            d.text((x, y), part, font=font, fill=fill)
            x += font.getlength(part)
            continue
        for ch in EMOJI_RE.findall(part) or [part]:
            tile = Image.new("RGBA", (strike * 2, strike * 2), (0, 0, 0, 0))
            ImageDraw.Draw(tile).text((0, 0), ch, font=ef, embedded_color=True)
            box = tile.getbbox()
            if box:
                side = max(box[2] - box[0], box[3] - box[1])
                glyph = tile.crop((box[0], box[1], box[0] + side, box[1] + side))
                glyph = glyph.resize((px, px), Image.LANCZOS)
                img.paste(glyph, (int(x), int(y + px * 0.12)), glyph)
            x += px * 1.18
    return x - xy[0]


def face(px: int) -> ImageFont.FreeTypeFont:
    for path in CAPTION_FONTS:
        try:
            return ImageFont.truetype(path, px)
        except OSError:
            continue
    return ImageFont.load_default(px)


# Named anchors. "l"/"r" are the SUBJECT's left and right, so mirrored in the
# image. Each value is either a pose index or a weighted mix of indices.
ANCHORS = {
    "nose": 0, "eye-l": 2, "eye-r": 5, "ear-l": 7, "ear-r": 8,
    "shoulder-l": 11, "shoulder-r": 12, "elbow-l": 13, "elbow-r": 14,
    "wrist-l": 15, "wrist-r": 16, "hand-l": 19, "hand-r": 20,
    "hip-l": 23, "hip-r": 24, "knee-l": 25, "knee-r": 26,
    "ankle-l": 27, "ankle-r": 28, "foot-l": 31, "foot-r": 32,
    "mouth": ((9, 0.5), (10, 0.5)),
    "face": tuple((i, 1.0 / len(HEAD_IDX)) for i in HEAD_IDX),
    "shoulders": ((11, 0.5), (12, 0.5)),
    "neck": ((11, 0.33), (12, 0.33), (0, 0.34)),
    "hips": ((23, 0.5), (24, 0.5)),
    "chest": ((11, 0.35), (12, 0.35), (23, 0.15), (24, 0.15)),
    "belly": ((11, 0.2), (12, 0.2), (23, 0.3), (24, 0.3)),
    "torso": ((11, 0.25), (12, 0.25), (23, 0.25), (24, 0.25)),
}


# AREA anchors: one name gives four coordinates, the extent of that part of
# the body. "body" uses the segmented outline when there is one, since it
# bounds the subject better than the pose landmarks alone.
ANCHOR_BOXES = {
    "body": tuple(range(33)),
    "upper-body": (11, 12, 23, 24),
    "legs": (23, 24, 25, 26, 27, 28, 29, 30, 31, 32),
    "arm-l": (11, 13, 15, 17, 19, 21),
    "arm-r": (12, 14, 16, 18, 20, 22),
    "head": tuple(HEAD_IDX),
}


def anchor_box(name: str, det: Detection) -> np.ndarray | None:
    """Resolves an area anchor to (x0, y0, x1, y1) pixels."""
    key = name.strip().lower()
    idx = ANCHOR_BOXES.get(key)
    if idx is None:
        return None
    if key == "body" and det.bbox is not None:
        return np.array(det.bbox, np.float32)
    if det.pose is None:
        return None
    pts = det.pose[list(idx)]
    if det.vis is not None:
        keep = pts[det.vis[list(idx)] > 0.2]
        pts = keep if len(keep) >= 2 else pts
    box = np.array([pts[:, 0].min(), pts[:, 1].min(),
                    pts[:, 0].max(), pts[:, 1].max()], np.float32)
    # An area with too few visible points gives a flattened rectangle: report
    # it as unavailable rather than print a meaningless extent.
    if box[2] - box[0] < 4.0 or box[3] - box[1] < 4.0:
        return None
    return box


def anchor_point(name: str, det: Detection) -> np.ndarray | None:
    """Resolves a named anchor to pixel coordinates."""
    spec = ANCHORS.get(name.strip().lower())
    if spec is None or det.pose is None:
        return None
    if isinstance(spec, int):
        return det.pose[spec].astype(np.float32)
    return sum(det.pose[i] * w for i, w in spec).astype(np.float32)


def anchor_conf(name: str, det: Detection) -> float:
    """Reliability of an anchor: the lowest visibility among its points.

    A point out of frame or badly seen comes out at a meaningless position
    without any warning, so it has to be possible to refuse it.
    """
    key = name.strip().lower()
    if key in ANCHOR_BOXES:
        if det.vis is None:
            return 1.0
        v = det.vis[list(ANCHOR_BOXES[key])]
        return float(np.percentile(v, 75))      # an area tolerates missed points
    spec = ANCHORS.get(key)
    if spec is None or det.pose is None or det.vis is None:
        return 1.0 if spec is not None else 0.0
    idx = [spec] if isinstance(spec, int) else [i for i, _ in spec]
    return float(min(det.vis[i] for i in idx))


def parse_notes(specs: list[str], shape: tuple[int, int],
                det: Detection | None = None) -> list[tuple[list[float], str]]:
    """'x,y:text' or 'x0,y0,x1,y1:text' -> [(px coordinates, text)].

    Coordinates are normalized (0..1) like --mask and --keep, and printed in
    pixels of the working image. Above 1 they are taken as pixels.
    """
    h, w = shape
    rows = []
    for spec in specs:
        head, sep, text = spec.partition(":")
        if not sep:
            sys.exit(f"invalid --note: {spec!r}. Expected 'x,y:text'")
        parts = [x.strip() for x in head.split(",")]
        if len(parts) == 1 and parts[0].lower() in ANCHOR_BOXES:
            if det is None:
                sys.exit("--note: anchors unavailable (no pose detected)")
            box = anchor_box(parts[0], det)
            if box is None:
                sys.exit(f"--note: the area '{parts[0]}' can't be used on this "
                         f"photo (too few points seen, out of frame?). "
                         f"See --anchors.")
            if anchor_conf(parts[0], det) < 0.25:
                print(f"  /!\\ area '{parts[0]}' is unreliable: rough extent.",
                      file=sys.stderr)
            rows.append(([float(c) for c in box], text.strip()))
            continue
        if parts and all(x.lower() in ANCHORS for x in parts):
            if det is None:
                sys.exit("--note: named anchors unavailable (no pose detected)")
            if len(parts) not in (1, 2):
                sys.exit(f"--note: 1 or 2 anchors expected in {spec!r}")
            pts = [anchor_point(x, det) for x in parts]
            if any(q is None for q in pts):
                sys.exit(f"--note: anchor not found in {spec!r}")
            for nm in parts:
                if anchor_conf(nm, det) < 0.25:
                    print(f"  /!\\ anchor '{nm}' is unreliable (out of frame or badly"
                          f" seen): its coordinate may be meaningless.", file=sys.stderr)
            px = [float(c) for q in pts for c in q]
        else:
            try:
                v = [float(x) for x in parts]
            except ValueError:
                sys.exit(f"invalid --note: {head!r} is neither coordinates nor a "
                         f"known anchor. List them with --anchors")
            if len(v) not in (2, 4):
                sys.exit(f"invalid --note: {len(v)} coordinates in {spec!r}, "
                         f"expected 2 or 4")
            px = [(n * (w if i % 2 == 0 else h)) if n <= 1.0 else n
                  for i, n in enumerate(v)]
        rows.append((px, text.strip()))
    return rows


MARK_ACCENT = (176, 58, 44)


def draw_marks(pil: Image.Image, notes: list, cfg, fs: int, ink) -> None:
    """An open circle for a point, a rectangle for an area.

    Each mark carries its number, the same as in the caption: without it,
    three notes and three marks connect to nothing.
    """
    mode = getattr(cfg, "mark", "ink")
    if mode == "none" or not notes:
        return
    col = tuple(MARK_ACCENT if mode == "accent" else ink)
    d = ImageDraw.Draw(pil, "RGBA")
    f = face(int(fs * 1.0))
    # sized to be seen at export size, not to the pixel
    lw = max(1, int(fs * 0.16))
    r = max(6, int(fs * 1.15))
    for i, (px, _) in enumerate(notes, 1):
        if len(px) >= 4:
            x0, y0, x1, y1 = px[:4]
            x0, x1 = sorted((x0, x1))
            y0, y1 = sorted((y0, y1))
            d.rectangle([x0, y0, x1, y1], outline=col + (210,), width=lw)
            lx, ly = x0 + lw + 2, y0 + lw + 2
        else:
            x, y = px[0], px[1]
            d.ellipse([x - r, y - r, x + r, y + r], outline=col + (230,), width=lw)
            lx, ly = x + r * 1.4, y - r * 1.1
        d.text((lx, ly), str(i), font=f, fill=col + (230,))


def with_caption(img: np.ndarray, det: Detection, cfg,
                 zone: np.ndarray | None = None) -> np.ndarray:
    """Bottom margin plus coordinates, set like a museum label: outside the image."""
    if cfg.caption_lines <= 0:
        return img
    h, w = img.shape[:2]
    fs = max(9, int(w * 0.0165))
    lead = int(fs * 1.7)
    inset = int(w * 0.055)
    n_rows = len(getattr(cfg, "note", []) or []) or cfg.caption_lines
    # a long note wraps, so plan for more room rather than crop the text
    wrap = 2 if getattr(cfg, "note", None) else 1
    margin = int(fs * 2.2) + n_rows * wrap * lead + int(fs * 1.4)

    ground = INK if cfg.polarity == "light" else PAPER
    text = PAPER if cfg.polarity == "light" else INK
    canvas = np.empty((h + margin, w, 3), np.float32)
    canvas[:] = np.array(ground, np.float32)
    canvas[:h] = img

    pil = Image.fromarray(canvas.astype(np.uint8))
    d = ImageDraw.Draw(pil, "RGBA")
    y = h + int(fs * 1.5)
    d.line([(inset, y), (w - inset, y)], fill=tuple(text) + (38,), width=1)

    # By default the coordinates come from the OUTLINE OF THE ERASED AREA, not
    # from the face mesh. The outline is already visible in the image, so the
    # numbers add no information. Face points, on the other hand, gave away
    # the real position and scale of a face that is otherwise destroyed: a
    # small leak, but a needless one.
    notes = parse_notes(getattr(cfg, "note", []) or [], (h, w), det)
    if notes:
        draw_marks(pil, notes, cfg, fs, text)          # on the canvas, not a copy
        fo = face(fs)
        maxw = w - 2 * inset
        y += int(fs * 1.3)
        # The number only links a line to its mark: without marks it points to
        # nothing and clutters the caption.
        numbered = getattr(cfg, "mark", "none") != "none"
        for i, (px, body_text) in enumerate(notes, 1):
            coords = " / ".join(f"{n:.1f}" for n in px)
            prefix = (f"{i}.  {coords} : " if numbered else f"{coords} : ")
            # fixed indent: aligning on the prefix width pushed the
            # continuation to the middle of the line when the prefix was long
            indent = fs * 1.6
            line, first = prefix, True
            for word in body_text.split():
                trial = line + word + " "
                if rich_width(trial, fo, fs) > maxw and line.strip():
                    draw_rich(pil, d, (inset if first else inset + indent, y),
                              line.rstrip(), fo, tuple(text) + (170,), fs)
                    y += lead
                    line, first = word + " ", False
                else:
                    line = trial
            if line.strip():
                draw_rich(pil, d, (inset if first else inset + indent, y),
                          line.rstrip(), fo, tuple(text) + (170,), fs)
                y += lead
        return np.asarray(pil).astype(np.float32)

    src = None
    if getattr(cfg, "caption_source", "zone") == "zone" and zone is not None:
        cont, _ = cv2.findContours((zone > 0.53).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if cont:
            src = max(cont, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    if src is None:
        src = det.faces[0] if det.faces else det.pose
    if src is None or not len(src):
        return np.asarray(pil).astype(np.float32)
    pick = np.linspace(0, len(src) - 1, cfg.caption_lines).astype(int)
    f = face(fs)
    y += int(fs * 1.3)
    for i in pick:
        q = src[i]
        d.text((inset, y), f"{q[0]:.1f}, {q[1]:.1f}", font=f, fill=tuple(text) + (150,))
        y += lead
    return np.asarray(pil).astype(np.float32)


# render

def render(img: np.ndarray, det: Detection, head: np.ndarray, body: np.ndarray,
           extra: np.ndarray, stage: float, cfg, zones: list | None = None,
           plans: list | None = None) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed)
    ident = np.clip(head + extra, 0, 1)
    base = img.astype(np.float32)

    if cfg.bg == "blur":
        base = kill_background(base, np.clip(body + ident, 0, 1), cfg.blur, ident)
    elif cfg.bg == "black":
        keep = np.clip(body + ident, 0, 1)[..., None]
        base = base * keep

    d = density_field(base, body, ident, zones or [head], stage, det.size, cfg,
                      rng, det.pose, det.vis)
    if plans:
        # --say: the bubble enters the dot field before stippling (same
        # texture, same drift); the text goes on afterwards, in solid ink
        from . import bubbles
        ink = PAPER if cfg.polarity == "light" else INK
        unit = max(1.0, min(d.shape) * cfg.grain / 1200.0)
        d = bubbles.machine_density(d, ident, plans, stage, unit, ink, cfg.speaker)
    out = compose(stipple(d, ident, cfg, rng), cfg, rng)
    if plans:
        out = bubbles.machine_text(out, plans, stage, ink, cfg.speaker)
    return np.clip(with_caption(out, det, cfg, ident), 0, 255).astype(np.uint8)


# CLI

def check_overlay(src: np.ndarray, det: Detection, head: np.ndarray) -> np.ndarray:
    """The outline of the erased area, drawn on the ORIGINAL photo.

    It's the only way to check that the mask falls on the face and not on an
    object in front of it. The image shows the face in clear: it is a local
    checking tool, never something to post.
    """
    out = src.copy()
    r = max(2, int(0.003 * max(out.shape[:2])))
    cont, _ = cv2.findContours((head > 0.53).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cont, -1, (255, 60, 60), max(2, r), cv2.LINE_AA)
    if det.pose is not None:
        for i in HEAD_IDX:
            x, y = det.pose[i]
            c = 1.0 if det.vis is None else float(np.clip(det.vis[i], 0, 1))
            cv2.circle(out, (int(x), int(y)), r,
                       (int(240 - 180 * c), int(50 + 60 * c), 255), -1, cv2.LINE_AA)
    for f in det.faces:
        cv2.polylines(out, [cv2.convexHull(f.astype(np.int32))], True,
                      (60, 220, 90), max(1, r), cv2.LINE_AA)
    return out


def mark_points(img: np.ndarray, det: Detection) -> np.ndarray:
    """Marks the estimated face points on the ALREADY anonymized render.

    Used to check that the machine located the face behind the object hiding
    it. The control image shows no face: it is made from the output, not from
    the photo.
    """
    out = img.copy()
    if det.pose is None:
        return out
    r = max(2, int(0.004 * max(out.shape[:2])))
    for i in HEAD_IDX:
        x, y = det.pose[i]
        c = 1.0 if det.vis is None else float(np.clip(det.vis[i], 0, 1))
        col = (int(235 - 180 * c), int(40 + 60 * c), int(40 + 60 * c))
        cv2.circle(out, (int(x), int(y)), r, col, -1, cv2.LINE_AA)
    for f in det.faces:
        cv2.polylines(out, [cv2.convexHull(f.astype(np.int32))], True,
                      (40, 90, 220), max(1, r // 2), cv2.LINE_AA)
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Rebuilds the body as a dot screen; the face area is "
                    "replaced by dots that carry nothing of the face.")
    p.add_argument("photo", type=Path)
    p.add_argument("--out", type=Path, help="default: <photo>_dots.png")
    p.add_argument("--stage", type=float, default=0.25,
                   help="how far the BODY dissolves towards formless, 0..1. Low = "
                        "readable body (default 0.25). Never affects the face.")
    p.add_argument("--variants", nargs="?", type=int, const=8, default=0,
                   metavar="N",
                   help="write a FOLDER with N separate images, one per value of "
                        "--stage (default 8), instead of a single render")
    p.add_argument("--face-floor", type=float, default=1.0,
                   help="MINIMUM dissolve of the identity area (default 1.0 = no "
                        "face structure left). Don't go below 0.95.")
    p.add_argument("--grain", type=float, default=1.8,
                   help="dot size (default 1.8). Low = finer screen.")
    p.add_argument("--face-grain", dest="face_grain", type=float, default=1.0,
                   metavar="N",
                   help="dot size in the identity area, as a multiple of --grain "
                        "(default 1 = as fine as the body). Above 1 the area is "
                        "rendered coarser, as if the machine lacked information "
                        "there.")
    p.add_argument("--density", type=float, default=1.0, help="amount of ink")
    p.add_argument("--gamma", type=float, default=1.0,
                   help="screen contrast (high = more white)")
    p.add_argument("--face-tone", dest="face_tone", type=float, default=None,
                   metavar="0..1",
                   help="fixed grey level for the face area. By default it is "
                        "matched to the body around it.")
    p.add_argument("--blotch", type=float, default=0.05,
                   help="scale of the formless field, as a fraction of the subject")
    p.add_argument("--polarity", choices=("ink", "light"), default="ink",
                   help="'ink' = dark dots on paper; 'light' = the reverse")
    p.add_argument("--bg", choices=("blur", "keep", "black"), default="blur",
                   help="background: 'blur' destroys the room's detail (default)")
    p.add_argument("--blur", type=float, default=1.0, help="background blur strength")
    p.add_argument("--bg-contrast", dest="bg_contrast", type=float, default=0.55,
                   help="background contrast, 0..1 (default 0.55). Low = flat, soft "
                        "background, which avoids a break at the edge of a mirror.")
    p.add_argument("--bg-density", dest="bg_density", type=float, default=0.15,
                   help="background ink density, 0..1 (default 0.15). The "
                        "background must stay well below the subject.")
    p.add_argument("--mark", choices=("none", "ink", "accent"), default="none",
                   help="marks on the image where notes point: a circle for a "
                        "point, a rectangle for an area, numbered like the "
                        "caption. 'none' = no marks (default), 'ink' = same ink "
                        "as the rest, 'accent' = a muted vermilion.")
    p.add_argument("--anchors", action="store_true",
                   help="list the named anchors usable in --note, with their "
                        "position on THIS photo, then stop")
    p.add_argument("--note", action="append", default=[], metavar="ANCHOR:TEXT",
                   help="note: an anchor and your text, written in the caption. "
                        "The anchor is a NAME ('neck:text', "
                        "'shoulder-l,hip-r:text', see --anchors) or normalized 0..1 "
                        "coordinates ('x,y:text'). Emoji are drawn in colour. "
                        "Repeatable; replaces the automatic coordinates.")
    p.add_argument("--say", action="append", default=[], metavar="[ANCHOR:]TEXT",
                   help="speech bubble INSIDE the dots. Your text stays sharp, in "
                        "solid ink, at every stage; the bubble dissolves with "
                        "--stage like the body, and its attribution line drifts "
                        "from the mouth until it points at no one. Default "
                        "anchor: mouth. '\\n' forces a line break. Repeatable.")
    p.add_argument("--say-size", dest="say_size", type=float, default=0.026,
                   help="bubble text size, as a fraction of the width")
    p.add_argument("--say-drift", dest="say_drift", type=float, default=1.0,
                   help="how far the attribution line drifts at the highest stage "
                        "(default 1 = leaves the head; 0 = no drift)")
    p.add_argument("--no-speaker", dest="speaker", action="store_false",
                   help="remove the 'speaker: ...' line under the bubble text")
    p.add_argument("--caption-source", dest="caption_source",
                   choices=("zone", "face"), default="zone",
                   help="where the caption coordinates come from: 'zone' = the "
                        "outline of the erased area, already visible so no added "
                        "information (default); 'face' = face mesh points, which "
                        "give away its real position and scale.")
    p.add_argument("--caption-lines", dest="caption_lines", type=int, default=3,
                   help="number of coordinates in the caption, 0 = no caption")
    p.add_argument("--faces", type=int, default=3,
                   help="max number of faces to look for (default 3). A photo taken "
                        "in a mirror often has two.")
    p.add_argument("--lines", type=float, default=0.0, metavar="0..1",
                   help="face lines guessed by the machine, drawn in the erased "
                        "area (default 0 = none). They come from the pose "
                        "landmarks, so they bring back a rough face geometry: "
                        "--lines 0 removes it.")
    p.add_argument("--bleed", type=float, default=0.0, metavar="0..2",
                   help="spread of the destruction into the body, as a fraction of "
                        "the subject scale (default 0 = none). WIDENS the destroyed "
                        "region well beyond the head: only use it for that "
                        "dissolving effect.")
    p.add_argument("--zone", choices=("face", "head"), default="face",
                   help="'face' keeps the destruction tight on the estimated face, "
                        "so an object held in front of it sticks out instead of "
                        "being swallowed (default); 'head' destroys the whole "
                        "head, hair included, which covers more.")
    p.add_argument("--fade", type=float, default=0.35, metavar="N",
                   help="fade from the edge of the area to the rest, as a fraction "
                        "of the face height (default 0.35, 0 = hard edge). The core "
                        "stays fully destroyed: only the outside gets a ramp.")
    p.add_argument("--head-grow", dest="head_grow", type=float, default=1.0,
                   help="enlarges the identity area (long hair, a hat...)")
    p.add_argument("--mask", action="append", default=[], metavar="SPEC",
                   help="EXTRA area to destroy (tattoo, reflection). 'x,y,r' or "
                        "'x0,y0,x1,y1', normalized. Repeatable.")
    p.add_argument("--keep", action="append", default=[], metavar="SPEC",
                   help="area to SPARE from the destruction (a phone held in front "
                        "of the face). Same format as --mask. What is spared stays "
                        "visible: check that no skin shows in it. Repeatable.")
    p.add_argument("--no-seal", dest="seal", action="store_false",
                   help="leave intact the objects the segmentation leaves out (a "
                        "phone held in front of the face) instead of absorbing "
                        "them into the erased area")
    p.set_defaults(seal=True)
    p.add_argument("--check", action="store_true",
                   help="write a CHECK overlay on the original photo: outline of "
                        "the erased area, pose landmarks, detected faces. This "
                        "image SHOWS THE FACE. It is for checking the placement, "
                        "never for posting.")
    p.add_argument("--show-points", dest="show_points", action="store_true",
                   help="also write a control image: the estimated face points "
                        "marked on the ALREADY anonymized render, to check that "
                        "the machine located the face")
    p.add_argument("--width", type=int, default=1440, help="long side of the export")
    p.add_argument("--seed", type=int, default=7)
    cfg = p.parse_args()

    if not cfg.photo.exists():
        sys.exit(f"Photo not found: {cfg.photo}")
    if cfg.face_floor < 0.95:
        print("WARNING: --face-floor < 0.95, some face structure remains.",
              file=sys.stderr)

    img = load_image(cfg.photo)
    print(f"image {img.shape[1]}x{img.shape[0]}, detecting...", file=sys.stderr)
    det = detect(img, cfg.faces)
    if det.pose is None and not det.faces:
        sys.exit("No person detected. Try a brighter photo, a looser crop, or a "
                 "less unusual pose.")

    head, zones = head_mask(det, img.shape[:2], cfg.head_grow, cfg.seal,
                            cfg.zone, cfg.fade)
    if head is None or float(head.max()) < 0.5:
        sys.exit("Head not located, so no image is made: anonymity would not be "
                 "guaranteed. Use --mask to mark the area by hand.")
    body = body_mask(det, img.shape[:2])
    extra = region_mask(cfg.mask, img.shape[:2], '--mask')

    # spared areas: taken out of the identity area AND of the per-head areas
    if cfg.keep:
        spare = region_mask(cfg.keep, img.shape[:2], '--keep')
        inv = 1.0 - np.clip(spare, 0, 1)
        before = float((head > 0.53).sum())
        head = head * inv
        zones = [z * inv for z in zones]
        extra = extra * inv
        after = float((head > 0.53).sum())
        print(f"  --keep: {100 * (1 - after / max(1.0, before)):.0f} % of the identity "
              f"area spared. WHAT IS SPARED STAYS VISIBLE: check that no skin "
              f"of the face shows in it.", file=sys.stderr)
        if head.max() < 0.5:
            sys.exit("--keep covers the whole identity area: nothing would be "
                     "destroyed. Make the spared area smaller.")
    print(f"  pose: {'yes' if det.pose is not None else 'NO'}  "
          f"faces: {len(det.faces)}  "
          f"segmentation: {'yes' if det.body is not None else 'fallback'}  "
          f"subject: {det.size:.0f}px", file=sys.stderr)
    h0, w0 = img.shape[:2]
    print(f"  face located by: {det.source}", file=sys.stderr)
    for i, g in enumerate(face_pose_gap(det), 1):
        if g > 1.2:
            print(f"  /!\\ face {i} found {g:.1f}x the head height away from the "
                  f"point estimated by the pose. A reflection in a screen, a "
                  f"pattern taken for a face, or a second person: CHECK with --check.",
                  file=sys.stderr)
    for i, f in enumerate(det.faces, 1):
        cx, cy = f.mean(axis=0)
        print(f"  face {i} erased, centre x={cx / w0:.2f} y={cy / h0:.2f}",
              file=sys.stderr)
    if cfg.zone == "face" and not det.faces:
        print("  --zone face: the area is ESTIMATED from the pose landmarks, not "
              "measured on the face. It covers less than --zone head: check that "
              "no part of the face sticks out.", file=sys.stderr)
    if not det.faces and det.pose is not None:
        pts = det.pose[HEAD_IDX]
        cf = det.vis[HEAD_IDX] if det.vis is not None else np.ones(len(HEAD_IDX))
        c = pts.mean(axis=0)
        print(f"  face mesh failed (something in front of the face?), but the pose "
              f"LOCATES the face: centre x={c[0] / w0:.2f} y={c[1] / h0:.2f}, "
              f"mean confidence {float(np.mean(cf)):.2f} over "
              f"{len(HEAD_IDX)} head points", file=sys.stderr)
        print(f"    nose x={det.pose[NOSE][0] / w0:.2f} y={det.pose[NOSE][1] / h0:.2f}"
              f"   eye spacing {np.linalg.norm(det.pose[R_EYE] - det.pose[L_EYE]) / w0:.3f}"
              f"   (--show-points to check by eye)", file=sys.stderr)
    if len(det.faces) <= 1:
        print("  -> one face only. If there is a mirror, check that no second face "
              "shows in it; cover it with --mask if needed.",
              file=sys.stderr)

    if cfg.anchors:
        print("\n  anchors usable in --note (l/r = the SUBJECT's left/right,"
              "\n  so mirrored in the image):\n", file=sys.stderr)
        print("    -- areas (one name = four coordinates) --", file=sys.stderr)
        for name in sorted(ANCHOR_BOXES):
            b = anchor_box(name, det)
            if b is None:
                continue
            cf = anchor_conf(name, det)
            flag = "   UNRELIABLE" if cf < 0.25 else ""
            print(f"    {name:<12} {b[0]/w0:.3f}, {b[1]/h0:.3f} -> "
                  f"{b[2]/w0:.3f}, {b[3]/h0:.3f}   confidence {cf:.2f}{flag}",
                  file=sys.stderr)
        print("\n    -- points --", file=sys.stderr)
        for name in sorted(ANCHORS):
            q = anchor_point(name, det)
            if q is None:
                continue
            vx, vy = q[0] / w0, q[1] / h0
            if not (0 <= vx <= 1 and 0 <= vy <= 1):
                print(f"    {name:<12} out of frame", file=sys.stderr)
                continue
            cf = anchor_conf(name, det)
            flag = "   UNRELIABLE" if cf < 0.25 else ""
            print(f"    {name:<12} {vx:.3f}, {vy:.3f}   confidence {cf:.2f}{flag}",
                  file=sys.stderr)
        print(f"\n  example: --note \"neck:Collar checked, no leash. \U0001F512\"",
              file=sys.stderr)
        return

    if cfg.check:
        chk = cfg.photo.with_name(f"{cfg.photo.stem}_CHECK_SHOWS_FACE.png")
        save_image(check_overlay(img, det, head), chk, None)
        print(f"-> {chk}\n     /!\\ THIS IMAGE SHOWS THE FACE. The red outline "
              f"must cover the face, not the object in front of it. Delete it "
              f"after checking, never post it.", file=sys.stderr)

    plans = None
    if cfg.say:
        from . import bubbles
        plans = bubbles.plan_machine(img, det, np.clip(head + extra, 0, 1), cfg.say, cfg)

    if cfg.variants:
        out_dir = cfg.out or cfg.photo.with_name(f"{cfg.photo.stem}_variants")
        out_dir.mkdir(parents=True, exist_ok=True)
        for s in np.linspace(0.05, 0.95, cfg.variants):
            f = out_dir / f"stage_{s:.2f}.png"
            save_image(render(img, det, head, body, extra, float(s), cfg, zones, plans),
                       f, cfg.width or None)
            who = ("   " + "  |  ".join(bubbles.speaker(pl, float(s))[0] for pl in plans)
                   if plans and cfg.speaker else "")
            print(f"  {f.name}{who}", file=sys.stderr)
        print(f"-> {out_dir}/  ({cfg.variants} images)", file=sys.stderr)
    else:
        out = cfg.out or cfg.photo.with_name(f"{cfg.photo.stem}_dots.png")
        r = render(img, det, head, body, extra, cfg.stage, cfg, zones, plans)
        save_image(r, out, cfg.width or None)
        print(f"-> {out}", file=sys.stderr)
        if cfg.show_points:
            ctl = cfg.photo.with_name(f"{cfg.photo.stem}_points.png")
            save_image(mark_points(r, det), ctl, cfg.width or None)
            print(f"-> {ctl}  (control image: the render is already anonymized)",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
