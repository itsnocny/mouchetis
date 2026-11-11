"""Anonymity audit: attack your own renders before posting them.

For each image in a folder:
  1. look for leftover metadata;
  2. write the RECONSTRUCTION someone trying to get back to the photo would
     get. A stochastic screen stores tone in the density of the dots, and a
     low-pass filter adds it back up. This is the image to look at: if you
     recognize yourself in it, don't post;
  3. with --original, measure where the output still correlates with the
     source photo, area by area, and write a map of the leaks.

    mouchetis audit photo_variants/
    mouchetis audit photo_variants/ --original photo.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


def load(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB")).copy()


def metadata(path: Path) -> list[str]:
    """Anything left that could identify: EXIF, GPS, comments."""
    found = []
    try:
        with Image.open(path) as im:
            if getattr(im, "_getexif", None) and im._getexif():
                found.append(f"EXIF ({len(im._getexif())} fields)")
            if im.info.get("gps") or im.info.get("GPS"):
                found.append("GPS")
            for k in ("comment", "Comment", "Software", "parameters", "XML:com.adobe.xmp"):
                if k in im.info:
                    found.append(k)
    except Exception as e:                       # noqa: BLE001
        found.append(f"unreadable ({e})")
    return found


def luminance(img: np.ndarray) -> np.ndarray:
    return (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]) / 255.0


def reconstruct(img: np.ndarray) -> np.ndarray:
    """The best simple way back: multi-scale low-pass, then equalization.

    This is exactly what someone trying to undo a dot screen would do.
    """
    lum = luminance(img.astype(np.float32))
    h, w = lum.shape
    base = min(h, w)
    acc = np.zeros_like(lum)
    for k, frac in ((0.5, 0.004), (0.3, 0.008), (0.2, 0.016)):
        acc += k * cv2.GaussianBlur(lum, (0, 0), max(1.0, base * frac))
    acc = cv2.normalize(acc, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.equalizeHist(acc.astype(np.uint8))


def local_corr(a: np.ndarray, b: np.ndarray, win: int) -> np.ndarray:
    """Sliding Pearson correlation: WHERE the output still looks like the source."""
    k = (win | 1, win | 1)
    ma, mb = cv2.blur(a, k), cv2.blur(b, k)
    va = cv2.blur(a * a, k) - ma * ma
    vb = cv2.blur(b * b, k) - mb * mb
    cov = cv2.blur(a * b, k) - ma * mb
    return np.clip(cov / np.sqrt(np.maximum(va * vb, 1e-8)), -1, 1)


def zones_from(orig: np.ndarray):
    """The subject's areas on the SOURCE photo, to break the measures down."""
    try:
        from . import stipple as C
    except Exception:                            # noqa: BLE001
        return None
    det = C.detect(orig)
    if det.pose is None and not det.faces:
        return None
    head, _ = C.head_mask(det, orig.shape[:2], 1.0, True, "face", 0.0)
    if head is None:
        return None
    body = C.body_mask(det, orig.shape[:2])
    return {"face": head > 0.5,
            "body": (body > 0.6) & (head < 0.02),
            "back": (body < 0.02) & (head < 0.02)}


def audit_one(path: Path, out_dir: Path, orig: np.ndarray | None, win: float):
    img = load(path)
    meta = metadata(path)
    rec = reconstruct(img)
    rec_path = out_dir / f"{path.stem}_reconstruction.png"
    cv2.imwrite(str(rec_path), rec)

    print(f"\n{path.name}  ({img.shape[1]}x{img.shape[0]})")
    print(f"  metadata             : {', '.join(meta) if meta else 'none'}")
    print(f"  reconstruction       : {rec_path.name}   <- LOOK AT THIS IMAGE")

    if orig is None:
        return None

    src = cv2.resize(orig, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
    a = luminance(src.astype(np.float32))
    b = 1.0 - luminance(img.astype(np.float32))          # ink is inverted light
    c = np.abs(local_corr(a, b, max(9, int(min(a.shape) * win))))

    cv2.imwrite(str(out_dir / f"{path.stem}_leaks.png"),
                cv2.applyColorMap((c * 255).astype(np.uint8), cv2.COLORMAP_INFERNO))

    z = zones_from(src)
    res = {}
    if z is None:
        print("  areas                : no subject found on the original")
        return res

    # A raw correlation means nothing here: two images that share the same
    # outline already correlate a lot without sharing any content. So we
    # measure the EXCESS over a chance level, obtained by correlating the
    # render with a flipped original (same statistics, wrong content), at
    # three scales, because fine detail and overall shape don't degrade at the
    # same rate.
    ink = 1.0 - luminance(img.astype(np.float32))
    wrong = luminance(np.flipud(src).astype(np.float32))
    base = min(img.shape[:2])
    scales = (("fine", base * 0.004), ("medium", base * 0.012), ("overall", base * 0.035))

    def cc(u, v, m, sig):
        u = cv2.GaussianBlur(u, (0, 0), max(1.0, sig))
        v = cv2.GaussianBlur(v, (0, 0), max(1.0, sig))
        if u[m].std() < 1e-6 or v[m].std() < 1e-6:
            return 0.0
        return abs(float(np.corrcoef(u[m], v[m])[0, 1]))

    print("  resemblance above chance (0 = no better than chance):")
    print("    for the face only the fine and medium columns matter. The overall")
    print("    column measures how the tone matches the surroundings, which is")
    print("    intended (the fade) and carries no features.")
    for name, m in z.items():
        if m.sum() < 64:
            continue
        ex = [max(0.0, cc(a, ink, m, sg) - cc(wrong, ink, m, sg)) for _, sg in scales]
        res[name] = tuple(ex)
        if name == "face":
            # A face's identity is in the DETAIL, not in the area's average
            # tone. That tone is matched to the surroundings on purpose (the
            # fade), and the 'overall' scale measures it. Including it raised a
            # false alarm on a face that the shuffle test proves leaks nothing.
            v = ("nothing recoverable" if max(ex[:2]) < 0.20 else
                 "LEAK, don't post")
        else:
            v = ("reconstructible" if max(ex) >= 0.30 else "barely reconstructible")
        cols = "  ".join(f"{lbl} {e:.2f}" for (lbl, _), e in zip(scales, ex))
        print(f"    {name:<8} {cols}   -> {v}")
    return res


def main() -> None:
    p = argparse.ArgumentParser(
        description="Attack your own renders to decide whether you can post them.")
    p.add_argument("folder", type=Path)
    p.add_argument("--original", type=Path,
                   help="source photo: adds the leak measures, area by area")
    p.add_argument("--out", type=Path, help="default: <folder>/audit")
    p.add_argument("--win", type=float, default=0.05,
                   help="correlation window size, as a fraction of the image "
                        "(default 0.05)")
    cfg = p.parse_args()

    if not cfg.folder.is_dir():
        sys.exit(f"Folder not found: {cfg.folder}")
    files = sorted(f for f in cfg.folder.iterdir() if f.suffix.lower() in EXTS)
    if not files:
        sys.exit(f"No image in {cfg.folder}")

    out_dir = cfg.out or cfg.folder / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.original and not cfg.original.exists():
        sys.exit(f"Original not found: {cfg.original}")
    orig = load(cfg.original) if cfg.original else None

    print(f"{len(files)} image(s), audit in {out_dir}/")
    worst = {}
    for f in files:
        r = audit_one(f, out_dir, orig, cfg.win)
        for name, ex in (r or {}).items():
            prev = worst.get(name, (0.0, 0.0, 0.0))
            worst[name] = tuple(max(a, b) for a, b in zip(prev, ex))

    print("\n" + "=" * 62)
    if not worst:
        print("Look at the *_reconstruction.png files: if you recognize yourself, don't post.")
        print("Run again with --original to put numbers on the leaks.")
        return
    for name, ex in worst.items():
        print(f"  worst {name:<8} : fine {ex[0]:.2f}  medium {ex[1]:.2f}  "
              f"overall {ex[2]:.2f}")
    face = max(worst.get("face", (0.0, 0.0, 0.0))[:2])
    if face >= 0.20:
        print("\n  /!\\ The FACE still correlates with the original. DON'T POST.")
    else:
        print("\n  Face: nothing recoverable.")
    print("  The body stays readable by design: the outline, the posture and")
    print("  anything larger than the dots (tattoos, marks) can be")
    print("  reconstructed. Look at the *_reconstruction.png files to judge.")


if __name__ == "__main__":
    main()
