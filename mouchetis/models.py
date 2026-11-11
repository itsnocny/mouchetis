"""MediaPipe models, downloaded on first use and then kept in a cache folder."""
from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

BASE = "https://storage.googleapis.com/mediapipe-models"
URLS = {
    "pose_landmarker_heavy.task":
        f"{BASE}/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "face_landmarker.task":
        f"{BASE}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "selfie_segmenter.tflite":
        f"{BASE}/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
}


def folder() -> Path:
    if os.environ.get("MOUCHETIS_MODELS"):
        return Path(os.environ["MOUCHETIS_MODELS"]).expanduser()
    cache = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(cache) / "mouchetis"


def path(name: str) -> Path:
    """Local path of a model, downloaded if missing."""
    p = folder() / name
    if p.exists():
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {name} (only once)...", file=sys.stderr)
    tmp = p.with_suffix(p.suffix + ".part")
    try:
        urllib.request.urlretrieve(URLS[name], tmp)
    except OSError as e:
        tmp.unlink(missing_ok=True)
        sys.exit(f"could not download {name}: {e}")
    tmp.rename(p)
    return p
