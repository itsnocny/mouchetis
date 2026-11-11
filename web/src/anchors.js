// Named anchors, as in mouchetis/stipple.py. "l"/"r" are the subject's left
// and right, so mirrored in the image. A value is a pose index or a weighted
// mix of indices.

const HEAD_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

export const ANCHORS = {
  nose: 0, "eye-l": 2, "eye-r": 5, "ear-l": 7, "ear-r": 8,
  "shoulder-l": 11, "shoulder-r": 12, "elbow-l": 13, "elbow-r": 14,
  "wrist-l": 15, "wrist-r": 16, "hand-l": 19, "hand-r": 20,
  "hip-l": 23, "hip-r": 24, "knee-l": 25, "knee-r": 26,
  "ankle-l": 27, "ankle-r": 28, "foot-l": 31, "foot-r": 32,
  mouth: [[9, 0.5], [10, 0.5]],
  face: HEAD_IDX.map((i) => [i, 1 / HEAD_IDX.length]),
  shoulders: [[11, 0.5], [12, 0.5]],
  neck: [[11, 0.33], [12, 0.33], [0, 0.34]],
  hips: [[23, 0.5], [24, 0.5]],
  chest: [[11, 0.35], [12, 0.35], [23, 0.15], [24, 0.15]],
  belly: [[11, 0.2], [12, 0.2], [23, 0.3], [24, 0.3]],
  torso: [[11, 0.25], [12, 0.25], [23, 0.25], [24, 0.25]],
};

// one name gives the extent of a whole part of the body
export const ANCHOR_BOXES = {
  body: [...Array(33).keys()],
  "upper-body": [11, 12, 23, 24],
  legs: [23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
  "arm-l": [11, 13, 15, 17, 19, 21],
  "arm-r": [12, 14, 16, 18, 20, 22],
  head: HEAD_IDX,
};

const indices = (spec) => (typeof spec === "number" ? [spec] : spec.map(([i]) => i));

export function anchorPoint(name, det) {
  const spec = ANCHORS[name];
  if (spec === undefined || !det.pose) return null;
  if (typeof spec === "number") return det.pose[spec].slice();
  let x = 0, y = 0;
  for (const [i, w] of spec) {
    x += det.pose[i][0] * w;
    y += det.pose[i][1] * w;
  }
  return [x, y];
}

export function anchorBox(name, det) {
  const idx = ANCHOR_BOXES[name];
  if (!idx || !det.pose) return null;
  let pts = idx.map((i) => det.pose[i]);
  if (det.vis) {
    const seen = idx.filter((i) => det.vis[i] > 0.2).map((i) => det.pose[i]);
    if (seen.length >= 2) pts = seen;
  }
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const box = [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
  // too few points seen gives a flattened rectangle: unavailable
  if (box[2] - box[0] < 4 || box[3] - box[1] < 4) return null;
  return box;
}

/** Reliability: the lowest visibility among the anchor's points. */
export function anchorConf(name, det) {
  if (!det.pose) return 0;
  if (!det.vis) return 1;
  if (ANCHOR_BOXES[name]) {
    // an area tolerates a few missed points: 75th percentile
    const v = ANCHOR_BOXES[name].map((i) => det.vis[i]).sort((a, b) => a - b);
    return v[Math.floor(0.75 * (v.length - 1))];
  }
  const spec = ANCHORS[name];
  return spec === undefined ? 0 : Math.min(...indices(spec).map((i) => det.vis[i]));
}

/** Point or box for a note, in pixels, or null when it can't be used. */
export function resolve(name, det) {
  if (ANCHOR_BOXES[name]) return anchorBox(name, det);
  const p = anchorPoint(name, det);
  if (!p) return null;
  if (p[0] < 0 || p[1] < 0 || p[0] > det.w || p[1] > det.h) return null;
  return p;
}
