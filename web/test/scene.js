// A synthetic scene: a body, a head, and a hand-made detection. Enough to run
// the whole stipple without MediaPipe.

import { fillEllipse, plane } from "../src/image.js";
import { rng } from "../src/rng.js";

export const W = 360, H = 480;
export const HEAD = { x: 180, y: 105, rx: 34, ry: 44 };

export function photo(seed = 1) {
  const R = rng(seed);
  const rgb = new Float32Array(W * H * 3);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = 3 * (y * W + x);
      const t = 120 + 60 * Math.sin(x / 23) * Math.cos(y / 31);
      for (let c = 0; c < 3; c++) rgb[i + c] = Math.min(255, Math.max(0, t + 25 * R.normal() + c * 10));
    }
  }
  return rgb;
}

function silhouette() {
  const m = plane(W, H);
  fillEllipse(m, W, H, HEAD.x, HEAD.y + 6, HEAD.rx + 8, HEAD.ry + 12, Math.PI / 2);
  fillEllipse(m, W, H, 180, 300, 150, 80, Math.PI / 2);
  fillEllipse(m, W, H, 180, 165, 14, 22, 0);
  return m;
}

function faceMesh() {
  const pts = [];
  for (let k = 0; k < 478; k++) {
    const a = (k * 2.399963) % (2 * Math.PI), r = Math.sqrt((k + 0.5) / 478);
    pts.push([HEAD.x + Math.cos(a) * r * (HEAD.rx - 4), HEAD.y + Math.sin(a) * r * (HEAD.ry - 4)]);
  }
  pts[10] = [HEAD.x, HEAD.y - HEAD.ry + 4];
  pts[152] = [HEAD.x, HEAD.y + HEAD.ry - 4];
  return pts;
}

function pose() {
  const p = Array.from({ length: 33 }, () => [180, 300]);
  p[0] = [180, 112];
  [1, 2, 3].forEach((i, k) => (p[i] = [168 - 4 * k, 96]));
  [4, 5, 6].forEach((i, k) => (p[i] = [192 + 4 * k, 96]));
  p[7] = [148, 104]; p[8] = [212, 104];
  p[9] = [172, 130]; p[10] = [188, 130];
  p[11] = [120, 190]; p[12] = [240, 190];
  p[23] = [140, 380]; p[24] = [220, 380];
  return p;
}

export function detection({ face = true, body = true } = {}) {
  return {
    w: W, h: H,
    pose: pose(),
    vis: new Array(33).fill(0.9),
    faces: face ? [faceMesh()] : [],
    body: body ? silhouette() : null,
    size: 190,
  };
}
