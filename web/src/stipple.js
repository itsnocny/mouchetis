// The stipple, ported from mouchetis/stipple.py. No browser dependency, so
// the tests run under Node.
//
// A detection is a plain object:
//   { w, h, pose: [[x, y]] x 33 | null, vis: [] | null,
//     faces: [[[x, y]] x 478], body: Float32Array 0/1 | null, size }

import {
  blur, channel, clamp, convexHull, dilate, dilateDisc, fillConvex, fillEllipse,
  keepComponent, luminance, mean, median, nearest, norm, percentiles, plane,
  shrink, sigmaOf, stretch, structureless, sub,
} from "./image.js";
import { inpaint } from "./telea.js";
import { rng } from "./rng.js";

export const PAPER = [236, 232, 224];
export const INK = [24, 22, 21];

const NOSE = 0, L_SHO = 11, R_SHO = 12;
const HEAD_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
// face mesh: top of the forehead and chin
const F_TOP = 10, F_CHIN = 152;

export const DEFAULTS = {
  stage: 0.25, grain: 1.8, faceGrain: 1, density: 1, gamma: 1,
  polarity: "ink", bg: "blur", zone: "face", fade: 0.35, grow: 1,
  bgContrast: 0.55, bgDensity: 0.15, blotch: 0.05, faceFloor: 1, seed: 7,
};

const dot = (a, b) => a[0] * b[0] + a[1] * b[1];

function feather(m, w, h, px) {
  const k = Math.max(3, Math.trunc(px) | 1);
  const out = blur(m, w, h, sigmaOf(k));
  for (let i = 0; i < out.length; i++) out[i] = clamp(out[i], 0, 1);
  return out;
}

// Softens the edge without eating into the hard area: a plain feather erodes
// the core, which is then no longer fully destroyed.
function soften(m, w, h, px) {
  const f = feather(m, w, h, px);
  for (let i = 0; i < f.length; i++) f[i] = Math.max(m[i], f[i]);
  return f;
}

// Core kept at exactly 1, then a fade outwards.
function withFade(m, w, h, scale, fade) {
  if (fade <= 0) return m;
  const b = blur(m, w, h, Math.max(2, fade * Math.max(scale, 8)));
  for (let i = 0; i < b.length; i++) b[i] = Math.max(m[i], b[i]);
  return b;
}

const headAxis = (pose) => {
  const sho = [(pose[L_SHO][0] + pose[R_SHO][0]) / 2, (pose[L_SHO][1] + pose[R_SHO][1]) / 2];
  const up = sub(pose[NOSE], sho);
  return { sho, up, n: norm(up) };
};

function headPoints(det) {
  const pts = HEAD_IDX.map((i) => det.pose[i]);
  if (!det.vis) return pts;
  const keep = pts.filter((_, k) => det.vis[HEAD_IDX[k]] > 0.2);
  return keep.length >= 3 ? keep : pts;
}

// Area of one face-mesh face. In "head" mode it is cut out of the person's
// outline, hair included, and cut off below the chin.
function faceZone(f, det, mode, grow) {
  const { w, h } = det;
  let covered = fillConvex(plane(w, h), w, h, convexHull(f));
  const c = mean(f);
  const r = Math.max(...f.map((p) => norm(sub(p, c))));
  // the face mesh already spans from the hairline to the chin
  const pad = Math.max(3, Math.trunc(r * (mode === "face" ? 0.07 : 0.14) * grow)) | 1;
  covered = dilate(covered, w, h, pad);
  if (mode === "face" || !det.body || f.length <= F_CHIN) return covered;

  const top = f[F_TOP], chin = f[F_CHIN];
  let up = sub(top, chin);
  const n = norm(up);
  if (n <= 1) return covered;
  up = [up[0] / n, up[1] / n];
  const side = [-up[1], up[0]];
  const across = Math.max(...f.map((p) => Math.abs(dot(sub(p, chin), side))));
  const cut = [chin[0] - up[0] * 0.07 * n * grow, chin[1] - up[1] * 0.07 * n * grow];

  const band = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (det.body[i] <= 0.5) continue;
      const above = (x - cut[0]) * up[0] + (y - cut[1]) * up[1] > 0;
      const lat = Math.abs((x - chin[0]) * side[0] + (y - chin[1]) * side[1]);
      band[i] = above && lat < across * 2.1 * grow ? 1 : 0;
    }
  }
  const zone = keepComponent(band, w, h, c[0], c[1]);
  for (let i = 0; i < zone.length; i++) zone[i] = Math.min(1, zone[i] + covered[i]);
  return zone;
}

// Without a face mesh: an ellipse over the pose's head landmarks, clipped by
// the outline unless the outline cuts too much of it.
function faceZoneFromPose(det, grow) {
  const { w, h, pose } = det;
  const pts = headPoints(det);
  if (pts.length < 3) return null;
  const { up: u, n } = headAxis(pose);
  if (n < 8) return null;
  const up = [u[0] / n, u[1] / n], side = [-up[1], up[0]];
  const c = mean(pts);
  const along = Math.max(...pts.map((p) => Math.abs(dot(sub(p, c), up))));
  const across = Math.max(...pts.map((p) => Math.abs(dot(sub(p, c), side))));
  const a = Math.max(along * 2, 0.13 * n) * grow;
  const b = Math.max(across * 1.12, 0.1 * n) * grow;
  const zone = fillEllipse(plane(w, h), w, h, c[0], c[1], Math.trunc(a), Math.trunc(b), Math.atan2(up[1], up[0]));
  if (!det.body) return zone;
  const clipped = zone.map((v, i) => (v && det.body[i] > 0.5 ? 1 : 0));
  let zs = 0, cs = 0;
  for (let i = 0; i < zone.length; i++) {
    zs += zone[i];
    cs += clipped[i];
  }
  return cs > 0.45 * zs ? clipped : zone;
}

// Without a face mesh, in "head" mode: the head cut out of the outline,
// along the shoulders-to-nose axis.
function headZoneFromPose(det, grow) {
  const { w, h, pose } = det;
  if (!det.body) return null;
  const { sho, up: u, n: d } = headAxis(pose);
  if (d < 8) return null;
  const up = [u[0] / d, u[1] / d], side = [-up[1], up[0]];
  const nose = pose[NOSE];
  const across = Math.max(0.16 * d, ...headPoints(det).map((p) => Math.abs(dot(sub(p, nose), side))));
  const k = 0.45 / Math.max(0.3, grow);
  const cut = [sho[0] + up[0] * k * d, sho[1] + up[1] * k * d];
  const band = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (det.body[i] <= 0.5) continue;
      const above = (x - cut[0]) * up[0] + (y - cut[1]) * up[1] > 0;
      const lat = Math.abs((x - nose[0]) * side[0] + (y - nose[1]) * side[1]);
      band[i] = above && lat < across * 2.9 * grow ? 1 : 0;
    }
  }
  const zone = keepComponent(band, w, h, nose[0], nose[1]);
  // feathering pushes the edge below the threshold, so dilate to cover the
  // head up to its outline
  return dilate(zone, w, h, Math.max(3, Math.trunc(0.028 * d)) | 1);
}

const peak = (m) => m.reduce((a, v) => (v > a ? v : a), 0);

/**
 * Identity area. Returns { head, zones }, or null when no head can be located,
 * in which case no image must be produced.
 */
export function headMask(det, { zone: mode = "face", fade = 0.35, grow = 1 } = {}) {
  const { w, h } = det;
  let m = plane(w, h);
  const zones = [];
  const heights = [];

  for (const f of det.faces) {
    if (f.length <= 10) continue;
    const z = faceZone(f, det, mode, grow);
    const fh = f.length > F_CHIN ? norm(sub(f[F_TOP], f[F_CHIN])) : det.size;
    heights.push(fh);
    zones.push(withFade(soften(z, w, h, Math.max(3, 0.07 * Math.max(fh, 8))), w, h, fh, fade));
    for (let i = 0; i < m.length; i++) m[i] = Math.min(1, m[i] + z[i]);
  }

  if (peak(m) < 0.5 && det.pose) {
    const d = headAxis(det.pose).n;
    const soft = Math.max(2.5, 0.022 * d);
    if (mode === "face") {
      const z = faceZoneFromPose(det, grow);
      if (z && peak(z) > 0.5) {
        zones.push(withFade(soften(z, w, h, soft), w, h, d * 0.3, fade));
        return { head: withFade(soften(z, w, h, soft), w, h, d * 0.3, fade), zones, source: "pose" };
      }
    }
    const z = headZoneFromPose(det, grow);
    if (z && peak(z) > 0.5) {
      const f = feather(z, w, h, soft);
      return { head: f, zones: [f], source: "pose" };
    }
  }

  if (peak(m) < 0.5) return null;
  const fh = heights.reduce((a, b) => a + b, 0) / heights.length;
  m = withFade(soften(m, w, h, Math.max(3, 0.07 * Math.max(fh, 8))), w, h, fh, fade);
  return { head: m, zones: zones.length ? zones : [m], source: "face" };
}

export function bodyMask(det) {
  const { w, h } = det;
  if (det.body) return feather(det.body, w, h, 0.04 * det.size);
  let m = plane(w, h);
  if (det.pose) {
    fillConvex(m, w, h, convexHull(det.pose));
    m = dilate(m, w, h, Math.max(5, Math.trunc(0.12 * det.size)) | 1);
  }
  return feather(m, w, h, 0.06 * det.size);
}

// Destroys the background's detail without blacking it out: shrink to 5 %,
// then blur. The identity area is removed before shrinking, otherwise a
// smeared trace of the face would end up in the background.
function killBackground(rgb, w, h, keep, ident) {
  const n = w * h;
  const out = new Float32Array(n * 3);
  const outside = new Uint8Array(n);
  let any = false;
  for (let i = 0; i < n; i++) if (ident[i] <= 0.5) outside[i] = any = 1;
  const sw = Math.max(1, Math.trunc(w * 0.05)), sh = Math.max(1, Math.trunc(h * 0.05));
  const k = Math.max(3, Math.trunc(Math.min(w, h) * 0.025) | 1);
  for (let c = 0; c < 3; c++) {
    const src = channel(rgb, n, c);
    if (any) {
      const med = median(src.map((v) => v / 255), outside) * 255;
      for (let i = 0; i < n; i++) if (!outside[i]) src[i] = med;
    }
    const back = blur(stretch(shrink(src, w, h, sw, sh), sw, sh, w, h), w, h, sigmaOf(k));
    for (let i = 0; i < n; i++) out[3 * i + c] = rgb[3 * i + c] * keep[i] + back[i] * (1 - keep[i]);
  }
  return out;
}

/**
 * The slow part, which depends on neither dissolve nor dot size: background,
 * ink density and filling of the identity areas.
 */
export function prepare(rgb, det, opts = {}) {
  const o = { ...DEFAULTS, ...opts };
  const { w, h } = det;
  const n = w * h;
  const found = headMask(det, o);
  if (!found) return null;
  const { head, zones } = found;
  const ident = head.map((v, i) => clamp(v + (o.extra ? o.extra[i] : 0), 0, 1));
  const body = bodyMask(det);
  const subject = body.map((v, i) => clamp(v + ident[i], 0, 1));

  let base = Float32Array.from(rgb);
  if (o.bg === "blur") base = killBackground(base, w, h, subject, ident);
  else if (o.bg === "black") for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) base[3 * i + c] *= subject[i];

  const lum = luminance(base, n);
  const raw = lum.map((v) => clamp(Math.pow(1 - v, o.gamma), 0, 1));

  // levels set on the body alone, identity area excluded
  const bodyOnly = new Uint8Array(n);
  for (let i = 0; i < n; i++) bodyOnly[i] = body[i] - ident[i] > 0.5 ? 1 : 0;
  let inkTrue = raw;
  const count = bodyOnly.reduce((a, v) => a + v, 0);
  if (count >= 64) {
    const [lo, hi] = percentiles(raw, bodyOnly, [2, 98]);
    if (hi - lo >= 1e-3) inkTrue = raw.map((v) => clamp((v - lo) / (hi - lo), 0, 1));
  }

  // the background is compressed, not stretched: stretching turns the edge
  // of a mirror (bright glass, dark room) into a hard break
  const out = subject.map((v) => (1 - v > 0.5 ? 1 : 0));
  const med = out.some((v) => v) ? median(raw, out) : median(raw, new Uint8Array(n).fill(1));
  const d = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const back = clamp(0.5 + (raw[i] - med) * o.bgContrast, 0, 1);
    d[i] = inkTrue[i] * subject[i] + back * o.bgDensity * (1 - subject[i]);
  }

  // the texture around each area, to match what replaces it
  const stats = (zone) => {
    let sel = bodyOnly;
    if (zone) {
      const hard = zone.map((v) => (v > 0.5 ? 1 : 0));
      const ring = dilate(hard, w, h, Math.max(3, Math.trunc(det.size * 0.45)) | 1);
      const s = new Uint8Array(n);
      let c = 0;
      for (let i = 0; i < n; i++) if (ring[i] && !hard[i] && bodyOnly[i]) s[i] = 1, c++;
      if (c >= 256) sel = s;
    }
    let s = 0, s2 = 0, c = 0;
    for (let i = 0; i < n; i++) {
      if (!sel[i]) continue;
      s += inkTrue[i];
      s2 += inkTrue[i] * inkTrue[i];
      c++;
    }
    if (c < 256) return { level: 0.5, spread: 0.16 };
    const mu = s / c;
    return { level: mu, spread: clamp(Math.sqrt(Math.max(0, s2 / c - mu * mu)), 0.1, 0.34) };
  };

  // Each area is filled by extending the texture around it. The fill reads
  // only pixels outside the area.
  const radius = Math.trunc(clamp(det.size * 0.06, 3, 24));
  const fills = zones.map((z) => {
    const m = z.map((v) => (v > 0.5 ? 1 : 0));
    const k = m.reduce((a, v) => a + v, 0);
    if (!k || k === n) return { zone: z, fill: null, stats: stats(z) };
    return { zone: z, fill: inpaint(inkTrue, Uint8Array.from(m), w, h, radius) };
  });

  return { w, h, ident, body, d, fills, stats: stats(null), size: det.size, source: found.source };
}

// Stochastic screen: one dot per cell, drawn with p = local tone.
function marks(d, w, h, gate, unit, density, R) {
  const gw = Math.max(1, Math.trunc(w / unit)), gh = Math.max(1, Math.trunc(h / unit));
  const cell = shrink(d, w, h, gw, gh);
  const hit = new Uint8Array(gw * gh);
  for (let i = 0; i < hit.length; i++) hit[i] = R.random() < clamp(cell[i] * density, 0, 1) ? 1 : 0;
  const full = nearest(hit, gw, gh, w, h);
  for (let i = 0; i < full.length; i++) full[i] &= gate[i];
  return full;
}

/**
 * Two sealed passes: the identity area is stippled on its own, coarser. Its
 * dots may spill outwards, but nothing gets in.
 */
function stipple(d, ident, w, h, o, R) {
  const unit = Math.max(1, (Math.min(w, h) * o.grain) / 1200);
  const n = w * h;
  const hard = new Uint8Array(n), soft = new Uint8Array(n);
  const dIn = new Float32Array(n), dOut = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    hard[i] = ident[i] > 0.53 ? 1 : 0;
    soft[i] = 1 - hard[i];
    if (hard[i]) dIn[i] = d[i];
    else dOut[i] = d[i];
  }
  const spill = dilateDisc(hard, w, h, Math.max(1, Math.round(unit)));
  const unitIn = Math.max(1, (Math.min(w, h) * o.grain * o.faceGrain) / 1200);
  const a = marks(dIn, w, h, spill, unitIn, o.density, R);
  const b = marks(dOut, w, h, soft, unit, o.density, R);
  for (let i = 0; i < n; i++) a[i] |= b[i];
  return a;
}

/** The render at a given dissolve, as RGBA ready for a canvas. */
export function draw(P, opts = {}) {
  const o = { ...DEFAULTS, ...opts };
  const { w, h, ident, body } = P;
  const n = w * h;
  const R = rng(o.seed);

  const sc = P.size * o.blotch;
  const flat = structureless(w, h, P.stats.level, sc, R, P.stats.spread);
  for (const { zone, fill, stats } of P.fills) {
    const f = fill || structureless(w, h, stats.level, sc, R, stats.spread);
    for (let i = 0; i < n; i++) if (zone[i] > 0.5) flat[i] = f[i];
  }

  // the body drifts towards formless with the dissolve; the core of the
  // identity area is always at 1
  const floor = Math.max(o.stage, o.faceFloor);
  const st = clamp(o.stage, 0, 1);
  const d = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const mix = Math.max(st * body[i], clamp(ident[i], 0, 1) * floor);
    d[i] = clamp(P.d[i] * (1 - mix) + flat[i] * mix, 0, 1);
  }

  const ink = stipple(d, ident, w, h, o, R);
  let paper = PAPER, dark = INK;
  if (o.polarity === "light") [paper, dark] = [dark, paper];
  const rgba = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) {
    const a = ink[i] * (0.86 + 0.14 * R.random());
    const noise = R.normal() * 1.5;
    for (let c = 0; c < 3; c++) rgba[4 * i + c] = paper[c] * (1 - a) + dark[c] * a + noise;
    rgba[4 * i + 3] = 255;
  }
  return rgba;
}

/**
 * What someone trying to undo the dots would get: a three-scale low-pass,
 * then histogram equalization. Grey RGBA.
 */
export function reconstruct(rgba, w, h) {
  const n = w * h;
  const lum = new Float32Array(n);
  for (let i = 0; i < n; i++) lum[i] = (0.2126 * rgba[4 * i] + 0.7152 * rgba[4 * i + 1] + 0.0722 * rgba[4 * i + 2]) / 255;
  const base = Math.min(w, h);
  const acc = new Float32Array(n);
  for (const [k, frac] of [[0.5, 0.004], [0.3, 0.008], [0.2, 0.016]]) {
    const b = blur(lum, w, h, Math.max(1, base * frac));
    for (let i = 0; i < n; i++) acc[i] += k * b[i];
  }
  let lo = Infinity, hi = -Infinity;
  for (const v of acc) {
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
  }
  const hist = new Uint32Array(256);
  const q = new Uint8Array(n);
  for (let i = 0; i < n; i++) hist[(q[i] = Math.round(((acc[i] - lo) / (hi - lo || 1)) * 255))]++;
  const cdf = new Float32Array(256);
  let s = 0;
  for (let v = 0; v < 256; v++) cdf[v] = s += hist[v];
  const c0 = cdf.find((v) => v > 0);
  const out = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) {
    const g = Math.round(((cdf[q[i]] - c0) / Math.max(1, n - c0)) * 255);
    out[4 * i] = out[4 * i + 1] = out[4 * i + 2] = g;
    out[4 * i + 3] = 255;
  }
  return out;
}

/**
 * Points on the outline of the erased area, for the caption. The outline is
 * already visible in the image, so these numbers add nothing.
 */
export function captionPoints(ident, w, h, count = 3) {
  const pts = [];
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      if (ident[i] <= 0.53) continue;
      if (ident[i - 1] <= 0.53 || ident[i + 1] <= 0.53 || ident[i - w] <= 0.53 || ident[i + w] <= 0.53) pts.push([x, y]);
    }
  }
  if (!pts.length) return [];
  const c = mean(pts);
  // sorted clockwise from the top
  const ang = (p) => (Math.atan2(p[0] - c[0], c[1] - p[1]) + 2 * Math.PI) % (2 * Math.PI);
  pts.sort((a, b) => ang(a) - ang(b));
  return Array.from({ length: count }, (_, k) => pts[Math.floor((k * pts.length) / count)]);
}
