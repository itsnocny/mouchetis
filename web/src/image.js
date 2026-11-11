// Image operations on Float32Array planes (one value per pixel, row by row).
// Just what the stipple needs, without OpenCV.

export const plane = (w, h, fill = 0) => {
  const p = new Float32Array(w * h);
  if (fill) p.fill(fill);
  return p;
};

export const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

// the sigma OpenCV derives from a kernel size
export const sigmaOf = (k) => 0.3 * ((k - 1) * 0.5 - 1) + 0.8;

function kernel(sigma) {
  const r = Math.max(1, Math.ceil(sigma * 3));
  const k = new Float32Array(2 * r + 1);
  let s = 0;
  for (let i = -r; i <= r; i++) s += k[i + r] = Math.exp(-(i * i) / (2 * sigma * sigma));
  for (let i = 0; i < k.length; i++) k[i] /= s;
  return { k, r };
}

// mirrored border without repeating the edge pixel, like BORDER_REFLECT_101
const reflect = (i, n) => {
  if (n === 1) return 0;
  while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - 2 - i;
  return i;
};

// Running mean of radius r along rows, mirrored borders.
function boxPass(src, dst, w, h, r) {
  const n = 2 * r + 1;
  for (let y = 0; y < h; y++) {
    const o = y * w;
    let s = 0;
    for (let i = -r; i <= r; i++) s += src[o + reflect(i, w)];
    for (let x = 0; x < w; x++) {
      dst[o + x] = s / n;
      s += src[o + reflect(x + r + 1, w)] - src[o + reflect(x - r, w)];
    }
  }
}

function transpose(src, w, h) {
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[x * h + y] = src[y * w + x];
  return out;
}

// Three successive running means approximate a Gaussian at constant cost per
// pixel. Radii follow Kovesi (2010).
function boxBlur(src, w, h, sigma) {
  const wl = Math.floor(Math.sqrt((12 * sigma * sigma) / 3 + 1));
  const wl1 = wl % 2 ? wl : wl - 1;
  const m =Math.round((12 * sigma * sigma - 3 * wl1 * wl1 - 12 * wl1 - 9) / (-4 * wl1 - 4));
  const radii = [0, 1, 2].map((i) => ((i < m ? wl1 : wl1 + 2) - 1) >> 1);
  let a = src.slice(), b = new Float32Array(w * h);
  for (const r of radii) {
    boxPass(a, b, w, h, r);
    [a, b] = [b, a];
  }
  a = transpose(a, w, h);
  b = new Float32Array(w * h);
  for (const r of radii) {
    boxPass(a, b, h, w, r);
    [a, b] = [b, a];
  }
  return transpose(a, h, w);
}

export function blur(src, w, h, sigma) {
  if (sigma <= 0) return src.slice();
  if (sigma > 6) return boxBlur(src, w, h, sigma);
  const { k, r } = kernel(sigma);
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);
  const row = new Float32Array(w + 2 * r);
  for (let y = 0; y < h; y++) {
    const o = y * w;
    for (let x = -r; x < w + r; x++) row[x + r] = src[o + reflect(x, w)];
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let i = 0; i < k.length; i++) s += row[x + i] * k[i];
      tmp[o + x] = s;
    }
  }
  const col = new Float32Array(h + 2 * r);
  for (let x = 0; x < w; x++) {
    for (let y = -r; y < h + r; y++) col[y + r] = tmp[reflect(y, h) * w + x];
    for (let y = 0; y < h; y++) {
      let s = 0;
      for (let i = 0; i < k.length; i++) s += col[y + i] * k[i];
      out[y * w + x] = s;
    }
  }
  return out;
}

// Downscale by averaging the covered pixels (INTER_AREA). Shrinking only.
export function shrink(src, w, h, nw, nh) {
  const out = new Float32Array(nw * nh);
  const sx = w / nw, sy = h / nh;
  for (let y = 0; y < nh; y++) {
    const y0 = Math.floor(y * sy), y1 = Math.max(y0 + 1, Math.floor((y + 1) * sy));
    for (let x = 0; x < nw; x++) {
      const x0 = Math.floor(x * sx), x1 = Math.max(x0 + 1, Math.floor((x + 1) * sx));
      let s = 0;
      for (let yy = y0; yy < y1; yy++) {
        const o = yy * w;
        for (let xx = x0; xx < x1; xx++) s += src[o + xx];
      }
      out[y * nw + x] = s / ((y1 - y0) * (x1 - x0));
    }
  }
  return out;
}

export function stretch(src, w, h, nw, nh) {
  const out = new Float32Array(nw * nh);
  const sx = w / nw, sy = h / nh;
  for (let y = 0; y < nh; y++) {
    const fy = clamp((y + 0.5) * sy - 0.5, 0, h - 1);
    const y0 = Math.floor(fy), y1 = Math.min(h - 1, y0 + 1), ty = fy - y0;
    for (let x = 0; x < nw; x++) {
      const fx = clamp((x + 0.5) * sx - 0.5, 0, w - 1);
      const x0 = Math.floor(fx), x1 = Math.min(w - 1, x0 + 1), tx = fx - x0;
      const a = src[y0 * w + x0] * (1 - tx) + src[y0 * w + x1] * tx;
      const b = src[y1 * w + x0] * (1 - tx) + src[y1 * w + x1] * tx;
      out[y * nw + x] = a * (1 - ty) + b * ty;
    }
  }
  return out;
}

export function nearest(src, w, h, nw, nh) {
  const out = new src.constructor(nw * nh);
  for (let y = 0; y < nh; y++) {
    const o = Math.min(h - 1, Math.floor((y * h) / nh)) * w;
    for (let x = 0; x < nw; x++) out[y * nw + x] = src[o + Math.min(w - 1, Math.floor((x * w) / nw))];
  }
  return out;
}

// Running maximum of radius r along a line (van Herk / Gil-Werman): constant
// cost per pixel whatever the kernel size.
// The line is padded with -Infinity so every window is complete.
function maxLine(src, n, r, out, pad, pre, suf) {
  const k = 2 * r + 1, m = n + 2 * r;
  pad.fill(-Infinity, 0, m);
  for (let i = 0; i < n; i++) pad[i + r] = src[i];
  for (let i = 0; i < m; i++) pre[i] = i % k === 0 ? pad[i] : Math.max(pre[i - 1], pad[i]);
  for (let i = m - 1; i >= 0; i--) suf[i] = i === m - 1 || (i + 1) % k === 0 ? pad[i] : Math.max(suf[i + 1], pad[i]);
  for (let i = 0; i < n; i++) out[i] = Math.max(suf[i], pre[i + 2 * r]);
}

// Dilation by a k x k square (k odd), in two passes.
export function dilate(src, w, h, k) {
  const r = k >> 1;
  const out = new Float32Array(w * h);
  if (r < 1) {
    out.set(src);
    return out;
  }
  const n = Math.max(w, h), m = n + 2 * r;
  const line = new Float32Array(n), res = new Float32Array(n);
  const pad = new Float32Array(m), pre = new Float32Array(m), suf = new Float32Array(m);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) line[x] = src[y * w + x];
    maxLine(line, w, r, res, pad, pre, suf);
    for (let x = 0; x < w; x++) out[y * w + x] = res[x];
  }
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) line[y] = out[y * w + x];
    maxLine(line, h, r, res, pad, pre, suf);
    for (let y = 0; y < h; y++) out[y * w + x] = res[y];
  }
  return out;
}

// Dilation of a binary mask by a disc of (small) radius r.
export function dilateDisc(mask, w, h, r) {
  const out = new Uint8Array(w * h);
  const offs = [];
  for (let dy = -r; dy <= r; dy++)
    for (let dx = -r; dx <= r; dx++) if (dx * dx + dy * dy <= r * r + r) offs.push([dx, dy]);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!mask[y * w + x]) continue;
      for (const [dx, dy] of offs) {
        const xx = x + dx, yy = y + dy;
        if (xx >= 0 && yy >= 0 && xx < w && yy < h) out[yy * w + xx] = 1;
      }
    }
  }
  return out;
}

export function convexHull(pts) {
  const p = pts.map(([x, y]) => [x, y]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (p.length < 3) return p;
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo = [], hi = [];
  for (const q of p) {
    while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop();
    lo.push(q);
  }
  for (let i = p.length - 1; i >= 0; i--) {
    const q = p[i];
    while (hi.length >= 2 && cross(hi[hi.length - 2], hi[hi.length - 1], q) <= 0) hi.pop();
    hi.push(q);
  }
  return lo.slice(0, -1).concat(hi.slice(0, -1));
}

// Fills a convex polygon: on each row, from the first crossing to the last.
export function fillConvex(out, w, h, poly, value = 1) {
  if (poly.length < 3) return out;
  const ys = poly.map((q) => q[1]);
  const y0 = Math.max(0, Math.floor(Math.min(...ys))), y1 = Math.min(h - 1, Math.ceil(Math.max(...ys)));
  for (let y = y0; y <= y1; y++) {
    const cy = y + 0.5;
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < poly.length; i++) {
      const [ax, ay] = poly[i], [bx, by] = poly[(i + 1) % poly.length];
      if ((ay <= cy && by > cy) || (by <= cy && ay > cy)) {
        const x = ax + ((cy - ay) / (by - ay)) * (bx - ax);
        lo = Math.min(lo, x);
        hi = Math.max(hi, x);
      }
    }
    if (lo > hi) continue;
    for (let x = Math.max(0, Math.ceil(lo - 0.5)); x <= Math.min(w - 1, Math.floor(hi - 0.5)); x++) out[y * w + x] = value;
  }
  return out;
}

// Filled ellipse with semi-axes a, b, rotated by `ang` radians.
export function fillEllipse(out, w, h, cx, cy, a, b, ang, value = 1) {
  const c = Math.cos(ang), s = Math.sin(ang), r = Math.max(a, b);
  for (let y = Math.max(0, Math.floor(cy - r)); y <= Math.min(h - 1, Math.ceil(cy + r)); y++) {
    for (let x = Math.max(0, Math.floor(cx - r)); x <= Math.min(w - 1, Math.ceil(cx + r)); x++) {
      const dx = x - cx, dy = y - cy;
      const u = (dx * c + dy * s) / a, v = (-dx * s + dy * c) / b;
      if (u * u + v * v <= 1) out[y * w + x] = value;
    }
  }
  return out;
}

// Connected components (8-neighbourhood) of a binary mask.
export function components(mask, w, h) {
  const lab = new Int32Array(w * h);
  const stack = new Int32Array(w * h);
  let n = 0;
  for (let i = 0; i < w * h; i++) {
    if (!mask[i] || lab[i]) continue;
    lab[i] = ++n;
    let top = 0;
    stack[top++] = i;
    while (top) {
      const j = stack[--top], x = j % w, y = (j - x) / w;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx, yy = y + dy;
          if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
          const k = yy * w + xx;
          if (mask[k] && !lab[k]) {
            lab[k] = n;
            stack[top++] = k;
          }
        }
      }
    }
  }
  return { lab, n };
}

// Keeps the component containing (x, y), or the largest if the point is outside.
export function keepComponent(mask, w, h, x, y) {
  const { lab, n } = components(mask, w, h);
  let l = lab[clamp(Math.round(y), 0, h - 1) * w + clamp(Math.round(x), 0, w - 1)];
  if (!l && n) {
    const sizes = new Int32Array(n + 1);
    for (let i = 0; i < lab.length; i++) sizes[lab[i]]++;
    sizes[0] = 0;
    l = sizes.indexOf(Math.max(...sizes));
  }
  const out = new Float32Array(w * h);
  if (!l) {
    for (let i = 0; i < out.length; i++) out[i] = mask[i] ? 1 : 0;
    return out;
  }
  for (let i = 0; i < out.length; i++) out[i] = lab[i] === l ? 1 : 0;
  return out;
}

// Percentiles of 0..1 values under a mask, from a histogram.
export function percentiles(x, sel, qs) {
  const bins = 4096;
  const hist = new Uint32Array(bins);
  let n = 0;
  for (let i = 0; i < x.length; i++) {
    if (!sel[i]) continue;
    hist[Math.min(bins - 1, Math.max(0, Math.floor(x[i] * bins)))]++;
    n++;
  }
  return qs.map((q) => {
    const target = (q / 100) * (n - 1);
    let acc = 0;
    for (let b = 0; b < bins; b++) {
      acc += hist[b];
      if (acc > target) return (b + 0.5) / bins;
    }
    return 1;
  });
}

export const median = (x, sel) => percentiles(x, sel, [50])[0];

export function luminance(rgb, n) {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = (0.2126 * rgb[3 * i] + 0.7152 * rgb[3 * i + 1] + 0.0722 * rgb[3 * i + 2]) / 255;
  return out;
}

export function channel(rgb, n, c) {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = rgb[3 * i + c];
  return out;
}

export const norm = (v) => Math.hypot(v[0], v[1]);
export const sub = (a, b) => [a[0] - b[0], a[1] - b[1]];
export const mean = (pts) => {
  let x = 0, y = 0;
  for (const p of pts) {
    x += p[0];
    y += p[1];
  }
  return [x / pts.length, y / pts.length];
};

// Smooth random field with no link at all to the photo.
export function structureless(w, h, level, scale, R, spread = 0.16) {
  const sc = Math.max(2, scale);
  const gw = Math.max(2, Math.trunc(w / sc)), gh = Math.max(2, Math.trunc(h / sc));
  const noise = new Float32Array(gw * gh);
  for (let i = 0; i < noise.length; i++) noise[i] = R.random();
  const f = blur(stretch(noise, gw, gh, w, h), w, h, sc * 0.6);
  let s = 0, s2 = 0;
  for (const v of f) {
    s += v;
    s2 += v * v;
  }
  const mu = s / f.length;
  const sd = Math.sqrt(Math.max(0, s2 / f.length - mu * mu)) || 1;
  for (let i = 0; i < f.length; i++) f[i] = clamp(((f[i] - mu) / sd) * spread + level, 0, 1);
  return f;
}

// Fills any polygon (even-odd rule), row by row.
export function fillPolygon(out, w, h, poly, value = 1) {
  const n = poly.length;
  if (n < 3) return out;
  const ys = poly.map((q) => q[1]);
  const y0 = Math.max(0, Math.floor(Math.min(...ys))), y1 = Math.min(h - 1, Math.ceil(Math.max(...ys)));
  const xs = [];
  for (let y = y0; y <= y1; y++) {
    const cy = y + 0.5;
    xs.length = 0;
    for (let i = 0; i < n; i++) {
      const [ax, ay] = poly[i], [bx, by] = poly[(i + 1) % n];
      if ((ay <= cy && by > cy) || (by <= cy && ay > cy)) xs.push(ax + ((cy - ay) / (by - ay)) * (bx - ax));
    }
    xs.sort((a, b) => a - b);
    for (let k = 0; k + 1 < xs.length; k += 2) {
      for (let x = Math.max(0, Math.ceil(xs[k] - 0.5)); x <= Math.min(w - 1, Math.floor(xs[k + 1] - 0.5)); x++) out[y * w + x] = value;
    }
  }
  return out;
}

// A thick segment (capsule of radius r) drawn into a mask.
export function fillSegment(out, w, h, a, b, r, value = 1) {
  const dx = b[0] - a[0], dy = b[1] - a[1], len2 = dx * dx + dy * dy || 1e-9;
  const x0 = Math.max(0, Math.floor(Math.min(a[0], b[0]) - r)), x1 = Math.min(w - 1, Math.ceil(Math.max(a[0], b[0]) + r));
  const y0 = Math.max(0, Math.floor(Math.min(a[1], b[1]) - r)), y1 = Math.min(h - 1, Math.ceil(Math.max(a[1], b[1]) + r));
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const t = clamp(((x + 0.5 - a[0]) * dx + (y + 0.5 - a[1]) * dy) / len2, 0, 1);
      const px = a[0] + t * dx - x - 0.5, py = a[1] + t * dy - y - 0.5;
      if (px * px + py * py <= r * r) out[y * w + x] = value;
    }
  }
  return out;
}
