// Telea (2004) inpainting, following OpenCV's implementation.
//
// The area is filled from its edge inwards (fast marching). Each pixel is
// computed from pixels already known: original ones outside the area, or ones
// already filled. The original values INSIDE the area are never read, which is
// what guarantees that nothing of the face reaches the result.

const KNOWN = 0, BAND = 1, INSIDE = 2, OFF = 3;
const FAR = 1e6;

class Heap {
  constructor() {
    this.t = [];
    this.i = [];
  }
  get size() {
    return this.t.length;
  }
  push(t, i) {
    const a = this.t, b = this.i;
    let k = a.length;
    a.push(t);
    b.push(i);
    while (k > 0) {
      const p = (k - 1) >> 1;
      if (a[p] <= a[k]) break;
      [a[p], a[k]] = [a[k], a[p]];
      [b[p], b[k]] = [b[k], b[p]];
      k = p;
    }
  }
  pop() {
    const a = this.t, b = this.i;
    const top = b[0];
    const lt = a.pop(), li = b.pop();
    if (a.length) {
      a[0] = lt;
      b[0] = li;
      let k = 0;
      for (;;) {
        const l = 2 * k + 1, r = l + 1;
        let m = k;
        if (l < a.length && a[l] < a[m]) m = l;
        if (r < a.length && a[r] < a[m]) m = r;
        if (m === k) break;
        [a[m], a[k]] = [a[k], a[m]];
        [b[m], b[k]] = [b[k], b[m]];
        k = m;
      }
    }
    return top;
  }
}

/**
 * values: Float32Array 0..1, mask: Uint8Array (1 = fill).
 * Returns a copy with the area filled.
 */
export function inpaint(values, mask, w, h, radius) {
  // grid padded by one pixel, so the image edge is handled like the interior
  const W = w + 2, H = h + 2;
  const I = new Float32Array(W * H);
  const f = new Uint8Array(W * H).fill(OFF);
  const T = new Float32Array(W * H).fill(FAR);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const j = (y + 1) * W + x + 1, i = y * w + x;
      if (mask[i]) {
        f[j] = INSIDE;
      } else {
        f[j] = KNOWN;
        T[j] = 0;
        I[j] = values[i] * 255;
      }
    }
  }
  const unknown = (j) => f[j] >= INSIDE;

  const heap = new Heap();
  for (let j = 0; j < W * H; j++) {
    if (f[j] !== KNOWN) continue;
    if (f[j - 1] === INSIDE || f[j + 1] === INSIDE || f[j - W] === INSIDE || f[j + W] === INSIDE) {
      f[j] = BAND;
      heap.push(0, j);
    }
  }

  // Outside, T is minus the (chamfer) distance to the band, so the gradient of
  // T is defined at the edge of the area.
  const range = 2 * radius;
  const d = new Float32Array(W * H).fill(FAR);
  for (let j = 0; j < W * H; j++) if (f[j] === BAND) d[j] = 0;
  const S = Math.SQRT2;
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const j = y * W + x;
      if (f[j] !== KNOWN) continue;
      d[j] = Math.min(d[j], d[j - 1] + 1, d[j - W] + 1, d[j - W - 1] + S, d[j - W + 1] + S);
    }
  }
  for (let y = H - 2; y >= 1; y--) {
    for (let x = W - 2; x >= 1; x--) {
      const j = y * W + x;
      if (f[j] !== KNOWN) continue;
      d[j] = Math.min(d[j], d[j + 1] + 1, d[j + W] + 1, d[j + W + 1] + S, d[j + W - 1] + S);
    }
  }
  for (let j = 0; j < W * H; j++) if (f[j] === KNOWN) T[j] = -Math.min(d[j], range);

  const solve = (a, b) => {
    const ta = T[a], tb = T[b];
    if (!unknown(a)) {
      if (!unknown(b)) {
        return Math.abs(ta - tb) >= 1 ? 1 + Math.min(ta, tb) : (ta + tb + Math.sqrt(2 - (ta - tb) * (ta - tb))) * 0.5;
      }
      return 1 + ta;
    }
    if (!unknown(b)) return 1 + tb;
    return 1 + Math.min(ta, tb);
  };

  const grad = (j, step, A) => {
    if (!unknown(j + step)) return unknown(j - step) ? A[j + step] - A[j] : (A[j + step] - A[j - step]) * 0.5;
    return unknown(j - step) ? 0 : A[j] - A[j - step];
  };
  // OpenCV doubles the central difference for the intensity gradient; kept as is
  const gradI = (j, step) => {
    if (!unknown(j + step)) return unknown(j - step) ? I[j + step] - I[j] : (I[j + step] - I[j - step]) * 2;
    return unknown(j - step) ? 0 : I[j] - I[j - step];
  };

  const r2 = radius * radius;
  while (heap.size) {
    const p = heap.pop();
    f[p] = KNOWN;
    for (const q of [p - W, p - 1, p + 1, p + W]) {
      if (f[q] !== INSIDE) continue;
      const qx = q % W, qy = (q - qx) / W;
      T[q] = Math.min(solve(q - W, q - 1), solve(q + W, q - 1), solve(q - W, q + 1), solve(q + W, q + 1));
      const gx = grad(q, 1, T), gy = grad(q, W, T);

      let Ia = 0, Jx = 0, Jy = 0, s = 1e-20;
      for (let ky = Math.max(1, qy - radius); ky <= Math.min(H - 2, qy + radius); ky++) {
        for (let kx = Math.max(1, qx - radius); kx <= Math.min(W - 2, qx + radius); kx++) {
          const k = ky * W + kx;
          if (unknown(k)) continue;
          const rx = qx - kx, ry = qy - ky;
          const len = rx * rx + ry * ry;
          if (len > r2) continue;
          let dir = rx * gx + ry * gy;
          if (Math.abs(dir) <= 0.01) dir = 1e-6;
          const wt = Math.abs((dir / (len * Math.sqrt(len))) / (1 + Math.abs(T[k] - T[q])));
          Ia += wt * I[k];
          Jx -= wt * gradI(k, 1) * rx;
          Jy -= wt * gradI(k, W) * ry;
          s += wt;
        }
      }
      const v = Ia / s + (Jx + Jy) / (Math.sqrt(Jx * Jx + Jy * Jy) + 1e-20);
      I[q] = Math.min(255, Math.max(0, v));
      f[q] = BAND;
      heap.push(T[q], q);
    }
  }

  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * w + x] = I[(y + 1) * W + x + 1] / 255;
  return out;
}
