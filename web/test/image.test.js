import { test } from "node:test";
import assert from "node:assert/strict";
import { blur, dilate } from "../src/image.js";
import { rng } from "../src/rng.js";

const random = (n, seed = 3) => {
  const R = rng(seed);
  return Float32Array.from({ length: n }, () => R.random());
};

test("dilate gives the maximum over the square, edges included", () => {
  const w = 37, h = 23, src = random(w * h);
  for (const k of [3, 5, 9, 31]) {
    const got = dilate(src, w, h, k), r = k >> 1;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let m = -Infinity;
        for (let yy = Math.max(0, y - r); yy <= Math.min(h - 1, y + r); yy++)
          for (let xx = Math.max(0, x - r); xx <= Math.min(w - 1, x + r); xx++) m = Math.max(m, src[yy * w + xx]);
        assert.equal(got[y * w + x], m);
      }
    }
  }
});

test("the fast blur stays close to a Gaussian", () => {
  const w = 200, h = 200, src = new Float32Array(w * h);
  src[100 * w + 100] = 1000;
  const b = blur(src, w, h, 12);
  let s = 0, mx = 0, my = 0, vx = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const v = b[y * w + x];
    s += v; mx += v * x; my += v * y;
  }
  mx /= s; my /= s;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) vx += b[y * w + x] * (x - mx) ** 2;
  assert.ok(Math.abs(s - 1000) < 1, `sum ${s}`);
  assert.ok(Math.abs(mx - 100) < 0.01 && Math.abs(my - 100) < 0.01);
  assert.ok(Math.abs(Math.sqrt(vx / s) - 12) < 0.6, `sigma ${Math.sqrt(vx / s)}`);
});
