// The main guarantee: the render depends on no pixel of the face.
//
// Replace the content of the identity area's core (where it is exactly 1),
// render again, and count the pixels that change.

import { test } from "node:test";
import assert from "node:assert/strict";
import { draw, prepare } from "../src/stipple.js";
import { rng } from "../src/rng.js";
import { detection, photo, W, H } from "./scene.js";

function swapCore(rgb, ident, how) {
  const out = rgb.slice();
  const core = [];
  for (let i = 0; i < W * H; i++) if (ident[i] >= 1) core.push(i);
  const R = rng(99);
  if (how === "shuffle") {
    const order = core.slice();
    for (let k = order.length - 1; k > 0; k--) {
      const j = Math.floor(R.random() * (k + 1));
      [order[k], order[j]] = [order[j], order[k]];
    }
    core.forEach((i, k) => {
      for (let c = 0; c < 3; c++) out[3 * i + c] = rgb[3 * order[k] + c];
    });
  } else {
    for (const i of core) for (let c = 0; c < 3; c++) out[3 * i + c] = R.random() * 255;
  }
  return { out, size: core.length };
}

const differing = (a, b) => {
  let n = 0;
  for (let i = 0; i < a.length; i += 4) if (a[i] !== b[i] || a[i + 1] !== b[i + 1] || a[i + 2] !== b[i + 2]) n++;
  return n;
};

const cases = [
  ["face mesh, face area", {}, {}],
  ["face mesh, whole head", {}, { zone: "head" }],
  ["background removed", {}, { bg: "black" }],
  ["background kept", {}, { bg: "keep" }],
  ["no fade", {}, { fade: 0 }],
  ["no face mesh", { face: false }, {}],
  ["no face mesh, whole head", { face: false }, { zone: "head" }],
  ["no outline", { body: false }, {}],
];

for (const [name, detOpts, opts] of cases) {
  for (const how of ["shuffle", "noise"]) {
    test(`${name} (${how}): no pixel changes`, () => {
      const det = detection(detOpts);
      const rgb = photo();
      const A = prepare(rgb, det, opts);
      assert.ok(A, "no identity area found");
      const { out, size } = swapCore(rgb, A.ident, how);
      assert.ok(size > 500, `core too small (${size} px)`);
      const B = prepare(out, det, opts);
      for (const stage of [0.05, 0.5, 0.95]) {
        for (const polarity of ["ink", "light"]) {
          const o = { ...opts, stage, polarity };
          assert.equal(differing(draw(A, o), draw(B, o)), 0, `dissolve ${stage}, ${polarity}`);
        }
      }
    });
  }
}

test("the check does see a change outside the area", () => {
  const det = detection();
  const rgb = photo();
  const A = prepare(rgb, det);
  const other = rgb.slice();
  for (let i = 0; i < W * H; i++) if (A.body[i] > 0.9 && A.ident[i] === 0) other[3 * i] = 255 - other[3 * i];
  assert.ok(differing(draw(A), draw(prepare(other, det))) > 1000);
});

// Head cut off by the top of the image: the fill must not copy edge pixels,
// which are inside the area.
test("head cut off by the edge: no pixel changes", () => {
  const dy = -90;
  const det = detection();
  const up = (p) => [p[0], p[1] + dy];
  det.pose = det.pose.map(up);
  det.faces = det.faces.map((f) => f.map(up));
  const body = new Float32Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (y - dy < H) body[y * W + x] = det.body[(y - dy) * W + x];
  det.body = body;
  const rgb = photo();
  const A = prepare(rgb, det);
  let top = 0;
  for (let x = 0; x < W; x++) if (A.ident[x] >= 1) top++;
  assert.ok(top > 10, "the area does not touch the edge, so the test proves nothing");
  for (const how of ["shuffle", "noise"]) {
    const B = prepare(swapCore(rgb, A.ident, how).out, det);
    assert.equal(differing(draw(A, { stage: 0.3 }), draw(B, { stage: 0.3 })), 0, how);
  }
});

