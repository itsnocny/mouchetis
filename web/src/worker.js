// The stipple runs here so the page stays responsive.

import { captionPoints, draw, prepare, reconstruct } from "./stipple.js";

let P = null;

self.onmessage = ({ data }) => {
  const { id, type } = data;
  try {
    if (type === "prepare") {
      P = prepare(data.rgb, data.det, data.opts);
      const res = P ? { ok: true, ident: P.ident, caption: captionPoints(P.ident, P.w, P.h), source: P.source } : { ok: false };
      self.postMessage({ id, ...res });
    } else if (type === "draw") {
      const rgba = draw(P, data.opts);
      self.postMessage({ id, rgba }, [rgba.buffer]);
    } else if (type === "reconstruct") {
      const rgba = reconstruct(data.rgba, data.w, data.h);
      self.postMessage({ id, rgba }, [rgba.buffer]);
    }
  } catch (e) {
    self.postMessage({ id, error: String(e && e.stack ? e.stack : e) });
  }
};
