// MediaPipe detection, in the browser. The models are downloaded once; the
// photo is never sent anywhere.

const VERSION = "1.2.2";
const CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}`;
const MODELS = "https://storage.googleapis.com/mediapipe-models";
const URLS = {
  pose: `${MODELS}/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task`,
  face: `${MODELS}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`,
  seg: `${MODELS}/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite`,
};

const L_SHO = 11, R_SHO = 12, L_HIP = 23, R_HIP = 24;
const HEAD_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

let tasks = null;

export async function load() {
  if (tasks) return tasks;
  const mp = await import(`${CDN}/vision_bundle.mjs`);
  const files = await mp.FilesetResolver.forVisionTasks(`${CDN}/wasm`);
  const base = (url) => ({ modelAssetPath: url, delegate: "CPU" });
  const [pose, face, seg] = await Promise.all([
    mp.PoseLandmarker.createFromOptions(files, {
      baseOptions: base(URLS.pose), runningMode: "IMAGE", numPoses: 1,
      minPoseDetectionConfidence: 0.3, minPosePresenceConfidence: 0.3,
    }),
    mp.FaceLandmarker.createFromOptions(files, {
      baseOptions: base(URLS.face), runningMode: "IMAGE", numFaces: 3,
      minFaceDetectionConfidence: 0.25, minFacePresenceConfidence: 0.25,
    }),
    mp.ImageSegmenter.createFromOptions(files, {
      baseOptions: base(URLS.seg), runningMode: "IMAGE",
      outputCategoryMask: true, outputConfidenceMasks: false,
    }),
  ]);
  tasks = { pose, face, seg };
  return tasks;
}

const toPx = (lms, w, h, k = 1, x0 = 0, y0 = 0) => lms.map((l) => [(l.x * w) / k + x0, (l.y * h) / k + y0]);

// Runs the face mesh again on an enlarged crop of the head: a face that is
// small in the frame (full-length shot, mirror) often goes undetected.
function faceOnCrop(image, w, h, center, rad) {
  const x0 = Math.max(0, Math.trunc(center[0] - rad)), y0 = Math.max(0, Math.trunc(center[1] - rad));
  const x1 = Math.min(w, Math.trunc(center[0] + rad)), y1 = Math.min(h, Math.trunc(center[1] + rad));
  if (x1 - x0 < 24 || y1 - y0 < 24) return [];
  const k = Math.min(4, Math.max(1, 640 / Math.max(x1 - x0, y1 - y0)));
  const cw = Math.round((x1 - x0) * k), ch = Math.round((y1 - y0) * k);
  const c = new OffscreenCanvas(cw, ch);
  const ctx = c.getContext("2d");
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(image, x0, y0, x1 - x0, y1 - y0, 0, 0, cw, ch);
  const r = tasks.face.detect(ctx.getImageData(0, 0, cw, ch));
  return (r.faceLandmarks || []).map((lm) => toPx(lm, cw, ch, k, x0, y0));
}

// The segmenter's class order is not guaranteed: keep the class that covers
// the pose landmarks, and reject an empty or full mask.
function pickPolarity(cat, w, h, pose) {
  const a = new Float32Array(w * h), b = new Float32Array(w * h);
  let sa = 0;
  for (let i = 0; i < w * h; i++) {
    a[i] = cat[i] > 0 ? 1 : 0;
    b[i] = 1 - a[i];
    sa += a[i];
  }
  let chosen;
  if (pose) {
    const at = (m) => pose.reduce((s, p) => {
      const x = Math.min(w - 1, Math.max(0, Math.trunc(p[0]))), y = Math.min(h - 1, Math.max(0, Math.trunc(p[1])));
      return s + m[y * w + x];
    }, 0) / pose.length;
    const ka = at(a), kb = at(b);
    if (Math.max(ka, kb) < 0.5) return null;
    chosen = ka >= kb ? a : b;
  } else {
    chosen = sa <= w * h - sa ? a : b;
  }
  const cover = chosen.reduce((s, v) => s + v, 0) / (w * h);
  return cover > 0.01 && cover < 0.9 ? chosen : null;
}

function segment(image, w, h, pose) {
  const r = tasks.seg.segment(image);
  try {
    const m = r.categoryMask;
    if (!m) return null;
    let cat = m.getAsUint8Array();
    if (m.width !== w || m.height !== h) {
      const out = new Uint8Array(w * h);
      for (let y = 0; y < h; y++)
        for (let x = 0; x < w; x++)
          out[y * w + x] = cat[Math.trunc((y * m.height) / h) * m.width + Math.trunc((x * m.width) / w)];
      cat = out;
    }
    return pickPolarity(cat, w, h, pose);
  } finally {
    r.close();
  }
}

/** Full detection on a w x h canvas. */
export async function detect(image, w, h) {
  await load();
  let pose = null, vis = null;
  const rp = tasks.pose.detect(image);
  if (rp.landmarks && rp.landmarks.length) {
    pose = toPx(rp.landmarks[0], w, h);
    vis = rp.landmarks[0].map((l) => l.visibility ?? 1);
  }

  const rf = tasks.face.detect(image);
  let faces = (rf.faceLandmarks || []).map((lm) => toPx(lm, w, h));
  let source = faces.length ? "face mesh" : null;
  if (!faces.length && pose) {
    let pts = HEAD_IDX.map((i) => pose[i]);
    const seen = pts.filter((_, k) => vis[HEAD_IDX[k]] > 0.2);
    if (seen.length >= 3) pts = seen;
    const c = [pts.reduce((s, p) => s + p[0], 0) / pts.length, pts.reduce((s, p) => s + p[1], 0) / pts.length];
    const rad = Math.max(Math.max(...pts.map((p) => Math.hypot(p[0] - c[0], p[1] - c[1]))) * 3.2, 0.04 * Math.max(w, h));
    faces = faceOnCrop(image, w, h, c, rad);
    if (faces.length) source = "face mesh on an enlarged crop";
  }
  if (!faces.length) source = "pose landmarks only";

  const body = segment(image, w, h, pose);
  let size = 0.25 * Math.min(w, h);
  if (pose) {
    const sh = [(pose[L_SHO][0] + pose[R_SHO][0]) / 2, (pose[L_SHO][1] + pose[R_SHO][1]) / 2];
    const hp = [(pose[L_HIP][0] + pose[R_HIP][0]) / 2, (pose[L_HIP][1] + pose[R_HIP][1]) / 2];
    const d = Math.hypot(sh[0] - hp[0], sh[1] - hp[1]);
    if (d > 10) size = d;
  }
  return { w, h, pose, vis, faces, body, size, source };
}

/**
 * Looks for faces in the render. Returns each face found, and whether it
 * falls in the erased area.
 */
export async function faceCheck(image, w, h, ident) {
  await load();
  const found = (tasks.face.detect(image).faceLandmarks || []).map((lm) => toPx(lm, w, h));
  // the area is small in a full-length shot, so also look at a close crop
  let x0 = w, y0 = h, x1 = 0, y1 = 0;
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      if (ident[y * w + x] > 0.5) {
        x0 = Math.min(x0, x); x1 = Math.max(x1, x);
        y0 = Math.min(y0, y); y1 = Math.max(y1, y);
      }
  if (x1 > x0) {
    const c = [(x0 + x1) / 2, (y0 + y1) / 2];
    found.push(...faceOnCrop(image, w, h, c, Math.max(x1 - x0, y1 - y0) * 1.2));
  }
  return found.map((f) => {
    const c = [f.reduce((s, p) => s + p[0], 0) / f.length, f.reduce((s, p) => s + p[1], 0) / f.length];
    const inZone = c[0] >= x0 && c[0] <= x1 && c[1] >= y0 && c[1] <= y1;
    return { center: c, inZone };
  });
}

