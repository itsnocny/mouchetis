import { anchorConf, resolve } from "./src/anchors.js";
import { detect, faceCheck, load } from "./src/detect.js";
import { INK, PAPER } from "./src/stipple.js";
import { rng } from "./src/rng.js";

const MAX_SIDE = 1600;   // working resolution, long side
const EXPORT_SIDE = 1440;

const $ = (id) => document.getElementById(id);

// segmented controls: one pressed button per group
const seg = (name) => document.querySelector(`.seg[data-name="${name}"] [aria-pressed="true"]`).dataset.value;
function bindSeg(name, onChange) {
  const group = document.querySelector(`.seg[data-name="${name}"]`);
  group.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b || b.getAttribute("aria-pressed") === "true") return;
    for (const x of group.querySelectorAll("button")) x.setAttribute("aria-pressed", String(x === b));
    onChange(b.dataset.value);
  });
}

// the dot screen is computed in a worker; each request waits for its own reply
const worker = new Worker(new URL("./src/worker.js", import.meta.url), { type: "module" });
const pending = new Map();
let seq = 0;
worker.onmessage = ({ data }) => {
  const p = pending.get(data.id);
  pending.delete(data.id);
  if (data.error) p.reject(new Error(data.error));
  else p.resolve(data);
};
const call = (msg, transfer = []) =>
  new Promise((resolve, reject) => {
    const id = ++seq;
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, ...msg }, transfer);
  });

const S = {
  name: "photo", src: null, w: 0, h: 0, det: null, ident: null, caption: [],
  rgba: null, seed: 7, version: 0, notes: [],
};

// what people see in the anchor menus; left and right are the person's own
const LABELS = {
  mouth: "Mouth", nose: "Nose", face: "Face", neck: "Neck", shoulders: "Shoulders",
  "shoulder-l": "Left shoulder", "shoulder-r": "Right shoulder", chest: "Chest", belly: "Belly",
  torso: "Torso", hips: "Hips", "hip-l": "Left hip", "hip-r": "Right hip",
  "elbow-l": "Left elbow", "elbow-r": "Right elbow", "wrist-l": "Left wrist", "wrist-r": "Right wrist",
  "hand-l": "Left hand", "hand-r": "Right hand", "knee-l": "Left knee", "knee-r": "Right knee",
  "ankle-l": "Left ankle", "ankle-r": "Right ankle", "foot-l": "Left foot", "foot-r": "Right foot",
  "eye-l": "Left eye", "eye-r": "Right eye", "ear-l": "Left ear", "ear-r": "Right ear",
  body: "Whole body", "upper-body": "Upper body", legs: "Legs", "arm-l": "Left arm",
  "arm-r": "Right arm", head: "Head",
};
const POINT_ORDER = ["mouth", "face", "neck", "shoulders", "shoulder-l", "shoulder-r", "chest", "belly",
  "torso", "hips", "hip-l", "hip-r", "hand-l", "hand-r", "wrist-l", "wrist-r", "elbow-l", "elbow-r",
  "knee-l", "knee-r", "ankle-l", "ankle-r", "foot-l", "foot-r", "nose", "eye-l", "eye-r", "ear-l", "ear-r"];
const AREA_ORDER = ["body", "upper-body", "head", "legs", "arm-l", "arm-r"];

// an anchor the detection can't place is shown but can't be picked
function anchorOptions(select, selected) {
  const group = (label, names) => {
    const g = document.createElement("optgroup");
    g.label = label;
    for (const name of names) {
      const o = new Option(LABELS[name] + (usable(name) ? "" : " (not visible)"), name);
      o.disabled = !usable(name);
      g.append(o);
    }
    return g;
  };
  select.replaceChildren();
  select.append(group("Points", POINT_ORDER), group("Areas", AREA_ORDER));
  const fallback = [...select.options].find((o) => !o.disabled);
  const wanted = selected && select.querySelector(`option[value="${selected}"]`);
  select.value = wanted && !wanted.disabled ? selected : fallback.value;
}

const usable = (name) => S.det && resolve(name, S.det) && anchorConf(name, S.det) >= 0.25;

// a new note points somewhere the other notes don't
function nextAnchor() {
  const used = new Set(S.notes.map((n) => n.anchor));
  const order = ["neck", "chest", "hips", "hand-l", "hand-r", "shoulder-l", "shoulder-r", "belly",
    "upper-body", "body", ...POINT_ORDER, ...AREA_ORDER];
  return order.find((a) => !used.has(a) && usable(a)) || "neck";
}

function busy(text) {
  $("busy").hidden = !text;
  $("busy").textContent = text || "";
}

function status(text, kind = "") {
  $("status-text").textContent = text;
  $("status").className = kind;
}

// The name drawn in dots, the same way as the screen: the chance of a dot
// follows the ink of the letters under it.
function drawMark() {
  const c = $("mark");
  const scale = Math.min(3, window.devicePixelRatio || 1);
  const fs = 64, pad = 4;
  const font = `700 ${fs}px ${getComputedStyle(document.body).fontFamily}`;
  const probe = document.createElement("canvas").getContext("2d");
  probe.font = font;
  const tw = Math.ceil(probe.measureText("mouchetis").width) + 2 * pad, th = Math.ceil(fs * 1.25);
  const src = document.createElement("canvas");
  src.width = tw;
  src.height = th;
  const g = src.getContext("2d");
  g.font = font;
  g.fillText("mouchetis", pad, Math.round(fs * 0.98));
  const a = g.getImageData(0, 0, tw, th).data;

  c.width = tw * scale;
  c.height = th * scale;
  c.style.width = `${(tw * parseFloat(getComputedStyle(c).height)) / th}px`;
  const ctx = c.getContext("2d");
  ctx.fillStyle = `rgb(${PAPER})`;
  const R = rng(11), cell = 1.25;
  for (let y = 0; y < th; y += cell) {
    for (let x = 0; x < tw; x += cell) {
      const cov = a[4 * (Math.floor(y) * tw + Math.floor(x)) + 3] / 255;
      if (R.random() < cov * 0.9) {
        ctx.beginPath();
        ctx.arc(x * scale, y * scale, 0.68 * scale * (0.75 + 0.5 * R.random()), 0, 2 * Math.PI);
        ctx.fill();
      }
    }
  }
}

async function open(file) {
  let bitmap;
  try {
    bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
  } catch {
    showEmpty("Your browser can't read this file. Try a JPEG or PNG photo.");
    return;
  }
  S.version++;
  S.name = file.name.replace(/\.[^.]+$/, "") || "photo";
  const k = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
  S.w = Math.round(bitmap.width * k);
  S.h = Math.round(bitmap.height * k);
  S.src = document.createElement("canvas");
  S.src.width = S.w;
  S.src.height = S.h;
  const ctx = S.src.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bitmap, 0, 0, S.w, S.h);
  bitmap.close();
  Object.assign(S, { det: null, ident: null, rgba: null });

  $("empty").hidden = true;
  $("out").hidden = false;
  $("bar").hidden = false;
  $("save").disabled = true;
  $("notes").replaceChildren();
  // the photo stays blurred on screen until the face is erased
  show(S.src, true);
  status("");

  try {
    busy("Loading detection models, about 50 MB the first time…");
    await load();
    busy("Looking for the person…");
    S.det = await detect(S.src, S.w, S.h);
  } catch (e) {
    console.error(e);
    busy("");
    status("The detection models could not be loaded. Check your connection and reload the page.", "warn");
    return;
  }
  if (!S.det.pose && !S.det.faces.length) {
    busy("");
    status("No person found in this photo. A brighter or less tightly cropped photo usually works better.", "warn");
    return;
  }
  notes();
  renderNotes();
  await prepare();
}

// Everything that depends on the photo and the slow settings (background, area).
async function prepare() {
  const version = ++S.version;
  $("save").disabled = true;
  busy("Placing the erased area…");
  const img = S.src.getContext("2d").getImageData(0, 0, S.w, S.h).data;
  const rgb = new Float32Array(S.w * S.h * 3);
  for (let i = 0, j = 0; i < img.length; i += 4, j += 3) {
    rgb[j] = img[i];
    rgb[j + 1] = img[i + 1];
    rgb[j + 2] = img[i + 2];
  }
  const det = { ...S.det, body: S.det.body && S.det.body.slice() };
  const res = await call({
    type: "prepare", rgb, det, opts: { bg: seg("bg"), zone: seg("zone") },
  }, [rgb.buffer]);
  if (version !== S.version) return;
  if (!res.ok) {
    S.ident = null;
    busy("");
    status("The head could not be located, so no image is made. Nothing would guarantee the face is hidden.", "warn");
    return;
  }
  S.ident = res.ident;
  S.caption = res.caption;
  await render();
}

// Runs on every slider move. One render at a time: if a setting changes
// during a render, render once more at the end.
let rendering = false, dirty = false;
async function render() {
  if (!S.ident) return;
  if (rendering) {
    dirty = true;
    return;
  }
  rendering = true;
  dirty = false;
  // what is on screen no longer matches the settings: nothing to save yet
  $("save").disabled = true;
  const version = ++S.version;
  busy("Rendering…");
  const opts = {
    stage: parseFloat($("stage-range").value), grain: parseFloat($("grain").value),
    polarity: seg("polarity"), seed: S.seed,
  };
  try {
    const { rgba } = await call({ type: "draw", opts });
    if (version === S.version) {
      S.rgba = rgba;
      await paint();
      busy("");
      status("Checking the result for faces…");
      scheduleCheck(version);
    }
  } finally {
    rendering = false;
  }
  if (dirty) render();
}

function rawCanvas(rgba) {
  const c = document.createElement("canvas");
  c.width = S.w;
  c.height = S.h;
  c.getContext("2d").putImageData(new ImageData(new Uint8ClampedArray(rgba), S.w, S.h), 0, 0);
  return c;
}

// Notes that point at something the detection can place, in pixels.
function liveNotes() {
  return S.notes
    .filter((n) => n.text.trim())
    .map((n) => ({ at: S.det && resolve(n.anchor, S.det), text: n.text.trim() }))
    .filter((n) => n.at);
}

// A circle for a point, a rectangle for an area, numbered like the caption.
function drawMarks(ctx, notes, fs, color) {
  ctx.save();
  ctx.strokeStyle = ctx.fillStyle = color;
  ctx.lineWidth = Math.max(1, fs * 0.16);
  ctx.font = `${fs}px Optima, Candara, "Gill Sans", sans-serif`;
  ctx.textBaseline = "top";
  const r = Math.max(6, fs * 1.15);
  notes.forEach(({ at }, i) => {
    let lx, ly;
    if (at.length === 4) {
      ctx.globalAlpha = 210 / 255;
      ctx.strokeRect(at[0], at[1], at[2] - at[0], at[3] - at[1]);
      [lx, ly] = [at[0] + ctx.lineWidth + 2, at[1] + ctx.lineWidth + 2];
    } else {
      ctx.globalAlpha = 230 / 255;
      ctx.beginPath();
      ctx.arc(at[0], at[1], r, 0, 2 * Math.PI);
      ctx.stroke();
      [lx, ly] = [at[0] + r * 1.4, at[1] - r * 1.1];
    }
    ctx.globalAlpha = 230 / 255;
    ctx.fillText(String(i + 1), lx, ly);
  });
  ctx.restore();
}

// The result, with a caption under the image like a museum label: your notes,
// or else three coordinates of the erased area's outline.
function finished() {
  const img = rawCanvas(S.rgba);
  const notesNow = liveNotes();
  if (!$("caption").checked || (!S.caption.length && !notesNow.length)) return img;
  const light = seg("polarity") === "light";
  const ground = light ? INK : PAPER, text = light ? PAPER : INK;
  const fs = Math.max(9, Math.trunc(S.w * 0.0165));
  const lead = Math.trunc(fs * 1.7), inset = Math.trunc(S.w * 0.055);
  const font = `${fs}px Optima, Candara, "Gill Sans", sans-serif`;
  const maxw = S.w - 2 * inset, indent = fs * 1.6;
  const marks = seg("mark");
  // lay the notes out first, so the margin fits them
  const probe = document.createElement("canvas").getContext("2d");
  probe.font = font;
  const rows = [];
  notesNow.forEach(({ at, text }, i) => {
    const coords = at.map((v) => v.toFixed(1)).join(" / ");
    let line = marks === "none" ? `${coords} : ` : `${i + 1}.  ${coords} : `, first = true;
    for (const word of text.split(/\s+/)) {
      const trial = line + word + " ";
      if (probe.measureText(trial).width > maxw - (first ? 0 : indent) && line.trim()) {
        rows.push([line.trimEnd(), first ? 0 : indent]);
        line = word + " ";
        first = false;
      } else line = trial;
    }
    if (line.trim()) rows.push([line.trimEnd(), first ? 0 : indent]);
  });
  const count = notesNow.length ? rows.length : S.caption.length;
  const margin = Math.trunc(fs * 2.2) + count * lead + Math.trunc(fs * 1.4);
  const c = document.createElement("canvas");
  c.width = S.w;
  c.height = S.h + margin;
  const ctx = c.getContext("2d");
  ctx.fillStyle = `rgb(${ground})`;
  ctx.fillRect(0, 0, c.width, c.height);
  ctx.drawImage(img, 0, 0);
  let y = S.h + Math.trunc(fs * 1.5);
  ctx.fillStyle = `rgba(${text}, ${38 / 255})`;
  ctx.fillRect(inset, y, S.w - 2 * inset, 1);
  ctx.font = font;
  ctx.textBaseline = "top";
  y += Math.trunc(fs * 1.3);
  if (notesNow.length) {
    if (marks !== "none") drawMarks(ctx, notesNow, fs, marks === "accent" ? "rgb(176, 58, 44)" : `rgb(${text})`);
    ctx.fillStyle = `rgba(${text}, ${170 / 255})`;
    for (const [row, dx] of rows) {
      ctx.fillText(row, inset + dx, y);
      y += lead;
    }
    return c;
  }
  ctx.fillStyle = `rgba(${text}, ${150 / 255})`;
  for (const [x, yy] of S.caption) {
    ctx.fillText(`${x.toFixed(1)}, ${yy.toFixed(1)}`, inset, y);
    y += lead;
  }
  return c;
}

function show(canvas, blurred = false) {
  const out = $("out");
  out.width = canvas.width;
  out.height = canvas.height;
  const ctx = out.getContext("2d");
  ctx.filter = blurred ? `blur(${Math.round(canvas.width / 60)}px)` : "none";
  ctx.drawImage(canvas, 0, 0);
  ctx.filter = "none";
}

async function paint() {
  if (!S.rgba) return;
  const view = seg("view");
  let note = "";
  if (view === "result") {
    show(finished());
  } else if (view === "recovered") {
    const { rgba } = await call({ type: "reconstruct", rgba: S.rgba.slice(), w: S.w, h: S.h });
    show(rawCanvas(rgba));
    note = "The result blurred and contrast-stretched, which is what someone trying to undo the dots would do. If you can recognize yourself here, don't post the image.";
  } else {
    const c = document.createElement("canvas");
    c.width = S.w;
    c.height = S.h;
    const ctx = c.getContext("2d");
    ctx.drawImage(S.src, 0, 0);
    const im = ctx.getImageData(0, 0, S.w, S.h);
    const r = Math.max(1, Math.round(S.w / 500));
    const inside = (i) => S.ident[i] > 0.53;
    for (let y = r; y < S.h - r; y++) {
      for (let x = r; x < S.w - r; x++) {
        const i = y * S.w + x;
        if (!inside(i) || (inside(i - r) && inside(i + r) && inside(i - r * S.w) && inside(i + r * S.w))) continue;
        im.data.set([255, 60, 40, 255], 4 * i);
      }
    }
    ctx.putImageData(im, 0, 0);
    show(c);
    note = "Your original photo with the erased area outlined in red. The outline should cover the whole face. This view is never saved.";
  }
  $("view-note").textContent = note;
}

function notes() {
  const d = S.det, items = [];
  const n = d.faces.length;
  if (!n) {
    items.push(["The face mesh found no face, so the erased area is estimated from the posture. Check Mask placement to make sure no part of the face sticks out.", "warn"]);
  } else {
    items.push([n === 1 ? "One face found." : `${n} faces found, all erased.`]);
    if (d.source.includes("crop")) items.push(["The face is small in the frame. It was found on an enlarged crop."]);
  }
  if (n <= 1) items.push(["If there is a mirror in the photo, check that a second face doesn't show in it."]);
  if (!d.body) items.push(["The outline of the person wasn't found. The body is estimated from the posture."]);
  $("notes").replaceChildren(...items.map(([text, cls]) => {
    const li = document.createElement("li");
    li.textContent = text;
    if (cls) li.className = cls;
    return li;
  }));
  // a warning in the notes must not stay hidden behind More
  if (items.some(([, cls]) => cls === "warn")) openMore(true);
}

// Once the render settles, look for a face in it with the same detector.
let checkTimer = 0;
function scheduleCheck(version) {
  clearTimeout(checkTimer);
  checkTimer = setTimeout(async () => {
    if (version !== S.version) return;
    const found = await faceCheck(rawCanvas(S.rgba), S.w, S.h, S.ident);
    if (version !== S.version) return;
    const save = $("save");
    const inZone = found.filter((f) => f.inZone);
    if (inZone.length) {
      status("The detector still finds a face where the face was erased. Don't post this. Try Whole head or bigger dots.", "warn");
      save.textContent = "Save anyway";
      save.classList.add("risky");
    } else {
      if (found.length) {
        const [x, y] = found[0].center;
        status(`The detector sees a face elsewhere, around ${Math.round((100 * x) / S.w)}% across and ${Math.round((100 * y) / S.h)}% down. Often it's just a pattern in the dots, but look there.`, "warn");
      } else {
        status("Checked: the face detector finds no face in the result.", "ok");
      }
      save.textContent = "Save";
      save.classList.remove("risky");
    }
    save.disabled = false;
  }, 500);
}

function save() {
  const img = finished();
  const k = Math.min(1, EXPORT_SIDE / Math.max(img.width, img.height));
  const c = document.createElement("canvas");
  c.width = Math.round(img.width * k);
  c.height = Math.round(img.height * k);
  const ctx = c.getContext("2d");
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(img, 0, 0, c.width, c.height);
  // a PNG exported from a canvas carries none of the photo's metadata
  c.toBlob((blob) => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${S.name}_dots.png`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  }, "image/png");
}

function showEmpty(message = "") {
  S.version++;
  Object.assign(S, { src: null, det: null, ident: null, rgba: null });
  $("file").value = "";
  $("empty").hidden = false;
  $("out").hidden = true;
  $("bar").hidden = true;
  busy(message);
}

// More and Text share the space under the controls: one open at a time.
function openPanel(name, open) {
  for (const p of ["more", "text"]) {
    const show = p === name && open;
    $(p).hidden = !show;
    $(`${p}-open`).setAttribute("aria-expanded", String(show));
  }
}
const openMore = (open) => openPanel("more", open);

function renderNotes() {
  const list = $("note-list");
  list.replaceChildren(...S.notes.map((n, i) => {
    const li = document.createElement("li");
    const sel = document.createElement("select");
    sel.className = "field";
    sel.setAttribute("aria-label", `Note ${i + 1} points at`);
    anchorOptions(sel, n.anchor);
    n.anchor = sel.value;
    sel.addEventListener("change", () => {
      n.anchor = sel.value;
      paint();
    });
    const input = document.createElement("input");
    input.type = "text";
    input.className = "field grow";
    input.maxLength = 200;
    input.placeholder = "Your note";
    input.value = n.text;
    input.setAttribute("aria-label", `Note ${i + 1}`);
    input.addEventListener("input", () => {
      n.text = input.value;
      paint();
    });
    const rm = document.createElement("button");
    rm.type = "button";
    rm.className = "ghost remove";
    rm.textContent = "Remove";
    rm.addEventListener("click", () => {
      S.notes.splice(i, 1);
      renderNotes();
      paint();
    });
    li.append(sel, input, rm);
    return li;
  }));
}

$("file").addEventListener("change", (e) => e.target.files[0] && open(e.target.files[0]));
document.addEventListener("dragover", (e) => {
  e.preventDefault();
  document.body.classList.add("dragging");
});
document.addEventListener("dragleave", (e) => {
  if (!e.relatedTarget) document.body.classList.remove("dragging");
});
document.addEventListener("drop", (e) => {
  e.preventDefault();
  document.body.classList.remove("dragging");
  const f = e.dataTransfer.files[0];
  if (f) open(f);
});

$("stage-range").addEventListener("input", render);
$("grain").addEventListener("input", render);
bindSeg("polarity", render);
bindSeg("zone", () => S.det && prepare());
bindSeg("bg", () => S.det && prepare());
bindSeg("view", paint);
$("caption").addEventListener("change", paint);
$("reroll").addEventListener("click", () => {
  S.seed = Math.floor(Math.random() * 1e9);
  render();
});
$("more-open").addEventListener("click", () => openPanel("more", $("more").hidden));
$("text-open").addEventListener("click", () => openPanel("text", $("text").hidden));
$("note-add").addEventListener("click", () => {
  S.notes.push({ anchor: nextAnchor(), text: "" });
  renderNotes();
  const last = $("note-list").lastElementChild;
  last?.scrollIntoView({ block: "nearest" });
  last?.querySelector("input")?.focus();
});
bindSeg("mark", paint);
$("save").addEventListener("click", save);
$("new-photo").addEventListener("click", () => showEmpty());
$("about-open").addEventListener("click", () => $("about").showModal());

document.fonts.ready.then(drawMark);
