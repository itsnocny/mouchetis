# mouchetis

mouchetis turns a photo of a person into a dot screen. The density of the dots
follows the light of the photo, so the body stays readable, but no original
pixel is left in the image. The face is replaced by dots computed only from
the pixels around it, never from the face itself. The coordinates found by the
pose detection are written in a caption under the image.

Everything runs locally. The saved image has no metadata.

There are two versions. The web page works on phones and computers with
nothing to install. The Python command has more options (speech bubbles,
an audit tool) and only runs on macOS for now.

## The web page

It lives in `web/`. The photo is processed in the browser and is not sent
anywhere. The page only downloads MediaPipe and the detection models the first
time, about 50 MB. They are the same models as the Python version.

Under Notes you can write notes. A note points at a part of the body (neck,
left hand, upper body…) and is written in the caption under the image with
that part's position. Numbered marks can show on the image where each note
points.

After each render, the page runs the face detector again on the result. If it
finds a face where the face was erased, it says so and the button becomes
"Save anyway". Under More, it can also show what someone gets by blurring the
dots, and the outline of the erased area on the original photo.

To run it on your own machine:

```sh
cd web
python3 -m http.server 8000
```

then open `http://localhost:8000`. Opening `index.html` straight from the disk
doesn't work, because the browser then blocks the background worker.

The repository publishes `web/` to GitHub Pages on every push to `main`, as
long as the tests pass (`.github/workflows/pages.yml`). Pages has to be turned
on once in the repository settings, with "GitHub Actions" as the source.

### Tests

```sh
cd web
npm test
```

The main test replaces the content of the face, either by shuffling its
pixels or with noise, then renders again. Not a single output pixel may
change. It runs on a synthetic scene in nine setups (face or whole head,
background blurred, kept or removed, no fade, no face mesh, no outline, head
cut off by the edge of the image), at three dissolve levels and in both inks.

On a real photo, with the same detection, the web page and the Python version
differ about as much as two Python renders with different seeds (correlation
0.98 at 24 px wide).

## The command

Tested on macOS (Apple Silicon). The caption fonts and emoji come from
`/System/Library/Fonts`.

### Installation

You need [uv](https://docs.astral.sh/uv/).

```sh
git clone <this repository> mouchetis
cd mouchetis
uv run mouchetis photo.jpg
```

The first run installs the dependencies and downloads three MediaPipe models
(about 35 MB) to `~/.cache/mouchetis`. Later runs don't use the network. Set
`MOUCHETIS_MODELS` to use another folder.

To get the `mouchetis` command everywhere, without `uv run`:

```sh
uv tool install .
```

### Usage

```sh
mouchetis photo.jpg                     # writes photo_dots.png
mouchetis photo.jpg --variants          # writes 8 images to photo_variants/
mouchetis photo.jpg --stage 0.6 --grain 1.2
```

`--stage` sets how far the body dissolves, from 0 (sharp) to 1 (almost
formless). The face is erased whatever the value. `--variants` writes one image
per `--stage` value, from 0.05 to 0.95.

`mouchetis -h` lists every option. The most useful ones:

| Option | Effect |
|---|---|
| `--stage` | how far the body dissolves, 0 to 1 (default 0.25) |
| `--grain` | dot size (default 1.8) |
| `--face-grain` | dot size in the face area, as a multiple of `--grain` (default 1) |
| `--density`, `--gamma` | amount of ink and contrast |
| `--polarity` | `ink` for dark dots on light paper, `light` for the reverse |
| `--bg` | background `blur` (default), `keep` or `black` |
| `--zone` | `face` erases the estimated face (default), `head` erases the whole head |
| `--fade` | width of the fade at the edge of the erased area (default 0.35) |
| `--mask` | extra area to erase, `x,y,r` or `x0,y0,x1,y1` in 0 to 1 coordinates |
| `--keep` | area to leave alone, same format |
| `--width` | long side of the saved image (default 1440) |
| `--seed` | random seed, to make exactly the same image again |

### Notes

`--note` writes a caption line under the image instead of the automatic
coordinates. It can be repeated.

```sh
mouchetis photo.jpg \
  --note "neck:Collar checked. 🔒" \
  --note "shoulder-l,hip-r:Kneeling pose."
```

Before the colon goes an anchor. It can be a name (`neck`, `nose`, `hand-l`,
`hips`…), two names that make a rectangle, or coordinates between 0 and 1.
There are also names for whole areas: `body`, `upper-body`, `legs`, `arm-l`,
`arm-r` and `head`. Left and right are the subject's, so they are mirrored in
the image.

```sh
mouchetis photo.jpg --anchors
```

This prints every anchor with its position on the given photo and its
confidence. A badly seen anchor is marked `UNRELIABLE`.

`--mark ink` or `--mark accent` also draws a numbered circle or rectangle on
the image at each note. By default the image has no marks.

### Speech bubbles

`--say` adds a speech bubble inside the dots, next to the head. The text stays
sharp at every stage. The bubble dissolves with `--stage` like the body, and
the line linking it to the mouth drifts away.

```sh
mouchetis photo.jpg --variants --say "I don't know what to say."
mouchetis photo.jpg --say "hand-l:Mine." --stage 0.6
```

A line under the text gives the detection's confidence for that anchor.
`--no-speaker` removes it.

The `bubbles` subcommand puts bubbles on a photo without turning it into dots.
It matches the bubble's colour, blur and grain to the photo.

```sh
mouchetis bubbles photo.jpg --say "I'm not posing."
mouchetis bubbles photo.jpg --think "It's cold." --style chalk
```

The styles are `glass` (default), `paper`, `manga` and `chalk`.

### Checking before posting

`--check` writes `photo_CHECK_SHOWS_FACE.png`. It shows the original photo,
the outline of the erased area in red, the pose landmarks and the detected
faces. This image shows the face. It is for checking where the area falls,
and should be deleted afterwards.

The `audit` subcommand tries to get back to the photo from the renders.

```sh
mouchetis audit photo_variants/
mouchetis audit photo_variants/ --original photo.jpg
```

For each image it looks for leftover metadata and writes a
`*_reconstruction.png`. That file shows what you get by blurring the dots back
into tones. If you recognize yourself in it, don't post. With `--original` it
also measures, area by area, what still looks like the original photo, and
writes a `*_leaks.png` map.

## What the erasing covers

The content of the face area is rebuilt from the pixels around it
(inpainting), never from the face. A test checks this. It shuffles the pixels
inside the area, renders again, and not a single output pixel changes.

The body stays recognizable by its outline and posture. That's intended.
Tattoos, moles, jewelry and scars larger than the dots stay visible. Cover
them with `--mask`.

Some cases need a look by eye.

- A photo taken in a mirror can contain the face twice. mouchetis looks for
  up to three faces (`--faces`), but a blurry or side-on reflection can slip
  through without a warning.
- When a phone or a hand covers the face, the face mesh often fails. The area
  is then estimated from the pose landmarks, and the console says so. Check
  with `--check` that no part of the face sticks out.
- `--keep` leaves visible whatever it spares. Make sure no skin of the face is
  in it.
- `--lines` draws estimated face guides inside the erased area. They bring
  back a rough face geometry (eye spacing, mouth position). With the default
  value, 0, nothing is drawn.

If the head can't be located at all, mouchetis refuses to make an image.

Finally, a photo already posted somewhere else can still be found, whatever
is done to it.

## MediaPipe version

`mediapipe` is pinned to 0.10.35. On macOS, version 1.0.1 installs but crashes
on the first call, while initializing Metal
(`graph_service.h:139 Check failed: service_ Service is unavailable`). It does
the same under Python 3.11 and 3.13, and with `delegate=CPU`.

In 0.10.35, reading a float32 segmentation mask kills the process
(`image_frame.cc:415 Check failed: 1 == ChannelSize()`), with no Python
exception to catch. So mouchetis asks for a uint8 category mask instead.
