# For Harriet: review before release

Things from the UI refresh (branch `ui-updates`) that need the app owner's eye.

## 1. All copy in the app

The wording is largely the original. It was only capitalised and lightly adjusted to match
the new buttons. Please read through all of it, especially:

- **The "?" help window** (`public/index.html`, `#helpDialog`): the intro and the Encode,
  Decode, Why Tessa, Morph and Interference Morph sections.
- **Why Tessa** (`why()` in `public/app.js`): the intro line, the table's column names and
  the footnote. It doesn't currently state its conclusion. A proposal is below.
- **Hover descriptions** on the setting options (the `tip:` text in `PANELS`, `public/app.js`),
  e.g. "Shape in θ, read in Z", "Direction in θ and φ, radius in θ".
- **Status and log messages** (`status(...)` calls in `public/app.js`), e.g. "Sphere ready.
  Press Run once to encode and read it, or click it for settings."
- **The Curves window's readings** (`shotsTakeaway`, `compressionTakeaway`, `depthTakeaway`
  in `public/app.js`). These are generated from the results, so check the wording reads
  right in each case.
- **Landing line:** "To begin, drop an .obj file anywhere or choose a preset".
- **Why a tool is greyed out** (`syncHud()` in `public/app.js`): the hover labels on the
  right-hand tools when they can't run, e.g. "Morph: add a second mesh first".

## 2. Why Tessa: proposed rework (not yet done)

The table has the data but leaves the reader to draw the conclusion. Proposed:

1. A short answer at the top, generated from the numbers, e.g. "Tessa fits big meshes
   onto real chips with shallow, SWAP-free circuits."
2. Highlight Tessa's row, and cut the table to the columns that make the argument:
   qubits, SWAPs, circuit depth and shots per value. Add plain-English column notes and
   put the full detail behind "Show all columns".
3. A "scaled up" line showing what each approach would need for a larger mesh, so the
   "one qubit per value won't fit" problem is visible. Only use numbers the app can compute.

## 3. Phones and narrow windows

The refresh was reviewed on a laptop-sized window in dark and light mode only. Small screens
have their own layouts, and none has been checked since the refresh. Please try a phone and
a narrow browser window (under about 900px wide) for:

- **Load screen:** the shapes in two rows, the "To begin…" line, and "Tessa3D" with the round
  buttons along the top.
- **Home:** the Low / Medium / High res row (moved under "Tessa3D" on narrow screens), and the
  log over the Run once / Run & play buttons.
- **Menus:** "Tap for settings" under the object, the Encode / Decode pills, the settings
  lines, and the back button on the left.
- **Morph:** the drop box and options stacked under the object.
- **Tools and windows:** the tools tab on the right edge, the View drawer (under the top bar),
  the help, Why Tessa, Curves and Qubit Map windows, and the greyed-tool reasons in the
  bottom-right corner.

## 4. Apple and Suzanne preset pictures

`public/assets/apple-dots.png` and `suzanne-dots.png` are rendered stand-ins and don't quite
match the cube, blob and sphere pictures. If you can export these two the same way as the
others (transparent background, same angle and framing), drop them in with the same file
names. The CSS in `public/style.css` (`.preset.apple`, `.preset.suzanne`) may need its box
sizes adjusted to fit.
