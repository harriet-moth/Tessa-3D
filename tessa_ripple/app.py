"""Matplotlib window: 3D mesh, controls for every Tessa setting, frame playback.

Three tiers: encoder settings re-encode, readout settings re-decode (on Run /
Run frames), post and animation settings only redraw cached frames.

Two views: **Fidelity** (gain 1, vertices coloured by error: the technical case)
and **Material** (gain exaggerates the noise: the artistic mode).
"""

import colorsys
import io
import subprocess
import sys
import time
from dataclasses import replace
from functools import cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.transforms import IdentityTransform
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from PIL import Image

from . import animate as anim
from . import interfere as interf
from .bridge import (BACKENDS, DEVICES, DRR_METHODS, MAX_COLS, MAX_ROWS, Lattice, Params, cells, encode_mesh,
                     fidelity_curves, mesh_error, morph_clamp, morph_expected, morph_frame, qubit_map,
                     read_edges, resources, restore, vertex_qubits)
from .mesh import PRESETS, load_decimated
from .morph import correspond

OUT = Path("exports")
RES = ("low", "mid", "high")
DECIMATIONS = {"low": 2, "mid": 1, "high": 0}  # halvings of an uploaded OBJ per Resolution
FIT_VERTICES = Lattice(MAX_ROWS, MAX_COLS).capacity  # 792: high fits radial on the largest lattice
COLOURS = ("error", "qubit", "morph Δ", "plain")
# Mode: what each slot's qubit holds (bridge.MODES)
MODE_LABELS = {"shape θ (Z)": "theta", "shape θ+φ": "split", "morph A→B": "morph"}
DELTA_CMAP = colormaps["viridis"]
PLAIN = np.array(colormaps["Blues"](0.45))

# Colour definitions, mirrored in public/app.js so both apps colour identically.
# Error: magma_r on [0, 0.88], lightest = no error (pure black is avoided so faces stay readable).
ERROR_CMAP = LinearSegmentedColormap.from_list(
    "error", ["#fcfdbf", "#fd9668", "#cf4070", "#752181", "#1a1042"])
EDGE_CMAP = colormaps["viridis"]
GOLDEN = 0.618034


@cache
def fit_passes(path) -> int:
    """Decimation passes that bring an uploaded OBJ within the largest lattice.

    0 when the file fits as it is (e.g. Suzanne, 507 vertices). A large export
    (thousands of vertices) is halved until one radial value per vertex fits the
    10x12 lattice, so Resolution high can still be encoded.

    Args:
        path: Path to the OBJ file.

    Returns:
        int: Passes for Resolution high (mid and low add 1 and 2).

    Raises:
        ValueError: The file cannot be read as a mesh (from ``mesh.load_obj``).
    """
    k = 0
    while len(load_decimated(path, k)[0]) > FIT_VERTICES and k < 20:
        k += 1
    return k


def obj_passes(path, res_name: str) -> int:
    """Decimation passes of an uploaded OBJ at one Resolution.

    Args:
        path: Path to the OBJ file.
        res_name: "low", "mid" or "high".

    Returns:
        int: :func:`fit_passes` plus the Resolution's own halvings (DECIMATIONS).
    """
    return fit_passes(path) + DECIMATIONS[res_name]


def obj_presets(path):
    """An uploaded OBJ at three resolutions: high is the file itself (decimated first if
    it is too big for the largest lattice, :func:`fit_passes`), mid has about half its
    vertices, low about a quarter.

    Args:
        path: Path to the OBJ file.

    Returns:
        dict: Resolution name ("low", "mid", "high") -> zero-argument callable returning
        (verts (N, 3), faces). Loading and decimation run on first call (so an unreadable
        file is reported then, in the status line) and are cached by ``mesh.load_decimated``.
    """
    return {r: (lambda r=r: load_decimated(path, obj_passes(path, r))[:2]) for r in DECIMATIONS}


def choose_obj():
    """A file dialog for one OBJ. macOS uses AppleScript, since the default macOS
    matplotlib backend does not mix well with a Tk dialog.

    Returns:
        str | None: The chosen file's path, or None if cancelled.
    """
    if sys.platform == "darwin":
        out = subprocess.run(["osascript", "-e", 'POSIX path of (choose file of type {"obj"} '
                              'with prompt "Choose mesh B (.obj)")'], capture_output=True, text=True)
        return out.stdout.strip() or None
    import tkinter
    from tkinter import filedialog

    root = tkinter.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(filetypes=[("OBJ mesh", "*.obj")])
    root.destroy()
    return path or None


def hls_rgba(hues, lightness, saturation):
    """RGBA colours from hue, lightness and saturation.

    Args:
        hues: Iterable of hues; taken modulo 1.
        lightness: HLS lightness in [0, 1], shared by every colour.
        saturation: HLS saturation in [0, 1], shared by every colour.

    Returns:
        np.ndarray: (len(hues), 4) RGBA rows, alpha 1.
    """
    return np.array([(*colorsys.hls_to_rgb(h % 1, lightness, saturation), 1.0) for h in hues])


def qubit_rgba(qubits):
    """One colour per data qubit in use: hues spread by the golden ratio over each
    qubit's rank, so no two qubits share a colour and neighbouring ranks contrast.

    Args:
        qubits: (N,) data qubit index per vertex.

    Returns:
        np.ndarray: (N, 4) RGBA per vertex.
    """
    rank = np.unique(qubits, return_inverse=True)[1]
    return hls_rgba(rank * GOLDEN, 0.55, 0.6)


# curves: one fixed colour and line style per layout, mirrored in public/app.js
LAYOUT_STYLE = {"mesh": dict(c="#3d5a98", ls="-", marker="o", ms=9, mfc="none", mew=1.8, zorder=3),
                "naive": dict(c="#e07b39", ls="--", marker="s", ms=4, zorder=4)}
HOT = "#e69600"  # qubit map: the hovered data qubit, its address qubits and its vertices


class App:
    """Matplotlib window: the mesh in 3D, every Tessa setting as a widget, frame playback.

    Args:
        objs: OBJ paths to add as extra Shape options (missing files are skipped).

    Attributes:
        w: Widget name -> matplotlib widget (radios, sliders, checks, buttons).
        verts, faces: The current mesh A, (N, 3) and a list of faces.
        encoded: The current ``bridge.Encoded``, or None before the first encode.
        enc_key: (shape, resolution, Params.encoder_key) ``encoded`` was built with.
        raw: Cached readouts from ``animate.run_frames``: (frames, N*comps) raw values,
            or None.
        raw_key, raw_ts: What ``raw`` was read with (``decode_key``) and each frame's morph t.
        decoded, refs: Decoded positions per frame and the noiseless reference each is
            compared against.
        seq, idx: Playback positions (list of (N, 3)) and the step shown.
        match, match_key: Morph correspondence (``morph.correspond``) and what it was built for.
        b_path: Mesh B's OBJ path (morph), or None.
        edge_data: (edges, measured, exact) from ``bridge.read_edges``, or None.
        qmap, vq, focus: ``bridge.qubit_map``, data qubit per vertex, and the hovered
            vertex/qubit highlight (or None).
    """
    def __init__(self, objs=()):
        """Build the window, its widgets and callbacks, then encode the default mesh.

        Args:
            objs: OBJ paths to add as extra Shape options; each opens at Resolution high.
        """
        for path in objs:  # e.g. suzanne exported as OBJ
            if not Path(path).is_file():
                print(f"tessa-ripple: no OBJ file at {Path(path).resolve()}; skipping it",
                      file=sys.stderr)
                continue
            PRESETS[Path(path).stem] = obj_presets(path)
        self.objs = {Path(p).stem: p for p in objs if Path(p).is_file()}
        self.shape = None
        self.b_path, self.match, self.match_key, self.mode = None, None, None, None
        self.error = None  # status message of the last failed encode (e.g. an unreadable OBJ)
        self.ie, self.ideal_cache = None, {}  # interference morph: its encoding, ideal frames per t
        self.kept = None  # interference: share of shots each frame was decoded from
        self.fig = fig = plt.figure("Tessa Ripple", figsize=(15, 9))
        self.ax = fig.add_axes([0.0, 0.07, 0.49, 0.88], projection="3d")
        self.encoded = self.raw = self.raw_key = self.edge_data = None
        self.idx, self.playing, self.sec_per_frame, self.decoded, self.w = 0, False, None, [], {}
        self.sec_per_run = {}  # backend -> seconds per run, for the Run frames estimate

        def radios(x, y, specs):
            """Labelled radio groups stacked down from (x, y) in figure coordinates.

            Args:
                x, y: Top-left of the stack.
                specs: (name, labels) per group; each becomes ``self.w[name]``.

            Returns:
                float: The y below the last group.
            """
            for name, labels in specs:
                h = 0.022 * len(labels) + 0.008
                fig.text(x, y - 0.014, name, fontsize=8, weight="bold")
                y -= 0.018 + h
                self.w[name] = RadioButtons(fig.add_axes([x, y, 0.09, h]), labels)
                y -= 0.01
            return y

        radios(0.50, 0.98, [("Shape", list(PRESETS)), ("Resolution", RES),
                            ("Payload", ["radial", "xyz", "spherical"]), ("Mode", list(MODE_LABELS)),
                            ("Layout", ["mesh", "naive"]), ("Backend", BACKENDS),
                            ("Placement", ["quality", "graph"]), ("DRR", DRR_METHODS)])
        y = radios(0.855, 0.98, [("View", ["fidelity", "material"]), ("Colour", COLOURS)])
        for name, y0, lo, hi, init, step in [
            # tier 1: encoder
            ("Clamp θ (π)", 0.95, 0, 0.45, 0.0, None), ("Clamp φ (π)", 0.92, 0, 0.45, 0.0, None),
            ("Compression %", 0.89, 0, 100, 0, 1), ("Quant bits", 0.86, 0, 8, 0, [0, 2, 3, 4, 5, 6, 7, 8]),
            ("Headroom", 0.83, 0, 0.5, 0.1, None),
            # tier 2: readout
            ("Shots (log2)", 0.79, 8, 16, 12, 1), ("Noise × (flat)", 0.76, 0, 3, 1, None),
            ("Seed", 0.73, 0, 999, 0, 1),
            # tier 3: post + animation
            ("Gain (material)", 0.69, 0, 10, 3, None),
            ("Frames", 0.65, 2, 60, 12, 1), ("In-between", 0.62, 0, 30, 8, 1),
            ("Speed (fps)", 0.59, 1, 60, 24, 1),
        ]:
            self.w[name] = s = Slider(fig.add_axes([0.665, y0, 0.12, 0.02]), name, lo, hi,
                                      valinit=init, valstep=step)
            s.label.set_fontsize(8)
        self.w["Toggles"] = CheckButtons(fig.add_axes([0.665, 0.42, 0.15, 0.115]),
                                         ["Loop", "Start from original", "Auto-run", "Qubit map"],
                                         [True, True, False, False])
        self.build_morph_box()
        # qubit map: the lattice or chip, with the hovered vertex's qubits in orange
        self.max = fig.add_axes([0.60, 0.15, 0.245, 0.255], visible=False)
        self.map_text = fig.text(0.60, 0.14, "", fontsize=7, family="monospace", va="top")
        self.focus = None
        buttons = [("Run", self.run), ("Run frames", self.run_frames), ("Play/Pause", self.toggle),
                   ("Read edges", self.on_edges), ("Curves", self.on_curves),
                   ("Why Tessa", self.on_resources), ("Export OBJ", self.export_obj),
                   ("Export GIF", self.export_gif)]
        for i, (label, cb) in enumerate(buttons):
            self.w[label] = b = Button(fig.add_axes([0.855, y - 0.045 - i * 0.045, 0.13, 0.037]), label)
            b.on_clicked(cb)
        self.status = fig.text(0.01, 0.02, "", fontsize=9, family="monospace")
        self.info = fig.text(0.855, y - 0.06 - len(buttons) * 0.045, "", fontsize=8,
                             family="monospace", va="top")

        for name in ("Shape", "Resolution", "Payload", "Mode", "Layout", "Backend", "Placement", "Clamp θ (π)",
                     "Clamp φ (π)", "Compression %", "Quant bits", "Headroom", "Shots (log2)",
                     "Noise × (flat)", "Seed", "Morph t"):
            w = self.w[name]
            (w.on_clicked if isinstance(w, RadioButtons) else w.on_changed)(self.quantum_changed)
        self.w["Morph"].on_clicked(self.quantum_changed)
        self.w["Upload B"].on_clicked(self.upload_b)
        for name in ("DRR", "View", "Colour", "Gain (material)", "In-between", "Toggles"):
            w = self.w[name]
            (w.on_clicked if hasattr(w, "on_clicked") else w.on_changed)(self.redraw)
        self.w["Speed (fps)"].on_changed(self.set_speed)
        fig.canvas.mpl_connect("motion_notify_event", self.hover)

        self.cax = fig.add_axes([0.04, 0.085, 0.19, 0.012])  # face colour legend
        self.eax = fig.add_axes([0.27, 0.085, 0.19, 0.012])  # edge colour legend
        self.coll = Poly3DCollection([], edgecolor="#20304a", lw=0.4)
        self.edge_coll = Line3DCollection([np.zeros((2, 3))], linewidths=2.5, visible=False)
        self.ax.add_collection3d(self.coll)
        self.ax.add_collection3d(self.edge_coll, autolim=False)
        self.mates, = self.ax.plot([], [], [], "o", ms=8, mfc=HOT, mec="#14161e", mew=1.5, zorder=10)
        lim = 1.4
        self.ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim), box_aspect=(1, 1, 1))
        self.ax.set_axis_off()
        px = IdentityTransform()  # hover readout (paused only), in display pixels
        self.marker = fig.add_artist(Line2D([], [], marker="o", ms=9, mfc="none", mec="#d0342c",
                                            mew=2, transform=px, visible=False))
        self.tip = fig.text(0, 0, "", transform=px, fontsize=8, family="monospace", visible=False,
                            bbox=dict(boxstyle="round", fc="white", ec="#888", alpha=0.95))
        self.anim = FuncAnimation(fig, self.tick, interval=1000 / 24, cache_frame_data=False)
        self.quantum_changed()

    def build_morph_box(self):
        """The Morph options inset (lower left of the 3D view, above the colour legends),
        shown only while Mode is morph.

        Adds the Morph, Upload B and Morph t widgets to ``self.w`` and sets
        ``self.morph_artists``, all hidden.
        """
        fig, x0, y0 = self.fig, 0.008, 0.11
        box = fig.add_axes([x0, y0 + 0.07, 0.215, 0.2], facecolor="#f4f2ec")
        box.set_xticks([])
        box.set_yticks([])
        box.text(0.03, 0.96, "Morph options", transform=box.transAxes, fontsize=8, weight="bold", va="top")
        self.morph_note = box.text(0.03, 0.87, "", transform=box.transAxes, fontsize=7, va="top")
        self.w["Morph"] = CheckButtons(fig.add_axes([x0 + 0.004, y0 + 0.152, 0.2, 0.07]),
                                       ["Shared topology", "Hold t", "A's directions only",
                                        "Interference readout"],
                                       [False, False, False, False])
        self.w["Upload B"] = Button(fig.add_axes([x0 + 0.006, y0 + 0.118, 0.07, 0.03]), "Upload B (.obj)")
        self.b_label = fig.text(x0 + 0.082, y0 + 0.133, "no B yet", fontsize=7, family="monospace", va="center")
        self.w["Morph t"] = Slider(fig.add_axes([x0 + 0.05, y0 + 0.085, 0.13, 0.02]), "Morph t", 0, 1, valinit=0)
        for t in self.w["Morph"].labels:
            t.set_fontsize(7.5)
        self.w["Upload B"].label.set_fontsize(7.5)
        self.w["Morph t"].label.set_fontsize(8)
        self.morph_artists = [box, self.b_label, *(self.w[k].ax for k in ("Morph", "Upload B", "Morph t"))]
        for a in self.morph_artists:
            a.set_visible(False)

    def show_morph_box(self, on: bool):
        """Show or hide the Morph options inset.

        Args:
            on: Show the inset (Mode is morph).
        """
        for a in self.morph_artists:
            a.set_visible(on)

    def upload_b(self, *_):
        """Pick mesh B with a file dialog and re-encode; a cancelled dialog changes nothing.

        Args:
            *_: Ignored button callback arguments.

        Updates self.b_path.
        """
        path = choose_obj()
        if not path:  # cancelled: nothing changes
            return
        self.b_path = path
        self.quantum_changed()

    # --- settings ----------------------------------------------------------
    def val(self, name):
        """Current value of a widget.

        Args:
            name: Key in ``self.w`` of a RadioButtons or Slider.

        Returns:
            The selected label (str) for a radio, else the slider value (float).
        """
        w = self.w[name]
        return w.value_selected if isinstance(w, RadioButtons) else w.val

    def params(self) -> Params:
        """The widgets' settings as Params.

        Returns:
            Params: Encoder and readout settings. Clamp sliders at 0 mean Tessa's default
            (None); in morph, clamp_b follows Clamp θ.
        """
        qb = int(self.val("Quant bits"))
        clamp = lambda name: self.val(name) * np.pi or None  # noqa: E731  0 = Tessa default
        mode = MODE_LABELS[self.val("Mode")]
        morph = self.morph_opts()
        return Params(payload=self.val("Payload"), layout=self.val("Layout"), mode=mode,
                      clamp=clamp("Clamp θ (π)"),
                      clamp_b=clamp("Clamp θ (π)" if mode == "morph" else "Clamp φ (π)"),
                      compression=self.val("Compression %"), quant_bits=qb or None,
                      headroom=self.val("Headroom"), backend=self.val("Backend"),
                      placement=self.val("Placement"),
                      target=self.b_path if mode == "morph" else None, shared_topology=morph["shared"],
                      shots=2 ** int(self.val("Shots (log2)")),
                      noise_strength=self.val("Noise × (flat)"), seed=int(self.val("Seed")),
                      morph_t=float(self.val("Morph t")))

    def toggles(self):
        """The Toggles check boxes.

        Returns:
            dict: "loop", "orig" (start from original), "auto" (auto-run), "map" (qubit map) -> bool.
        """
        return dict(zip(["loop", "orig", "auto", "map"], self.w["Toggles"].get_status()))

    def morph_opts(self):
        """The Morph check boxes.

        Returns:
            dict: "shared" (shared topology), "hold" (hold t), "a_dirs" (A's directions only),
            "interf" (interference readout, :mod:`~tessa_ripple.interfere`) -> bool.
        """
        return dict(zip(["shared", "hold", "a_dirs", "interf"], self.w["Morph"].get_status()))

    @property
    def morphing(self):
        """Whether the current encoding is a morph.

        Returns:
            bool: True if a morph mesh is encoded.
        """
        return self.encoded is not None and self.encoded.mode == "morph"

    @property
    def interfering(self):
        """Whether the current encoding is an interference morph (``interfere.encode_interfere``).

        Returns:
            bool: True if the encoded morph is read by turning selector qubits.
        """
        return self.morphing and self.ie is not None and self.ie.e is self.encoded

    def frame_ts(self, n):
        """Morph t of each of n frames: the slider's t (Hold t, or one frame), else 0 -> 1
        over the run.

        Args:
            n: Number of frames.

        Returns:
            list[float]: One t per frame.
        """
        return [float(self.val("Morph t"))] * n if self.morph_opts()["hold"] or n == 1 else \
            list(np.linspace(0, 1, n))

    def set_quiet(self, name, value):
        """Move a widget without firing its callbacks.

        Args:
            name: Key in ``self.w``.
            value: Radio index to activate, or slider value.
        """
        w = self.w[name]
        w.eventson = False
        w.set_active(value) if isinstance(w, RadioButtons) else w.set_val(value)
        w.eventson = True

    def sync_morph_controls(self, mode):
        """Mode morph: payload to xyz on entry, Clamp φ greyed out and following Clamp θ.

        Args:
            mode: The canonical mode now selected (one of ``bridge.MODES``).

        Updates the Payload and Clamp φ widgets and self.mode.
        """
        if mode == "morph" and self.mode != "morph":
            self.set_quiet("Payload", 1)  # xyz
        phi = self.w["Clamp φ (π)"]
        if mode == "morph":
            self.set_quiet("Clamp φ (π)", self.val("Clamp θ (π)"))
        phi.set_active(mode != "morph")
        grey = "#a8a8a8" if mode == "morph" else "black"
        phi.label.set_color(grey)
        phi.valtext.set_color(grey)
        phi.poly.set_alpha(0.3 if mode == "morph" else 1.0)
        self.mode = mode

    def a_source(self, shape, res_name):
        """A's decimation logs and full vertex count (an uploaded OBJ at mid/low), for
        Shared topology.

        Args:
            shape: Shape name.
            res_name: Resolution name of the shape's preset ("low", "mid", "high").

        Returns:
            tuple: (logs, full_count): one collapse log per decimation pass and A's vertex count
            before decimation; ((), None) for a built-in shape.
        """
        if shape not in self.objs:
            return (), None
        path = self.objs[shape]  # both cached: the Shape preset already decimated it
        return load_decimated(path, obj_passes(path, res_name))[2], len(load_decimated(path)[0])

    @property
    def fidelity(self):
        """Whether View is fidelity (gain 1, true decoded positions).

        Returns:
            bool: True for the fidelity view, False for material.
        """
        return self.val("View") == "fidelity"

    def set_resolution(self, label):
        """Move the Resolution radio without triggering a re-encode.

        Args:
            label: Resolution name, one of ``RES``.
        """
        self.set_quiet("Resolution", RES.index(label))

    def quantum_changed(self, *_):
        """Encoder or readout setting changed: re-encode if the encoder key changed (and,
        in morph, re-match B if needed), refresh the info panel, then auto-run or redraw.

        Args:
            *_: Ignored widget callback arguments.

        Updates self.verts, self.faces, self.encoded, self.enc_key, self.match, self.qmap
        and drops cached frames and edges when the mesh is re-encoded.

        Returns:
            None (early returns only report a status message).
        """
        self.sync_morph_controls(MODE_LABELS[self.val("Mode")])
        p, shape = self.params(), self.val("Shape")
        if shape != self.shape and shape in self.objs:  # an uploaded mesh opens at its own size
            self.set_resolution("high")
        self.shape = shape
        presets = list(PRESETS[shape].items())
        name, make = presets[min(RES.index(self.val("Resolution")), len(presets) - 1)]
        interf_on = p.mode == "morph" and self.morph_opts()["interf"]
        key = (shape, name, p.encoder_key, interf_on)
        self.b_label.set_text(f"{Path(self.b_path).name}" if self.b_path else "no B yet")
        self.error = None  # set again below if this encode fails
        if p.mode == "morph" and self.b_path is None:
            self.show_morph_box(True)
            return self.say("Morph: upload a B mesh")
        if self.encoded is None or key != self.enc_key:
            try:
                verts, faces = make()
                target = None
                if p.mode == "morph":
                    mkey = (shape, name, p.target, p.shared_topology)
                    if mkey != self.match_key:
                        logs, full = self.a_source(shape, name)
                        self.say("Morph: matching B to A …")
                        self.match = correspond(verts, faces, p.target, p.shared_topology, logs, full)
                        self.match_key = mkey
                    target = self.match
                if interf_on:  # A and B in the two slots of each selector, one field
                    ie = interf.encode_interfere(verts, faces, p, target)
                    encoded = ie.e
                else:
                    ie, encoded = None, encode_mesh(verts, faces, p, target)
            except (ValueError, OSError) as err:  # e.g. an unreadable OBJ (Shape or B)
                self.show_morph_box(p.mode == "morph")
                self.error = f"⚠ {err}"
                return self.say(self.error)
            self.error = None
            # swap everything together, before say()/flush_events lets a tick draw
            self.verts, self.faces, self.encoded, self.enc_key = verts, faces, encoded, key
            self.ie, self.ideal_cache, self.kept = ie, {}, None
            self.face_mean = np.zeros((len(faces), len(verts)))  # face colour = mean of its corners
            for k, f in enumerate(faces):
                self.face_mean[k, f] = 1 / len(f)
            self.vq, self.qmap, self.focus = vertex_qubits(encoded), qubit_map(encoded), None
            self.raw = None
            if "Read edges" in self.w:  # a new mesh drops the old mesh's edges
                self.set_edges(None)
            self.seq, self.idx = [verts], 0
        s = self.encoded.stats
        self.show_morph_box(self.morphing)
        self.morph_note.set_text("Interference: t turns the selectors from A (0) to B (1)" if self.interfering
                                 else "Basis sweep: t turns the readout from Z (A) to X (B)")
        lines = [f"{shape} {name}: {len(self.verts)} verts", f"payload {p.payload}, {p.layout} layout",
                 s["graph"], *([f"data-qubit quality {s['quality']:.2f}"] if s["quality"] is not None else []),
                 f"{s['values']} values on {s['qubits']} qubits"
                 + (f" (of {s['graph_qubits']})" if s["qubits"] < s["graph_qubits"] else ""),
                 f"= {s['ratio']:.2f} values / qubit", f"(angle encoding: {s['values']} qubits)",
                 f"capacity used {s['values']}/{s['capacity']}",
                 f"depth {s['depth']}, 2q gates {s['twoq']}"]
        if self.encoded.pair:  # a second field in φ costs gates (see README Findings)
            lines.append(f"θ field alone: {s['twoq_single']} 2q")
        if self.encoded.mode == "split":
            lines.append(f"{int(self.encoded.fields.sum())} values in φ")
        elif self.morphing:
            m, e = self.match, self.encoded
            n_vals = e.slots.size
            at = next(k for k, line in enumerate(lines) if " values on " in line)
            lines[at] = f"values: 2 × {n_vals} on {s['qubits']} qubits"
            if s["runs"] > 1:  # one run's values share the qubits, so count per run
                top = max(s["values_per_run"])
                lines[at + 1] = f"= {top / s['qubits']:.2f} values / qubit per run"
                lines[at + 3] = f"capacity used {top}/{s['capacity']} per run"
                lines.insert(at + 1, f"{s['runs']} runs: " + " + ".join(map(str, s["values_per_run"])) + " values")
            cov = f", covered {m['coverage']:.0%}" if m["method"] == "closest point" else ""
            lines += [f"B {Path(p.target).name}: {len(m['verts'])} verts", f"match: {m['method']}",
                      f"|B − A| {m['distance']:.3f}{cov}"]
            if self.interfering:
                c = interf.cost(self.ie)
                lines[next(k for k, line in enumerate(lines) if line.startswith("capacity used"))] = (
                    f"A/B pairs used {e.slots.size}/{c['capacity']}" + (" per run" if s["runs"] > 1 else ""))
                lines += [f"{c['stars']} stars: selector 0 = A, 1 = B", f"clamp {morph_clamp(p) / np.pi:.2f}π"]
            else:
                lines.append(f"θ = A, φ = B, clamp {morph_clamp(p) / np.pi:.2f}π")
        self.info.set_text("\n".join(lines))
        if self.toggles()["auto"]:
            return self.run()
        stale = self.raw is not None and self.raw_key != self.decode_key(p)
        est, per_run = "", self.sec_per_run.get(p.backend)
        if per_run:  # last measured time of one run on this backend, times this mesh's runs
            n, sec = int(self.val("Frames")), per_run * s["runs"]
            est = f"  ≈{sec:.1f}s/frame, Run frames ≈{sec * n:.0f}s"
        elif s["runs"] > 1:
            est = f"  ({s['runs']} runs per frame: frame time × {s['runs']})"
        if self.interfering and p.backend in DEVICES and not per_run:  # noisy star clusters
            est += "  interference on a chip: a minute or more per frame"
        self.say(("frames stale: press Run / Run frames" if stale else "ready") + est)
        self.redraw()

    def decode_key(self, p, n=None):
        """What the cached frames were read with. Morph frames also depend on each
        frame's t (the basis sweep reads at that t).

        Args:
            p: The readout's Params.
            n: Number of frames; defaults to ``len(self.raw)``.

        Returns:
            tuple: (enc_key, p.readout_key, ts marker) to compare with ``self.raw_key``.
        """
        n, ts = n or len(self.raw), ()
        if p.mode == "morph" and not self.morph_opts()["hold"] and n > 1:
            p, ts = replace(p, morph_t=0.0), ("t 0 → 1", n)  # the slider's t is not used
        return (self.enc_key, p.readout_key, ts)

    # --- quantum work --------------------------------------------------------
    def _decode(self, n):
        """Read n frames from the current encoding with ``animate.run_frames``.

        Args:
            n: Number of readouts (seeds seed, seed+1, ...).

        Updates self.raw, self.raw_ts, self.raw_key, timing estimates and
        self.idx, then redraws.
        """
        if self.error:  # the last encode failed: keep its message, not a stale mesh
            return self.say(self.error)
        if self.encoded is None or (MODE_LABELS[self.val("Mode")] == "morph" and not self.morphing):
            return self.say("Morph: upload a B mesh")
        p, t0 = self.params(), time.time()
        runs = self.encoded.runs
        prog = lambda i, n: self.say(f"frame {i + 1} / {n}" + (f" ({runs} runs each)" if runs > 1 else "")  # noqa: E731
                                     + " …")
        ts = self.frame_ts(n) if self.morphing else None
        if self.interfering:  # each frame turns the selectors to its own t
            self.raw, self.kept = interf.run_frames(self.ie, p, ts, progress=prog)
        else:
            self.raw, self.kept = anim.run_frames(self.encoded, p, n, progress=prog, ts=ts), None
        self.raw_ts = ts
        self.raw_key = self.decode_key(p, n)
        self.sec_per_frame = (time.time() - t0) / n
        self.sec_per_run[p.backend] = self.sec_per_frame / runs
        self.idx = 0
        kept = f", decoded from {self.kept.mean():.0%} of shots (selector 0)" if self.kept is not None else ""
        self.say(f"{n} frame(s) in {time.time() - t0:.1f}s ({self.sec_per_frame:.2f}s/frame){kept}")
        self.redraw()

    def run(self, *_):
        """Run button: one readout, shown at once (playback stopped).

        Args:
            *_: Ignored button callback arguments.
        """
        self.playing = False
        self._decode(1)
        self.idx = len(self.seq) - 1  # show the result, not the start-from-original key
        self.draw()

    def run_frames(self, *_):
        """Run frames button: ``Frames`` readouts, then play them.

        Args:
            *_: Ignored button callback arguments.
        """
        self._decode(int(self.val("Frames")))
        self.playing = True

    def set_edges(self, data):
        """Show (data) or hide (None) the edge overlay; the button names the next action.

        Args:
            data: (edges (E, 2), measured (E,), exact (E,)) from ``bridge.read_edges``, or None.
        """
        self.edge_data = data
        self.w["Read edges"].label.set_text("Hide edges" if data else "Read edges")

    def on_edges(self, *_):
        """Read edges button: read every edge length from coherences, or hide the overlay
        if it is shown.

        Args:
            *_: Ignored button callback arguments.
        """
        if self.edge_data:  # the same button turns the overlay off
            self.set_edges(None)
            self.say("edge overlay hidden")
            return self.redraw()
        if self.error:  # the last encode failed: never act on the stale mesh
            return self.say(self.error)
        self.playing, t0 = False, time.time()
        self.say("reading edge lengths from coherences …")
        E, measured, exact = read_edges(self.verts, self.faces, self.params())
        self.set_edges((E, measured, exact))
        rel = np.abs(measured - exact).mean() / exact.mean()
        self.say(f"{len(E)} edge lengths read from the state ({3 * len(E)} pair circuits, "
                 f"{time.time() - t0:.1f}s): mean error {rel:.1%} of the mean edge")
        self.redraw()

    def on_resources(self, *_):
        """Why Tessa button: open a table of Tessa's resources against four baseline
        encodings (``bridge.resources``), transpiled onto ibm_phoenix.

        Args:
            *_: Ignored button callback arguments.
        """
        if self.error:  # the last encode failed: never act on the stale mesh
            return self.say(self.error)
        self.say("transpiling Tessa and four baselines onto ibm_phoenix …")
        where = [np.concatenate([self.encoded.slots.ravel(), self.ie.b_slots.ravel()])] if self.interfering else None
        rows, shots = resources(self.encoded, where), self.params().shots
        fig = plt.figure("Why Tessa", figsize=(11, 3.8), clear=True)
        ax = fig.add_axes([0.01, 0.2, 0.98, 0.68])
        ax.axis("off")
        cols = ["encoding", "qubits", "fits on chip", "2q gates", "depth", "SWAPs",
                "settings", "outcomes / cell", f"shots / value ({shots})"]
        body = [[r["name"], r["qubits"], "yes" if r["fits"] else "no", r["twoq"], r["depth"],
                 r["swaps"], r["settings"], r["configs"], shots // r["configs"]] for r in rows]
        table = ax.table(cellText=body, colLabels=cols, loc="center", cellLoc="center",
                         colWidths=[0.2] + [0.1] * 8)
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.6)
        s = self.encoded.stats
        fig.suptitle(f"{s['values']} values ({self.encoded.payload}, mode {self.encoded.mode}) "
                     f"transpiled onto the ibm_phoenix snapshot (120 qubits, square lattice)", fontsize=10)
        fig.text(0.01, 0.02, "Generic random data in every row (no compression). SWAPs ≈ extra CZs from "
                 "routing / 3. 'outcomes / cell': address configurations each shot is split over, so "
                 "each value gets about shots / outcomes.\nAmplitude: Z readout gives magnitudes only. "
                 "With a chip backend, Tessa's row is on that chip (identity layout); the baselines stay on Phoenix.",
                 fontsize=7.5)
        fig.show()
        self.say("Why Tessa: resource table opened")

    def on_curves(self, *_):
        """Curves button: plot error vs shots and error/depth vs compression for both
        layouts (``bridge.fidelity_curves``). Not available in morph mode.

        Args:
            *_: Ignored button callback arguments.
        """
        if self.error:  # the last encode failed: never act on the stale mesh
            return self.say(self.error)
        if self.morphing:
            return self.say("Curves compare against one fixed mesh; not available in morph mode")
        prog = lambda k, n: self.say(f"curves: run {k} / {n} …")  # noqa: E731
        data = fidelity_curves(self.verts, self.faces, self.params(), progress=prog)
        fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 4.4), num="Fidelity curves", clear=True)
        s, err = map(np.array, zip(*data["shots"]))
        a1.loglog(s, err, "o-", c=LAYOUT_STYLE["mesh"]["c"], label="measured")
        a1.loglog(s, err[0] * np.sqrt(s[0] / s), ":", c="grey", label="1/√shots")
        a1.set(xlabel="shots", ylabel="mean error (fraction of bbox)", title="error vs shots")
        a1.legend(fontsize=8)
        for layout in ("mesh", "naive"):  # mesh: solid, hollow circles; naive: dashed squares,
            c, depth, e = zip(*data[layout])  # so both show even where the lines coincide
            kw = dict(LAYOUT_STYLE[layout], label=f"{layout} layout")
            a2.plot(c, e, **kw)
            a3.plot(c, depth, **kw)
        a2.set(xlabel="compression %", ylabel="mean error (fraction of bbox)", title="error vs compression")
        a3.set(xlabel="compression %", ylabel="circuit depth", title="depth vs compression")
        for ax in (a2, a3):
            ax.legend(fontsize=8)
        if data["same"]:
            fig.text(0.5, 0.01, "Mesh and naive layouts place every value identically for this mesh, "
                     "so their lines coincide.",
                     ha="center", fontsize=8, color="#6b6f78")
        fig.tight_layout(rect=(0, 0.04 if data["same"] else 0, 1, 1))
        fig.show()
        self.say("Fidelity curves opened")

    # --- classical layer -----------------------------------------------------
    def morph_refs(self, ts):
        """Noiseless morph positions at each t: what each frame is compared against.

        Args:
            ts: Morph t per frame.

        Returns:
            list[np.ndarray]: (N, 3) expected positions per t.
        """
        a_dirs = self.morph_opts()["a_dirs"]
        if self.interfering:  # exact noiseless frame (star statevectors), cached per t
            ideal = [self.ideal_cache.setdefault(float(t), interf.ideal(self.ie, float(t))) for t in ts]
            return [interf.frame(self.ie, x, t, "none", a_dirs) for x, t in zip(ideal, ts)]
        return [morph_expected(self.encoded, t, a_dirs) for t in ts]

    def positions(self):
        """Playback positions. Every frame has its own reference (the original mesh, or
        in morph the noiseless morph at that frame's t); error and gain act on the
        deviation from it.

        Returns:
            list[np.ndarray]: (N, 3) positions per playback step, including in-betweens and
            the optional leading reference.

        Updates self.decoded, self.refs, self.err, self.ref_seq, self.err_seq, self.err_top.
        """
        if self.raw is None:
            self.decoded, self.err_seq, self.refs = [], None, None
            if self.morphing:  # classical preview of the morph at the slider's t
                return self.morph_refs([float(self.val("Morph t"))])
            return [self.verts]
        e, drr, n = self.encoded, self.val("DRR"), len(self.raw)
        if self.morphing:
            a_dirs = self.morph_opts()["a_dirs"]
            if self.interfering:
                self.decoded = [interf.frame(self.ie, r, t, drr, a_dirs) for r, t in zip(self.raw, self.raw_ts)]
            else:
                self.decoded = [morph_frame(e, r, t, drr, a_dirs) for r, t in zip(self.raw, self.raw_ts)]
            refs = self.morph_refs(self.raw_ts)
        else:
            self.decoded = [restore(e, r, drr) for r in self.raw]
            refs = [self.verts] * n
        self.refs = refs
        self.err = np.mean([mesh_error(ref, d) for ref, d in zip(refs, self.decoded)])
        if self.fidelity:  # true decoded positions, nothing exaggerated
            keys = self.decoded
        else:
            keys = [anim.blend(ref, d, self.val("Gain (material)")) for ref, d in zip(refs, self.decoded)]
        t = self.toggles()
        lead = [refs[0]] if t["orig"] else []
        glide = lambda k: k if len(k) == 1 else anim.timeline(k, int(self.val("In-between")), t["loop"])  # noqa: E731
        # error colour follows playback: each step's true decoded error (same timeline,
        # no gain) from its reference, on one scale fixed to the largest error in the run
        self.ref_seq = glide(lead + refs)
        self.err_seq = np.array([np.linalg.norm(v - r, axis=1)
                                 for v, r in zip(glide(lead + self.decoded), self.ref_seq)])
        self.err_top = self.err_seq.max() or 1.0
        return glide(lead + list(keys))

    def vertex_rgba(self):
        """Per-vertex colours for the Colour mode.

        Returns:
            tuple: (rgba (N, 4), title label, legend spec (cmap, low, high, caption) or None).
        """
        mode, e = self.val("Colour"), self.encoded
        if mode == "error" and self.decoded:  # true decoded error, whatever the view's gain
            n = len(self.decoded)
            return (ERROR_CMAP(self.err_seq[self.idx] / self.err_top),
                    f"error per vertex at this step (scale: largest over {n} readout{'s' * (n > 1)})",
                    (ERROR_CMAP, 0, self.err_top, "vertex error at this step (distance, mesh units)"))
        if mode == "qubit":
            # naive xyz puts x, y, z on different qubits
            what = "data qubit holding each vertex's x (naive layout splits x, y, z)" if e.straddles \
                else "data qubit holding each vertex"
            return qubit_rgba(self.vq), f"{what}, {len(np.unique(self.vq))} qubits", None
        if mode == "morph Δ" and e.target is not None:
            d = np.linalg.norm(e.target - self.verts, axis=1)
            top = d.max() or 1.0
            return (DELTA_CMAP(d / top), "morph Δ: |B − A| per vertex after alignment",
                    (DELTA_CMAP, 0, top, "|B − A| (mesh units)"))
        why = {"error": " (run to see error)", "morph Δ": " (Mode morph only)"}.get(mode, "")
        return np.tile(PLAIN, (len(self.verts), 1)), "plain" + why, None

    def legend(self, ax, spec):
        """Draw a horizontal colour bar, or hide it.

        Args:
            ax: The legend axes.
            spec: (cmap, low, high, caption), or None to hide.
        """
        ax.clear()
        ax.set_visible(spec is not None)
        if spec:
            cmap, lo, hi, caption = spec
            self.fig.colorbar(ScalarMappable(Normalize(lo, hi), cmap), cax=ax, orientation="horizontal")
            ax.tick_params(labelsize=7)
            ax.set_xlabel(caption, fontsize=7, labelpad=1)

    def redraw(self, *_):
        """Rebuild playback positions, colours, legends and the qubit map, then draw.

        Args:
            *_: Ignored widget callback arguments.

        Updates self.seq, self.idx, self.face_rgba, self.edge_rgba, self.colour_label.
        """
        self.seq = self.positions()
        self.idx %= len(self.seq)
        rgba, self.colour_label, spec = self.vertex_rgba()
        self.face_rgba = self.face_mean @ rgba
        self.legend(self.cax, spec)
        if self.edge_data:
            E, measured, exact = self.edge_data
            norm = Normalize(exact.min(), exact.max())  # true edge range; off-range values clip
            self.edge_rgba = EDGE_CMAP(norm(measured))
            self.legend(self.eax, (EDGE_CMAP, exact.min(), exact.max(), "edge length read from coherences"))
        else:
            self.legend(self.eax, None)
        if not self.toggles()["map"]:
            self.focus = None
        self.draw_map()
        self.draw()

    def draw(self):
        """Draw the current step: faces, edge overlay, highlighted vertices and title."""
        v = self.seq[self.idx]
        self.tip.set_visible(False)  # mesh moved, so any hover readout is stale
        self.marker.set_visible(False)
        self.coll.set_verts([v[f] for f in self.faces])
        if self.val("Colour") == "error" and self.decoded:  # recolour for this step
            self.face_rgba = self.face_mean @ ERROR_CMAP(self.err_seq[self.idx] / self.err_top)
        rgba = self.face_rgba.copy()
        rgba[:, 3] = 0.35 if self.edge_data else 1.0
        self.coll.set_facecolor(rgba)
        edge_txt = ""
        if self.edge_data:
            E, measured, exact = self.edge_data
            self.edge_coll.set_segments(v[E])
            self.edge_coll.set_color(self.edge_rgba)
            edge_txt = " · edges: length read from coherences"
        self.edge_coll.set_visible(bool(self.edge_data))
        mates = self.focus["mates"] if self.focus else []
        self.mates.set_data_3d(*v[mates].T) if len(mates) else self.mates.set_data_3d([], [], [])
        if self.fidelity:
            head = "Fidelity · gain 1 (true decoded positions)"
        else:
            head = f"Material · gain {self.val('Gain (material)'):.1f}× (noise exaggerated)"
        err = f" · mean err {self.err:.4f} of bbox" if self.raw is not None else ""
        if self.morphing:
            if self.raw is None:
                head = f"Morph · noiseless preview at t = {self.val('Morph t'):.2f} (press Run)"
            else:
                head = ("Morph interference (selectors) · " if self.interfering
                        else "Morph basis sweep (transition decode) · ") + head
        self.ax.set_title(f"{head}{err}\nstep {self.idx + 1}/{len(self.seq)} · colour: "
                          f"{self.colour_label}{edge_txt}", fontsize=9)
        self.fig.canvas.draw_idle()

    def tick(self, _):
        """Animation timer: advance one step while playing; stop at the end unless Loop.

        Args:
            _: Frame number from FuncAnimation (unused).

        Returns:
            tuple: No artists (the canvas redraws itself).
        """
        if self.playing and len(self.seq) > 1:
            if self.idx + 1 >= len(self.seq) and not self.toggles()["loop"]:
                self.playing = False
            else:
                self.idx = (self.idx + 1) % len(self.seq)
                self.draw()
        return ()

    def toggle(self, *_):
        """Play/Pause button; playing from the last step restarts at the first.

        Args:
            *_: Ignored button callback arguments.
        """
        self.playing = not self.playing
        self.hide_tip()
        if self.playing and self.idx + 1 >= len(self.seq):
            self.idx = 0

    def set_speed(self, fps):
        # macOS timers ignore interval changes while running, so restart the timer
        """Set the playback rate. macOS timers ignore interval changes while running, so
        the timer is restarted.

        Args:
            fps: Frames (steps) per second.
        """
        timer = self.anim.event_source
        timer.stop()
        timer.interval = 1000 / fps
        timer.start()

    def hide_tip(self):
        """Hide the hover readout and its marker if shown."""
        if self.tip.get_visible():
            self.tip.set_visible(False)
            self.marker.set_visible(False)
            self.fig.canvas.draw_idle()

    # --- qubit map -------------------------------------------------------------
    # qmap["vertices"][i][c] = [group, address, field]: value c of vertex i is in data qubit
    # groups[group]["data"], in the slot where address qubit groups[group]["addresses"][j]
    # reads bit j of address (Tessa's decoder bit order).
    def set_focus(self, vertex=None, group=None):
        """Highlight a vertex's data qubit(s), or one data qubit, on the map and mesh.

        Args:
            vertex: Vertex index, or None.
            group: Data qubit (``graph.groups`` index), or None. Both None clears the focus.

        Updates self.focus (only with the qubit map on) and redraws if it changed.
        """
        focus = None
        if self.toggles()["map"] and (vertex is not None or group is not None):
            vs = self.qmap["vertices"]
            groups = sorted({g for g, *_ in vs[vertex]}) if vertex is not None else [group]
            focus = {"vertex": vertex, "groups": groups,
                     "mates": [i for i, v in enumerate(vs) if any(g in groups for g, *_ in v)]}
        if focus == self.focus:
            return
        self.focus = focus
        self.draw_map()
        self.draw()

    def draw_map(self):
        """Draw the lattice or chip qubit map (if the Qubit map toggle is on): couplers,
        address and data qubits, the focused qubits in orange with their address bits.
        """
        ax, on = self.max, self.toggles()["map"]
        ax.clear()
        ax.set_visible(on)
        self.map_text.set_visible(on)
        if not on:
            return
        m, f = self.qmap, self.focus
        xy = np.array(m["coords"], float)
        hot_groups = set(f["groups"]) if f else set()
        hot_data = {m["groups"][g]["data"] for g in hot_groups}
        bits = {q: None for g in hot_groups for q in m["groups"][g]["addresses"]}
        if f and f["vertex"] is not None and len(m["vertices"][f["vertex"]]) == 1:
            g, a, _ = m["vertices"][f["vertex"]][0]  # one value: label each address bit
            bits.update({q: (a >> j) & 1 for j, q in enumerate(m["groups"][g]["addresses"])})
        hot_edge = [(a in hot_data and b in bits) or (b in hot_data and a in bits) for a, b in m["edges"]]
        ax.add_collection(LineCollection(xy[m["edges"]], colors=[HOT if h else "#c9c6bf" for h in hot_edge],
                                         linewidths=[2.2 if h else 1.0 for h in hot_edge], zorder=1))
        span = xy.max(0) + 1
        w, h = ax.get_window_extent().size
        step = min(w / span[0], h / span[1]) * 72 / self.fig.dpi  # points per lattice step
        data = [g["data"] for g in m["groups"]]
        addr = [q for q in range(len(xy)) if q not in set(data)]
        ax.scatter(*xy[addr].T, marker="s", s=[(step * (0.5 if q in bits else 0.34)) ** 2 for q in addr],
                   c=[HOT if q in bits else "#6b6f78" for q in addr], zorder=2)
        used = {g for v in m["vertices"] for g, *_ in v}
        ax.scatter(*xy[data].T, s=(step * 0.6) ** 2, zorder=3,
                   c=[HOT if k in hot_groups else "#3d5a98" if k in used else "#dcdad4"
                      for k in range(len(data))])
        for q, b in bits.items():
            if b is not None:
                ax.text(*xy[q], str(b), ha="center", va="center", fontsize=max(5, step * 0.35),
                        color="white", weight="bold", zorder=4)
        ax.set(xlim=(-0.6, span[0] - 0.4), ylim=(span[1] - 0.4, -0.6), aspect="equal")
        ax.set_axis_off()
        self.map_text.set_text(self.focus_text() + "\n● data qubit (2^k slots, k = its address "
                               "qubits)\n■ address qubit   blue ● = holds part of this mesh")

    def focus_text(self):
        """Caption under the qubit map for the current focus.

        Returns:
            str: Where the hovered vertex's values sit (data qubit, slot, address bits), or
            what the hovered data qubit holds, or a hint when nothing is focused.
        """
        f, m = self.focus, self.qmap
        pair = {"split": " (shape over θ and φ)", "morph": " (θ: A, φ: B)"}.get(self.encoded.mode, "")
        if self.encoded.payload == "spherical":
            pair = " (direction in θ and φ, radius in θ; read in Z, X, Y)"
        if self.interfering:
            pair = " (A at selector 0, B at selector 1)"
        if not f:
            return "Hover a vertex on the mesh, or a data qubit here."
        addrs = lambda g: m["groups"][g]["addresses"]  # noqa: E731
        if f["vertex"] is None:
            g = f["groups"][0]
            held = sum(gg == g for v in m["vertices"] for gg, *_ in v)
            return (f"data q{m['groups'][g]['data']} · address " + " ".join(f"q{q}" for q in addrs(g))
                    + f"\n{2 ** len(addrs(g))} slots, {held} values of this mesh{pair}"
                    + f"\n{len(f['mates'])} vertices on it (orange on the mesh)")
        e, i = self.encoded, f["vertex"]
        names = (["polar", "azimuth", "r"] if e.payload == "spherical"
                 else ["r"] if len(m["vertices"][0]) == 1 else ["x", "y", "z"])
        split, morph = e.mode == "split" or e.payload == "spherical", e.mode == "morph"
        where_b = (lambda c: f" in slot {self.ie.b_slots[i, c] - cells(e.graph)[m['vertices'][i][c][0], 0]}"  # noqa: E731
                   if self.interfering else lambda c: " in φ")
        both = lambda c: (f": A {e.prep.reference.reshape(e.slots.shape)[i, c]:+.3f}"  # noqa: E731
                          + ("" if self.interfering else " in θ")
                          + f", B {e.prep_b.reference.reshape(e.slots.shape)[i, c]:+.3f}{where_b(c)}") if morph else ""
        lines = [f"{names[c]} → data q{m['groups'][g]['data']}, slot {a} of {2 ** len(addrs(g))}"
                 + (f" in {'θφ'[fld]}" if split else "") + ", address "
                 + " ".join(f"q{q}={(a >> j) & 1}" for j, q in enumerate(addrs(g))) + both(c)
                 for c, (g, a, fld) in enumerate(m["vertices"][i])]
        share = "these data qubits" if len(f["groups"]) > 1 else "this data qubit"
        run = f", run {e.batches[i] + 1} of {e.runs}" if e.runs > 1 else ""
        return (f"vertex {i}{pair}{run}\n" + "\n".join(lines)
                + f"\n{len(f['mates'])} vertices share {share} (orange)")

    def hover_map(self, event):
        """Focus the data qubit under the mouse on the qubit map (within 0.45 lattice steps).

        Args:
            event: Matplotlib MouseEvent in the map axes.
        """
        xy = np.array(self.qmap["coords"], float)
        data = np.array([g["data"] for g in self.qmap["groups"]])
        d = np.hypot(*(xy[data] - (event.xdata, event.ydata)).T)
        self.set_focus(group=int(d.argmin()) if d.min() < 0.45 else None)

    def hover(self, event):
        """Nearest vertex: original vs current coordinates while paused; with the qubit
        map on, also its data and address qubits (while playing too).

        Args:
            event: Matplotlib MouseEvent (motion_notify_event).
        """
        map_on = self.toggles()["map"]
        if map_on and event.inaxes is self.max:
            return self.hover_map(event)
        if event.inaxes is not self.ax or (self.playing and not map_on):
            self.set_focus(None)
            return self.hide_tip()
        v = self.seq[self.idx]
        xs, ys, _ = proj3d.proj_transform(*v.T, self.ax.get_proj())
        pix = self.ax.transData.transform(np.column_stack([xs, ys]))
        dist = np.hypot(*(pix - (event.x, event.y)).T)
        i = int(dist.argmin())
        if dist[i] > 12:
            self.set_focus(None)
            return self.hide_tip()
        self.set_focus(vertex=i)
        if self.playing:
            return self.hide_tip()
        o, n = self.verts[i], v[i]
        ref = "original"
        if self.morphing:  # compare with the noiseless morph at this step
            o = self.ref_seq[self.idx][i] if self.raw is not None else self.verts[i]
            ref = "expected" if self.raw is not None else "A       "
        fmt = lambda a: "(" + ", ".join(f"{c:+.4f}" for c in a) + ")"  # noqa: E731
        gain = 1.0 if self.fidelity else self.val("Gain (material)")
        self.tip.set_text(f"vertex {i} on data qubit {self.vq[i]}   (gain {gain:.1f}x, step {self.idx + 1})\n"
                          f"{ref} {fmt(o)}\ncurrent  {fmt(n)}\n"
                          f"delta    {fmt(n - o)}\n|delta|  {np.linalg.norm(n - o):.4f}")
        self.tip.set_position(pix[i] + (14, 6))
        self.marker.set_data([pix[i][0]], [pix[i][1]])
        self.tip.set_visible(True)
        self.marker.set_visible(True)
        self.fig.canvas.draw_idle()

    def say(self, msg):
        """Show a status line and flush the canvas so it appears during long work.

        Args:
            msg: Status text.
        """
        self.status.set_text(msg)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    # --- export --------------------------------------------------------------
    def export_obj(self, *_):
        """Export OBJ button: write every playback step to ``exports/obj/frame_###.obj``.

        Args:
            *_: Ignored button callback arguments.
        """
        paths = anim.export_obj_sequence(OUT / "obj", self.seq, self.faces)
        self.say(f"wrote {len(paths)} OBJ file(s) to {OUT / 'obj'}/")

    def export_gif(self, *_):
        """Export GIF button: the playback as ``exports/ripple.gif``, the 3D view only (no
        controls), at the Speed (fps) setting.

        Args:
            *_: Ignored button callback arguments.
        """
        self.playing = False
        path = OUT / "ripple.gif"
        path.parent.mkdir(exist_ok=True)
        box = self.ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        frames = []
        for i in range(len(self.seq)):
            self.say(f"GIF frame {i + 1} / {len(self.seq)} …")
            self.idx = i
            self.draw()
            buf = io.BytesIO()
            self.fig.savefig(buf, format="png", bbox_inches=box, dpi=72)
            frames.append(Image.open(buf).convert("RGB"))
        frames[0].save(path, save_all=True, append_images=frames[1:], loop=0,
                       duration=int(1000 / self.val("Speed (fps)")))
        self.say(f"wrote {path} ({len(frames)} frames)")


def main():
    """Console entry point: open the app with any OBJ paths given on the command line."""
    app = App(sys.argv[1:])  # keep a reference: matplotlib holds widget callbacks weakly
    plt.show()
    del app


if __name__ == "__main__":
    main()
