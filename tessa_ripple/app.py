"""Matplotlib window: 3D mesh, controls for every Tessa setting, frame playback.

Three tiers: encoder settings re-encode, readout settings re-decode (on Run /
Run frames), post and animation settings only redraw cached frames.

Two views: **Fidelity** (gain 1, vertices coloured by error: the technical case)
and **Material** (gain exaggerates the noise: the artistic mode).
"""

import colorsys
import io
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.transforms import IdentityTransform
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from PIL import Image

from . import animate as anim
from .bridge import (BACKENDS, DRR_METHODS, cells, Params, encode_mesh, fidelity_curves, mesh_error,
                     qubit_map, read_edges, resources, restore, vertex_qubits)
from .mesh import PRESETS, load_obj

OUT = Path("exports")
RES = ("low", "mid", "high")
COLOURS = ("error", "qubit", "φ colour", "plain")
PLAIN = np.array(colormaps["Blues"](0.45))

# Colour definitions, mirrored in public/app.js so both apps colour identically.
# Error: magma_r on [0, 0.88], lightest = no error (pure black is avoided so faces stay readable).
ERROR_CMAP = LinearSegmentedColormap.from_list(
    "error", ["#fcfdbf", "#fd9668", "#cf4070", "#752181", "#1a1042"])
EDGE_CMAP = colormaps["viridis"]
GOLDEN = 0.618034


def hls_rgba(hues, lightness, saturation):
    return np.array([(*colorsys.hls_to_rgb(h % 1, lightness, saturation), 1.0) for h in hues])


def qubit_rgba(qubits):
    """One colour per data qubit in use: hues spread by the golden ratio over each
    qubit's rank, so no two qubits share a colour and neighbouring ranks contrast."""
    rank = np.unique(qubits, return_inverse=True)[1]
    return hls_rgba(rank * GOLDEN, 0.55, 0.6)


HOT = "#e69600"  # qubit map: the hovered data qubit, its address qubits and its vertices
HUE_CMAP = ListedColormap(hls_rgba(np.linspace(0, 1, 256), 0.55, 0.75))  # φ colour value -> hue


class App:
    def __init__(self, objs=()):
        for path in objs:  # e.g. Blender's Suzanne exported as OBJ
            if not Path(path).is_file():
                print(f"tessa-ripple: no OBJ file at {Path(path).resolve()}; skipping it",
                      file=sys.stderr)
                continue
            PRESETS[Path(path).stem] = {"(OBJ)": lambda path=path: load_obj(path)}
        self.fig = fig = plt.figure("Tessa Ripple", figsize=(15, 9))
        self.ax = fig.add_axes([0.0, 0.07, 0.49, 0.88], projection="3d")
        self.encoded = self.raw = self.colours = self.raw_key = self.edge_data = None
        self.idx, self.playing, self.sec_per_frame, self.decoded, self.w = 0, False, None, [], {}

        def radios(x, y, specs):  # labelled radio groups stacked down from y
            for name, labels in specs:
                h = 0.024 * len(labels) + 0.008
                fig.text(x, y - 0.014, name, fontsize=8, weight="bold")
                y -= 0.018 + h
                self.w[name] = RadioButtons(fig.add_axes([x, y, 0.085, h]), labels)
                y -= 0.01
            return y

        radios(0.50, 0.98, [("Shape", list(PRESETS)), ("Resolution", RES),
                            ("Payload", ["radial", "xyz"]), ("Mode", ["single", "pair"]),
                            ("Layout", ["mesh", "naive"]), ("Backend", BACKENDS),
                            ("Placement", ["quality", "importance", "graph"]), ("DRR", DRR_METHODS)])
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
            ("Gain (material)", 0.69, 0, 10, 3, None), ("Mask %", 0.66, 0, 100, 100, 1),
            ("Frames", 0.62, 2, 60, 12, 1), ("In-between", 0.59, 0, 30, 8, 1),
            ("Speed (fps)", 0.56, 1, 60, 24, 1),
        ]:
            self.w[name] = s = Slider(fig.add_axes([0.665, y0, 0.12, 0.02]), name, lo, hi,
                                      valinit=init, valstep=step)
            s.label.set_fontsize(8)
        self.w["Toggles"] = CheckButtons(fig.add_axes([0.665, 0.42, 0.15, 0.115]),
                                         ["Loop", "Start from original", "Auto-run", "Qubit map"],
                                         [True, True, False, False])
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
                     "Noise × (flat)", "Seed"):
            w = self.w[name]
            (w.on_clicked if isinstance(w, RadioButtons) else w.on_changed)(self.quantum_changed)
        for name in ("DRR", "View", "Colour", "Gain (material)", "Mask %", "In-between", "Toggles"):
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

    # --- settings ----------------------------------------------------------
    def val(self, name):
        w = self.w[name]
        return w.value_selected if isinstance(w, RadioButtons) else w.val

    def params(self) -> Params:
        qb = int(self.val("Quant bits"))
        clamp = lambda name: self.val(name) * np.pi or None  # noqa: E731  0 = Tessa default
        return Params(payload=self.val("Payload"), layout=self.val("Layout"), mode=self.val("Mode"),
                      clamp=clamp("Clamp θ (π)"), clamp_b=clamp("Clamp φ (π)"),
                      compression=self.val("Compression %"), quant_bits=qb or None,
                      headroom=self.val("Headroom"), backend=self.val("Backend"),
                      placement=self.val("Placement"),
                      shots=2 ** int(self.val("Shots (log2)")),
                      noise_strength=self.val("Noise × (flat)"), seed=int(self.val("Seed")))

    def toggles(self):
        return dict(zip(["loop", "orig", "auto", "map"], self.w["Toggles"].get_status()))

    @property
    def fidelity(self):
        return self.val("View") == "fidelity"

    def quantum_changed(self, *_):
        p, shape = self.params(), self.val("Shape")
        presets = list(PRESETS[shape].items())
        name, make = presets[min(RES.index(self.val("Resolution")), len(presets) - 1)]
        key = (shape, name, p.encoder_key)
        if self.encoded is None or key != self.enc_key:
            verts, faces = make()
            try:
                encoded = encode_mesh(verts, faces, p)
            except ValueError as err:
                return self.say(f"⚠ {err}")
            # swap everything together, before say()/flush_events lets a tick draw
            self.verts, self.faces, self.encoded, self.enc_key = verts, faces, encoded, key
            self.face_mean = np.zeros((len(faces), len(verts)))  # face colour = mean of its corners
            for k, f in enumerate(faces):
                self.face_mean[k, f] = 1 / len(f)
            self.vq, self.qmap, self.focus = vertex_qubits(encoded), qubit_map(encoded), None
            self.raw = self.colours = None
            if "Read edges" in self.w:  # a new mesh drops the old mesh's edges
                self.set_edges(None)
            self.seq, self.idx = [verts], 0
        s = self.encoded.stats
        lines = [f"{shape} {name}: {len(self.verts)} verts", f"payload {p.payload}, {p.layout} layout",
                 s["graph"], *([f"data-qubit quality {s['quality']:.2f}"] if s["quality"] is not None else []),
                 f"{s['values']} values on {s['qubits']} qubits"
                 + (f" (of {s['graph_qubits']})" if s["qubits"] < s["graph_qubits"] else ""),
                 f"= {s['ratio']:.2f} values / qubit", f"(angle encoding: {s['values']} qubits)",
                 f"capacity used {s['values']}/{s['capacity']}",
                 f"depth {s['depth']}, 2q gates {s['twoq']}"]
        if self.encoded.pair:
            lines.append(f"θ field alone: {s['twoq_single']} 2q")
        self.info.set_text("\n".join(lines))
        if self.toggles()["auto"]:
            return self.run()
        stale = self.raw is not None and self.raw_key != (self.enc_key, p.readout_key)
        est = f"  ≈{self.sec_per_frame:.1f}s/frame" if self.sec_per_frame else ""
        self.say(("frames stale: press Run / Run frames" if stale else "ready") + est)
        self.redraw()

    # --- quantum work --------------------------------------------------------
    def _decode(self, n):
        p, t0 = self.params(), time.time()
        prog = lambda i, n: self.say(f"frame {i + 1} / {n} …")  # noqa: E731
        self.raw, self.colours = anim.run_frames(self.encoded, p, n, progress=prog)
        self.raw_key = (self.enc_key, p.readout_key)
        self.sec_per_frame = (time.time() - t0) / n
        self.idx = 0
        self.say(f"{n} frame(s) in {time.time() - t0:.1f}s ({self.sec_per_frame:.2f}s/frame)")
        self.redraw()

    def run(self, *_):
        self.playing = False
        self._decode(1)
        self.idx = len(self.seq) - 1  # show the result, not the start-from-original key
        self.draw()

    def run_frames(self, *_):
        self._decode(int(self.val("Frames")))
        self.playing = True

    def set_edges(self, data):
        """Show (data) or hide (None) the edge overlay; the button names the next action."""
        self.edge_data = data
        self.w["Read edges"].label.set_text("Hide edges" if data else "Read edges")

    def on_edges(self, *_):
        if self.edge_data:  # the same button turns the overlay off
            self.set_edges(None)
            self.say("edge overlay hidden")
            return self.redraw()
        self.playing, t0 = False, time.time()
        self.say("reading edge lengths from coherences …")
        E, measured, exact = read_edges(self.verts, self.faces, self.params())
        self.set_edges((E, measured, exact))
        rel = np.abs(measured - exact).mean() / exact.mean()
        self.say(f"{len(E)} edge lengths read from the state ({3 * len(E)} pair circuits, "
                 f"{time.time() - t0:.1f}s): mean error {rel:.1%} of the mean edge")
        self.redraw()

    def on_resources(self, *_):
        self.say("transpiling Tessa and four baselines onto ibm_phoenix …")
        rows, shots = resources(self.encoded), self.params().shots
        fig = plt.figure("Why Tessa", figsize=(11, 3.8))
        ax = fig.add_axes([0.01, 0.2, 0.98, 0.68])
        ax.axis("off")
        cols = ["encoding", "qubits", "fits on chip", "2q gates", "depth", "SWAPs",
                "settings", "outcomes / cell", f"shots / value ({shots})"]
        cells = [[r["name"], r["qubits"], "yes" if r["fits"] else "no", r["twoq"], r["depth"],
                  r["swaps"], r["settings"], r["configs"], shots // r["configs"]] for r in rows]
        table = ax.table(cellText=cells, colLabels=cols, loc="center", cellLoc="center",
                         colWidths=[0.2] + [0.1] * 8)
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.6)
        s = self.encoded.stats
        fig.suptitle(f"{s['values']} values ({self.encoded.payload}, {self.encoded.enc.mode}) "
                     f"transpiled onto the ibm_phoenix snapshot (120 qubits, square lattice)", fontsize=10)
        fig.text(0.01, 0.02, "Generic random data in every row (no compression). SWAPs ≈ extra CZs from "
                 "routing / 3. 'outcomes / cell': address configurations each shot is split over, so "
                 "each value gets about shots / outcomes.\nAmplitude: Z readout gives magnitudes only. "
                 "With a chip backend, Tessa's row is on that chip (identity layout); the baselines stay on Phoenix.",
                 fontsize=7.5)
        fig.show()
        self.say("Why Tessa: resource table opened")

    def on_curves(self, *_):
        prog = lambda k, n: self.say(f"curves: run {k} / {n} …")  # noqa: E731
        data = fidelity_curves(self.verts, self.faces, self.params(), progress=prog)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), num="Fidelity curves")
        s, err = map(np.array, zip(*data["shots"]))
        a1.loglog(s, err, "o-", label="measured")
        a1.loglog(s, err[0] * np.sqrt(s[0] / s), ":", c="grey", label="1/√shots")
        a1.set(xlabel="shots", ylabel="mean error (fraction of bbox)", title="error vs shots")
        a1.legend()
        a3 = a2.twinx()
        for layout, mark in (("mesh", "o"), ("naive", "s")):
            c, depth, e = zip(*data[layout])
            a2.plot(c, e, f"{mark}-", label=f"{layout} layout: error")
            a3.plot(c, depth, f"{mark}:", alpha=0.6, label=f"{layout} layout: depth")
        a2.set(xlabel="compression %", ylabel="mean error", title="compression: error and depth")
        a3.set_ylabel("circuit depth")
        a2.legend(loc="upper left", fontsize=8)
        a3.legend(loc="upper center", fontsize=8)
        fig.tight_layout()
        fig.show()
        self.say("Fidelity curves opened")

    # --- classical layer -----------------------------------------------------
    def positions(self):
        if self.raw is None:
            self.decoded, self.err_seq = [], None
            return [self.verts]
        e, drr = self.encoded, self.val("DRR")
        self.decoded = [restore(e, r, drr) for r in self.raw]
        self.err = np.mean([mesh_error(self.verts, d) for d in self.decoded])
        if self.fidelity:  # true decoded positions, nothing exaggerated or masked
            keys = self.decoded
        else:
            mask = anim.random_mask(len(self.verts), self.val("Mask %") / 100, self.params().seed)
            keys = [anim.blend(self.verts, d, self.val("Gain (material)"), mask) for d in self.decoded]
        t = self.toggles()
        lead = [self.verts] if t["orig"] else []
        glide = lambda k: k if len(k) == 1 else anim.timeline(k, int(self.val("In-between")), t["loop"])  # noqa: E731
        # error colour follows playback: each step's true decoded error (same timeline,
        # no gain or mask), on one scale fixed to the largest error anywhere in the run
        self.err_seq = np.array([np.linalg.norm(v - self.verts, axis=1) for v in glide(lead + self.decoded)])
        self.err_top = self.err_seq.max() or 1.0
        return glide(lead + list(keys))

    def vertex_rgba(self):
        """Per-vertex colours for the Colour mode, a title label, and a legend spec
        (cmap, low, high, caption) or None."""
        mode, e = self.val("Colour"), self.encoded
        if mode == "error" and self.decoded:  # true decoded error, whatever the view's gain
            n = len(self.decoded)
            return (ERROR_CMAP(self.err_seq[self.idx] / self.err_top),
                    f"error per vertex at this step (scale: largest over {n} readout{'s' * (n > 1)})",
                    (ERROR_CMAP, 0, self.err_top, "vertex error at this step (distance, mesh units)"))
        if mode == "qubit":
            cell = np.searchsorted(cells(e.graph)[:, 1], e.slots, side="right")
            split = (cell != cell[:, :1]).any()  # naive xyz puts x, y, z on different qubits
            what = "data qubit holding each vertex's x (naive layout splits x, y, z)" if split \
                else "data qubit holding each vertex"
            return qubit_rgba(self.vq), f"{what}, {len(np.unique(self.vq))} qubits", None
        if mode == "φ colour" and e.colour is not None:
            if self.colours is None:
                c, src = e.colour, "original colour field"
            else:
                c, src = self.colours.mean(0), f"colour decoded from φ (mean of {len(self.colours)} readouts)"
            return hls_rgba(c, 0.55, 0.75), src, (HUE_CMAP, 0, 1, "colour value stored in φ")
        why = {"error": " (run to see error)", "φ colour": " (pair mode only)"}.get(mode, "")
        return np.tile(PLAIN, (len(self.verts), 1)), "plain" + why, None

    def legend(self, ax, spec):
        ax.clear()
        ax.set_visible(spec is not None)
        if spec:
            cmap, lo, hi, caption = spec
            self.fig.colorbar(ScalarMappable(Normalize(lo, hi), cmap), cax=ax, orientation="horizontal")
            ax.tick_params(labelsize=7)
            ax.set_xlabel(caption, fontsize=7, labelpad=1)

    def redraw(self, *_):
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
        self.ax.set_title(f"{head}{err}\nstep {self.idx + 1}/{len(self.seq)} · colour: "
                          f"{self.colour_label}{edge_txt}", fontsize=9)
        self.fig.canvas.draw_idle()

    def tick(self, _):
        if self.playing and len(self.seq) > 1:
            if self.idx + 1 >= len(self.seq) and not self.toggles()["loop"]:
                self.playing = False
            else:
                self.idx = (self.idx + 1) % len(self.seq)
                self.draw()
        return ()

    def toggle(self, *_):
        self.playing = not self.playing
        self.hide_tip()
        if self.playing and self.idx + 1 >= len(self.seq):
            self.idx = 0

    def set_speed(self, fps):
        # macOS timers ignore interval changes while running, so restart the timer
        timer = self.anim.event_source
        timer.stop()
        timer.interval = 1000 / fps
        timer.start()

    def hide_tip(self):
        if self.tip.get_visible():
            self.tip.set_visible(False)
            self.marker.set_visible(False)
            self.fig.canvas.draw_idle()

    # --- qubit map -------------------------------------------------------------
    # qmap["vertices"][i][c] = [group, address]: value c of vertex i is in data qubit
    # groups[group]["data"], in the slot where address qubit groups[group]["addresses"][j]
    # reads bit j of address (Tessa's decoder bit order).
    def set_focus(self, vertex=None, group=None):
        """Highlight a vertex's data qubit(s), or one data qubit, on the map and mesh."""
        focus = None
        if self.toggles()["map"] and (vertex is not None or group is not None):
            vs = self.qmap["vertices"]
            groups = sorted({g for g, _ in vs[vertex]}) if vertex is not None else [group]
            focus = {"vertex": vertex, "groups": groups,
                     "mates": [i for i, v in enumerate(vs) if any(g in groups for g, _ in v)]}
        if focus == self.focus:
            return
        self.focus = focus
        self.draw_map()
        self.draw()

    def draw_map(self):
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
            g, a = m["vertices"][f["vertex"]][0]  # one value: label each address bit
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
        used = {g for v in m["vertices"] for g, _ in v}
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
                               "qubits)  ■ address qubit\nblue ● = holds part of this mesh")

    def focus_text(self):
        f, m = self.focus, self.qmap
        pair = " (θ: shape, φ: colour)" if self.encoded.pair else ""
        if not f:
            return "Hover a vertex on the mesh, or a data qubit here."
        addrs = lambda g: m["groups"][g]["addresses"]  # noqa: E731
        if f["vertex"] is None:
            g = f["groups"][0]
            held = sum(gg == g for v in m["vertices"] for gg, _ in v)
            return (f"data q{m['groups'][g]['data']} · address " + " ".join(f"q{q}" for q in addrs(g))
                    + f"\n{2 ** len(addrs(g))} slots, {held} hold values of this mesh{pair}"
                    + f"\n{len(f['mates'])} vertices on it (orange on the mesh)")
        names = ["r"] if len(m["vertices"][0]) == 1 else ["x", "y", "z"]
        lines = [f"{names[c]} → data q{m['groups'][g]['data']}, slot {a} of {2 ** len(addrs(g))}, address "
                 + " ".join(f"q{q}={(a >> j) & 1}" for j, q in enumerate(addrs(g)))
                 for c, (g, a) in enumerate(m["vertices"][f["vertex"]])]
        share = "these data qubits" if len(f["groups"]) > 1 else "this data qubit"
        return (f"vertex {f['vertex']}{pair}\n" + "\n".join(lines)
                + f"\n{len(f['mates'])} vertices share {share} (orange)")

    def hover_map(self, event):
        xy = np.array(self.qmap["coords"], float)
        data = np.array([g["data"] for g in self.qmap["groups"]])
        d = np.hypot(*(xy[data] - (event.xdata, event.ydata)).T)
        self.set_focus(group=int(d.argmin()) if d.min() < 0.45 else None)

    def hover(self, event):
        """Nearest vertex: original vs current coordinates while paused; with the qubit
        map on, also its data and address qubits (while playing too)."""
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
        fmt = lambda a: "(" + ", ".join(f"{c:+.4f}" for c in a) + ")"  # noqa: E731
        gain = 1.0 if self.fidelity else self.val("Gain (material)")
        self.tip.set_text(f"vertex {i} on data qubit {self.vq[i]}   (gain {gain:.1f}x, step {self.idx + 1})\n"
                          f"original {fmt(o)}\ncurrent  {fmt(n)}\n"
                          f"delta    {fmt(n - o)}\n|delta|  {np.linalg.norm(n - o):.4f}")
        self.tip.set_position(pix[i] + (14, 6))
        self.marker.set_data([pix[i][0]], [pix[i][1]])
        self.tip.set_visible(True)
        self.marker.set_visible(True)
        self.fig.canvas.draw_idle()

    def say(self, msg):
        self.status.set_text(msg)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    # --- export --------------------------------------------------------------
    def export_obj(self, *_):
        paths = anim.export_obj_sequence(OUT / "obj", self.seq, self.faces)
        self.say(f"wrote {len(paths)} OBJ file(s) to {OUT / 'obj'}/")

    def export_gif(self, *_):
        """The playback as a GIF of the 3D view only (no controls)."""
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
    app = App(sys.argv[1:])  # keep a reference: matplotlib holds widget callbacks weakly
    plt.show()
    del app


if __name__ == "__main__":
    main()
