"""Matplotlib window: 3D mesh, sliders for every Tessa setting, frame playback.

Three tiers: encoder settings re-encode, readout settings re-decode (on Run /
Run frames), post and animation settings only redraw cached frames.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.transforms import IdentityTransform
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from . import animate as anim
from .bridge import DRR_METHODS, Params, encode_mesh, restore
from .mesh import PRESETS

OUT = Path("exports")


class App:
    def __init__(self):
        self.fig = fig = plt.figure("Tessa Ripple", figsize=(13, 8))
        self.ax = fig.add_axes([0.0, 0.08, 0.55, 0.9], projection="3d")
        self.encoded, self.raw, self.raw_key, self.idx, self.playing = None, None, None, 0, False
        self.sec_per_frame = None
        self.w = {}

        def radio(name, rect, labels, active=0):
            fig.text(rect[0], rect[1] + rect[3] + 0.005, name, fontsize=8, weight="bold")
            self.w[name] = r = RadioButtons(fig.add_axes(rect), labels, active=active)
            return r

        def slider(name, y, lo, hi, init, step=None, x=0.71):
            self.w[name] = s = Slider(fig.add_axes([x, y, 0.12, 0.022]), name, lo, hi,
                                      valinit=init, valstep=step)
            s.label.set_fontsize(8)
            return s

        # tier 1: encoder ----------------------------------------------------
        radio("Shape", [0.56, 0.86, 0.07, 0.07], ["cube", "sphere"])
        radio("Resolution", [0.56, 0.74, 0.07, 0.09], ["low", "mid", "high"])
        radio("Mode", [0.56, 0.64, 0.07, 0.07], ["single", "pair"])
        radio("Backend", [0.56, 0.54, 0.07, 0.07], ["noiseless", "noisy"])
        radio("DRR", [0.56, 0.24, 0.07, 0.27], DRR_METHODS)
        for name, y, lo, hi, init, step in [
            ("Clamp θ (π)", 0.92, 0, 0.45, 0.0, None), ("Clamp φ (π)", 0.89, 0, 0.45, 0.0, None),
            ("Compression %", 0.86, 0, 100, 0, 1), ("Quant bits", 0.83, 0, 8, 0, [0, 2, 3, 4, 5, 6, 7, 8]),
            ("Headroom", 0.80, 0, 0.5, 0.1, None),
            # tier 2: readout
            ("Shots (log2)", 0.75, 8, 16, 12, 1), ("Noise ×", 0.72, 0, 3, 1, None),
            ("Seed", 0.69, 0, 999, 0, 1),
            # tier 3: post + animation
            ("Gain", 0.64, 0, 10, 3, None), ("Mask %", 0.61, 0, 100, 100, 1),
            ("Frames", 0.56, 2, 60, 12, 1), ("In-between", 0.53, 0, 30, 8, 1),
            ("Speed (fps)", 0.50, 1, 60, 24, 1),
        ]:
            slider(name, y, lo, hi, init, step)
        self.w["Toggles"] = CheckButtons(fig.add_axes([0.71, 0.38, 0.14, 0.09]),
                                         ["Loop", "Start from original", "Auto-run"], [True, True, False])
        buttons = [("Run", self.run), ("Run frames", self.run_frames), ("Play/Pause", self.toggle),
                   ("Export OBJ", self.export_obj), ("Export GIF", self.export_gif)]
        for i, (label, cb) in enumerate(buttons):
            self.w[label] = b = Button(fig.add_axes([0.88, 0.88 - i * 0.06, 0.11, 0.045]), label)
            b.on_clicked(cb)
        self.status = fig.text(0.01, 0.02, "", fontsize=9, family="monospace")
        self.info = fig.text(0.88, 0.30, "", fontsize=8, family="monospace", va="top")

        for name in ("Shape", "Resolution", "Mode", "Clamp θ (π)", "Clamp φ (π)", "Compression %",
                     "Quant bits", "Headroom", "Backend", "Shots (log2)", "Noise ×", "Seed"):
            w = self.w[name]
            (w.on_clicked if isinstance(w, RadioButtons) else w.on_changed)(self.quantum_changed)
        for name in ("DRR", "Gain", "Mask %", "In-between", "Toggles"):
            w = self.w[name]
            (w.on_clicked if hasattr(w, "on_clicked") else w.on_changed)(self.redraw)
        self.w["Speed (fps)"].on_changed(self.set_speed)
        fig.canvas.mpl_connect("motion_notify_event", self.hover)

        self.coll = Poly3DCollection([], facecolor="#9ab8d8", edgecolor="#20304a", lw=0.5, alpha=0.9)
        self.ax.add_collection3d(self.coll)
        lim = 1.6
        self.ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim), box_aspect=(1, 1, 1))
        self.ax.set_axis_off()
        # hover readout (paused only), drawn in display pixels over the 3D view
        px = IdentityTransform()
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
        return Params(mode=self.val("Mode"), clamp=clamp("Clamp θ (π)"),
                      clamp_b=clamp("Clamp φ (π)"), compression=self.val("Compression %"),
                      quant_bits=qb or None, headroom=self.val("Headroom"),
                      shots=2 ** int(self.val("Shots (log2)")), noisy=self.val("Backend") == "noisy",
                      noise_strength=self.val("Noise ×"), seed=int(self.val("Seed")))

    def toggles(self):
        return dict(zip(["loop", "orig", "auto"], self.w["Toggles"].get_status()))

    def quantum_changed(self, *_):
        p = self.params()
        shape = self.val("Shape")
        preset = list(PRESETS[shape].items())[["low", "mid", "high"].index(self.val("Resolution"))]
        key = (shape, preset[0], p.encoder_key)
        if self.encoded is None or key != self.enc_key:
            verts, faces = preset[1]()
            try:
                encoded = encode_mesh(verts, p)
            except ValueError as err:
                return self.say(f"⚠ {err}")
            # swap everything together, before say()/flush_events lets a tick draw
            self.verts, self.faces, self.encoded, self.enc_key = verts, faces, encoded, key
            self.raw, self.seq, self.idx = None, [verts], 0
        s = self.encoded.stats
        self.info.set_text(f"{shape} {preset[0]}: {len(self.verts)} verts\nlattice {s['lattice']}"
                           f"\nqubits {s['qubits']}\ncapacity {s['used']:.0f}/{s['capacity']}"
                           f"\ndepth {s['depth']}")
        if self.toggles()["auto"]:
            return self.run()
        stale = self.raw is not None and self.raw_key != (self.enc_key, p.readout_key)
        est = f"  ≈{self.sec_per_frame:.1f}s/frame" if self.sec_per_frame else ""
        self.say(("frames stale — press Run / Run frames" if stale else "ready") + est)
        self.redraw()

    # --- quantum work --------------------------------------------------------
    def _decode(self, n):
        p, t0 = self.params(), time.time()
        prog = lambda i, n: self.say(f"frame {i + 1} / {n} …")  # noqa: E731
        self.raw = anim.run_frames(self.encoded, p, n, progress=prog)
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

    # --- classical layer -----------------------------------------------------
    def positions(self):
        if self.raw is None:
            return [self.verts]
        e, drr = self.encoded, self.val("DRR")
        mask = anim.random_mask(len(self.verts), self.val("Mask %") / 100, self.params().seed)
        decoded = [restore(e, r, drr) for r in self.raw]
        self.err = np.mean([np.abs(d - self.verts).mean() for d in decoded]) / np.ptp(self.verts)
        keys = [anim.blend(self.verts, d, self.val("Gain"), mask) for d in decoded]
        t = self.toggles()
        keys = ([self.verts] if t["orig"] else []) + keys
        if len(keys) == 1:
            return keys
        return anim.timeline(keys, int(self.val("In-between")), t["loop"])

    def redraw(self, *_):
        self.seq = self.positions()
        self.idx %= len(self.seq)
        self.draw()

    def draw(self):
        v = self.seq[self.idx]
        self.tip.set_visible(False)  # mesh moved, so any hover readout is stale
        self.marker.set_visible(False)
        self.coll.set_verts([v[f] for f in self.faces])
        err = f"  mean err {self.err:.3f}" if self.raw is not None else ""
        self.ax.set_title(f"step {self.idx + 1}/{len(self.seq)}{err}", fontsize=9)
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

    def hover(self, event):
        """While paused, show the nearest vertex's original vs current coordinates."""
        if self.playing or event.inaxes is not self.ax:
            return self.hide_tip()
        v = self.seq[self.idx]
        xs, ys, _ = proj3d.proj_transform(*v.T, self.ax.get_proj())
        pix = self.ax.transData.transform(np.column_stack([xs, ys]))
        dist = np.hypot(*(pix - (event.x, event.y)).T)
        i = int(dist.argmin())
        if dist[i] > 12:
            return self.hide_tip()
        o, n = self.verts[i], v[i]
        fmt = lambda a: "(" + ", ".join(f"{c:+.4f}" for c in a) + ")"  # noqa: E731
        self.tip.set_text(f"vertex {i}   (gain {self.val('Gain'):.1f}x, step {self.idx + 1})\n"
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
        self.playing = False
        path = OUT / "ripple.gif"
        path.parent.mkdir(exist_ok=True)

        def frame(i):
            self.idx = i
            self.draw()

        FuncAnimation(self.fig, frame, frames=len(self.seq), cache_frame_data=False).save(
            path, writer=PillowWriter(fps=int(self.val("Speed (fps)"))))
        self.say(f"wrote {path}")


def main():
    app = App()  # keep a reference: matplotlib holds widget callbacks weakly
    plt.show()
    del app


if __name__ == "__main__":
    main()
