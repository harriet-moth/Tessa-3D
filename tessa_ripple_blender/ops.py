"""Operators. A run copies the mesh out on the main thread, does the quantum work in a
worker thread (engine.run, numpy only) and writes the result back on the main thread
from a modal timer, so Blender stays responsive and bpy is never touched off-thread.
"""

import threading
import traceback

import bpy
from bpy.props import BoolProperty, StringProperty

from . import engine, mesh_io

STATE = {"running": False, "text": "", "frac": 0.0, "cancel": False, "summary": []}
RESULTS = {}  # object name -> last engine result (to re-apply gain / spacing / colour, and export)
LARGE = 2000  # vertices: above this, warn that a run will be slow


def _mesh_object(context):
    """The active object if it is a mesh.

    Args:
        context: Blender context.

    Returns:
        bpy.types.Object | None: The mesh object, else None.
    """
    obj = context.active_object
    return obj if obj is not None and obj.type == "MESH" else None


def frames_hint(s) -> str:
    """What Make frames will animate, for the panel.

    Args:
        s: The settings (``scene.tessa``).

    Returns:
        str: One line.
    """
    if s.mode == "morph" and s.morph_frames == "sweep":
        return f"A → B over {s.frames} frames"
    if s.mode == "morph":
        return f"{s.frames} frames of readout noise at t = {s.morph_t:.2f}"
    return f"{s.frames} frames of shot-to-shot readout variance"


def make_job(context, frames: int) -> engine.Job:
    """Copy everything a run needs out of Blender (main thread only).

    Args:
        context: Blender context; the active object is A.
        frames: Readouts to make.

    Returns:
        engine.Job: The run's inputs.

    Raises:
        ValueError: Settings that cannot run, reported before the thread starts.
    """
    s, obj = context.scene.tessa, _mesh_object(context)
    if s.payload == "spherical" and s.mode != "theta":
        raise ValueError("The spherical payload fills θ and φ itself: use Mode θ (Z)")
    verts, faces = mesh_io.base_coords(obj.data), mesh_io.polygons(obj.data)
    b_verts = b_faces = None
    if s.mode == "morph":
        if s.target is None:
            raise ValueError("Morph: pick a mesh object as B")
        if s.target == obj:
            raise ValueError("Morph: B must be a different object from A (the active object)")
        if s.morph_front and s.front_source >= len(verts):
            raise ValueError(f"Front source {s.front_source} is not a vertex of {obj.name} "
                             f"(0..{len(verts) - 1}); pick one or use Auto")
        b_verts, b_faces = mesh_io.coords_in(s.target, obj), mesh_io.polygons(s.target.data)
    return engine.Job(verts=verts, faces=faces, params=s.params(), frames=frames, drr=s.drr,
                      interference=s.interference, hold_t=s.morph_frames == "hold",
                      a_dirs_only=s.a_dirs_only, b_verts=b_verts, b_faces=b_faces)


def summary(result, scene=None, step: int = 1) -> list[str]:
    """The Results panel's lines for a run.

    Args:
        result: ``engine.run``'s result.
        scene: The scene, for the timeline range (None leaves it out).
        step: Timeline spacing.

    Returns:
        list[str]: Lines, the mean error first.
    """
    st, err = result["stats"], result["errors"]
    n = len(err)
    lines = [f"Error {100 * sum(err) / n:.2f} %"
             + (f"  ({100 * min(err):.2f}–{100 * max(err):.2f})" if n > 1 else ""),
             f"{st['values']} values on {st['qubits']} of {st['graph_qubits']} qubits ({st['graph']})",
             f"{st['ratio']:.1f} values/qubit, {st['runs']} circuit(s)/readout",
             f"Depth {st['depth']}, {st['twoq']} two-qubit gates",
             f"{n} readout(s) in {result['seconds']:.1f} s"]
    if n > 1 and scene is not None:
        start = scene.frame_start
        what = "A → B" if result["ts"] and result["ts"][0] != result["ts"][-1] else "readout noise"
        lines.append(f"Timeline {start}–{start + (n - 1) * step}: {what} (Space to play)")
    if result["match"]:
        m = result["match"]
        lines.append(f"B matched by {m['method']}, coverage {100 * m['coverage']:.0f} %")
    if result["spectral"]:
        sp = result["spectral"]
        lines.append(f"Spectral k {sp['k']}: truncation {100 * sp['error_a']:.2f} %"
                     + (f" (B {100 * sp['error_b']:.2f} %)" if sp.get("error_b") is not None else ""))
    if result["front_source"] is not None:
        lines.append(f"Front from vertex {result['front_source']}")
    if result["kept"]:
        lines.append(f"Decoded from {100 * sum(result['kept']) / len(result['kept']):.0f} % of shots")
    return lines


def reapply(context, part: str):
    """Re-write one part of the active object's last result, without a new readout.

    Only that part is touched: "gain" re-writes the keys' positions, "timeline" their
    keyframes and "colour" the attributes. If Tessa's keys were removed, everything is
    re-written; if the vertex count changed, the result no longer fits and is dropped.

    Args:
        context: Blender context.
        part: "gain", "timeline" or "colour".
    """
    obj = _mesh_object(context)
    if STATE["running"] or obj is None or obj.name not in RESULTS or context.mode != "OBJECT":
        return
    s, scene, result = context.scene.tessa, context.scene, RESULTS[obj.name]
    if len(result["decoded"][0]) != len(obj.data.vertices):
        RESULTS.pop(obj.name)
        return
    if not mesh_io.fits(obj, result):
        mesh_io.write_frames(obj, scene, result, s.gain, s.frame_step, s.colour)
    elif part == "gain":
        mesh_io.set_gain(obj, result, s.gain)
    elif part == "timeline":
        mesh_io.set_timeline(obj, scene, s.frame_step)
    else:
        mesh_io.write_attributes(obj.data, result, s.colour)
    STATE["summary"] = summary(result, scene, s.frame_step)


def _redraw(context):
    """Redraw every 3D view (the sidebar shows the progress).

    Args:
        context: Blender context.
    """
    for area in context.screen.areas if context.screen else ():
        if area.type == "VIEW_3D":
            area.tag_redraw()


class _Run:
    """A threaded run: ``n_frames`` readouts written as shape keys (Esc cancels)."""

    bl_options = {"REGISTER", "UNDO"}  # one undo step, pushed when the result is written

    @classmethod
    def poll(cls, context):
        """Runnable on a mesh in Object Mode while no other run is going.

        Args:
            context: Blender context.

        Returns:
            bool: Whether the operator can run.
        """
        return _mesh_object(context) is not None and context.mode == "OBJECT" and not STATE["running"]

    def n_frames(self, context) -> int:
        """Readouts this operator makes.

        Args:
            context: Blender context.

        Returns:
            int: 1 here; Make frames overrides it.
        """
        return 1

    def execute(self, context):
        """Copy the inputs out, then start the worker thread and the modal timer.

        Args:
            context: Blender context.

        Returns:
            set: ``{"RUNNING_MODAL"}``, or ``{"CANCELLED"}`` for settings that cannot run.
        """
        try:
            job = make_job(context, self.n_frames(context))
        except ValueError as err:
            self.report({"ERROR"}, str(err))
            return {"CANCELLED"}
        engine.prepare()  # qiskit must first load on this (main) thread, or a later worker crashes
        self.obj_name, self.out = context.active_object.name, {}
        STATE.update(running=True, cancel=False, text="Starting…", frac=0.0)

        def work():
            """The worker thread: engine.run only, never bpy."""
            try:
                self.out["result"] = engine.run(
                    job, lambda text, frac: STATE.update(text=text, frac=frac), lambda: STATE["cancel"])
            except engine.Cancelled:
                self.out["cancelled"] = True
            except ValueError as err:  # settings Tessa cannot encode: a message for the user
                self.out["error"] = str(err)
            except Exception as err:  # unexpected: full traceback to the terminal
                traceback.print_exc()
                self.out["error"] = f"{type(err).__name__}: {err} (see the terminal)"

        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()
        wm = context.window_manager
        self.timer = wm.event_timer_add(0.2, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        """Esc cancels; each timer tick shows progress, and the end writes the result.

        Args:
            context: Blender context.
            event: The event.

        Returns:
            set: Blender's modal return set.
        """
        if event.type == "ESC" and event.value == "PRESS":
            STATE["cancel"] = True
            return {"RUNNING_MODAL"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        context.workspace.status_text_set(f"Tessa: {STATE['text']}  (Esc to cancel)")
        _redraw(context)
        if self.thread.is_alive():
            return {"PASS_THROUGH"}
        return self.finish(context)

    def finish(self, context):
        """Stop the timer and write the result (or report why not) on the main thread.

        Args:
            context: Blender context.

        Returns:
            set: ``{"FINISHED"}`` when written, else ``{"CANCELLED"}``.
        """
        context.window_manager.event_timer_remove(self.timer)
        context.workspace.status_text_set(None)
        STATE.update(running=False, text="")
        _redraw(context)
        if "error" in self.out:
            self.report({"ERROR"}, self.out["error"])
            return {"CANCELLED"}
        if self.out.get("cancelled"):
            self.report({"WARNING"}, "Tessa run cancelled")
            return {"CANCELLED"}
        obj = bpy.data.objects.get(self.obj_name)
        if obj is None or context.mode != "OBJECT":
            self.report({"WARNING"}, "Finished, but the object is gone or not in Object Mode")
            return {"CANCELLED"}
        result, s = self.out["result"], context.scene.tessa
        if len(result["decoded"][0]) != len(obj.data.vertices):
            self.report({"ERROR"}, "The mesh changed during the run; run again")
            return {"CANCELLED"}
        RESULTS[obj.name] = result
        mesh_io.write_frames(obj, context.scene, result, s.gain, s.frame_step, s.colour)
        if len(result["decoded"]) > 1:
            context.scene.frame_set(context.scene.frame_start)
        STATE["summary"] = summary(result, context.scene, s.frame_step)
        self.report({"INFO"}, "Tessa: " + STATE["summary"][0])
        return {"FINISHED"}


class TESSA_OT_run(_Run, bpy.types.Operator):
    """Encode the active mesh and read it back once (Morph: at Morph t). Esc cancels"""

    bl_idname = "tessa.run"
    bl_label = "Encode & read"


class TESSA_OT_make_frames(_Run, bpy.types.Operator):
    """Make animated frames: Morph A→B over time, or the readout noise from frame to frame.
    One shape key per frame, keyframed along the timeline. Esc cancels"""

    bl_idname = "tessa.make_frames"
    bl_label = "Make frames"

    def n_frames(self, context) -> int:
        """Readouts to make.

        Args:
            context: Blender context.

        Returns:
            int: The Frames setting.
        """
        return context.scene.tessa.frames


class TESSA_OT_cancel(bpy.types.Operator):
    """Stop the running readouts after the current one"""

    bl_idname = "tessa.cancel"
    bl_label = "Cancel"

    def execute(self, context):
        """Flag the worker to stop before its next readout.

        Args:
            context: Blender context.

        Returns:
            set: ``{"FINISHED"}``.
        """
        STATE["cancel"] = True
        return {"FINISHED"}


class TESSA_OT_front_source(bpy.types.Operator):
    """Start the morph front from the selected vertex (Edit Mode: the active one), or Auto"""

    bl_idname = "tessa.front_source"
    bl_label = "Front source"
    bl_options = {"REGISTER", "UNDO"}

    auto: BoolProperty(name="Auto", default=False,
                       description="Use the vertex that moves most instead of a picked one")

    @classmethod
    def poll(cls, context):
        """Needs a mesh as the active object.

        Args:
            context: Blender context.

        Returns:
            bool: Whether the operator can run.
        """
        return _mesh_object(context) is not None

    def execute(self, context):
        """Set ``front_source`` to the selected vertex, or -1 (auto).

        Args:
            context: Blender context.

        Returns:
            set: ``{"FINISHED"}``, or ``{"CANCELLED"}`` with no vertex selected.
        """
        s = context.scene.tessa
        if self.auto:
            s.front_source = -1
            return {"FINISHED"}
        index = mesh_io.selected_vertex(context.active_object)
        if index is None:
            self.report({"WARNING"}, "Select a vertex of A first (Edit Mode)")
            return {"CANCELLED"}
        s.front_source = index
        self.report({"INFO"}, f"Front source: vertex {index}")
        return {"FINISHED"}


class TESSA_OT_clear(bpy.types.Operator):
    """Remove Tessa's shape keys, their animation and its attributes from the active mesh"""

    bl_idname = "tessa.clear"
    bl_label = "Clear"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        """Needs a mesh in Object Mode and no run going.

        Args:
            context: Blender context.

        Returns:
            bool: Whether the operator can run.
        """
        return _mesh_object(context) is not None and context.mode == "OBJECT" and not STATE["running"]

    def execute(self, context):
        """Clear the active mesh and forget its result and the cached encoding.

        Args:
            context: Blender context.

        Returns:
            set: ``{"FINISHED"}``.
        """
        obj = context.active_object
        mesh_io.clear(obj)
        RESULTS.pop(obj.name, None)
        engine.clear_cache()
        STATE["summary"] = []
        return {"FINISHED"}


class TESSA_OT_export_obj(bpy.types.Operator):
    """Write each readout (with the current gain) as a numbered OBJ file"""

    bl_idname = "tessa.export_obj"
    bl_label = "Export OBJ sequence"

    directory: StringProperty(subtype="DIR_PATH")

    @classmethod
    def poll(cls, context):
        """Needs a result for the active mesh.

        Args:
            context: Blender context.

        Returns:
            bool: Whether the operator can run.
        """
        obj = _mesh_object(context)
        return obj is not None and obj.name in RESULTS

    def invoke(self, context, event):
        """Open the folder picker.

        Args:
            context: Blender context.
            event: The event.

        Returns:
            set: ``{"RUNNING_MODAL"}``.
        """
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        """Write ``frame_000.obj``, ... (Y-up) into the chosen folder.

        Args:
            context: Blender context.

        Returns:
            set: ``{"FINISHED"}``.
        """
        from .core.animate import export_obj_sequence
        obj, gain = context.active_object, context.scene.tessa.gain
        result = RESULTS[obj.name]
        positions = [r + gain * (d - r) for d, r in zip(result["decoded"], result["refs"])]
        paths = export_obj_sequence(self.directory, positions, mesh_io.polygons(obj.data))
        self.report({"INFO"}, f"Wrote {len(paths)} OBJ file(s) to {self.directory}")
        return {"FINISHED"}


class TESSA_OT_check(bpy.types.Operator):
    """Load qiskit, qiskit-aer and scipy and run a small Bell-state circuit"""

    bl_idname = "tessa.check"
    bl_label = "Check quantum backend"

    def execute(self, context):
        """Import the bundled wheels and sample a Bell state.

        Args:
            context: Blender context.

        Returns:
            set: ``{"FINISHED"}``, or ``{"CANCELLED"}`` when an import fails.
        """
        try:
            import numpy as np
            import qiskit
            import qiskit_aer
            import scipy
            from qiskit import QuantumCircuit
            from qiskit_aer import AerSimulator
        except Exception as exc:
            self.report({"ERROR"}, f"Import failed: {exc}")
            return {"CANCELLED"}
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        counts = AerSimulator().run(qc, shots=1000, seed_simulator=1).result().get_counts()
        STATE["summary"] = [f"qiskit {qiskit.__version__}, aer {qiskit_aer.__version__}",
                            f"scipy {scipy.__version__}, numpy {np.__version__}",
                            f"Bell counts {dict(sorted(counts.items()))}"]
        self.report({"INFO"}, " | ".join(STATE["summary"]))
        return {"FINISHED"}


CLASSES = (TESSA_OT_run, TESSA_OT_make_frames, TESSA_OT_cancel, TESSA_OT_front_source, TESSA_OT_clear,
           TESSA_OT_export_obj, TESSA_OT_check)
