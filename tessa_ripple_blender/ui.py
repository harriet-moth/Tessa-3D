"""The sidebar panels: View3D > N > Tessa."""

import bpy

from . import ops, props


class _Tessa:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Tessa"


class TESSA_PT_main(_Tessa, bpy.types.Panel):
    bl_label = "Tessa Ripple"

    def draw(self, context):
        col = self.layout.column()
        obj = context.active_object
        if obj is None or obj.type != "MESH":
            col.label(text="Select a mesh object (A)", icon="INFO")
            return
        n = len(obj.data.vertices)
        col.label(text=f"A: {obj.name}  ({n} vertices)", icon="MESH_DATA")
        if n > ops.LARGE:
            col.label(text="Large mesh: many circuits per readout. Apply a Decimate", icon="ERROR")
            col.label(text="modifier first to speed it up", icon="BLANK1")
        if context.mode != "OBJECT":
            col.label(text="Switch to Object Mode to run", icon="INFO")
        s = context.scene.tessa
        if ops.STATE["running"]:
            col.progress(factor=ops.STATE["frac"], text=ops.STATE["text"])
            col.operator("tessa.cancel", icon="CANCEL")
        else:
            col.operator("tessa.run", icon="PLAY",
                         text="Encode & read" + (f" (t = {s.morph_t:.2f})" if s.mode == "morph" else ""))
            box = col.box().column()
            row = box.row(align=True)
            row.prop(s, "frames")
            row.operator("tessa.make_frames", icon="RENDER_ANIMATION")
            box.label(text=ops.frames_hint(s), icon="TIME")
        row = col.row(align=True)
        row.operator("tessa.clear", icon="TRASH")
        row.operator("tessa.export_obj", icon="EXPORT", text="Export OBJs")


class TESSA_PT_encoding(_Tessa, bpy.types.Panel):
    bl_label = "Encoding"
    bl_parent_id = "TESSA_PT_main"

    def draw(self, context):
        s, col = context.scene.tessa, self.layout.column()
        col.use_property_split = True
        sub = col.column()
        sub.enabled = s.payload != "spherical"  # spherical fills θ and φ itself: Mode θ only
        sub.prop(s, "mode")
        col.prop(s, "payload")
        if s.payload == "spherical":
            col.label(text="Spherical fills θ and φ itself: Mode θ (Z) only", icon="INFO")
        col.prop(s, "layout")
        sub = col.column()
        sub.enabled = props.reads_one_field(s)
        sub.prop(s, "readout")
        col.prop(s, "clamp_theta")
        sub = col.column()
        sub.enabled = s.mode == "split"
        sub.prop(s, "clamp_phi")
        sub = col.column()
        sub.enabled = s.payload == "spectral"
        sub.prop(s, "spectral_k")
        col.prop(s, "compression")
        col.prop(s, "quant_bits")
        col.prop(s, "headroom")


class TESSA_PT_readout(_Tessa, bpy.types.Panel):
    bl_label = "Readout"
    bl_parent_id = "TESSA_PT_main"

    def draw(self, context):
        s, col = context.scene.tessa, self.layout.column()
        col.use_property_split = True
        col.prop(s, "backend")
        if s.backend in ("phoenix", "fez"):
            col.prop(s, "placement")
        if s.backend == "flat_noise":
            col.prop(s, "noise_strength")
        col.prop(s, "shots_log2", text=f"Shots ({2 ** s.shots_log2:,})")
        col.prop(s, "seed")
        col.prop(s, "drr")


class TESSA_PT_morph(_Tessa, bpy.types.Panel):
    bl_label = "Morph A→B"
    bl_parent_id = "TESSA_PT_main"

    @classmethod
    def poll(cls, context):
        return context.scene.tessa.mode == "morph"

    def draw(self, context):
        s, col = context.scene.tessa, self.layout.column()
        col.use_property_split = True
        col.prop(s, "target")
        col.prop(s, "shared_topology")
        col.prop(s, "interference")
        col.prop(s, "morph_frames", expand=True)
        col.prop(s, "morph_t")
        sub = col.column()
        sub.enabled = s.payload == "radial"
        sub.prop(s, "a_dirs_only")
        sub = col.column()
        sub.enabled = not s.interference and s.payload != "spectral"  # the front needs the basis sweep
        sub.prop(s, "morph_front")
        if s.morph_front:
            sub.prop(s, "front_width")
            row = sub.row(align=True)
            row.label(text="Source: " + ("auto" if s.front_source < 0 else f"vertex {s.front_source}"))
            row.operator("tessa.front_source", text="Selected", icon="VERTEXSEL").auto = False
            row.operator("tessa.front_source", text="Auto").auto = True


class TESSA_PT_output(_Tessa, bpy.types.Panel):
    bl_label = "Output"
    bl_parent_id = "TESSA_PT_main"

    def draw(self, context):
        s, col = context.scene.tessa, self.layout.column()
        col.use_property_split = True
        col.prop(s, "gain")
        col.prop(s, "frame_step")
        col.prop(s, "fps")
        col.prop(s, "loop")
        col.prop(s, "from_original")
        col.prop(s, "ease")
        col.prop(s, "colour")
        if s.colour == "qubit" and s.payload == "spectral":
            col.label(text="Spectral: qubits hold modes, not vertices", icon="INFO")
        elif s.colour == "delta" and s.mode != "morph":
            col.label(text="Morph Δ needs Mode Morph A→B", icon="INFO")
        elif s.colour != "none":
            col.label(text="View: Solid shading > Color > Attribute", icon="INFO")


class TESSA_PT_results(_Tessa, bpy.types.Panel):
    bl_label = "Results"
    bl_parent_id = "TESSA_PT_main"

    def draw(self, context):
        col = self.layout.column(align=True)
        for line in ops.STATE["summary"] or ["Run a readout to see results"]:
            col.label(text=line)


class TESSA_PT_diagnostics(_Tessa, bpy.types.Panel):
    bl_label = "Diagnostics"
    bl_parent_id = "TESSA_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        self.layout.operator("tessa.check", icon="CHECKMARK")


CLASSES = (TESSA_PT_main, TESSA_PT_encoding, TESSA_PT_readout, TESSA_PT_morph,
           TESSA_PT_output, TESSA_PT_results, TESSA_PT_diagnostics)
