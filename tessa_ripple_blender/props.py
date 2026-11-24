"""Scene settings (``scene.tessa``): the desktop app's widgets as Blender properties.

Enum identifiers match ``bridge`` strings where possible (``params`` maps the rest), and
are listed here rather than imported, so registering the add-on never imports qiskit.
"""

import math

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty

PAYLOADS = [
    ("radial", "Radial", "One radius per vertex from the centroid; directions kept classically"),
    ("xyz", "xyz", "Three coordinates per vertex, every one from the readout"),
    ("spherical", "Spherical", "Direction as θ, φ of one slot, radius in a second (Mode θ only)"),
    ("spectral", "Spectral", "xyz coefficients of the mesh's first k Laplacian modes"),
]
LAYOUTS = [("mesh", "Mesh", "Neighbourhood patches per cell"), ("naive", "Naive", "Axis-major order")]
MODES = [
    ("theta", "θ (Z)", "One value per slot, read in Z"),
    ("split", "Shape θ+φ", "Two values per slot: twice the vertices fit"),
    ("morph", "Morph A→B", "A in θ, a second mesh B in φ, read by a basis sweep"),
]
BACKENDS = [
    ("noiseless", "Noiseless", "Shot noise only"),
    ("flat_noise", "Flat noise", "Typical IBM-like error rates, scaled by Noise ×"),
    ("phoenix", "ibm_phoenix", "Whole-chip calibration snapshot (156 qubits)"),
    ("fez", "ibm_fez", "Whole-chip calibration snapshot"),
]
BACKEND_NAMES = {"flat_noise": "flat noise"}  # Blender identifier -> bridge.BACKENDS
PLACEMENTS = [("quality", "Quality", "Best data qubits first"), ("graph", "Graph", "Tessa's own order")]
DRR = [(m, m.replace("_", " ").title(), "") for m in
       ("none", "min_max", "percentile", "mean_std", "abs_max", "min_avg_max")]
COLOURS = [  # the desktop app's Colour modes that hold still (its front colour plays along t)
    ("error", "Error", "Mean readout error per vertex (light = none, dark = largest)"),
    ("qubit", "Qubit", "One colour per data qubit: shows how the Layout places vertices (not spectral)"),
    ("delta", "Morph Δ", "|B − A| per vertex after alignment (Morph only)"),
    ("none", "None", "No colour attribute"),
]


def _update(part):
    """A property update that re-writes one part of the last result, without a readout.

    Args:
        part: "gain", "timeline" or "colour" (see ``ops.reapply``).

    Returns:
        Callable: ``update(self, context)`` for a Blender property.
    """
    def update(self, context):
        """Re-write ``part`` of the active object's last result."""
        from . import ops
        ops.reapply(context, part)
    return update


def _is_mesh(self, obj):
    """Poll for the B picker: mesh objects only.

    Args:
        self: The settings.
        obj: A candidate object.

    Returns:
        bool: True for meshes.
    """
    return obj.type == "MESH"


class TessaSettings(bpy.types.PropertyGroup):
    """Every Tessa Ripple setting, stored on the scene as ``scene.tessa``."""

    # encoder (changing any re-encodes)
    payload: EnumProperty(name="Payload", items=PAYLOADS, default="radial")
    layout: EnumProperty(name="Layout", items=LAYOUTS, default="mesh")
    mode: EnumProperty(name="Mode", items=MODES, default="theta")
    clamp_theta: FloatProperty(name="Clamp θ (π)", min=0, max=0.45, default=0,
                               description="θ clamp as a fraction of π; 0 uses Tessa's default")
    clamp_phi: FloatProperty(name="Clamp φ (π)", min=0, max=0.45, default=0,
                             description="φ clamp (Shape θ+φ) as a fraction of π; 0 uses Tessa's default")
    compression: FloatProperty(name="Compression %", min=0, max=100, default=0,
                               description="Share of multiplexer coefficients dropped")
    quant_bits: IntProperty(name="Quant bits", min=0, max=16, default=0, description="0 = no quantisation")
    headroom: FloatProperty(name="Headroom", min=0, max=1, default=0.1,
                            description="Fractional padding on the value range")
    spectral_k: IntProperty(name="Spectral k", min=0, max=500, default=0,
                            description="Modes to encode; 0 picks the smallest k that meets the tolerance")
    # readout
    backend: EnumProperty(name="Backend", items=BACKENDS, default="noiseless")
    placement: EnumProperty(name="Placement", items=PLACEMENTS, default="quality")
    shots_log2: IntProperty(name="Shots (2ⁿ)", min=6, max=16, default=12,
                            description="Shots per measurement setting, as a power of two")
    noise_strength: FloatProperty(name="Noise ×", min=0, max=10, default=1,
                                  description="Scales the flat-noise rates")
    seed: IntProperty(name="Seed", min=0, default=0, description="Seed of the first readout")
    frames: IntProperty(name="Frames", min=2, max=500, default=24,
                        description="Make frames: readouts to animate (seeds seed, seed + 1, …), "
                                    "one shape key each")
    drr: EnumProperty(name="DRR", items=DRR, default="none", description="Dynamic range recovery")
    # morph
    target: PointerProperty(name="B", type=bpy.types.Object, poll=_is_mesh,
                            description="The mesh A morphs into (A is the active object)")
    shared_topology: BoolProperty(name="Shared topology", default=False,
                                  description="Pair B with A by vertex index instead of closest point")
    interference: BoolProperty(name="Interference readout", default=False,
                               description="Read the morph by interference instead of the basis sweep")
    morph_frames: EnumProperty(name="Frames show", default="sweep", items=[
        ("sweep", "Morph A→B", "Make frames: t runs 0 → 1 across the frames, so A turns into B"),
        ("hold", "Noise at t", "Make frames: every frame reads at Morph t; only readout noise moves"),
    ])
    morph_t: FloatProperty(name="Morph t", min=0, max=1, default=0.5, subtype="FACTOR",
                           description="Where Encode & read (and Noise at t frames) read the morph")
    a_dirs_only: BoolProperty(name="A's directions only", default=False,
                              description="Radial payload: keep A's directions instead of turning to B's")
    morph_front: BoolProperty(name="Front", default=False,
                              description="A front spreads over the surface from the largest |B − A|")
    front_width: FloatProperty(name="Front width", min=0.02, max=1, default=0.15,
                               description="How soft the front's edge is, as a share of the surface distance")
    front_source: IntProperty(name="Front source", min=-1, default=-1,
                              description="Vertex the front starts from; -1 = auto (the vertex that moves most)")
    # output (re-applied to the last result without a new readout)
    gain: FloatProperty(name="Gain", min=0, soft_max=10, default=1, update=_update("gain"),
                        description="1 = true decoded positions; above 1 exaggerates the readout noise")
    frame_step: IntProperty(name="Timeline spacing", min=1, max=240, default=5, update=_update("timeline"),
                            description="Timeline frames between successive readouts (blended linearly)")
    colour: EnumProperty(name="Colour", items=COLOURS, default="error", update=_update("colour"),
                         description="What the tessa_colour attribute shows")

    def params(self) -> dict:
        """Keyword arguments for ``bridge.Params`` (as the desktop app's ``params``).

        Returns:
            dict: Clamps of 0 are Tessa's default (None); in morph, ``clamp_b``
            follows Clamp θ. The front is on only in morph with the basis sweep
            (not interference) and a payload other than spectral.
        """
        clamp = lambda x: x * math.pi or None  # noqa: E731  0 = Tessa's default
        morph = self.mode == "morph"
        return dict(
            payload=self.payload, layout=self.layout, mode=self.mode,
            clamp=clamp(self.clamp_theta),
            clamp_b=clamp(self.clamp_theta if morph else self.clamp_phi),
            compression=self.compression, quant_bits=self.quant_bits or None, headroom=self.headroom,
            backend=BACKEND_NAMES.get(self.backend, self.backend), placement=self.placement,
            shared_topology=self.shared_topology, shots=2 ** self.shots_log2,
            noise_strength=self.noise_strength, seed=self.seed, morph_t=self.morph_t,
            spectral_k=self.spectral_k or None,
            morph_front=morph and self.morph_front and not self.interference and self.payload != "spectral",
            front_width=round(self.front_width, 2),
            front_source=self.front_source if self.front_source >= 0 else None,
        )


CLASSES = (TessaSettings,)
