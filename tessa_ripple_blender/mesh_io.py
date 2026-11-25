"""Blender mesh data <-> numpy, and writing results as shape keys and colours.

A result is written in three independent parts, so changing one output setting only
re-writes its own part: the shape keys' positions (gain), their keyframes (timeline
spacing) and the per-vertex attributes (colour).
"""

import colorsys

import bpy
import numpy as np

KEY_PREFIX = "Tessa "
ERROR_ATTR = "tessa_error"         # float per vertex: mean deviation / mesh size
QUBIT_ATTR = "tessa_qubit"         # int per vertex: data qubit holding it (not spectral)
DELTA_ATTR = "tessa_morph_delta"   # float per vertex: |B - A| / mesh size (morph only)
COLOUR_ATTR = "tessa_colour"       # the chosen one as a colour, for Solid > Attribute
ATTRS = (ERROR_ATTR, QUBIT_ATTR, DELTA_ATTR, COLOUR_ATTR, "tessa_error_colour")  # last: 0.1 name
# Ramps mirrored from the desktop app (tessa_ripple.app) and public/app.js, in sRGB.
ERROR_RAMP = ("#fcfdbf", "#fd9668", "#cf4070", "#752181", "#1a1042")  # ERROR_CMAP: light = no error
DELTA_RAMP = ("#440154", "#3b528b", "#21918c", "#5ec962", "#fde725")  # viridis (DELTA_CMAP)
GOLDEN = 0.618034  # qubit colours: hue steps by the golden ratio per qubit rank (qubit_rgba)


def base_coords(mesh) -> np.ndarray:
    """Rest positions: the Basis shape key if there is one, else the vertices.

    Args:
        mesh: A ``bpy.types.Mesh``.

    Returns:
        np.ndarray: (N, 3) float positions in object coordinates.
    """
    src = mesh.shape_keys.reference_key.data if mesh.shape_keys else mesh.vertices
    co = np.empty(len(mesh.vertices) * 3, np.float32)
    src.foreach_get("co", co)
    return co.reshape(-1, 3).astype(float)


def polygons(mesh) -> list:
    """Faces as lists of vertex indices.

    Args:
        mesh: A ``bpy.types.Mesh``.

    Returns:
        list[list[int]]: One vertex-index list per polygon.
    """
    loops = np.empty(len(mesh.loops), np.int32)
    mesh.loops.foreach_get("vertex_index", loops)
    starts = np.empty(len(mesh.polygons), np.int32)
    totals = np.empty(len(mesh.polygons), np.int32)
    mesh.polygons.foreach_get("loop_start", starts)
    mesh.polygons.foreach_get("loop_total", totals)
    return [loops[s:s + t].tolist() for s, t in zip(starts, totals)]


def coords_in(obj_b, obj_a) -> np.ndarray:
    """B's rest positions in A's object coordinates, as they sit in the scene.

    The morph then centres and scales B onto A (``morph.align``), so of B's placement
    only its rotation relative to A (and any non-uniform scale) changes the result.

    Args:
        obj_b: Mesh object B.
        obj_a: Mesh object A.

    Returns:
        np.ndarray: (M, 3) positions.
    """
    m = np.array(obj_a.matrix_world.inverted() @ obj_b.matrix_world)
    return base_coords(obj_b.data) @ m[:3, :3].T + m[:3, 3]


def selected_vertex(obj):
    """The active vertex in Edit Mode, else the first selected vertex.

    Args:
        obj: A mesh object.

    Returns:
        int | None: Vertex index, or None when no vertex is selected.
    """
    if obj.mode == "EDIT":
        import bmesh
        bm = bmesh.from_edit_mesh(obj.data)
        active = bm.select_history.active
        if isinstance(active, bmesh.types.BMVert):
            return active.index
        return next((v.index for v in bm.verts if v.select), None)
    sel = np.zeros(len(obj.data.vertices), bool)
    obj.data.vertices.foreach_get("select", sel)
    hits = np.flatnonzero(sel)
    return int(hits[0]) if len(hits) else None


def tessa_keys(obj) -> list:
    """Tessa's shape keys on ``obj``, in readout order.

    Args:
        obj: A mesh object.

    Returns:
        list[bpy.types.ShapeKey]: Keys named with ``KEY_PREFIX``.
    """
    sk = obj.data.shape_keys
    return [kb for kb in sk.key_blocks if kb.name.startswith(KEY_PREFIX)] if sk else []


def _key_fcurves(obj) -> list:
    """The F-curves that animate Tessa's keys (other shape-key animation is left out).

    Args:
        obj: A mesh object.

    Returns:
        list[bpy.types.FCurve]: Empty when there is no such animation or the action
        has an unexpected layout.
    """
    sk = obj.data.shape_keys
    ad = sk.animation_data if sk else None
    if not ad or not ad.action:
        return []
    try:
        from bpy_extras import anim_utils
        bag = anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot)
        return [fc for fc in bag.fcurves if fc.data_path.startswith(f'key_blocks["{KEY_PREFIX}')]
    except Exception:  # unexpected action layout
        return []


def _drop_key_animation(obj):
    """Remove the F-curves that animate Tessa's keys (other shape-key animation stays).

    Args:
        obj: A mesh object.
    """
    sk = obj.data.shape_keys
    ad = sk.animation_data if sk else None
    if not ad or not ad.action:
        return
    try:
        from bpy_extras import anim_utils
        bag = anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot)
        for fc in [fc for fc in bag.fcurves if fc.data_path.startswith(f'key_blocks["{KEY_PREFIX}')]:
            bag.fcurves.remove(fc)
    except Exception:  # unexpected action layout: Tessa owns the shape-key animation
        sk.animation_data_clear()


def _drop_attributes(mesh):
    """Remove every Tessa attribute (current and 0.1 names) from ``mesh``.

    Args:
        mesh: A ``bpy.types.Mesh``.
    """
    for name in ATTRS:
        if name in mesh.attributes:
            mesh.attributes.remove(mesh.attributes[name])


def clear(obj):
    """Remove Tessa's shape keys, their animation and its attributes.

    Args:
        obj: A mesh object.
    """
    _drop_key_animation(obj)
    for kb in tessa_keys(obj):
        obj.shape_key_remove(kb)
    _drop_attributes(obj.data)
    obj.data.update()


def write_frames(obj, scene, result, gain: float, step: int, colour: str, **playback):
    """One shape key per readout, keyframed in turn, plus the per-vertex attributes.

    Args:
        obj: Mesh object A (Object Mode).
        scene: The scene whose frame range the animation sets.
        result: ``engine.run``'s result for ``obj``.
        gain: See :func:`set_gain`.
        step: See :func:`set_timeline`.
        colour: See :func:`write_attributes`.
        **playback: ``loop``, ``original`` and ``ease`` for :func:`set_timeline`.
    """
    clear(obj)
    if not obj.data.shape_keys:
        obj.shape_key_add(name="Basis", from_mix=False)
    for i in range(len(result["decoded"])):
        obj.shape_key_add(name=f"{KEY_PREFIX}{i:03d}", from_mix=False).slider_max = 1.0
    set_gain(obj, result, gain)
    set_timeline(obj, scene, step, **playback)
    write_attributes(obj.data, result, colour)


def fits(obj, result) -> bool:
    """Whether ``obj`` still holds ``result``'s keys (same readouts and vertex count).

    Args:
        obj: A mesh object.
        result: ``engine.run``'s result.

    Returns:
        bool: True when the shape keys can be updated in place.
    """
    return (len(tessa_keys(obj)) == len(result["decoded"])
            and len(result["decoded"][0]) == len(obj.data.vertices))


def set_gain(obj, result, gain: float):
    """Write each key's positions: ref + gain * (decoded - ref).

    At gain 1 a key holds exactly the decoded positions; above 1 the deviation from
    the noiseless reference is exaggerated.

    Args:
        obj: A mesh object whose Tessa keys match ``result`` (:func:`fits`).
        result: ``engine.run``'s result.
        gain: Deviation multiplier.
    """
    for kb, d, r in zip(tessa_keys(obj), result["decoded"], result["refs"]):
        kb.data.foreach_set("co", (r + gain * (d - r)).astype(np.float32).ravel())
    obj.data.update()


def set_timeline(obj, scene, step: int, loop: bool = False, original: bool = False, ease: bool = False):
    """Keyframe the Tessa keys ``step`` frames apart from the scene's start frame.

    One key: its value is set to 1, without animation. Several: key i is 1 at its
    readout's frame and 0 at its neighbours' frames (the first and last hold before and
    after the run), so the shape blends from one readout to the next.

    ``original`` puts the unmodified mesh on the start frame, so readout 0 sits one step
    later. ``loop`` adds one more step that glides from the last readout back to where
    the run began (readout 0, or the original); the scene's end frame is then one frame
    short of that, so playback wraps without holding a frame twice. Otherwise the end
    frame is the last readout. ``ease`` eases in and out of every readout (sine) instead
    of blending linearly; the two weights still sum to 1, so it is a true ease between
    the two shapes.

    Args:
        obj: A mesh object with Tessa keys.
        scene: The scene whose frame range is set.
        step: Timeline frames between successive readouts.
        loop: Glide back to the start after the last readout.
        original: Start from the unmodified mesh.
        ease: Ease between readouts instead of blending linearly.
    """
    _drop_key_animation(obj)
    keys = tessa_keys(obj)
    if len(keys) == 1:
        keys[0].value = 1.0
        return
    n, start = len(keys), scene.frame_start
    first = start + (step if original else 0)  # readout 0's frame
    after = first + n * step                   # the frame the loop returns to the start on
    prefs = bpy.context.preferences.edit
    old = prefs.keyframe_new_interpolation_type
    prefs.keyframe_new_interpolation_type = "LINEAR"
    try:
        for i, kb in enumerate(keys):
            f = first + i * step
            frames = [(f, 1.0)]
            if i > 0 or original:
                frames.insert(0, (f - step, 0.0))
            if i < n - 1 or loop:
                frames.append((f + step, 0.0))
            if loop and i == 0 and not original:  # back to readout 0, in the last step only
                if after - step != f + step:
                    frames.append((after - step, 0.0))
                frames.append((after, 1.0))
            for frame, value in frames:
                kb.value = value
                kb.keyframe_insert("value", frame=frame)
    finally:
        prefs.keyframe_new_interpolation_type = old
    if ease:
        for fc in _key_fcurves(obj):
            for kp in fc.keyframe_points:
                kp.interpolation, kp.easing = "SINE", "EASE_IN_OUT"
            fc.update()
    scene.frame_end = after - 1 if loop else first + (n - 1) * step


def _ramp(x, stops) -> np.ndarray:
    """Evenly spaced sRGB hex stops, interpolated at ``x``, as linear RGBA.

    Args:
        x: (N,) values in [0, 1].
        stops: Hex colour strings, first at 0 and last at 1.

    Returns:
        np.ndarray: (N, 4) scene-linear RGBA.
    """
    rgb = np.array([[int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)] for h in stops])
    pos = np.linspace(0, 1, len(stops))
    x = np.clip(np.asarray(x, float), 0, 1)
    return _rgba(np.stack([np.interp(x, pos, rgb[:, c]) for c in range(3)], axis=1))


def _rgba(srgb) -> np.ndarray:
    """sRGB rows to scene-linear RGBA (Blender's float colour attributes are linear).

    Args:
        srgb: (N, 3) sRGB values in [0, 1].

    Returns:
        np.ndarray: (N, 4) linear RGBA, alpha 1.
    """
    c = np.asarray(srgb, float)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return np.hstack([lin, np.ones((len(c), 1))])


def qubit_colours(qubits) -> np.ndarray:
    """One colour per data qubit in use, as the desktop app's ``qubit_rgba``.

    Args:
        qubits: (N,) data qubit per vertex.

    Returns:
        np.ndarray: (N, 4) linear RGBA.
    """
    rank = np.unique(qubits, return_inverse=True)[1]
    return _rgba([colorsys.hls_to_rgb((r * GOLDEN) % 1, 0.55, 0.6) for r in rank])


def colours(result, colour: str):
    """Per-vertex colours for a Colour setting, or None when it has nothing to show.

    Args:
        result: ``engine.run``'s result.
        colour: "error", "qubit", "delta" or "none".

    Returns:
        np.ndarray | None: (N, 4) linear RGBA.
    """
    if colour == "error":
        err = result["vertex_error"]
        return _ramp(err / (err.max() or 1.0), ERROR_RAMP)
    if colour == "qubit" and result["vertex_qubit"] is not None:
        return qubit_colours(result["vertex_qubit"])
    if colour == "delta" and result["delta"] is not None:
        d = result["delta"]
        return _ramp(d / (d.max() or 1.0), DELTA_RAMP)
    return None


def write_attributes(mesh, result, colour: str):
    """Write the per-vertex data attributes and the chosen colour.

    ``tessa_error`` always; ``tessa_qubit`` unless the payload is spectral (its
    qubits hold modes, not vertices); ``tessa_morph_delta`` in morph. The colour goes
    to ``tessa_colour``, made the active colour attribute.

    Args:
        mesh: A ``bpy.types.Mesh``.
        result: ``engine.run``'s result.
        colour: "error", "qubit", "delta" or "none".
    """
    _drop_attributes(mesh)
    for name, kind, field, values in ((ERROR_ATTR, "FLOAT", "value", result["vertex_error"]),
                                      (QUBIT_ATTR, "INT", "value", result["vertex_qubit"]),
                                      (DELTA_ATTR, "FLOAT", "value", result["delta"])):
        if values is not None:
            dtype = np.int32 if kind == "INT" else np.float32
            mesh.attributes.new(name, kind, "POINT").data.foreach_set(field, np.asarray(values, dtype))
    rgba = colours(result, colour)
    if rgba is not None:
        c = mesh.color_attributes.new(COLOUR_ATTR, "FLOAT_COLOR", "POINT")
        c.data.foreach_set("color", rgba.astype(np.float32).ravel())
        mesh.color_attributes.active_color = c
    mesh.update()
