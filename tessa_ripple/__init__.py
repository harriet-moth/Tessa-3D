"""Tessa Ripple: mesh vertices encoded with Tessa, moved only by its readout noise.

Modules: :mod:`~tessa_ripple.mesh` (preset meshes, OBJ input/output, topology,
decimation), :mod:`~tessa_ripple.morph` (matching a second mesh B to A, classical
only), :mod:`~tessa_ripple.bridge` (every Tessa call: placement on a lattice or
chip, encode, decode, resources, coherence edge readout),
:mod:`~tessa_ripple.animate` (multi-frame readouts, blending, timelines, OBJ
sequences) and :mod:`~tessa_ripple.app` (the matplotlib desktop app). The web
app's API is ``api/tessa.py``, which calls the same bridge.
"""
