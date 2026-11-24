"""Copy the quantum code into the extension, so the add-on ships it as source.

Run from this folder whenever you change the core in ../tessa_ripple:

    python3 build_ext.py

It (re)creates two generated folders. Edit the originals, never these copies:

    core/   ../tessa_ripple's bpy-free modules (not app.py, the matplotlib app),
            with ``from tessa ...`` rewritten to ``from ..tessa ...``
    tessa/  Moth's Tessa package (MIT), copied from ../.venv (or $TESSA_SRC)

Tessa is not on PyPI, and Blender's extension rules only allow unmodified PyPI wheels,
so it is vendored as source instead of bundled as a wheel.
"""

import os
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CORE_SRC = REPO / "tessa_ripple"
CORE_MODULES = ("__init__", "bridge", "mesh", "morph", "spectral", "interfere", "animate")
CORE_DATA = ("quality.json",)  # probe_qubits scores per chip snapshot (bridge.QUALITY_FILE)
TESSA_IMPORT = re.compile(r"^(\s*)from tessa(\.|\s)", re.M)
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")


def tessa_source() -> Path:
    """The installed Tessa package: $TESSA_SRC, else ../.venv's site-packages."""
    if os.environ.get("TESSA_SRC"):
        return Path(os.environ["TESSA_SRC"]).expanduser().resolve()
    found = sorted(REPO.glob(".venv/lib/python3*/site-packages/tessa"))
    if not found:
        sys.exit("Tessa not found: create ../.venv (see the README) or set $TESSA_SRC "
                 "to a folder named tessa")
    return found[-1]


def tessa_license(src: Path) -> Path | None:
    """Tessa's licence file, from its dist-info or a checkout named by $TESSA_REPO."""
    places = list(src.parent.glob("tessa_paper-*.dist-info/licenses/*"))
    if os.environ.get("TESSA_REPO"):
        places += list(Path(os.environ["TESSA_REPO"]).expanduser().glob("LICEN[CS]E*"))
    return places[0] if places else None


def fresh(folder: Path):
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir()


def build_core():
    out = HERE / "core"
    fresh(out)
    for name in CORE_MODULES:
        text = (CORE_SRC / f"{name}.py").read_text()
        (out / f"{name}.py").write_text(TESSA_IMPORT.sub(r"\1from ..tessa\2", text))
    for name in CORE_DATA:
        shutil.copy2(CORE_SRC / name, out / name)
    print(f"core/   {len(CORE_MODULES)} modules from {CORE_SRC}")


def build_tessa():
    src, out = tessa_source(), HERE / "tessa"
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src, out, ignore=SKIP)
    lic = tessa_license(src)
    if lic:
        shutil.copy2(lic, out / "LICENSE")
        print(f"tessa/  from {src} (licence: {lic.name})")
    else:
        print(f"tessa/  from {src}\n"
              "        WARNING: no licence file found. Tessa's metadata says MIT; copy its "
              "LICENSE into tessa/ (or set $TESSA_REPO to a checkout) before sharing.")


if __name__ == "__main__":
    build_core()
    build_tessa()
