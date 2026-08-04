#!/usr/bin/env python3
"""Enable "shrink text on overflow" on every text box of a pandoc-generated .pptx.

Pandoc writes bare `<a:bodyPr />` elements, which means "do not autofit": any slide
whose content is taller than its placeholder silently overflows off the bottom, and
long code lines wrap into unreadable soup. PowerPoint and Impress both honour
`<a:normAutofit/>`, so injecting it makes over-full slides scale down instead.

Usage: fit_pptx.py slides.pptx [...]
"""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

BARE_BODY_PR = "<a:bodyPr />"
AUTOFIT_BODY_PR = "<a:bodyPr><a:normAutofit /></a:bodyPr>"


def fit(pptx: Path) -> int:
    """Rewrite pptx in place; return the number of text boxes patched."""
    patched = 0
    tmp = pptx.with_suffix(".pptx.tmp")

    with zipfile.ZipFile(pptx) as src, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename.startswith("ppt/slides/slide") and item.filename.endswith(".xml"):
                text = data.decode("utf-8")
                patched += text.count(BARE_BODY_PR)
                data = text.replace(BARE_BODY_PR, AUTOFIT_BODY_PR).encode("utf-8")
            dst.writestr(item, data)

    shutil.move(tmp, pptx)
    return patched


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for arg in sys.argv[1:]:
        path = Path(arg)
        print(f"{path}: autofit enabled on {fit(path)} text boxes")


if __name__ == "__main__":
    main()
