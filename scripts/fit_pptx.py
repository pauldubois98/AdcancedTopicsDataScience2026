#!/usr/bin/env python3
"""Post-process a pandoc-generated .pptx so it renders the way the slide reads.

Two fixes, both of which pandoc cannot express from markdown:

1. Autofit. Pandoc writes bare `<a:bodyPr />` elements, which means "do not
   autofit": any slide whose content is taller than its placeholder silently
   overflows off the bottom, and long code lines wrap into unreadable soup.
   PowerPoint and Impress both honour `<a:normAutofit/>`, so injecting it makes
   over-full slides scale down instead.

2. Question slides. A `>` blockquote is how the deck asks the room a question, and
   pandoc renders it as a left-aligned, indented paragraph sitting at the top of an
   otherwise empty placeholder. Centring it — horizontally, and vertically in the
   box — makes the slide read as the single question it is.

Usage: fit_pptx.py slides.pptx [...]
"""

from __future__ import annotations

import re
import shutil
import sys
import zipfile
from pathlib import Path

BARE_BODY_PR = "<a:bodyPr />"
AUTOFIT_BODY_PR = "<a:bodyPr><a:normAutofit /></a:bodyPr>"

# pandoc's blockquote indent, in EMU; the only thing marking a quote in the output
QUOTE_MARL = 'marL="1270000"'
SHAPE = re.compile(r"<p:sp>.*?</p:sp>", re.DOTALL)


def centre_quotes(shape: str) -> str:
    """Centre a blockquote shape: drop the indent, centre the text, anchor middle."""
    shape = shape.replace(QUOTE_MARL, 'marL="0" algn="ctr"')
    return re.sub(r"<a:bodyPr(?=[ />])", '<a:bodyPr anchor="ctr"', shape, count=1)


def patch(text: str) -> tuple[str, int, int]:
    """Return the patched slide XML plus (text boxes autofitted, quotes centred)."""
    boxes = text.count(BARE_BODY_PR)
    text = text.replace(BARE_BODY_PR, AUTOFIT_BODY_PR)

    quotes = 0

    def one(m: re.Match[str]) -> str:
        nonlocal quotes
        if QUOTE_MARL not in m.group(0):
            return m.group(0)
        quotes += 1
        return centre_quotes(m.group(0))

    return SHAPE.sub(one, text), boxes, quotes


def fit(pptx: Path) -> tuple[int, int]:
    """Rewrite pptx in place; return (text boxes autofitted, quotes centred)."""
    boxes = quotes = 0
    tmp = pptx.with_suffix(".pptx.tmp")

    with zipfile.ZipFile(pptx) as src, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename.startswith("ppt/slides/slide") and item.filename.endswith(".xml"):
                text, b, q = patch(data.decode("utf-8"))
                boxes += b
                quotes += q
                data = text.encode("utf-8")
            dst.writestr(item, data)

    shutil.move(tmp, pptx)
    return boxes, quotes


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for arg in sys.argv[1:]:
        path = Path(arg)
        boxes, quotes = fit(path)
        print(f"{path}: autofit enabled on {boxes} text boxes, "
              f"{quotes} question slides centred")


if __name__ == "__main__":
    main()
