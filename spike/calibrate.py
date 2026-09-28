"""Draw the field template on a flattened card so the ROI boxes can be tuned by eye.

    python spike/calibrate.py path/to/card.jpg out.png

Adjust the fractions in src/khmer_kyc_ocr/pipeline/template.py until every box tightly
contains its field (label bleed is fine; the parsers strip labels) on several real cards,
then bump TEMPLATE_VERSION and drop the "-uncalibrated" suffix.
Output images contain personal data: keep them out of the repo (*.png is gitignored).
"""

import sys
from pathlib import Path

import cv2

from khmer_kyc_ocr.pipeline.decode import decode_image
from khmer_kyc_ocr.pipeline.template import TEMPLATE_V1
from khmer_kyc_ocr.pipeline.warp import CARD_H, CARD_W, find_card


def main(src: Path, dst: Path) -> int:
    image = decode_image(src.read_bytes())
    if image is None:
        print("could not decode image")
        return 1
    card = find_card(image)
    if card is None:
        print("no card found: use a photo of the whole card on a darker background")
        return 1
    for name, roi in TEMPLATE_V1.items():
        x0, y0, x1, y1 = roi.to_pixels(CARD_W, CARD_H)
        cv2.rectangle(card, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(card, name, (x0 + 4, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite(str(dst), card)
    print(f"wrote {dst} (if the card is upside down, the service will flip it; tune upright)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
