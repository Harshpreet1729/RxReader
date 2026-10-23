"""Optional page detection or explicit user-defined regions. No fixed universal slip."""

import json
import math
from functools import lru_cache
from pathlib import Path


def checked_box(box, size, normalized=False):
    if len(box) != 4 or any(not math.isfinite(float(v)) for v in box):
        raise ValueError("Bounding boxes need four finite coordinates.")
    width, height = size
    if normalized:
        if any(v < 0 or v > 1 for v in box):
            raise ValueError("Normalized coordinates must be between zero and one.")
        box = [box[0] * width, box[1] * height, box[2] * width, box[3] * height]
    x1, y1, x2, y2 = [round(v) for v in box]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"Invalid/out-of-image bounding box: {box}")
    return [x1, y1, x2, y2]


@lru_cache(maxsize=1)
def page_detector():
    from paddleocr import TextDetection

    return TextDetection(
        model_name="PP-OCRv5_mobile_det", device="cpu", enable_mkldnn=False
    )


def detect_regions(image, mode="word", template=None):
    if template:
        payload = (
            json.loads(Path(template).read_text(encoding="utf-8-sig"))
            if isinstance(template, (str, Path))
            else template
        )
        if payload.get("coordinate_space") not in ("normalized", "pixels"):
            raise ValueError("Specify coordinate_space: normalized or pixels.")
        regions = []
        normalized = payload["coordinate_space"] == "normalized"
        for region in payload["regions"]:
            box = checked_box(region["box"], image.size, normalized)
            regions.append({"id": str(region["id"]), "box": box})
        if not regions or len({r["id"] for r in regions}) != len(regions):
            raise ValueError("Regions must be nonempty and have unique IDs.")
        return regions
    if mode == "word":
        return [{"id": "1", "box": [0, 0, *image.size]}]
    if mode != "page":
        raise ValueError("Mode must be word or page.")
    try:
        import numpy as np
        from paddleocr import TextDetection
    except ImportError as error:
        raise RuntimeError(
            "Page detection needs requirements-page.txt (Python 3.11 recommended), or supply a regions JSON file."
        ) from error
    detector = page_detector()
    regions = []
    # Paddle expects BGR channel order; Pillow images use RGB.
    bgr_image = np.asarray(image)[:, :, ::-1]
    for result in detector.predict(bgr_image):
        payload = result.json
        if isinstance(payload, str):
            payload = json.loads(payload)
        payload = payload.get("res", payload)
        for polygon in payload.get("dt_polys", []):
            xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
            box = [
                max(0, min(xs)),
                max(0, min(ys)),
                min(image.width, max(xs)),
                min(image.height, max(ys)),
            ]
            regions.append({"box": checked_box(box, image.size)})
    regions.sort(key=lambda r: (r["box"][1], r["box"][0]))
    for i, region in enumerate(regions, 1):
        region["id"] = str(i)
    if not regions:
        raise ValueError(
            "No text regions detected. Supply manually selected line regions."
        )
    return regions
