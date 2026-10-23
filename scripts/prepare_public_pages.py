"""Audit the pinned prescription-page archive and prepare UNREVIEWED annotations.

No training, automatic transcription, source relabeling, or split assignment.
"""

import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import tempfile
from zipfile import ZipFile

from PIL import Image
from src.prescription.data import ROOT
from src.prescription.documents import decode_document
from src.prescription.layout import checked_box


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def parse_box(line, size):
    """Source uses class 1 despite listing only one class. Reject other classes."""
    fields = line.split()
    if len(fields) != 5 or fields[0] != "1":
        raise ValueError("Expected source medicine class 1 and four coordinates.")
    x, y, w, h = map(float, fields[1:])
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        raise ValueError("Invalid source box.")
    box = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
    # Six-decimal YOLO values can round an edge a fraction outside the page.
    if any(v < -0.000001 or v > 1.000001 for v in box):
        raise ValueError("Source box extends outside the page.")
    box = [min(1, max(0, v)) for v in box]
    return checked_box(box, size, normalized=True)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def prepare(archive, output):
    archive, output = Path(archive), Path(output)
    if output.exists():
        raise ValueError("Output already exists; choose a new folder.")
    source = json.loads((ROOT / "data/prescription_pages_bd/SOURCE.json").read_text())
    raw = archive.read_bytes()
    if len(raw) != source["archive_bytes"] or sha256(raw) != source["archive_sha256"]:
        raise ValueError("Archive differs from the pinned publisher file.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix="page-preparation-"
    ) as tmp:
        target = Path(tmp) / "prepared"
        for folder in ("pages", "labels", "crops", "reviews", "regions"):
            (target / folder).mkdir(parents=True)
        pages, crops = [], []
        page_hashes, crop_hashes = defaultdict(list), defaultdict(list)
        with ZipFile(io.BytesIO(raw)) as z:
            names = z.namelist()
            if (
                len(names) != len(set(names))
                or sum(i.file_size for i in z.infolist()) > 100_000_000
            ):
                raise ValueError("Unexpected archive entries or expanded size.")
            images = sorted(
                n
                for n in names
                if n.startswith("Dataset/images/")
                and PurePosixPath(n).suffix.lower() in {".png", ".jpeg", ".jpg"}
            )
            stems = [PurePosixPath(n).stem for n in images]
            if (
                len(set(stems)) != source["expected_pages"]
                or len(stems) != source["expected_pages"]
                or not all(s.isdigit() for s in stems)
            ):
                raise ValueError("Unexpected page identities/count.")
            expected_labels = {f"Dataset/labels/{s}.txt" for s in stems}
            actual_labels = {
                n
                for n in names
                if n.startswith("Dataset/labels/")
                and n.endswith(".txt")
                and not n.endswith("/classes.txt")
            }
            if (
                actual_labels != expected_labels
                or z.read("Dataset/labels/classes.txt").decode().strip() != "medicine"
            ):
                raise ValueError("Image/label pairing or class description differs.")
            count_rows = list(
                csv.DictReader(
                    io.StringIO(
                        z.read("Dataset/prescription_box_counts.csv").decode(
                            "utf-8-sig"
                        )
                    )
                )
            )
            counts = {
                r["image_number"]: int(r["num_medicine_boxes"]) for r in count_rows
            }
            if len(count_rows) != len(stems) or set(counts) != set(stems):
                raise ValueError("Count CSV page identities differ.")
            for name in images:
                stem = PurePosixPath(name).stem
                data = z.read(name)
                with Image.open(io.BytesIO(data)) as original:
                    if original.getexif().get(274, 1) != 1:
                        raise ValueError(
                            "EXIF rotation requires explicit box transformation."
                        )
                image, png = decode_document(data)
                page_hash = sha256(str(image.size).encode() + image.tobytes())
                page_hashes[page_hash].append(stem)
                (target / "pages" / f"{stem}.png").write_bytes(png)
                lines = z.read(f"Dataset/labels/{stem}.txt").decode().splitlines()
                boxes = [parse_box(line, image.size) for line in lines if line.strip()]
                if len(boxes) != counts[stem] or len({tuple(b) for b in boxes}) != len(
                    boxes
                ):
                    raise ValueError(f"Box count/duplicates differ on page {stem}.")
                labels, regions = [], []
                for index, box in enumerate(boxes, 1):
                    region_id = str(index)
                    regions.append(
                        {
                            "id": region_id,
                            "box": box,
                            "role": "medicine",
                            "ocr_text": "",
                            "text": "",
                            "medicine_id": "",
                            "reviewed": False,
                        }
                    )
                    crop = image.crop(box)
                    crop_name = f"crops/{stem}_{index}.png"
                    crop.save(target / crop_name)
                    digest = sha256(str(crop.size).encode() + crop.tobytes())
                    crop_hashes[digest].append(crop_name)
                    crops.append(
                        {
                            "image": crop_name,
                            "page": stem,
                            "region": region_id,
                            "box": box,
                            "text": "",
                            "reviewed": False,
                            "sha256": sha256((target / crop_name).read_bytes()),
                        }
                    )
                    x1, y1, x2, y2 = box
                    labels.append(
                        f"0 {(x1+x2)/(2*image.width):.8f} {(y1+y2)/(2*image.height):.8f} {(x2-x1)/image.width:.8f} {(y2-y1)/image.height:.8f}"
                    )
                (target / "labels" / f"{stem}.txt").write_text(
                    "\n".join(labels) + "\n", encoding="utf-8"
                )
                write_json(
                    target / "reviews" / f"{stem}.json",
                    {
                        "schema_version": 1,
                        "image_sha256": sha256(png),
                        "image_size": list(image.size),
                        "regions": regions,
                    },
                )
                write_json(
                    target / "regions" / f"{stem}.json",
                    {
                        "coordinate_space": "pixels",
                        "regions": [{"id": r["id"], "box": r["box"]} for r in regions],
                    },
                )
                pages.append(
                    {
                        "image": f"pages/{stem}.png",
                        "review": f"reviews/{stem}.json",
                        "group": "",
                        "split": "unassigned",
                        "source_member": name,
                        "source_sha256": sha256(data),
                        "image_sha256": sha256(png),
                        "medicine_boxes": len(boxes),
                    }
                )
        if len(crops) != source["expected_boxes"]:
            raise ValueError("Unexpected total medicine box count.")
        report = {
            "pages": len(pages),
            "medicine_boxes": len(crops),
            "archive_sha256_verified": True,
            "image_label_pairs_verified": True,
            "count_csv_verified": True,
            "all_boxes_in_bounds": True,
            "duplicate_pages": [v for v in page_hashes.values() if len(v) > 1],
            "duplicate_crops": [v for v in crop_hashes.values() if len(v) > 1],
            "source_class": 1,
            "derived_class": 0,
            "transcriptions_reviewed": 0,
            "privacy_review_complete": False,
            "split_assigned": False,
            "training_ready": False,
        }
        write_json(target / "audit.json", report)
        write_json(target / "SOURCE.json", source)
        write_json(
            target / "manifest.draft.json",
            {"permission_confirmed": True, "deidentified": False, "documents": pages},
        )
        write_json(target / "crop_annotations.draft.json", crops)
        (target / "labels/classes.txt").write_text("medicine\n", encoding="utf-8")
        (target / "README.txt").write_text(
            "UNREVIEWED SUPPLEMENTARY DATA - NOT TRAINING READY\nSource: "
            + source["source_url"]
            + "\nAuthors: "
            + ", ".join(source["authors"])
            + "\nLicense: CC BY 4.0, "
            + source["license_url"]
            + "\nChanges: canonical PNG pages, integer-coordinate crops, source class 1 mapped to class 0 in derived labels only.\nThe original ZIP is unchanged. Full pages need privacy and box review. All crop transcriptions are empty/unreviewed. No train/test split or writer independence is claimed.\nSee docs/PUBLIC_PAGE_DATA.md for workflow.\n",
            encoding="utf-8",
        )
        target.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=ROOT / "Dataset/PrescriptionPagesBD/source/Dataset.zip",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "Dataset/PrescriptionPagesBD/prepared"
    )
    args = parser.parse_args()
    print(json.dumps(prepare(args.archive, args.output), indent=2))


if __name__ == "__main__":
    main()
