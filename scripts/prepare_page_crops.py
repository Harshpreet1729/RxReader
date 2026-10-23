"""Turn explicitly reviewed, de-identified page annotations into OCR crop splits.

This prepares data only. It never trains or alters the existing RxHandBD dataset.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile
from src.prescription.documents import decode_document
from src.prescription.layout import checked_box
from src.prescription.page_review import assemble_medications


def contained(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Manifest file must exist inside the manifest directory.")
    return path


def prepare(manifest_path, output):
    manifest_path = Path(manifest_path)
    output = Path(output)
    if output.exists():
        raise ValueError(
            "Choose a new output folder; existing data is never overwritten."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if (
        manifest.get("permission_confirmed") is not True
        or manifest.get("deidentified") is not True
    ):
        raise ValueError(
            "Confirm data permission and de-identification in the manifest."
        )
    documents = manifest.get("documents", [])
    if not documents:
        raise ValueError("Manifest has no documents.")
    groups = {}
    image_splits = {}
    crops = {}
    prepared = []
    source_ids = set()
    for index, document in enumerate(documents):
        group = str(document.get("group", "")).strip()
        split = document.get("split")
        if not group or split not in {"train", "validation", "test"}:
            raise ValueError(
                "Every page needs a writer/clinic/document group and an explicit split."
            )
        if group in groups and groups[group] != split:
            raise ValueError("A document/writer group crosses splits.")
        groups[group] = split
        image, png = decode_document(
            contained(manifest_path.parent, document["image"]).read_bytes()
        )
        digest = hashlib.sha256(png).hexdigest()
        if digest in source_ids:
            raise ValueError("Duplicate page in manifest.")
        source_ids.add(digest)
        image_splits[digest] = split
        review = json.loads(
            contained(manifest_path.parent, document["review"]).read_text(
                encoding="utf-8-sig"
            )
        )
        if (
            review.get("schema_version") != 1
            or review.get("image_sha256") != digest
            or review.get("image_size") != list(image.size)
        ):
            raise ValueError("Review must match the exact uploaded page and schema.")
        regions = review.get("regions", [])
        if not regions or any(
            r.get("role") not in {"medicine", "instruction", "other"}
            or not r.get("reviewed")
            for r in regions
        ):
            raise ValueError(
                "Every region must have a reviewed role; unreadable pages need separate adjudication."
            )
        records = assemble_medications(regions)
        if not records or not all(r["transcription_reviewed"] for r in records):
            raise ValueError("All medicine records require reviewed transcriptions.")
        if any(
            r["role"] == "instruction" and not r.get("medicine_id") for r in regions
        ):
            raise ValueError(
                "Instruction regions must be linked before preparing training data."
            )
        for region in regions:
            box = checked_box(region["box"], image.size)
            if region["role"] == "other":
                continue
            text = " ".join(region["text"].split())
            if not text or "?" in text:
                raise ValueError(
                    "Empty or uncertain crop labels must not enter training."
                )
            crop = image.crop(box)
            fingerprint = hashlib.sha256(
                str(crop.size).encode() + crop.tobytes()
            ).hexdigest()
            if fingerprint in crops:
                previous = crops[fingerprint]
                if previous != (split, text):
                    raise ValueError(
                        "Duplicate crop crosses splits or has conflicting labels."
                    )
                continue
            crops[fingerprint] = (split, text)
            prepared.append(
                (
                    crop,
                    {
                        "filename": f"{index:04d}_{len(prepared):05d}.png",
                        "label": text,
                        "split": split,
                        "role": region["role"],
                        "group": group,
                        "page_sha256": digest,
                        "box": box,
                    },
                )
            )
    if not prepared:
        raise ValueError("No eligible crops.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix="page-crops-"
    ) as staging:
        target = Path(staging) / "prepared"
        (target / "images").mkdir(parents=True)
        for image, row in prepared:
            path = target / "images" / row["filename"]
            image.save(path)
            row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        for split in ("train", "validation", "test"):
            with (target / f"{split}.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=["filename", "label"])
                writer.writeheader()
                writer.writerows(
                    {k: r[k] for k in ("filename", "label")}
                    for _, r in prepared
                    if r["split"] == split
                )
        evidence = {
            "documents": len(documents),
            "crops": len(prepared),
            "counts": {
                s: sum(r["split"] == s for _, r in prepared)
                for s in ("train", "validation", "test")
            },
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "regions": [r for _, r in prepared],
            "limitations": "User-reviewed transcriptions, not independently adjudicated ground truth. No near-duplicate or unknown-writer independence claim.",
        }
        (target / "preparation.json").write_text(
            json.dumps(evidence, indent=2), encoding="utf-8"
        )
        target.rename(output)
    return {k: v for k, v in evidence.items() if k != "regions"}


def verify(root):
    """Verify frozen crop bytes, labels and group splits before training."""
    root = Path(root)
    metadata = json.loads((root / "preparation.json").read_text(encoding="utf-8"))
    records = metadata["regions"]
    seen = set()
    groups = {}
    pixels = {}
    counts = {}
    for split in ("train", "validation", "test"):
        with (root / f"{split}.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        expected = [
            {"filename": r["filename"], "label": r["label"]}
            for r in records
            if r["split"] == split
        ]
        if rows != expected:
            raise ValueError(
                "Crop labels or split membership changed after preparation."
            )
        counts[split] = len(rows)
    for row in records:
        if row["filename"] in seen:
            raise ValueError("Repeated crop filename.")
        seen.add(row["filename"])
        path = contained(root / "images", row["filename"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Crop bytes changed after preparation.")
        group = row["group"]
        split = row["split"]
        if group in groups and groups[group] != split:
            raise ValueError("A group crosses splits.")
        groups[group] = split
        from PIL import Image

        with Image.open(path) as image:
            image = image.convert("RGB")
            fingerprint = hashlib.sha256(
                str(image.size).encode() + image.tobytes()
            ).hexdigest()
        if fingerprint in pixels:
            raise ValueError("Duplicate crop pixels in prepared data.")
        pixels[fingerprint] = split
    if not counts["train"] or not counts["validation"]:
        raise ValueError(
            "Training needs nonempty train and validation splits from different groups."
        )
    if len(seen) != sum(counts.values()):
        raise ValueError("Invalid crop split metadata.")
    return {
        "format": "reviewed_pages",
        "crops": len(seen),
        "groups": len(groups),
        "splits": counts,
        "all_crop_hashes_verified": True,
        "training_started": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest, args.output), indent=2))


if __name__ == "__main__":
    main()
