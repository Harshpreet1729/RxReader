"""Prepare redacted page copies and resumable, unreviewed OCR annotations.

Run from the project root. No training or source-file modification occurs.
"""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile

from PIL import Image
from src.prescription.annotations import load_annotations
from src.prescription.data import ROOT
from src.prescription.documents import decode_document
from src.prescription.layout import checked_box


def digest(data):
    return hashlib.sha256(data).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True)
        stream.write("\n")
        temp = Path(stream.name)
    temp.replace(path)


def redacted_copy(image, retain_box, regions, redactions=()):
    """Use actual source pixels inside reviewed bounds; white everywhere else."""
    retain_box = checked_box(retain_box, image.size)
    for region in regions:
        x1, y1, x2, y2 = checked_box(region["box"], image.size)
        a, b, c, d = retain_box
        if x1 < a or y1 < b or x2 > c or y2 > d:
            raise ValueError("Retained block would cut a medicine annotation.")
    out = Image.new("RGB", image.size, "white")
    out.paste(image.crop(retain_box), retain_box[:2])
    for box in redactions:
        box = checked_box(box, image.size)
        for r in regions:
            a, b, c, d = r["box"]
            if max(a, box[0]) < min(c, box[2]) and max(b, box[1]) < min(d, box[3]):
                raise ValueError("Redaction overlaps a medicine annotation.")
        out.paste("white", box)
    buffer = io.BytesIO()
    out.save(buffer, format="PNG")
    return buffer.getvalue()


def prepare(source, plan_path, output):
    source, plan_path, output = Path(source), Path(plan_path), Path(output)
    if output.exists():
        raise ValueError("Choose a new batch directory; existing files are preserved.")
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    page_ids = [p["page_id"] for p in plan["pages"]]
    if (
        not page_ids
        or len(page_ids) != len(set(page_ids))
        or any(
            not isinstance(p, str) or not p.isascii() or not p.isdigit()
            for p in page_ids
        )
    ):
        raise ValueError("Plan needs unique numeric page IDs.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix="annotation-batch-"
    ) as tmp:
        target = Path(tmp) / "batch"
        for folder in ("pages", "reviews"):
            (target / folder).mkdir(parents=True)
        documents = []
        for entry in plan["pages"]:
            identifier = entry["page_id"]
            source_image = (source / "pages" / f"{identifier}.png").read_bytes()
            source_review = (source / "reviews" / f"{identifier}.json").read_bytes()
            if (
                digest(source_image) != entry["source_image_sha256"]
                or digest(source_review) != entry["source_review_sha256"]
            ):
                raise ValueError(
                    "Source image or annotations changed since visual screening."
                )
            image, canonical = decode_document(source_image)
            regions = load_annotations(source_review, canonical, image.size)
            if any(r["text"] or r["reviewed"] for r in regions):
                raise ValueError(
                    "Preparation requires blank source drafts, not existing corrections."
                )
            png = redacted_copy(
                image, entry["retain_box"], regions, entry.get("redactions", [])
            )
            image_path = f"pages/{identifier}.png"
            review_path = f"reviews/{identifier}.json"
            (target / image_path).write_bytes(png)
            review = {
                "schema_version": 1,
                "image_sha256": digest(png),
                "image_size": list(image.size),
                "status": "draft",
                "regions": regions,
                "review_notes": entry["review_notes"],
                "privacy_screen": entry["privacy_screen"],
                "source_page_id": identifier,
            }
            atomic_json(target / review_path, review)
            documents.append(
                {
                    "page_id": identifier,
                    "image": image_path,
                    "review": review_path,
                    "image_sha256": digest(png),
                    "regions": len(regions),
                    "group": "",
                    "split": "unassigned",
                    "review_notes": entry["review_notes"],
                }
            )
        manifest = {
            "schema_version": 1,
            "plan_sha256": digest(plan_bytes),
            "permission_confirmed": True,
            "deidentified": False,
            "training_ready": False,
            "selection": plan["selection"],
            "source": plan["source"],
            "excluded": plan["excluded"],
            "documents": documents,
        }
        atomic_json(target / "manifest.draft.json", manifest)
        (target / "redaction_plan.json").write_bytes(plan_bytes)
        source_card = ROOT / "data/prescription_pages_bd/SOURCE.json"
        (target / "SOURCE.json").write_bytes(source_card.read_bytes())
        with (target / "review_queue.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(["page", "regions", "image", "review", "notes"])
            for row in documents:
                writer.writerow(
                    [
                        row["page_id"],
                        row["regions"],
                        row["image"],
                        row["review"],
                        " | ".join(row["review_notes"]),
                    ]
                )
        (target / "START_HERE.md").write_text(
            "# First annotation batch\n\n20 redacted working copies; all transcriptions require review. Original pages are unchanged.\n\nStart with page 7, which has comparatively clear printed medicine names. Open http://127.0.0.1:8000/ and upload a file from pages/ with the same-numbered JSON from reviews/. Turn automatic reading off to inspect the generated draft. Correct each reading, role and link before checking its review box. Download the corrected JSON; keep it with that exact page. Save corrections separately from generated drafts.\n\nRead review_queue.csv before starting a page. Pages 12, 17 and 20 contain crossed-out/edited entries. Page 20 also needs a box for a handwritten replacement. Do not treat crossed-out entries as active medicines. Unreadable entries need adjudication, not guesses.\n\nThe working copy preserves the medicine block and removes pixels outside it. Some schedules may be clipped or in Bangla; this is a medicine-transcription pilot, not an English schedule benchmark. Original full-page detection cannot be evaluated on these masked pages.\n\nNo train/validation/test split has been assigned; writer/clinic grouping and independent privacy review are still needed. Model drafts must not be used as verified training labels.\n\nSource: Dipto saha, Sourav Debnath, Anirban Roy, DOI 10.17632/k62rfd23kz.2, CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Changes: redacted context outside the selected medicine block; generated unreviewed OCR text. See SOURCE.json and redaction_plan.json.\n",
            encoding="utf-8",
        )
        target.rename(output)
    return manifest


def batch_files(root, manifest):
    root = Path(root).resolve()
    for row in manifest["documents"]:
        paths = [(root / row[key]).resolve() for key in ("image", "review")]
        if any(not path.is_relative_to(root) or not path.is_file() for path in paths):
            raise ValueError("Batch files must exist inside the batch directory.")
        data = paths[0].read_bytes()
        if digest(data) != row["image_sha256"]:
            raise ValueError("Batch image changed; make a new matching review.")
        image, png = decode_document(data)
        review = json.loads(paths[1].read_text(encoding="utf-8"))
        load_annotations(json.dumps(review).encode(), png, image.size)
        yield row, image, paths[1], review


def read_batch(output, reader, model_name):
    """Checkpoint each crop; resume never replaces existing readings/corrections."""
    output = Path(output)
    manifest = json.loads((output / "manifest.draft.json").read_text())
    # Validate every image/review before any mutation or inference.
    for _ in batch_files(output, manifest):
        pass
    total = sum(row["regions"] for row in manifest["documents"])
    completed = 0
    for row, image, path, review in batch_files(output, manifest):
        for region in review["regions"]:
            if (
                not region["text"].strip()
                and not region["reviewed"]
                and not region.get("ocr_attempted", False)
            ):
                result = reader.read(image.crop(region["box"]))
                region.update(
                    text=result["text"],
                    ocr_text=result["text"],
                    alternatives=result.get("alternatives", []),
                    possibly_truncated=result.get("possibly_truncated", False),
                    reviewed=False,
                    ocr_attempted=True,
                )
                review["draft_model"] = model_name
                review["status"] = "draft"
                atomic_json(path, review)
            completed += 1
            print(
                f"[{completed}/{total}] page {row['page_id']} region {region['id']} saved/kept; review status preserved",
                flush=True,
            )
    report = {
        "pages": len(manifest["documents"]),
        "regions": total,
        "ocr_attempted": 0,
        "nonempty_readings": 0,
        "reviewed_regions": 0,
        "training_started": False,
    }
    for _, _, _, review in batch_files(output, manifest):
        for region in review["regions"]:
            report["ocr_attempted"] += bool(region.get("ocr_attempted"))
            report["nonempty_readings"] += bool(region["text"].strip())
            report["reviewed_regions"] += region["reviewed"]
    atomic_json(output / "progress.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=ROOT / "Dataset/PrescriptionPagesBD/prepared"
    )
    parser.add_argument(
        "--plan", type=Path, default=ROOT / "data/prescription_pages_bd/batch_001.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "Dataset/PrescriptionPagesBD/annotation_batch_001",
    )
    parser.add_argument(
        "--read",
        action="store_true",
        help="Generate or resume unreviewed OCR drafts using the installed model.",
    )
    args = parser.parse_args()
    if not args.output.exists():
        prepare(args.source, args.plan, args.output)
    else:
        manifest = json.loads((args.output / "manifest.draft.json").read_text())
        if manifest["plan_sha256"] != digest(args.plan.read_bytes()):
            raise ValueError("Existing batch uses a different redaction plan.")
        for _ in batch_files(args.output, manifest):
            pass
    if args.read:
        import torch
        from src.prescription.ocr import TrOCRReader

        config = json.loads((ROOT / "configs/app.json").read_text())
        model = (ROOT / config["model_path"]).resolve()
        if not model.is_dir():
            raise ValueError("Installed local model is missing.")
        torch.set_num_threads(4)
        print("Loading installed model for draft OCR (CPU); no training.", flush=True)
        reader = TrOCRReader(str(model), device="cpu")
        print(json.dumps(read_batch(args.output, reader, str(model)), indent=2))
    else:
        print(
            f"Batch prepared/verified at {args.output}. Use --read to generate drafts."
        )


if __name__ == "__main__":
    main()
