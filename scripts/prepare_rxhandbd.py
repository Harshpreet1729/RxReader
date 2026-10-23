"""Prepare/verify RxHandBD without executing archive contents or modifying source labels."""

import argparse
import csv
import hashlib
import io
import json
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image
from src.prescription.progress import emit, progress_rows

PROJECT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://zenodo.org/records/18478741"
EXPECTED_MD5 = "c096859fcccb33655edaf34e4c1e7a06"
SOURCE_SPLITS = [
    ("train", "Train_Label.csv", "Train_Set"),
    ("test", "Test_Labels.csv", "Test_Set"),
]
FIELDS = [
    "filename",
    "label",
    "source_split",
    "split",
    "reason",
    "sha256",
    "pixel_sha256",
    "width",
    "height",
]


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


#^ spilling into train, test, validation
def assign_splits(rows, seed=42):
    groups = defaultdict(list)

    #* group images by pixel hatch
    for row in rows:
        groups[row["pixel_sha256"]].append(row)


    representatives = []
    for group in groups.values():
        labels = {row["label"] for row in group}
        conflict = len(labels) > 1
        uncertain = any(
            not row["label"].strip() or "?" in row["label"] for row in group
        )
        if conflict or uncertain:
            for row in group:
                row.update(
                    split="quarantine",
                    reason="conflicting_labels" if conflict else "uncertain_label",
                )
            continue
        # Preserve publisher test membership if a future archive has cross-split duplicates.

        #& Select one representative and mark duplicates
        ordered = sorted(
            group, key=lambda r: (r["source_split"] != "test", r["filename"])
        )
        representative = ordered[0]
        representatives.append(representative)
        for row in ordered[1:]:
            row.update(
                split="duplicate",
                reason="exact_pixel_duplicate_of:" + representative["filename"],
            )

    #^ assigning traiing data and vlidation data from training data
    training = sorted(
        [r for r in representatives if r["source_split"] == "train"],
        key=lambda r: r["filename"],
    )
    random.Random(seed).shuffle(training)
    val_count = max(1, round(len(training) * 0.1)) if len(training) > 1 else 0
    for index, row in enumerate(training):
        row.update(split="validation" if index < val_count else "train", reason="")
    for row in representatives:
        if row["source_split"] == "test":
            row.update(split="test", reason="")
    return groups


def prepare(archive_path, destination, metadata_path):
    emit("Checking archive checksum and ZIP integrity")
    #* Verifies archive integrity(start)
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")

    if hashlib.md5(archive_path.read_bytes()).hexdigest() != EXPECTED_MD5:
        raise ValueError("Archive checksum does not match publisher")
    rows = []
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC verification failed")
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP members")
        #* Verifies archive integrity(end)



        #^ reading the csv in ZIPs validating file name
        for split, csv_name, folder in SOURCE_SPLITS:
            reader = csv.DictReader(
                io.StringIO(archive.read(csv_name).decode("utf-8-sig"), newline=None)
            )
            for item in progress_rows(list(reader), f"Reading {split} source images"):
                name = item["Images"]
                if (
                    Path(name).name != name
                    or "/" in name
                    or "\\" in name
                    or ":" in name
                    or not name.endswith(".jpg")
                ):
                    raise ValueError("Unsafe image filename")
                content = archive.read(folder + "/" + name)
                with Image.open(io.BytesIO(content)) as image:
                    image.load()
                    width, height = image.size
                    pixels = hashlib.sha256(
                        f"{width}x{height}:RGB:".encode()
                        + image.convert("RGB").tobytes()
                    ).hexdigest()
                rows.append(
                    dict(
                        filename=name,
                        label=item["Text"],
                        source_split=split,
                        sha256=hashlib.sha256(content).hexdigest(),
                        pixel_sha256=pixels,
                        width=width,
                        height=height,
                    )
                )

        #! Verify total images and filenames
        if len(rows) != 5578 or len({r["filename"] for r in rows}) != 5578:
            raise ValueError("Unexpected image count or nonunique image names")


        labelled_names = {
            folder + "/" + r["filename"]
            for split, _, folder in SOURCE_SPLITS
            for r in rows
            if r["source_split"] == split
        }
        if labelled_names != {n for n in names if n.lower().endswith(".jpg")}:
            raise ValueError("Unlabelled or missing images")

        #& Extract and save images
        groups = assign_splits(rows)
        (destination / "images").mkdir(parents=True)
        (destination / "source").mkdir()
        for row in progress_rows(rows, "Extracting images"):
            folder = "Train_Set" if row["source_split"] == "train" else "Test_Set"
            (destination / "images" / row["filename"]).write_bytes(
                archive.read(folder + "/" + row["filename"])
            )

        #& Save CSV files and dataset metadata
        for _, csv_name, _ in SOURCE_SPLITS:
            (destination / "source" / csv_name).write_bytes(archive.read(csv_name))
    write_csv(destination / "manifest.csv", rows, FIELDS)
    for split in ["train", "validation", "test", "quarantine"]:
        write_csv(
            destination / f"{split}.csv",
            [r for r in rows if r["split"] == split],
            ["filename", "label"],
        )


    shutil.copyfile(metadata_path, destination / "source/publisher_metadata.json")
    source = {
        "name": "RxHandBD",
        "url": SOURCE_URL,
        "doi": "10.5281/zenodo.18478741",
        "publication_date": "2026-02-04",
        "publisher_license_id": "mit-license",
        "author": "Md. Masudul Islam",
        "download_url": "https://zenodo.org/api/records/18478741/files/RxHandBD.zip/content",
        "archive_md5": EXPECTED_MD5,
        "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "preparation_seed": 42,
        "label_semantics": "verbatim word transcription; NOT validated medicine or generic identity",
        "generic_name_annotations": False,
        "writer_and_document_ids": False,
    }
    (destination / "SOURCE.json").write_text(
        json.dumps(source, indent=2) + "\n", encoding="utf-8"
    )
    report = verify(destination)
    report["source_pixel_duplicate_groups"] = sum(len(g) > 1 for g in groups.values())
    report["source_cross_split_duplicate_groups"] = sum(
        len({r["source_split"] for r in g}) > 1 for g in groups.values()
    )
    report["source_conflicting_label_groups"] = sum(
        len({r["label"] for r in g}) > 1 for g in groups.values()
    )
    (destination / "audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def verify(root):
    root = Path(root)
    with (root / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(
        {r["filename"] for r in rows}
    ), "Duplicate manifest filenames"
    seen = {}
    counts = Counter()
    for row in progress_rows(rows, "Verifying image hashes"):
        image_path = (root / "images" / row["filename"]).resolve()
        assert image_path.is_relative_to(
            (root / "images").resolve()
        ), "Unsafe manifest path"
        data = image_path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == row["sha256"], row["filename"]
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            w, h = image.size
            assert (w, h) == (int(row["width"]), int(row["height"]))
            assert (
                hashlib.sha256(
                    f"{w}x{h}:RGB:".encode() + image.convert("RGB").tobytes()
                ).hexdigest()
                == row["pixel_sha256"]
            )
        counts[row["split"]] += 1
        if row["split"] in ["train", "validation", "test"]:
            assert (
                row["label"].strip() and "?" not in row["label"]
            ), "Uncertain active label"
            assert row["pixel_sha256"] not in seen, "Exact duplicate in active data"
            seen[row["pixel_sha256"]] = row["split"]
    assert {p.name for p in (root / "images").iterdir()} == {
        r["filename"] for r in rows
    }, "Unreferenced image"
    for split in ["train", "validation", "test", "quarantine"]:
        with (root / f"{split}.csv").open(encoding="utf-8", newline="") as stream:
            actual = list(csv.DictReader(stream))
        expected = [
            {"filename": r["filename"], "label": r["label"]}
            for r in rows
            if r["split"] == split
        ]
        assert actual == expected, f"{split} manifest mismatch"
    source_rows = {}
    for split, csv_name, _ in SOURCE_SPLITS:
        with (root / "source" / csv_name).open(
            encoding="utf-8-sig", newline=""
        ) as stream:
            for row in csv.DictReader(stream):
                source_rows[row["Images"]] = (row["Text"], split)
    assert source_rows == {
        r["filename"]: (r["label"], r["source_split"]) for r in rows
    }, "Source label mismatch"
    train_labels = {r["label"] for r in rows if r["split"] == "train"}
    train_chars = set("".join(train_labels))
    report = {
        "images": len(rows),
        "source_labels_case_sensitive": len({r["label"] for r in rows}),
        "source_labels_casefolded": len({r["label"].casefold() for r in rows}),
        "prepared_splits": dict(counts),
        "quarantine_reasons": dict(
            Counter(r["reason"] for r in rows if r["split"] == "quarantine")
        ),
        "active_pixel_duplicates": 0,
        "all_images_decoded_and_hash_verified": True,
        "all_source_transcriptions_preserved": True,
    }
    for split in ["validation", "test"]:
        labels = {r["label"] for r in rows if r["split"] == split}
        report[split + "_unseen_label_count"] = len(labels - train_labels)
        report[split + "_unseen_characters"] = sorted(
            set("".join(labels)) - train_chars
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", type=Path, default=PROJECT / "Dataset/RxHandBD")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        report = verify(args.output)
    else:
        if not args.archive or not args.metadata:
            parser.error("--archive and --metadata are required for preparation")
        report = prepare(args.archive, args.output, args.metadata)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
