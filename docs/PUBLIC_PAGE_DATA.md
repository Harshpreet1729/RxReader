# Supplementary full-page prescription data

## Selected source

[A Curated Bangladesh-Based Dataset of Handwritten and Printed Prescription Images, version 2](https://data.mendeley.com/datasets/k62rfd23kz/2), Dipto saha, Sourav Debnath and Anirban Roy. Published 27 March 2026. DOI: 10.17632/k62rfd23kz.2. Licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Retain attribution and describe modifications when sharing derivatives.

This is supplementary Bangladesh data, not an Indian prescription benchmark. The archive contains 200 pages with 1,086 medicine-region boxes. Handwritten and printed pages have English, Bangla and mixed content. There are no medicine transcriptions, schedule links, writer identities or clinic group IDs supplied for training. Boxes sometimes include strengths or treatment text; their semantic consistency still needs review.

The publisher claims de-identification. Visual sampling found remaining signatures/stamps, so **independent privacy review is incomplete**. Original files and derived pages stay in the ignored local Dataset directory; do not publish these pages or use them for training until reviewed and, where needed, redacted. Review crops too. Do not treat publisher boxes as proof of correct medicine spelling.

## Local files

- `Dataset/PrescriptionPagesBD/source/Dataset.zip`: unchanged publisher archive; SHA256 `ced4394523998386b274e549469c144fa9718402b4c52764552914182bb331db`.
- `Dataset/PrescriptionPagesBD/source/publisher_files.json`: downloaded publisher file metadata.
- `Dataset/PrescriptionPagesBD/prepared/pages/`: canonical PNG pages.
- `Dataset/PrescriptionPagesBD/prepared/crops/`: 1,086 medicine-region crops; no transcription labels claimed.
- `Dataset/PrescriptionPagesBD/prepared/labels/`: derived YOLO labels with medicine class 0. Source labels use class 1 while classes.txt lists only medicine. Source labels are preserved inside the ZIP.
- `Dataset/PrescriptionPagesBD/prepared/regions/`: pixel box templates compatible with the existing inference pipeline / Gradio saved-regions input.
- `Dataset/PrescriptionPagesBD/prepared/reviews/`: draft page annotations with empty text and `reviewed: false`.
- `Dataset/PrescriptionPagesBD/prepared/manifest.draft.json`: deliberately has `deidentified: false`, unassigned splits and empty groups. It cannot pass training preparation.
- `Dataset/PrescriptionPagesBD/prepared/audit.json`: machine-readable checks and readiness flags.
- `data/prescription_pages_bd/SOURCE.json`: pinned provenance, license and limitations without prescription content.

## Reproduce preparation

Download **Dataset.zip** from the source page into the source directory above. Then, from the project root:

```powershell
python -m scripts.prepare_public_pages
```

The script checks the pinned archive checksum, image/label pairing, CSV box counts, dimensions, EXIF orientation, box bounds and exact pixel duplicates. It refuses an existing output directory. To independently reproduce, choose a new directory using `--output`. It never trains or modifies RxHandBD or existing checkpoints. It writes no training CSV or training YAML because review and split assignment are unfinished.

Derived coordinates are converted to integer pixel boxes and normalized again for YOLO; minor rounding is documented. Preparation does not measure model accuracy. Duplicate checks cover exact pixels only, not near-duplicates or shared writers.

## Next annotation steps

1. Review/redact each page and inspect box coverage, language, legibility and any repeated pages. Keep a separate redacted working copy. Redaction changes page hashes, so export new annotations against that exact image.
2. Begin with a small English-medicine subset, including printed and handwritten pages. Upload the page to the Django workspace with its matching `prepared/reviews/<page>.json` in the optional saved-review field. Imported boxes appear immediately when automatic reading is off. With automatic reading on, only empty regions are read; existing text, roles, links and review checks are preserved. Use `Read empty regions` to retry unfinished crops. A saved review must match the exact image hash; after redaction, make a new review against the changed image rather than editing its hash. The existing Gradio saved-regions input accepts the prepared region template when image dimensions are unchanged.
3. Draft files in `Dataset/PrescriptionPagesBD/ocr_drafts/`, when present, are model predictions only. Upload one with the same numbered image in `prepared/pages/` to continue correcting it. Independently check medicine spelling, strengths, schedule text and medicine/schedule links. Mark unreadable text for adjudication rather than guessing. Current TrOCR output may assist transcription but cannot be used as ground truth without review.
4. Export the reviewed page JSON. Populate a reviewed manifest with permission, completed de-identification, groups and explicit splits. Keep related pages, near-duplicates and augmentations together. Without writer metadata, document-level splitting cannot establish writer-independent performance.
5. Use `scripts.prepare_page_crops` only after review, then training preflight. Build a separate detector experiment after full-page privacy/box review. Keep the original RxHandBD baseline and an untouched final test split.

Representative Indian pages are still needed to establish performance for the intended users. More Bangladesh pages alone do not establish Indian brand coverage or accurate dosage extraction.

## First annotation batch

The first local pilot is `Dataset/PrescriptionPagesBD/annotation_batch_001/`. It contains 20 pages (source IDs 1–22 excluding 4 and 15) and 86 medicine regions. This is a convenience sample for annotation, not a train/test split or representative benchmark. Source page 4 appears to label examination text; page 15 labels procedures. They are excluded from this batch only; originals remain unchanged.

`data/prescription_pages_bd/batch_001.json` records the selected source hashes, retained medicine-block rectangles, excluded-page reasons and visual-screening notes. `scripts.prepare_annotation_batch` creates full-size PNG working copies with white pixels outside each selected block. The crop pixels inside the medicine annotations are preserved. Retained blocks were visually screened for obvious direct identifiers, but the manifest deliberately leaves independent privacy review incomplete. Some schedules are Bangla or clipped at the block boundary. Pages 12, 17 and 20 need adjudication of crossed-out/edited text; page 20 has an unboxed handwritten replacement.

```powershell
python -m scripts.prepare_annotation_batch --read
```

This runs inference on CPU using the installed local model, not training. Progress is printed per region and every OCR result is saved atomically. Re-running with the same plan validates the batch and skips existing text, reviewed regions and already attempted empty readings. It does not overwrite corrections. Use the app's individual crop reread for an attempted empty reading. If interrupted, run the same command again. Do not run concurrent writers against one batch. Human corrections should be exported separately from the generated drafts.

Open `START_HERE.md` in the batch and follow `review_queue.csv`. Upload a page from `pages/` with the same-numbered JSON from `reviews/` into the Django app. Turn automatic reading off to inspect saved drafts. Review notes are in the queue/JSON and must be checked separately; they are not imported into the app's displayed region fields. Every generated reading starts unreviewed. `progress.json` describes inference completion, not annotation completion or accuracy.

No schedule transcription labels, verified drug identities or dataset splits are generated by this step. Masked pages are not an original full-page detection benchmark. Before training, finish medicine/strength review, add and link any eligible schedule regions, adjudicate crossed-out or unreadable entries, complete privacy checks, and group related pages before splitting.

## Other candidates checked

- [MedOCR Vision](https://huggingface.co/datasets/naazimsnh02/medocr-vision-dataset): publisher tags it MIT, but its viewer mixes prescriptions, restaurant receipts, invoices and laboratory reports. Some displayed labels contain patient identifiers. Not imported wholesale or accepted as reviewed Indian prescription ground truth.
- [Automated Digitization of Unstructured Medical Prescriptions](https://aclanthology.org/2023.acl-industry.76/): describes 1,359 Indian prescriptions, but explicitly uses proprietary data; no public download was verified.
- [Synthetic medical prescriptions](https://huggingface.co/datasets/chinmays18/medical-prescription-dataset): potentially useful for interface/layout experiments, not evidence of real handwritten prescription recognition.

The selected archive adds actual page layouts and box annotations. It does not replace the word-crop dataset or justify starting another recognition training run yet.
