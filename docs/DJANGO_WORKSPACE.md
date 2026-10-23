# Django prescription workspace

The first full-page backend milestone uses Django templates, HTML, one CSS file and a small JavaScript file. No React, TypeScript, Node build step, database or account is required. The existing Gradio interface is still available.

## Run locally

Use the project environment, from the project root:

```powershell
python -m pip install -r requirements-web.txt
python -m webapp runserver 127.0.0.1:8000 --noreload --insecure
```

Open http://127.0.0.1:8000. `--insecure` here enables Django's development static-file server with DEBUG disabled; it is not a public deployment command. Keep this server bound to loopback. `--noreload` keeps one model/worker instance. Model selection uses configs/app.json when present; otherwise the public pretrained baseline is used. CUDA is used if available; CPU reading can take minutes for a page. Installed fine-tuned weights are not included in Git.

## Use it

1. Upload a clear JPG, PNG, WebP or single-page PDF, maximum 10 MB/16 megapixels. Images receive EXIF orientation correction; perspective/blur repair is not implemented yet.
2. Automatic mode detects text regions and reads them in a background worker, with real completed-region progress. Role proposals use conservative text rules, not a trained medicine-region classifier. General text detection does not guarantee every medicine is found.
3. Check the original and every crop. Correct the text; assign Medicine, Instruction or Other. Bare names without a medication prefix intentionally remain Unreviewed. Save new Medicine roles once so they appear in the instruction link menu.
4. Explicitly link each separate instruction to its medicine. Same-row suggestions are not automatically applied. Check the review boxes and save. Adding/changing text, re-reading or removing a linked medicine invalidates the relevant review.
5. A missed region can be drawn on the image, or entered using normalized coordinates. Read that crop again or transcribe it manually. Save pending edits before adding, removing or re-reading a region.
6. The medicine list separates strength, dose quantity, raw schedule, explicit times/day and duration. Ambiguous, conditional or changing schedules do not yield a fixed daily count. Review status is not a clinical approval or confidence percentage.
7. Confirm spelling to use the existing RxNorm/DailyMed lookup. Ingredient-only matches cannot select product labels; refine to a product/strength/form and verify its label. These US sources do not cover all Indian brands. A verified Indian catalogue is still outstanding; the app does not invent missing mappings.
8. Download the review JSON if needed. It includes all region text, including anything marked Other; de-identify before sharing. Delete the job when finished.

## Data lifetime and limitations

Images and OCR/review records live in process memory, scoped to a signed browser session. Streaming uploads are bounded and do not spool prescription files to disk. Jobs expire after 30 minutes; a background reaper clears them within about 30 seconds of expiry, and expired jobs are immediately inaccessible on requests. Restart clears everything. Exported JSON is the user's explicit download. Original filenames are not retained. External lookup sends medicine queries and label IDs, not the page.

There is one daemon inference worker, a bounded queue and at most six active jobs. Cancel/delete discards the job immediately, but cannot interrupt a model's currently executing native call; processing stops between calls. This is a local prototype, not a multi-worker or publicly hardened service. Browser close does not immediately delete the server job; expiry/delete/restart does. Camera capture, public hosting, confidence calibration, automatic perspective correction and a representative Indian page benchmark remain future work.

## Code map

- webapp/settings.py: local Django settings and session rules.
- webapp/views.py: upload, review, manual regions, export and confirmed lookup.
- webapp/jobs.py: cached OCR model and bounded background worker.
- webapp/templates/: ordinary Django HTML templates.
- webapp/static/: basic CSS and progress/drawing JavaScript.
- src/prescription/documents.py: bounded image/PDF decoding.
- src/prescription/page_review.py: role suggestions and explicit medicine/instruction relationships.
- src/prescription/extraction.py: literal fields; no inferred treatment.
- scripts/prepare_page_crops.py: reviewed page annotations to separate OCR crop data.

## Resume an annotation review

Upload the original image and the matching review JSON in the optional saved-review field. It accepts version 1 exports from this workspace and the prepared public-page drafts. A region template or dataset manifest is not a page review. The file must be at most 512 KB, match the canonical image hash and dimensions, and contain at most 80 valid regions. Invalid imports create no job.

Automatic reading with a saved review fills only empty, unchecked regions. Corrections, assigned roles, explicit links and existing review checks are retained. Each completed draft is kept if a later region fails; use **Read empty regions** to retry. Save pending form edits before triggering this action. A fully populated saved review opens immediately without loading the model. Imported status and medicine identity fields are ignored; medicine records are recomputed from the regions.

Download review JSON before the 30-minute session expires. Reopen it with the same image to resume later. Uploads remain in memory; JSON is stored only when you download/export it. Redaction changes the image hash, so a changed page needs a new matching review. No imported or automatic reading is considered independent clinical verification.

## Prepare new training data

Keep the current RxHandBD data and checkpoint. Collect representative, permitted, de-identified Indian pages with independent review of uncertain handwriting. Review every region in the app, including non-medicine regions, and export each page's JSON. Maintain the exact uploaded image beside its export. Copy examples/page_manifest.json into that directory and list every image/review pair, its writer/clinic/document group and an explicit train/validation/test split. Keep every related page and augmentation in the same group. Confirm permission/de-identification only after checking them.

```powershell
python -m scripts.prepare_page_crops --manifest Dataset/IndianPages/manifest.json --output Dataset/IndianPrescriptionCrops
python -m scripts.train --config configs/page_finetune.example.json
```

The first command writes to a new directory only. It checks page hashes, all reviewed roles, explicit links, ambiguous labels and conflicting/duplicate crop pixels. It excludes Other regions from crop training and retains role/page/group evidence. The second command is **preflight only** and requires separate nonempty train/validation groups. It verifies frozen crop hashes and labels. The example configuration starts from the installed local best checkpoint and selects a new output directory; it is a starting configuration, not a tuned recipe. Page annotation and crop transcription are not a full-page detection training recipe or a substitute for held-out page evaluation.

Actual training must be run on CUDA after the data and splits are ready, using `--execute`. No new training was run for this milestone. The existing Colab notebook still trains RxHandBD; it is not silently switched to IndianPages. A new page-data Colab package can be made once those data exist. Keep the final test split untouched during model selection.

## Checks

```powershell
python -m webapp check
python -m unittest webapp.tests -v
python -m unittest discover -s tests -v
```

Website tests do not download models. They test decoding, CSRF, ownership, expiry, deletion, escaping, linking, lookup gating, worker results and training-data integrity. Browser checks use clearly labeled synthetic pages for workflow testing; they do not establish real-prescription accuracy.
