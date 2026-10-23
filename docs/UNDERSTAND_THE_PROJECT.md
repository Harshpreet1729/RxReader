# Understand RxReader

This guide follows the actual code. Start with the recognition flow, then training,
then the website. You do not need to memorize every line of a library, but you
should be able to explain every part you use, its inputs, outputs and limitations.

## The project in 30 seconds

"I fine-tuned a pretrained TrOCR handwriting model on prescription word crops.
The app can read a crop or detect text regions on a page. Python rules extract
possible medicine names and written instructions. Users review the text and
explicitly link separate instruction regions. After spelling confirmation, the
app retrieves medicine information from RxNorm and DailyMed. I evaluated OCR
using character error rate and exact match. Full-page Indian prescription
accuracy is still future work."

## 1. Follow one image

```text
Image
  -> layout.py: find boxes around text (or use a user-drawn box)
  -> ocr.py: convert each crop into text with TrOCR
  -> extraction.py: find literal fields in that text
  -> human review: correct spelling and link instructions
  -> knowledge.py: retrieve information for a confirmed name
```

`pipeline.py:process_image` is the short version used by the command line and
Gradio demo. Read that function first. It loops over regions, crops the image,
calls the reader and attaches extracted fields to each result.

Django adds an editable review workflow: `webapp/views.py` accepts the upload,
`webapp/jobs.py` runs detection/OCR in the background, and `page_review.py`
groups only the instructions explicitly linked to a medicine. Django does not
call `process_image` because it must keep individual regions editable.

| Part | What it receives | What it returns | Is it learned? |
|---|---|---|---|
| Text detector | Whole page | Pixel boxes | Pretrained Paddle detector; not trained here |
| TrOCR | One image crop | Text and alternative readings | Yes; pretrained then fine-tuned |
| Field extraction | Text | Name candidate, strength, schedule | No; regular expressions |
| Region role proposal | Text and boxes | Suggested medicine/instruction roles | No; rules |
| Medicine lookup | Confirmed spelling | Source matches and label text | No; HTTP requests |

## 2. Understand the OCR code

Read `src/prescription/ocr.py`, especially `TrOCRReader.read`.

1. `TrOCRProcessor` prepares the image and handles text tokenization. A tensor
   is a multidimensional array; the processed image includes a batch dimension.
2. The encoder converts image patches into features. The decoder uses these
   features and previous tokens to predict the next token. It generates a text
   sequence; it does not choose from a fixed list of medicine classes.
3. `.to(device)` puts data/model on the CPU or GPU. `.eval()` switches the model
   to evaluation behavior, while `inference_mode()` avoids gradient tracking.
4. `generate()` produces token sequences. Three beams retain several candidate
   sequences during decoding; they are not three separately trained models.
5. `batch_decode()` converts token IDs to readable text. Duplicate alternatives
   are removed. If no end token appears, the result is marked possibly truncated.

Beam scores rank generated sequences. They are **not accuracy percentages** and
must not be presented as a probability that a medicine identity is correct.

Imports inside model-loading functions delay expensive dependencies. The cached
reader keeps the loaded model in memory so each button click does not reload it.

## 3. Understand the rules

Read `src/prescription/extraction.py:extract_line`. It has four stages: find
explicit frequencies, find other written fields, isolate the possible name,
and decide whether an unambiguous daily count is available.

For the illustrative text `Tab Example 500 mg twice daily for 3 days`, the rules
produce a name candidate `Example`, strength `500 mg`, daily count `2`, and
duration `for 3 days`. This is a parsing example, not a treatment instruction.

The regex notation is small: `\b` means a word boundary, `\d` a digit, `\s`
whitespace, `+` one or more, `?` optional, and `|` alternatives. `re.I` ignores
case. `match.group()` returns the original matched text; `match.start()` gives
its position so the name can be cut off before the strength/instruction.

Abbreviations such as `BD`, interval instructions and changing/conditional
schedules remain verbatim with no fixed daily count. These are intentional
limits. A rule cannot determine whether OCR read an abbreviation correctly.

`page_review.py` retains the original OCR text alongside the corrected text.
An instruction on a separate line is not automatically attached to the nearest
medicine. A reviewed transcription also does not prove clinical identity.

## 4. Understand training

Read these in order:

1. `configs/colab_base.json`: the settings used by the base-model notebook.
2. `src/prescription/data.py:OCRDataset`: one training example.
3. `scripts/train.py:train`: the five commented training stages.
4. `src/prescription/metrics.py`: how predictions are scored.

The dataset returns an image tensor and expected text tokens. All target
sequences are padded to a configured length so they can be batched. Padding
positions become `-100`, which the model's loss ignores. Overlong labels are
rejected instead of silently cutting away the answer.

The Hugging Face `Seq2SeqTrainer` implements batching, forward passes, loss,
backpropagation and optimizer updates. The model learns token prediction using
the supplied labels. During evaluation, generation predicts text without being
given the correct answer. This is fine-tuning existing weights, not implementing
or training a Transformer from scratch.

| Setting | Meaning in this project |
|---|---|
| Learning rate `2e-5` | Size of optimizer updates |
| Batch size `1` | One example per forward/backward pass |
| Gradient accumulation `16` | Accumulate gradients over 16 batches before a normal optimizer step; effective batch 16 on one GPU |
| Epochs `12` | Maximum number of passes through training data |
| Evaluation/save every `100` steps | Check validation performance and save progress |
| Early stopping patience `5` | Stop after enough validation checks without improvement, according to the Trainer callback |
| FP16 | Use mixed precision to reduce GPU memory needs |
| Gradient checkpointing | Recompute some activations during backward to save memory |
| Seed `42` | Make random choices more reproducible; not a guarantee across all hardware |

The best checkpoint is selected by **lowest validation CER**, not training loss.
The test set is for final reporting, not choosing settings. Resume checks ensure
the configuration and split files still match the interrupted run. The model
and processor/tokenizer are saved together because inference needs both.

Running `python -m scripts.train --config configs/colab_base.json` only audits
the data. `--execute` starts training and requires CUDA. Do not run it merely to
prepare for an interview: the trained checkpoint already exists locally.

## 5. Explain the results correctly

CER = total character edit operations / total reference characters.
WER uses words instead. Exact match is the fraction of completely correct
transcriptions. A single wrong character fails exact match, even if CER is low.

`distance()` computes edit distance with insertions, deletions and replacements.
It builds one row of a dynamic-programming table at a time. The totals are
computed over the corpus, so short words do not receive the same weight as long
words in CER. Normalization ignores case and normalizes whitespace/Unicode.

| Recorded run | Images | Normalized CER | Normalized exact match |
|---|---:|---:|---:|
| Pretrained small, validation | 444 | 59.54% | 9.91% |
| Pretrained base, validation | 444 | 54.39% | 12.16% |
| Fine-tuned base, validation | 444 | 16.57% | 48.87% |
| Fine-tuned base, test | 1,111 | 28.78% | 41.22% |

These are saved results in `docs/results.json`, not a new evaluation from this
refactor. Do not call `100 - CER` whole-prescription accuracy. Do not claim the
base architecture is better than a fine-tuned small model: that comparison was
not run. We know test performance is worse than validation; we have not proved
one specific cause. Split difficulty and generalization need investigation.

RxHandBD has 5,578 original word crops. Active splits contain 3,994 train,
444 validation and 1,111 test examples. Twelve uncertain/conflicting examples
are quarantined and 17 redundant copies excluded. Originals remain intact.
Writer IDs are unavailable, so writer-independent evaluation is not established.

## 6. Understand the website without learning React

| File | Responsibility |
|---|---|
| `webapp/urls.py` | URL to Python view mapping |
| `webapp/forms.py` | Upload fields and basic validation |
| `webapp/views.py` | Receive requests, validate edits, return HTML or JSON |
| `webapp/jobs.py` | Temporary uploads, one OCR worker, progress and expiry |
| `webapp/templates/` | Django HTML pages |
| `webapp/static/reader.js` | Draw boxes and request progress updates |
| `webapp/static/reader.css` | Appearance |

A `Job` dataclass holds one uploaded page and its readings. The `JobStore` keeps
jobs in a dictionary, not a database. A queue lets a worker perform slow OCR
while the browser asks for progress. A lock prevents simultaneous edits by the
request and worker threads. A second thread removes expired uploads. One worker
avoids simultaneous model calls and unbounded memory use.

Jobs belong to a browser session. Uploaded pages are temporary in-memory data
and expire after 30 minutes; restarting the server clears them. Export JSON to
keep a review. Reopening a review checks its image hash and dimensions so text
cannot accidentally be attached to another page. This is a local prototype,
not a multi-user production deployment.

Gradio is an optional, smaller demo. In `app/gradio_app.py`, `transcribe` calls
the backend, `make_theme` sets colors, `build_ui` creates components and connects
callbacks, and `main` starts the local server. The lists passed to `.click()`
map component inputs and returned outputs in order. Changing an image/name
clears old results to avoid showing information for a previous input.

## 7. What is optional research code?

The ordinary demo does not use the `prepare_public_pages`,
`prepare_annotation_batch` or `prepare_page_crops` scripts. They support future
training on whole-page material. The local batch has 20 working pages and 86
model-generated draft readings, not verified ground truth. It has not produced
a newly trained model. See `PUBLIC_PAGE_DATA.md` before using it.

`build_colab.py` embeds the training source into a portable notebook.
`notebook_runner.py`, `progress.py` and `training_progress.py` display real
progress and handle subprocess interruption. They are support code, not part of
the recognition algorithm. Older classifier/CRNN/ViT experiments and their
checkpoints have been removed; the active model is the fine-tuned TrOCR checkpoint.

## 8. Interview practice

- **Why TrOCR?** It provides pretrained image-to-text handwriting recognition,
  which can be adapted with a relatively small labeled dataset.
- **Why not a medicine classifier?** Text generation can output names that were
  not fixed classes in training; correct recognition is still not guaranteed.
- **Where are uses generated?** They are retrieved from a selected source label,
  not generated by the OCR model. Indian brand coverage is incomplete.
- **What did you implement?** Explain the data checks, fine-tuning configuration,
  evaluation, extraction rules and app integration. Credit the pretrained model
  and libraries. Be open about development assistance if asked.
- **What would you improve?** Representative Indian page data, verified text and
  links, grouped evaluation, detection errors and source coverage. Measure each
  change before claiming better results.
- **Why keep checks and tests?** They protect dataset integrity, reviewed edits,
  checkpoint resume and the association between names and instructions.

For a demo, start Django, upload a local image, review/correct a crop, and explain
the raw text versus parsed fields. Show the recorded test metrics and one known
failure too. Avoid presenting a training example as evidence of generalization.

## Suggested reading order

First session: `pipeline.py`, `ocr.py`, `extraction.py`, `knowledge.py`.
Second: `data.py`, `configs/colab_base.json`, `train.py`, `metrics.py`.
Third: trace one upload through `urls.py`, `views.py`, `jobs.py` and the template.
Then read `tests/test_pipeline.py` and `webapp/tests.py` to see failure examples.

Useful checks from the project root:

```powershell
python -m scripts.audit_dataset
python -m scripts.train --config configs/colab_base.json
python -m unittest discover -s tests -v
python -m unittest webapp.tests -v
```
