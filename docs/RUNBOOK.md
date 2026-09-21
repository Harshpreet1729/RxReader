# Runbook

Use Python 3.11 and run Python entry points from the repository root with `python -m`. Install requirements.txt; optional page detection uses requirements-page.txt.

## Local interface

`python -m app.gradio_app` starts http://127.0.0.1:7860 with public sharing disabled. A fresh clone uses pretrained TrOCR-small. A local configs/app.json overrides that default; copy app.example.json only after installing the trained best directory. Explicit `--model path/to/best` takes precedence. Missing configured weights produce an error rather than silently switching models. CPU inference works; first loading is slower than subsequent requests.

Upload a word crop, review its transcription/alternatives, then enter spelling checked against the image to look up ingredients and labels. Select the matching formulation to retrieve published uses. Images stay local for a local app; entered medicine names/label IDs go to NLM. Unknown fields display Not found. Raw output remains in expandable details.

## Data

```bash
python -m scripts.download_dataset
python -m scripts.audit_dataset
python -m scripts.check_setup
```

Existing datasets are verified rather than overwritten. Public archive and split checksums are under data/rxhandbd. Dataset labels are preserved; OCR targets normalize whitespace while retaining case. Label padding is masked with -100 and overlength targets raise errors.

## Inference and evaluation

```bash
python -m scripts.infer --image path/to/crop.jpg --model checkpoints/trocr_base_rxhandbd/best
python -m scripts.infer --image path/to/page.jpg --mode page --regions examples/regions.json --model checkpoints/trocr_base_rxhandbd/best
python -m scripts.evaluate --model checkpoints/trocr_base_rxhandbd/best --split validation
python -m scripts.lookup_medicine --confirmed-name "acetaminophen 500 MG Oral Tablet"
```

The example region file must be adapted to the actual image. It is not a universal prescription template. Omitting --regions in page mode uses optional PP-OCRv5 detection; CPU MKLDNN is disabled due to an observed Windows Paddle failure. Page detection and separate-line schedule association are not validated by the cropped-word dataset.

Use --limit 2 only for smoke checks. Full metrics must use the complete split. Preserve the held-out test result after model selection; do not repeatedly tune against it.

## Training

```bash
python -m scripts.train --config configs/colab_base.json
python -m scripts.train --config configs/colab_base.json --execute
```

The first command only verifies data. Actual training requires --execute and CUDA. Output directories must be new or explicitly resumed with --resume path/to/checkpoint-N. Resume verifies configuration and data hashes. The selected best model and processor are saved together. See COLAB_TRAINING.md for notebook persistence and progress.

## Tests and frontend

`python -m unittest discover -s tests -v` runs unit/integration regression fixtures without downloading the dataset or starting project-model training. Gradio styles and escaped result presentation live in app/; recognition, retrieval and metrics stay in src/prescription for a future React/API layer. No React app currently exists.
