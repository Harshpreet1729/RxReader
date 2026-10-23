# RxReader - Prescription OCR

A Python ML project that reads handwritten medicine text, lets you review it,
and retrieves medicine information from named sources. The main website uses
**Django, HTML, CSS and a small amount of JavaScript**. React is not required.

**Start learning here: [Understand the project](docs/UNDERSTAND_THE_PROJECT.md).**
It explains the code flow, training, results and interview questions.

## How it works

```text
Upload image -> Find text regions -> TrOCR -> Extract written fields
             -> Review/correct text -> Confirm medicine -> Source lookup
```

- Read a word crop or a full page with automatic/manual regions.
- Correct readings and link separate medicine/instruction regions.
- Extract written strengths and explicit daily frequencies; leave unclear values unknown.
- Look up a confirmed name through RxNorm and a selected DailyMed product label.
- Export/reopen review JSON, with background OCR progress in Django.
- Fine-tune and evaluate TrOCR using the included Colab notebook.

The OCR model generates text. Field extraction is Python rules; medicine uses
come from source labels. This is a local prototype, not a clinical decision tool.
Indian brand coverage and full-page accuracy are not established.

## Run the website

Use Python 3.11 for a new environment. Run commands from the project root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-web.txt
python -m webapp runserver 127.0.0.1:8000 --noreload --insecure
```

Open [the local website](http://127.0.0.1:8000/). In an existing configured
`.venv`, activate it and run the last command. This command is for local use.

The fine-tuned weights are not included in Git. Put the complete exported
`best` folder in `checkpoints/trocr_base_rxhandbd/best`, then copy
`configs/app.example.json` to `configs/app.json`. A fresh clone without this
local configuration uses pretrained TrOCR-small, which does **not** reproduce
the fine-tuned results below. Keep tokenizer/processor files with the weights.

The optional Gradio demo is smaller and useful for exploring the basic flow:

```powershell
python -m app.gradio_app
```

Open [Gradio](http://127.0.0.1:7860/). Both interfaces use the same OCR and
extraction modules. Images stay local; confirmed names and label IDs go to NLM
for lookup. Django uploads expire after 30 minutes or on server restart.

## Results already measured

| Model | Split | Images | Normalized CER | Exact match |
|---|---|---:|---:|---:|
| Pretrained TrOCR-small | Validation | 444 | 59.54% | 9.91% |
| Pretrained TrOCR-base | Validation | 444 | 54.39% | 12.16% |
| Fine-tuned TrOCR-base | Validation | 444 | **16.57%** | **48.87%** |
| Fine-tuned TrOCR-base | Test | 1,111 | **28.78%** | **41.22%** |

CER measures character edits; lower is better. Exact match requires the whole
transcription to match. Normalization ignores case and normalizes whitespace
and Unicode. These are word-crop results, not whole-prescription accuracy.
There is no comparison against a separately fine-tuned small model.
See [model limitations](docs/MODEL_CARD.md) and [saved metrics](docs/results.json).

## Code map

| Folder | Purpose |
|---|---|
| `src/prescription/` | OCR, detection, extraction, lookup, data and metrics |
| `webapp/` | Main Django website and temporary background jobs |
| `app/` | Optional Gradio demo and presentation |
| `scripts/` | Inference, data checks, training and evaluation commands |
| `configs/` | Model paths and training settings |
| `notebooks/` | One self-contained Colab training notebook |
| `tests/` | Backend regression tests; Django tests are in `webapp/tests.py` |
| `data/` | Dataset provenance and reproducible split checksums |
| `docs/` | Learning guide, setup details and results |

Local `Dataset/`, `checkpoints/` and `test_images/` contain the datasets, current
trained model and demo images. They are excluded from Git. Commands create
`outputs/` for new reports and `.cache/` for downloaded dependencies when needed.
Old model experiments, duplicate exports and stale reports have been removed.

## Training and checks

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Harshpreet1729/Prescription-Scanner-Medicine-Identifier/blob/main/notebooks/RxHandBD_TrOCR_Base_Colab.ipynb)

The badge opens the GitHub version. For local changes that have not been pushed,
upload `notebooks/RxHandBD_TrOCR_Base_Colab.ipynb` directly to Colab.
**Run all starts training** after GPU selection and Drive authorization.
See [Colab setup](docs/COLAB_TRAINING.md).

```powershell
python -m scripts.audit_dataset
python -m scripts.train --config configs/colab_base.json  # checks only
python -m unittest discover -s tests -v
python -m unittest webapp.tests -v
```

Detailed commands: [runbook](docs/RUNBOOK.md).
Website details: [Django guide](docs/DJANGO_WORKSPACE.md).
Future page annotation work: [page data guide](docs/PUBLIC_PAGE_DATA.md).
The 20-page annotation batch contains draft model readings, not verified
training labels; it has not produced a new trained model.

Optional formatting tools: install `requirements-dev.txt`, then run
`python -m black app src scripts tests webapp --workers 1`.
After editing embedded training/backend sources, rebuild the notebook with
`python -m scripts.build_colab` before running tests or uploading to Colab.

## Credits

- [RxHandBD](https://doi.org/10.5281/zenodo.18478741), Md. Masudul Islam;
  [retained provenance](data/rxhandbd/README.md).
- [Microsoft TrOCR](https://github.com/microsoft/unilm/tree/master/trocr),
  Hugging Face Transformers and PyTorch.
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR),
  [RxNorm](https://lhncbc.nlm.nih.gov/RxNav/) and
  [DailyMed](https://dailymed.nlm.nih.gov/).

Third-party datasets, weights and services retain their own terms and licenses.
