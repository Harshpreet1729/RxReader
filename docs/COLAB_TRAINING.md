# Colab training

Open notebooks/RxHandBD_TrOCR_Base_Colab.ipynb through the README's Colab button, select a GPU runtime, then Run all. This starts actual training after GPU checks and Drive authorization. Do not run two jobs with the same RUN_NAME.

The notebook embeds a reviewed project snapshot, downloads the checksummed publisher archive, verifies all images and frozen splits, evaluates pretrained baselines, trains TrOCR-base, and compares validation results. No manual dataset upload or API token is required. Keep Colab's supplied CUDA PyTorch/torchvision installation.

## Persistence and recovery

Default Drive folder: MyDrive/medical_AI/trocr-base-rxhandbd-v2. Allow several GB free space for model/optimizer checkpoints, exports and logs; approximately 12 GB headroom is a planning estimate, not a guarantee. Checkpoints are written periodically. Resume requires a completion marker and matching configuration/data. Re-run with the same run name after interruption; work since the last complete checkpoint may be lost. If interrupted before the first complete checkpoint, preserve those files and choose a new run name.

Changes to the embedded code hash require a new run directory. Rebuild after source edits using `python -m scripts.build_colab`; public provenance and split checksums are sufficient to build the notebook without locally downloaded images.

## Visible progress

Dataset downloads show bytes. Preparation/audit show image counts. Training reports actual optimizer steps, gradient-accumulation batches, epoch and logged loss. Validation reports batches. Model loading and Drive checkpoint writes use indeterminate bars, elapsed time and live output rather than invented percentages. A silence notice does not prove the process is healthy or stuck. Full output appends to live.log under the run folder.

The live runner explicitly forwards unbuffered subprocess stdout/stderr into notebook HTML. Interrupting normally stops the launched process and its process group on Linux; runtime disconnects remain governed by Colab. Check for lingering older jobs before launching another session.

## Evaluation and export

Validation compares pretrained small, pretrained base and fine-tuned base. This is not a controlled comparison against fine-tuned small. Enable RUN_FINAL_TEST only after model selection. Download final_test.json and comparison.json for your run's evidence.

Enable DOWNLOAD_MODEL to export the complete best directory, including processor/tokenizer files. Fine-tuned weights are not hosted by this repository. Put the exported directory under checkpoints/trocr_base_rxhandbd/best or pass its path with --model. Optional image upload and confirmed-name lookup cells are disabled until explicitly enabled/filled in.

Training settings and known limits are described in MODEL_CARD.md. Free GPU availability and session length vary: https://research.google.com/colaboratory/faq.html . This project does not create paid cloud resources.
