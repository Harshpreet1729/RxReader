import argparse
import hashlib
import json
from pathlib import Path
from src.prescription.data import (
    ROOT,
    OCRDataset,
    load_config,
    project_path,
    read_split,
)
from src.prescription.metrics import metrics
from src.prescription.progress import emit, progress_rows



def preflight(config):
    """Check the dataset and settings without loading or training the model."""
    root = project_path(config["dataset_root"])
    if config.get("dataset_format", "rxhandbd") == "reviewed_pages":
        from scripts.prepare_page_crops import verify
    elif config.get("dataset_format", "rxhandbd") == "rxhandbd":
        from scripts.prepare_rxhandbd import verify
    else:
        raise ValueError("Unknown dataset_format.")
    audit = verify(root)
    return {
        "mode": "preflight_only",
        "training_started": False,
        "model": config["model_id"],
        "dataset": audit,
        "max_target_length": config["max_target_length"],
        "output_dir": str(project_path(config["output_dir"])),
        "next_step": "On a CUDA GPU only: python -m scripts.train --execute",
    }


def check_output_directory(config, root, output, resume):
    """Avoid overwriting a run or resuming with different data/settings."""
    if resume:
        resume = Path(resume).resolve()
        if (
            not resume.is_relative_to(output.resolve())
            or not (resume / "trainer_state.json").exists()
        ):
            raise ValueError(
                "Resume must identify a Trainer checkpoint under the configured output directory."
            )
    elif output.exists() and any(output.iterdir()):
        raise ValueError(
            "Output is not empty. Select a new output_dir or explicitly resume a checkpoint."
        )
    if resume:
        previous = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
        if previous["config"] != config:
            raise ValueError(
                "Resume configuration differs from the saved run. Restore it or use a new run directory."
            )
        for split in ("train", "validation"):
            digest = hashlib.sha256((root / f"{split}.csv").read_bytes()).hexdigest()
            if digest != previous[f"{split}_csv_sha256"]:
                raise ValueError("Resume dataset differs from the saved run.")
    return resume


def train(config, resume=None):
    """Fine-tune TrOCR using the Hugging Face training loop."""
    # 1. Require a GPU, fix the seed, and check the output directory.
    emit("Initializing training dependencies")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Training is disabled on CPU. Use a CUDA notebook/GPU; preflight needs no GPU."
        )
    print("Training device:", torch.cuda.get_device_name(0), flush=True)
    from transformers import (
        TrOCRProcessor,
        PreTrainedTokenizerBase,
        VisionEncoderDecoderModel,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        default_data_collator,
        set_seed,
        EarlyStoppingCallback,
        TrainerCallback,
    )

    set_seed(config["seed"])
    root = project_path(config["dataset_root"])
    output = project_path(config["output_dir"])
    resume = check_output_directory(config, root, output, resume)


    # 2. Load pretrained weights and match special tokens to the tokenizer.
    model_options = {
        "revision": config["model_revision"],
        "cache_dir": str(ROOT / ".cache/huggingface"),
    }
    emit("Loading processor")
    processor = TrOCRProcessor.from_pretrained(
        config["model_id"], use_fast=False, **model_options
    )
    emit("Loading model weights")
    model = VisionEncoderDecoderModel.from_pretrained(
        config["model_id"], weights_only=True, **model_options
    )
    # Transformers attaches the tokenizer dynamically. Check it before use.
    tokenizer = getattr(processor, "tokenizer", None)
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise TypeError("The TrOCR processor must contain a text tokenizer.")

    start_token_id = tokenizer.cls_token_id
    pad_token_id = tokenizer.pad_token_id
    end_token_id = tokenizer.sep_token_id
    # Each setting needs one integer ID, not a token string or a list of IDs.
    if not isinstance(start_token_id, int):
        raise ValueError("The tokenizer must have one integer start-token ID.")
    if not isinstance(pad_token_id, int):
        raise ValueError("The tokenizer must have one integer padding-token ID.")
    if not isinstance(end_token_id, int):
        raise ValueError("The tokenizer must have one integer end-token ID.")

    model.config.decoder_start_token_id = start_token_id
    model.config.pad_token_id = pad_token_id
    model.config.eos_token_id = end_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    # The decoder shares this configuration with the parent model.
    model.config.decoder.use_cache = False

    generation_config = model.generation_config
    if generation_config is None:
        raise ValueError("The loaded OCR model has no text generation configuration.")
    generation_config.decoder_start_token_id = start_token_id
    generation_config.pad_token_id = pad_token_id
    generation_config.eos_token_id = end_token_id
    generation_config.max_length = config["max_target_length"]
    generation_config.num_beams = 3

    # 3. Load training/validation examples and reject overlong target text.
    train_rows = read_split(root, "train")
    validation_rows = read_split(root, "validation")
    for row in progress_rows(
        train_rows + validation_rows, "Checking target token lengths"
    ):
        if (
            len(tokenizer(row["text"]).input_ids)
            > config["max_target_length"]
        ):
            raise ValueError(f'Target exceeds token limit: {row["filename"]}')

    def score(evaluation):
        """The Trainer calls this with predicted tokens and expected tokens."""
        import numpy as np

        predicted_ids = (
            evaluation.predictions[0]
            if isinstance(evaluation.predictions, tuple)
            else evaluation.predictions
        )
        # Replace ignored loss positions with padding before decoding text.
        labels = np.where(
            evaluation.label_ids == -100,
            pad_token_id,
            evaluation.label_ids,
        )
        return metrics(
            processor.batch_decode(labels, skip_special_tokens=True),
            processor.batch_decode(predicted_ids, skip_special_tokens=True),
        )

    # 4. Choose the learning settings. Lower validation CER selects the best model.
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output),
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        eval_strategy=config.get("eval_strategy", "epoch"),
        save_strategy=config.get("eval_strategy", "epoch"),
        eval_steps=config.get("eval_steps", 100),
        save_steps=config.get("save_steps", 100),
        warmup_ratio=config.get("warmup_ratio", 0.0),
        weight_decay=config.get("weight_decay", 0.0),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="normalized_cer",
        greater_is_better=False,
        predict_with_generate=True,
        # Mixed precision and checkpointing reduce GPU memory use.
        fp16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=[],
        disable_tqdm=False,
        logging_first_step=True,
        logging_steps=config.get("logging_steps", 25),
        seed=config["seed"],
    )

    class CompletedCheckpoint(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            # Written only after Trainer has finished writing model/optimizer/state files.
            if args.output_dir is None:
                raise ValueError("A checkpoint output directory is required.")
            path = (
                Path(args.output_dir)
                / f"checkpoint-{state.global_step}"
                / "checkpoint_complete.json"
            )
            path.write_text(
                json.dumps({"global_step": state.global_step}), encoding="utf-8"
            )

    from src.prescription.training_progress import TrainingProgress

    callbacks: list[TrainerCallback] = [
        CompletedCheckpoint(),
        TrainingProgress(len(validation_rows), config["batch_size"]),
    ]
    if config.get("early_stopping_patience"):
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config["early_stopping_patience"]
            )
        )
    emit("Preparing CUDA model and trainer")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        callbacks=callbacks,
        processing_class=processor,
        data_collator=default_data_collator,
        train_dataset=OCRDataset(train_rows, processor, config["max_target_length"]),
        eval_dataset=OCRDataset(
            validation_rows, processor, config["max_target_length"]
        ),
        compute_metrics=score,
    )
    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "config": config,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "train_csv_sha256": hashlib.sha256(
            (root / "train.csv").read_bytes()
        ).hexdigest(),
        "validation_csv_sha256": hashlib.sha256(
            (root / "validation.csv").read_bytes()
        ).hexdigest(),
    }
    (output / "run_config.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    # 5. Train, save the best weights + processor, then record validation results.
    result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    emit("Saving best model and processor to Drive/output folder")
    trainer.save_model(str(output / "best"))
    processor.save_pretrained(str(output / "best"))
    validation = trainer.evaluate()
    summary = {
        "training_completed": True,
        "global_step": trainer.state.global_step,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "train_metrics": result.metrics,
        "validation_metrics": validation,
    }
    (output / "training_complete.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually train; requires CUDA. Omit for a safe preflight.",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.resume and not args.execute:
        parser.error("--resume requires --execute")
    config = load_config(args.config)
    report = preflight(config)
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.execute:
        train(config, args.resume)


if __name__ == "__main__":
    main()
