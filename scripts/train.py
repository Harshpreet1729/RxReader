"""Default: data-only preflight. Training requires --execute and CUDA."""
import argparse
import hashlib
import json
from pathlib import Path
from src.prescription.data import ROOT, OCRDataset, load_config, project_path, read_split
from src.prescription.metrics import metrics
from src.prescription.progress import emit, progress_rows


def preflight(cfg):
    root = project_path(cfg['dataset_root'])
    from scripts.prepare_rxhandbd import verify
    audit = verify(root)
    return {'mode': 'preflight_only', 'training_started': False, 'model': cfg['model_id'],
            'dataset': audit, 'max_target_length': cfg['max_target_length'],
            'output_dir': str(project_path(cfg['output_dir'])),
            'next_step': 'On a CUDA GPU only: python -m scripts.train --execute'}


def train(cfg, resume=None):
    emit('Initializing training dependencies')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Training is disabled on CPU. Use a CUDA notebook/GPU; preflight needs no GPU.')
    print('Training device:',torch.cuda.get_device_name(0),flush=True)
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator, set_seed, EarlyStoppingCallback, TrainerCallback
    set_seed(cfg['seed'])
    root = project_path(cfg['dataset_root'])
    output = project_path(cfg['output_dir'])
    if resume:
        resume = Path(resume).resolve()
        if not resume.is_relative_to(output.resolve()) or not (resume/'trainer_state.json').exists():
            raise ValueError('Resume must identify a Trainer checkpoint under the configured output directory.')
    elif output.exists() and any(output.iterdir()):
        raise ValueError('Output is not empty. Select a new output_dir or explicitly resume a checkpoint.')
    if resume:
        previous = json.loads((output/'run_config.json').read_text(encoding='utf-8'))
        if previous['config'] != cfg:
            raise ValueError('Resume configuration differs from the saved run. Restore it or use a new run directory.')
        for split in ('train', 'validation'):
            digest = hashlib.sha256((root/f'{split}.csv').read_bytes()).hexdigest()
            if digest != previous[f'{split}_csv_sha256']:
                raise ValueError('Resume dataset differs from the saved run.')
    opts = {'revision': cfg['model_revision'], 'cache_dir': str(ROOT/'.cache/huggingface')}
    emit('Loading processor')
    processor = TrOCRProcessor.from_pretrained(cfg['model_id'], use_fast=False, **opts)
    emit('Loading model weights')
    model = VisionEncoderDecoderModel.from_pretrained(cfg['model_id'], weights_only=True, **opts)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.decoder.config.use_cache = False
    model.generation_config.decoder_start_token_id = model.config.decoder_start_token_id
    model.generation_config.pad_token_id = model.config.pad_token_id
    model.generation_config.eos_token_id = model.config.eos_token_id
    model.generation_config.max_length = cfg['max_target_length']
    model.generation_config.num_beams = 3
    train_rows, val_rows = read_split(root, 'train'), read_split(root, 'validation')
    for row in progress_rows(train_rows + val_rows,'Checking target token lengths'):
        if len(processor.tokenizer(row['text']).input_ids) > cfg['max_target_length']:
            raise ValueError(f'Target exceeds token limit: {row["filename"]}')
    def score(output):
        import numpy as np
        preds = output.predictions[0] if isinstance(output.predictions, tuple) else output.predictions
        labels = np.where(output.label_ids == -100, processor.tokenizer.pad_token_id, output.label_ids)
        return metrics(processor.batch_decode(labels, skip_special_tokens=True), processor.batch_decode(preds, skip_special_tokens=True))
    args = Seq2SeqTrainingArguments(output_dir=str(output), num_train_epochs=cfg['epochs'],
        per_device_train_batch_size=cfg['batch_size'], per_device_eval_batch_size=cfg['batch_size'],
        gradient_accumulation_steps=cfg['gradient_accumulation_steps'], learning_rate=cfg['learning_rate'],
        eval_strategy=cfg.get('eval_strategy','epoch'), save_strategy=cfg.get('eval_strategy','epoch'),
        eval_steps=cfg.get('eval_steps',100), save_steps=cfg.get('save_steps',100),
        warmup_ratio=cfg.get('warmup_ratio',0.0), weight_decay=cfg.get('weight_decay',0.0),
        save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model='normalized_cer', greater_is_better=False, predict_with_generate=True,
        fp16=True, gradient_checkpointing=True, dataloader_num_workers=0, remove_unused_columns=False,
        report_to=[], disable_tqdm=False, logging_first_step=True, logging_steps=cfg.get('logging_steps',25), seed=cfg['seed'])
    class CompletedCheckpoint(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            # Written only after Trainer has finished writing model/optimizer/state files.
            path = Path(args.output_dir)/f'checkpoint-{state.global_step}'/'checkpoint_complete.json'
            path.write_text(json.dumps({'global_step':state.global_step}),encoding='utf-8')
    from src.prescription.training_progress import TrainingProgress
    callbacks = [CompletedCheckpoint(), TrainingProgress(len(val_rows),cfg['batch_size'])]
    if cfg.get('early_stopping_patience'):
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg['early_stopping_patience']))
    emit('Preparing CUDA model and trainer')
    trainer = Seq2SeqTrainer(model=model, args=args, callbacks=callbacks, processing_class=processor, data_collator=default_data_collator,
        train_dataset=OCRDataset(train_rows,processor,cfg['max_target_length']),
        eval_dataset=OCRDataset(val_rows,processor,cfg['max_target_length']), compute_metrics=score)
    output.mkdir(parents=True,exist_ok=True)
    provenance = {'config': cfg, 'resolved_revision': getattr(model.config,'_commit_hash',None),
                  'train_csv_sha256': hashlib.sha256((root/'train.csv').read_bytes()).hexdigest(),
                  'validation_csv_sha256': hashlib.sha256((root/'validation.csv').read_bytes()).hexdigest()}
    (output/'run_config.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    emit('Saving best model and processor to Drive/output folder')
    trainer.save_model(str(output/'best'))
    processor.save_pretrained(str(output/'best'))
    validation = trainer.evaluate()
    summary = {'training_completed':True,'global_step':trainer.state.global_step,
               'best_model_checkpoint':trainer.state.best_model_checkpoint,
               'train_metrics':result.metrics,'validation_metrics':validation}
    (output/'training_complete.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--execute',action='store_true',help='Actually train; requires CUDA. Omit for a safe preflight.')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    if args.resume and not args.execute:
        parser.error('--resume requires --execute')
    cfg=load_config(args.config)
    report=preflight(cfg)
    print(json.dumps(report,indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
    if args.execute:
        train(cfg,args.resume)

if __name__=='__main__':
    main()
