# RxReader model card

## Intended use

A research and portfolio prototype for transcribing handwritten medicine-word crops. All outputs require review. It is not validated for diagnosis, medication selection, dosing, dispensing or unsupervised clinical use.

## Model and training

Base: microsoft/trocr-base-handwritten, revision eaacaf452b06415df8f10bb6fad3a4c11e609406. The Colab configuration uses image-to-text fine-tuning with batch 1, gradient accumulation 16, fp16, gradient checkpointing, learning rate 2e-5, warmup ratio 0.1, weight decay 0.01, a 12-epoch maximum, validation/save every 100 optimizer steps and early stopping after five evaluations without improved normalized CER. These are configured limits; completed epoch/step counts are not established by the exported evaluation summaries alone.

Selection metric: validation normalized CER. Only training/validation enter optimization and model selection. The user ran the held-out test afterwards. Weights and processor were exported together and verified locally with a real inference smoke check.

## Results

See results.json for validation comparisons and held-out metrics. Final test: 1,111 samples, 28.78% normalized CER, 62.21% normalized WER and 41.22% normalized exact match (458 complete transcriptions). Raw exact match is 34.29% (381 samples). Full-test inference reported approximately 209 seconds in the user's Colab run; this is not a portable hardware benchmark.

Validation CER fell from 59.54% for pretrained small to 16.57% after base fine-tuning, a 72.2% relative reduction. This comparison combines architecture and domain fine-tuning effects. No separately fine-tuned small comparison exists. No pretrained held-out test comparison is claimed.

## Limitations

- Most test images were not completely transcribed correctly; a small spelling error can change a medicine identity.
- Labels may contain regional brands or non-drug words and are not authoritative generic mappings.
- Word-crop training does not validate page localization, multi-line relationships, writer independence or real clinical photographs.
- Beam log scores are not calibrated confidence probabilities.
- Abbreviations are retained rather than expanded; no schedule is invented or associated across separate regions.
- RxNorm/DailyMed have US-focused coverage. Names are looked up only after user confirmation; unresolved names are not fuzzily substituted.

Freeze this test result for this model version. Develop improvements using training/validation and use a separate untouched evaluation set for further unbiased comparisons.
