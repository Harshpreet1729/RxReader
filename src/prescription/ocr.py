"""Lazy TrOCR inference; generation scores are NOT correctness probabilities."""

from pathlib import Path
from .data import ROOT


class TrOCRReader:
    """Load the model once, then reuse it for individual image crops."""

    def __init__(
        self,
        model_id="microsoft/trocr-small-handwritten",
        device="cpu",
        revision=None,
        max_new_tokens=64,
    ):
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self.torch = torch
        self.device = device
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        cache = str(ROOT / ".cache/huggingface")
        options = {"cache_dir": cache}
        if revision and not Path(model_id).exists():
            options["revision"] = revision
        self.processor = TrOCRProcessor.from_pretrained(
            model_id, use_fast=False, **options
        )
        self.model = (
            VisionEncoderDecoderModel.from_pretrained(
                model_id, weights_only=True, **options
            )
            .to(device)
            .eval()
        )
        self.max_new_tokens = max_new_tokens
        self.model_id = model_id
        self.revision = getattr(self.model.config, "_commit_hash", None)

    def read(self, image):
        """Convert a crop into text and return up to three possible readings."""
        # The processor resizes/normalizes the image and creates a PyTorch tensor.
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self.device)

        # No gradients are needed when reading an image. Beam search keeps
        # three candidate sequences instead of choosing only one at every step.
        with self.torch.inference_mode():
            output = self.model.generate(
                pixel_values,
                max_new_tokens=self.max_new_tokens,
                num_beams=3,
                num_return_sequences=3,
                output_scores=True,
                return_dict_in_generate=True,
            )
        # Turn token IDs back into words, without start/end/padding tokens.
        texts = self.processor.batch_decode(output.sequences, skip_special_tokens=True)
        scores = output.sequences_scores.tolist()
        candidates = []
        seen = set()
        for text, score in zip(texts, scores):
            if text not in seen:
                candidates.append({"text": text, "beam_log_score": float(score)})
                seen.add(text)
        sequence = output.sequences[0].tolist()
        # Ignore the starting token when checking whether generation finished.
        ended = self.processor.tokenizer.eos_token_id in sequence[1:]
        return {
            "text": texts[0],
            "alternatives": candidates,
            "possibly_truncated": not ended,
            "score_is_calibrated": False,
            "model": self.model_id,
            "revision": self.revision,
        }
