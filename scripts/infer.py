"""Read a word crop or prescription page; output candidates for review."""

import argparse
import json
from pathlib import Path
from src.prescription.data import load_config, load_image
from src.prescription.ocr import TrOCRReader
from src.prescription.pipeline import process_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mode", choices=["word", "page"], default="word")
    parser.add_argument(
        "--regions", type=Path, help="Explicit per-line boxes; bypass page detector"
    )
    parser.add_argument("--model")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/prescription.json")
    )
    args = parser.parse_args()
    config = load_config(args.config)
    reader = TrOCRReader(
        args.model or config["model_id"],
        args.device,
        revision=None if args.model else config["model_revision"],
        max_new_tokens=config["max_target_length"],
    )
    result = process_image(load_image(args.image), reader, args.mode, args.regions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
