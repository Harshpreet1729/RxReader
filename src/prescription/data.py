"""Portable configuration and validated image/label loading."""

import csv
import json
from pathlib import Path
from PIL import Image, ImageOps
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    path = Path(path) if path else ROOT / "configs/pipeline.json"
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    return config


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_image(path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if image.width * image.height > 30_000_000:
            raise ValueError(
                "Image exceeds 30 megapixels; resize it before processing."
            )
        return image.convert("RGB")

#! ready images paths and labels
def read_split(root, split):
    if split not in {"train", "validation", "test"}:
        raise ValueError("Only active train/validation/test splits are allowed.")

    root = Path(root).resolve()

    #^ opening the files eg root/"train.csv"
    with (root / f"{split}.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream)) #* like this [{'filename': 'P1116.jpg', 'label': 'Dexter'},
    if not rows:
        raise ValueError(f"Empty {split} split")


    records = []
    for row in rows:
        image = (root / "images" / row["filename"]).resolve() #& D:/RxReader/Dataset/RxHandBD/images/img001.jpg
        if not image.is_relative_to(root / "images") or not image.is_file():
            raise ValueError(f'Invalid image: {row["filename"]}')

        text = " ".join(row["label"].split())
        if not text or "?" in text:
            raise ValueError(f'Unreviewed target: {row["filename"]}')

        records.append({"image": image, "text": text, "filename": row["filename"]}) #^ [{'image': WindowsPath('D:/medical_AI/Dataset/RxHandBD/images/P1116.jpg'), 'text': 'Dexter', 'filename': 'P1116.jpg'}
    return records


class OCRDataset(Dataset):
    """Give the Trainer one image tensor and its expected text tokens at a time."""

    def __init__(self, records, processor, max_length=64):
        self.records = records
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.records) # 3994

    def __getitem__(self, index):
        import torch


        #^ Image preprocessing
        row = self.records[index]
        image = load_image(row["image"])
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values[0]
        token_ids = self.processor.tokenizer(
            row["text"], add_special_tokens=True
        ).input_ids


        if len(token_ids) > self.max_length:
            raise ValueError(
                f'Target too long; increase max_target_length: {row["filename"]}'
            )

        #& padding
        token_ids += [self.processor.tokenizer.pad_token_id] * (self.max_length - len(token_ids))

        #^ list to pytorch tensor
        labels = torch.tensor(token_ids)
        # -100 tells the loss function to ignore padding positions.
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}
