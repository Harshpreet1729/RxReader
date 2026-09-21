"""Portable configuration and validated image/label loading."""
import csv
import json
from pathlib import Path
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    path = Path(path) if path else ROOT / 'configs/pipeline.json'
    cfg = json.loads(path.read_text(encoding='utf-8-sig'))
    return cfg


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_image(path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if image.width * image.height > 30_000_000:
            raise ValueError('Image exceeds 30 megapixels; resize it before processing.')
        return image.convert('RGB')


def read_split(root, split):
    if split not in {'train', 'validation', 'test'}:
        raise ValueError('Only active train/validation/test splits are allowed.')
    root = Path(root).resolve()
    with (root / f'{split}.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f'Empty {split} split')
    records = []
    for row in rows:
        image = (root / 'images' / row['filename']).resolve()
        if not image.is_relative_to(root / 'images') or not image.is_file():
            raise ValueError(f'Invalid image: {row["filename"]}')
        # Normalize whitespace for OCR targets, never alter source files or casing.
        text = ' '.join(row['label'].split())
        if not text or '?' in text:
            raise ValueError(f'Unreviewed target: {row["filename"]}')
        records.append({'image': image, 'text': text, 'filename': row['filename']})
    return records


class OCRDataset:
    def __init__(self, records, processor, max_length=64):
        self.records, self.processor, self.max_length = records, processor, max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import torch
        row = self.records[index]
        image = load_image(row['image'])
        values = self.processor(images=image, return_tensors='pt').pixel_values[0]
        ids = self.processor.tokenizer(row['text'], add_special_tokens=True).input_ids
        if len(ids) > self.max_length:
            raise ValueError(f'Target too long; increase max_target_length: {row["filename"]}')
        ids += [self.processor.tokenizer.pad_token_id] * (self.max_length - len(ids))
        labels = torch.tensor(ids)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {'pixel_values': values, 'labels': labels}
