"""Read-only dataset audit. Run: python -m scripts.audit_dataset."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
SPLITS = {'train': ('Training', 'training'), 'validation': ('Validation', 'validation'), 'test': ('Testing', 'testing')}


def duplicate_groups(groups, cross_split=False):
    result = []
    for members in groups.values():
        if len(members) > 1 and (not cross_split or len({m['split'] for m in members}) > 1):
            result.append(members)
    return result


def audit(root):
    root = Path(root).resolve()
    report = {'dataset_root': str(root), 'splits': {}, 'issues': [], 'method': 'SHA256 file bytes and decoded RGB pixels with dimensions; no perceptual similarity or writer identification.'}
    byte_groups, pixel_groups = defaultdict(list), defaultdict(list)
    medicine_generics = defaultdict(set)
    all_labels = {}
    for split, (directory, prefix) in SPLITS.items():
        csv_path = root / directory / f'{prefix}_labels.csv'
        images_dir = root / directory / f'{prefix}_words'
        with csv_path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            columns = reader.fieldnames or []
            required = {'IMAGE', 'MEDICINE_NAME', 'GENERIC_NAME'}
            if not required.issubset(columns):
                raise ValueError(f'{csv_path}: missing columns {required - set(columns)}')
            rows = list(reader)
        labels, generics, modes = Counter(), Counter(), Counter()
        widths, heights, ratios, referenced = [], [], [], []
        for line, row in enumerate(rows, 2):
            fields = {key: (row.get(key) or '').strip() for key in required}
            member = {'split': split, 'image': fields['IMAGE'], 'medicine': fields['MEDICINE_NAME'], 'generic': fields['GENERIC_NAME'], 'csv_line': line}
            if not all(fields.values()):
                report['issues'].append({'kind': 'empty_field', **member})
            label, generic = fields['MEDICINE_NAME'], fields['GENERIC_NAME']
            labels[label] += 1
            generics[generic] += 1
            medicine_generics[label].add(generic)
            path = (images_dir / fields['IMAGE']).resolve()
            if not path.is_relative_to(images_dir.resolve()) or not fields['IMAGE']:
                report['issues'].append({'kind': 'invalid_image_path', **member})
                continue
            referenced.append(path)
            if not path.is_file():
                report['issues'].append({'kind': 'missing_image', **member})
                continue
            byte_groups[hashlib.sha256(path.read_bytes()).hexdigest()].append(member)
            try:
                with Image.open(path) as image:
                    image.load()
                    width, height = image.size
                    widths.append(width)
                    heights.append(height)
                    ratios.append(width / height)
                    modes[image.mode] += 1
                    pixel_hash = hashlib.sha256(f'{width}x{height}:RGB:'.encode() + image.convert('RGB').tobytes()).hexdigest()
                    pixel_groups[pixel_hash].append(member)
            except (OSError, ValueError) as error:
                report['issues'].append({'kind': 'unreadable_image', 'error': str(error), **member})
        references = Counter(referenced)
        for path, count in references.items():
            if count > 1:
                report['issues'].append({'kind': 'repeated_image_reference', 'split': split, 'image': path.name, 'count': count})
        actual = {path.resolve() for path in images_dir.rglob('*') if path.is_file()}
        def summary(values):
            return {'min': min(values), 'median': median(values), 'max': max(values)} if values else None
        report['splits'][split] = {
            'rows': len(rows), 'image_files': len(actual), 'columns': columns,
            'medicine_classes': len(labels), 'generic_classes': len(generics),
            'medicine_counts': dict(sorted(labels.items())), 'generic_counts': dict(sorted(generics.items())),
            'width': summary(widths), 'height': summary(heights), 'aspect_ratio': summary(ratios), 'modes': dict(modes),
            'orphan_files': sorted(str(path.relative_to(root)) for path in actual - set(referenced)),
            'labels_sha256': hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        }
        all_labels[split] = set(labels)
    report['medicine_to_generics'] = {k: sorted(v) for k, v in sorted(medicine_generics.items())}
    report['ambiguous_generic_mappings'] = {k: sorted(v) for k, v in medicine_generics.items() if len(v) > 1}
    report['labels_absent_from_train'] = {split: sorted(labels - all_labels['train']) for split, labels in all_labels.items() if split != 'train'}
    report['byte_duplicate_groups'] = duplicate_groups(byte_groups)
    report['pixel_duplicate_groups'] = duplicate_groups(pixel_groups)
    report['cross_split_pixel_duplicate_groups'] = duplicate_groups(pixel_groups, cross_split=True)
    report['conflicting_pixel_labels'] = [group for group in duplicate_groups(pixel_groups) if len({(r['medicine'], r['generic']) for r in group}) > 1]
    report['summary'] = {
        'rows': sum(s['rows'] for s in report['splits'].values()),
        'medicine_classes': len(medicine_generics),
        'generic_classes': len(set().union(*medicine_generics.values())),
        'byte_duplicate_groups': len(report['byte_duplicate_groups']),
        'pixel_duplicate_groups': len(report['pixel_duplicate_groups']),
        'cross_split_pixel_duplicate_groups': len(report['cross_split_pixel_duplicate_groups']),
        'conflicting_pixel_label_groups': len(report['conflicting_pixel_labels']),
        'issues': len(report['issues']),
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=PROJECT / 'Dataset/RxHandBD')
    parser.add_argument('--output', type=Path, default=PROJECT / 'codex/context/rxhandbd_audit.json')
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.root.resolve()):
        parser.error('Output must be outside the source dataset.')
    if (args.root / 'SOURCE.json').is_file():
        from scripts.prepare_rxhandbd import verify
        report = verify(args.root)
    else:
        report = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(report.get('summary', report), indent=2))
    print(f'Report: {args.output}')


if __name__ == '__main__':
    main()
