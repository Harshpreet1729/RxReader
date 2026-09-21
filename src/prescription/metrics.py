"""Metrics independent of training libraries. CER is corpus weighted."""
import unicodedata


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (left != right)))
        previous = current
    return previous[-1]


def metrics(references, predictions):
    if len(references) != len(predictions) or not references:
        raise ValueError('Supply equally sized, non-empty reference/prediction lists.')
    result = {'samples': len(references)}
    for mode, transform in [('raw', str), ('normalized', normalize)]:
        refs, preds = list(map(transform, references)), list(map(transform, predictions))
        characters = sum(map(len, refs))
        words = sum(len(s.split()) for s in refs)
        result[mode + '_cer'] = sum(distance(a, b) for a, b in zip(refs, preds)) / max(1, characters)
        result[mode + '_wer'] = sum(distance(a.split(), b.split()) for a, b in zip(refs, preds)) / max(1, words)
        result[mode + '_exact_match'] = sum(a == b for a, b in zip(refs, preds)) / len(refs)
    return result
