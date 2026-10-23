"""Measure OCR mistakes using character errors, word errors and exact matches."""

import unicodedata


def normalize(text):
    """Ignore letter case, repeated whitespace and equivalent Unicode forms."""
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.casefold().split())


def distance(reference, prediction):
    """Count the fewest insertions, deletions and replacements (edit distance).

    Works with strings for CER and lists of words for WER. Each row stores
    the best answer for every prefix of the prediction. We only need to keep
    the previous row, rather than a whole table.
    """
    previous_row = list(range(len(prediction) + 1))

    for row_number, expected in enumerate(reference, start=1):
        current_row = [row_number]

        for column_number, actual in enumerate(prediction, start=1):
            insertion = current_row[column_number - 1] + 1
            deletion = previous_row[column_number] + 1
            replacement = previous_row[column_number - 1]
            if expected != actual:
                replacement += 1

            current_row.append(min(insertion, deletion, replacement))

        previous_row = current_row

    return previous_row[-1]


def metrics(references, predictions):
    """Calculate errors over the whole dataset, not an average of image rates."""
    if len(references) != len(predictions) or not references:
        raise ValueError("Supply equally sized, non-empty reference/prediction lists.")

    result = {"samples": len(references)}

    for mode in ("raw", "normalized"):
        character_errors = 0
        word_errors = 0
        total_characters = 0
        total_words = 0
        exact_matches = 0

        for reference, prediction in zip(references, predictions):
            if mode == "normalized":
                reference = normalize(reference)
                prediction = normalize(prediction)
            else:
                reference = str(reference)
                prediction = str(prediction)

            character_errors += distance(reference, prediction)
            word_errors += distance(reference.split(), prediction.split())
            total_characters += len(reference)
            total_words += len(reference.split())
            if reference == prediction:
                exact_matches += 1

        # max(1, ...) avoids division by zero for an empty reference string.
        result[mode + "_cer"] = character_errors / max(1, total_characters)
        result[mode + "_wer"] = word_errors / max(1, total_words)
        result[mode + "_exact_match"] = exact_matches / len(references)

    return result
