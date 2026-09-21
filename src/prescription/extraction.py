"""Extract literal fields; never infer a dose or expand ambiguous abbreviations."""
import re

FREQUENCIES = [
    (r'\bonce\s+(?:a\s+day|daily|per\s+day)\b', 1),
    (r'\btwice\s+(?:a\s+day|daily|per\s+day)\b', 2),
    (r'\bthree\s+times\s+(?:a\s+day|daily|per\s+day)\b', 3),
    (r'\bfour\s+times\s+(?:a\s+day|daily|per\s+day)\b', 4),
    (r'\b([1-9])\s+times\s+(?:a\s+day|daily|per\s+day)\b', None),
]
DOSE = re.compile(r'\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units?)\b', re.I)
AMBIGUOUS = re.compile(r'\b(?:OD|BD|BID|TDS|TID|QID|HS|PRN|SOS)\b|\b\d+\s*-\s*\d+\s*-\s*\d+\b', re.I)


def extract_line(text):
    text = ' '.join(text.split())
    frequency_matches = []
    for pattern, count in FREQUENCIES:
        for match in re.finditer(pattern, text, re.I):
            frequency_matches.append((match, count if count is not None else int(match.group(1))))
    intervals = list(re.finditer(r'\bevery\s+(\d+)\s+hours?\b', text, re.I))
    ambiguous = list(AMBIGUOUS.finditer(text))
    doses = list(DOSE.finditer(text))
    duration = re.search(r'\bfor\s+\d+\s+(?:days?|weeks?|months?)\b', text, re.I)
    delimiters = [m.start() for m, _ in frequency_matches] + [m.start() for m in doses + ambiguous + intervals]
    if duration:
        delimiters.append(duration.start())
    candidate = text[:min(delimiters)] if delimiters else text
    candidate = re.sub(r'^\s*(?:(?:\d+[.)]|[-*])\s*)?(?:(?:tab(?:let)?|cap(?:sule)?|syrup|inj(?:ection)?|rx)\.?\s+)?', '', candidate, flags=re.I).strip(' :;-')
    reasons = ['OCR transcription and medicine identity require confirmation.']
    if ambiguous:
        reasons.append('Abbreviated schedule retained verbatim; no dosing interpretation supplied.')
    if len(frequency_matches) > 1 or (frequency_matches and (intervals or ambiguous)):
        reasons.append('Multiple schedule expressions require review.')
    times = frequency_matches[0][1] if len(frequency_matches) == 1 and not intervals and not ambiguous else None
    return {'raw_text': text, 'medicine_candidate': candidate or None,
            'strength_as_written': [m.group() for m in doses],
            'frequency_as_written': [m.group() for m, _ in frequency_matches] + [m.group() for m in intervals + ambiguous],
            'times_per_day_as_written': times, 'duration_as_written': duration.group() if duration else None,
            'needs_review': True, 'review_reasons': reasons}
