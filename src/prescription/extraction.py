"""Extract literal fields; never infer a dose or expand ambiguous abbreviations."""

import re

FREQUENCIES = [
    (r"\bonce\s+(?:a\s+day|daily|per\s+day)\b", 1),
    (r"\btwice\s+(?:a\s+day|daily|per\s+day)\b", 2),
    (r"\bthree\s+times\s+(?:a\s+day|daily|per\s+day)\b", 3),
    (r"\bfour\s+times\s+(?:a\s+day|daily|per\s+day)\b", 4),
    (r"\b([1-9])\s+times\s+(?:a\s+day|daily|per\s+day)\b", None),
]
DOSE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units?)\b", re.I)
AMBIGUOUS = re.compile(
    r"\b(?:OD|BD|BID|TDS|TID|QID|HS|PRN|SOS)\b|\b\d+\s*-\s*\d+\s*-\s*\d+\b", re.I
)


def extract_line(text):
    """Copy fields from one reading; missing or ambiguous values stay unknown."""
    text = " ".join(text.split())

    # 1. Find explicit phrases such as "twice daily" or "3 times a day".
    frequency_matches = []
    for pattern, count in FREQUENCIES:
        for match in re.finditer(pattern, text, re.I):
            daily_count = count
            if daily_count is None:
                daily_count = int(match.group(1))
            frequency_matches.append((match, daily_count))

    # 2. Keep strengths, quantities and unclear schedules exactly as written.
    intervals = list(re.finditer(r"\bevery\s+(\d+)\s+hours?\b", text, re.I))
    ambiguous = list(AMBIGUOUS.finditer(text))
    doses = list(DOSE.finditer(text))
    quantities = list(
        re.finditer(
            r"\b\d+(?:\.\d+)?\s*(?:tablets?|capsules?|drops?|puffs?)\b", text, re.I
        )
    )
    conditional = re.search(
        r"\b(?:as needed|if needed|when needed|if|when|required|then|alternate|weekly|every other|taper)\b",
        text,
        re.I,
    )
    duration = re.search(r"\bfor\s+\d+\s+(?:days?|weeks?|months?)\b", text, re.I)
    # 3. The possible name ends where a strength or instruction begins.
    field_positions = []
    for match, count in frequency_matches:
        field_positions.append(match.start())
    for match in doses + quantities + ambiguous + intervals:
        field_positions.append(match.start())
    if duration:
        field_positions.append(duration.start())

    candidate = text
    if field_positions:
        candidate = text[: min(field_positions)]

    # Remove a list number and a prefix such as "Tab" or "Cap".
    candidate = re.sub(
        r"^\s*(?:(?:\d+[.)]|[-*])\s*)?(?:(?:tab(?:let)?|cap(?:sule)?|syrup|inj(?:ection)?|rx)\.?\s+)?",
        "",
        candidate,
        flags=re.I,
    ).strip(" :;-")
    if re.fullmatch(r"(?:take|use|apply|instil|instill)?\s*", candidate, re.I):
        candidate = ""
    reasons = ["OCR transcription and medicine identity require confirmation."]
    if conditional:
        reasons.append(
            "Conditional or changing schedule retained verbatim; no fixed daily count supplied."
        )
    if ambiguous:
        reasons.append(
            "Abbreviated schedule retained verbatim; no dosing interpretation supplied."
        )
    if len(frequency_matches) > 1 or (frequency_matches and (intervals or ambiguous)):
        reasons.append("Multiple schedule expressions require review.")
    # 4. Only one unambiguous, fixed daily phrase can produce a daily count.
    times_per_day = None
    has_one_frequency = len(frequency_matches) == 1
    has_unclear_schedule = bool(intervals or ambiguous or conditional)
    if has_one_frequency and not has_unclear_schedule:
        times_per_day = frequency_matches[0][1]

    frequency_text = []
    for match, count in frequency_matches:
        frequency_text.append(match.group())
    for match in intervals + ambiguous:
        frequency_text.append(match.group())

    duration_text = None
    if duration:
        duration_text = duration.group()

    return {
        "raw_text": text,
        "medicine_candidate": candidate or None,
        "strength_as_written": [match.group() for match in doses],
        "dose_quantity_as_written": [match.group() for match in quantities],
        "frequency_as_written": frequency_text,
        "times_per_day_as_written": times_per_day,
        "duration_as_written": duration_text,
        "needs_review": True,
        "review_reasons": reasons,
    }
