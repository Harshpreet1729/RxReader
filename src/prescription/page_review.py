"""Conservative region proposals and explicit, evidence-preserving medicine links.

Proposals are heuristics, not a trained medicine detector. The user must review
region roles and links before the app calls a medicine record reviewed.
"""

import re
from .extraction import extract_line

HEADER = re.compile(
    r"^(?:dr[. ]|doctor\b|patient\b|name\s*:|age\s*:|date\s*:|address\b|phone\b|diagnosis\b|hospital\b|clinic\b)",
    re.I,
)
MEDICINE = re.compile(
    r"^\s*(?:\d+[.)]\s*)?(?:tab(?:let)?|cap(?:sule)?|syrup|syp|inj(?:ection)?|cream|ointment|drops)\.?\s+",
    re.I,
)


def propose_regions(regions):
    for region in regions:
        text = region["ocr_text"]
        fields = extract_line(text)
        if HEADER.search(text):
            role = "other"
        elif MEDICINE.search(text):
            role = "medicine"
        elif not fields["medicine_candidate"] and (
            fields["frequency_as_written"]
            or fields["strength_as_written"]
            or fields["duration_as_written"]
            or fields["dose_quantity_as_written"]
        ):
            role = "instruction"
        else:
            role = "unreviewed"
        region.update(
            text=text,
            role=role,
            medicine_id="",
            reviewed=False,
            suggested_medicine_id="",
        )
    medicines = [r for r in regions if r["role"] == "medicine"]
    for region in regions:
        if region["role"] != "instruction":
            continue
        x1, y1, x2, y2 = region["box"]
        height = y2 - y1
        matches = []
        for med in medicines:
            a, b, c, d = med["box"]
            overlap = max(0, min(y2, d) - max(y1, b))
            if overlap >= 0.5 * min(height, d - b) and x1 >= c - 5:
                matches.append(med["id"])
        # No nearest-neighbor linking across rows. Even a unique same-row suggestion
        # remains unapplied until the user explicitly selects its medicine.
        if len(matches) == 1:
            region["suggested_medicine_id"] = matches[0]
    return regions


def assemble_medications(regions):
    identifiers = {r["id"] for r in regions}
    if len(identifiers) != len(regions):
        raise ValueError("Region IDs must be unique.")
    medicines = {r["id"]: r for r in regions if r["role"] == "medicine"}
    for region in regions:
        parent = region.get("medicine_id", "")
        if parent and (region["role"] != "instruction" or parent not in medicines):
            raise ValueError("An instruction must link to an existing medicine region.")
    records = []
    for identifier, medicine in medicines.items():
        linked = [medicine] + [
            r
            for r in regions
            if r["role"] == "instruction" and r.get("medicine_id") == identifier
        ]
        # Preserve the user's explicit association, not global page reading order.
        combined = " ; ".join(r["text"] for r in linked)
        parsed = extract_line(combined)
        name = extract_line(medicine["text"])["medicine_candidate"]
        reviewed = all(r.get("reviewed") and r["text"].strip() for r in linked)
        records.append(
            {
                **parsed,
                "id": identifier,
                "medicine_candidate": name,
                "transcription_reviewed": reviewed,
                "identity_confirmed": False,
                "source_regions": [r["id"] for r in linked],
                "evidence": [
                    {
                        "region_id": r["id"],
                        "box": r["box"],
                        "ocr_text": r["ocr_text"],
                        "reviewed_text": r["text"],
                    }
                    for r in linked
                ],
                "times_per_day_as_written": (
                    parsed["times_per_day_as_written"] if reviewed else None
                ),
                "review_reasons": (
                    ["Confirm medicine identity before looking up uses."]
                    if reviewed
                    else ["Review the medicine and every linked instruction."]
                )
                + parsed["review_reasons"][1:],
            }
        )
    return records
