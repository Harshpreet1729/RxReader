"""Validate saved page annotations before admitting them to the review workspace."""

import hashlib
import json
import re

from .layout import checked_box
from .page_review import assemble_medications

MAX_ANNOTATION_BYTES = 512 * 1024
MAX_REGIONS = 80
ROLES = {"medicine", "instruction", "other", "unreviewed"}


def load_annotations(data, image_png, size):
    if len(data) > MAX_ANNOTATION_BYTES:
        raise ValueError("Saved review must be no larger than 512 KB.")
    try:
        payload = json.loads(data.decode("utf-8-sig"))
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ValueError("Saved review must be a valid JSON file.") from exc
    if (
        not isinstance(payload, dict)
        or type(payload.get("schema_version")) is not int
        or payload["schema_version"] != 1
    ):
        raise ValueError(
            "Use a version 1 page review JSON, not a region template or manifest."
        )
    if payload.get("image_sha256") != hashlib.sha256(
        image_png
    ).hexdigest() or payload.get("image_size") != list(size):
        raise ValueError(
            "This review belongs to a different image. Upload the exact original image with its saved review."
        )
    rows = payload.get("regions")
    if not isinstance(rows, list) or len(rows) > MAX_REGIONS:
        raise ValueError("Saved review must contain at most 80 regions.")
    regions = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each saved region must be an object.")
        identifier = row.get("id")
        if not isinstance(identifier, str) or not re.fullmatch(
            r"[1-9][0-9]{0,8}", identifier
        ):
            raise ValueError("Region IDs must be positive integers written as text.")
        box = row.get("box")
        if (
            not isinstance(box, list)
            or len(box) != 4
            or any(type(v) is not int for v in box)
        ):
            raise ValueError("Saved boxes must have four integer pixel coordinates.")
        box = checked_box(box, size)
        text, ocr = row.get("text", ""), row.get("ocr_text", "")
        if any(not isinstance(v, str) or len(v) > 1000 for v in (text, ocr)):
            raise ValueError(
                "Each saved reading must be text of at most 1,000 characters."
            )
        role, parent = row.get("role"), row.get("medicine_id", "")
        if (
            not isinstance(role, str)
            or role not in ROLES
            or not isinstance(parent, str)
        ):
            raise ValueError("Saved region role or medicine link is invalid.")
        reviewed = row.get("reviewed", False)
        truncated = row.get("possibly_truncated", False)
        if type(reviewed) is not bool or type(truncated) is not bool:
            raise ValueError("Review flags must be true or false.")
        if reviewed and (not text.strip() or role == "unreviewed"):
            raise ValueError("A reviewed region needs text and an assigned role.")
        alternatives = row.get("alternatives", [])
        if (
            not isinstance(alternatives, list)
            or len(alternatives) > 10
            or any(
                not isinstance(a, dict)
                or not isinstance(a.get("text"), str)
                or len(a["text"]) > 1000
                for a in alternatives
            )
        ):
            raise ValueError("Invalid saved alternative readings.")
        regions.append(
            {
                "id": identifier,
                "box": box,
                "text": text,
                "ocr_text": ocr,
                "role": role,
                "medicine_id": parent,
                "reviewed": reviewed,
                "possibly_truncated": truncated,
                "alternatives": [{"text": a["text"]} for a in alternatives],
                "suggested_medicine_id": "",
            }
        )
    # Recompute medicine records later; do not trust imported status/identity fields.
    assemble_medications(regions)
    return regions
