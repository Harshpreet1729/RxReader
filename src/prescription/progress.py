"""Opt-in structured progress; normal CLI/audit output stays backward-compatible."""

import json
import os

PREFIX = "@@RX_PROGRESS@@"


def emit(stage, current=None, total=None, **details):
    if os.environ.get("RX_PROGRESS") == "1":
        print(
            PREFIX
            + json.dumps(
                {"stage": stage, "current": current, "total": total, **details}
            ),
            flush=True,
        )


def progress_rows(rows, stage, every=50):
    total = len(rows)
    emit(stage, 0, total)
    for index, row in enumerate(rows, 1):
        yield row
        if index % every == 0 or index == total:
            emit(stage, index, total)
