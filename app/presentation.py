"""Convert backend results into HTML cards for the Gradio demo."""

from html import escape

DOCUMENT = (
    '<svg width="38" height="44" viewBox="0 0 38 44" fill="none" aria-hidden="true">'
    '<path d="M8 2h15l9 9v29H8zM23 2v10h9M13 21h14M13 27h14M13 33h10" '
    'stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>'
)
EMPTY_READING = (
    f'<div class="reading-empty">{DOCUMENT}'
    "<h3>Your reading will appear here</h3>"
    "<p>Upload an image, then select Read handwriting.</p></div>"
)
EMPTY_MATCHES = (
    '<p class="muted">Confirm the spelling to see generic ingredients '
    "and source labels.</p>"
)


def text(value, fallback="Not found"):
    """Escape OCR/API text so the browser displays it rather than running HTML."""
    if value is None or not str(value).strip():
        return fallback
    return escape(str(value))


def render_readings(report):
    cards = []

    for line in report["lines"]:
        strengths = ", ".join(line.get("strength_as_written") or [])
        schedules = ", ".join(line.get("frequency_as_written") or [])
        fields = [
            ("Possible medicine", line.get("medicine_candidate")),
            ("Strength as written", strengths),
            ("Schedule as written", schedules),
            ("Explicit times/day", line.get("times_per_day_as_written")),
        ]

        fields_html = ""
        for label, value in fields:
            fields_html += f"<div><dt>{label}</dt><dd>{text(value)}</dd></div>"

        alternatives_html = ""
        alternatives = line.get("ocr", {}).get("alternatives", [])
        for alternative in alternatives:
            alternatives_html += f'<li>{text(alternative["text"])}</li>'
        if not alternatives_html:
            alternatives_html = "<li>None available</li>"

        card = (
            '<article class="reading">'
            '<div class="reading-meta">'
            f'<span>Region {text(line["region_id"])}</span>'
            '<span class="review-label">Review required</span></div>'
            f'<h3 class="reading-text">{text(line["raw_text"], "No readable text")}</h3>'
            f"<dl>{fields_html}</dl>"
            "<details><summary>Alternative readings</summary>"
            f"<ul>{alternatives_html}</ul>"
            '<p class="muted">Alternatives are possible readings, '
            "not verified medicine names.</p></details></article>"
        )
        cards.append(card)

    return "".join(cards) or EMPTY_READING


def render_matches(report):
    if not report.get("candidates"):
        return (
            '<div class="lookup-note"><b>No exact match found.</b>'
            "<p>Check the spelling and formulation. Regional brands may "
            "be absent from these sources.</p></div>"
        )

    items = []
    for candidate in report["candidates"]:
        ingredients = ", ".join(candidate.get("ingredients") or [])
        label_count = len(candidate.get("labels", []))
        card = (
            '<div class="source-match">'
            f'<b>{text(candidate.get("name"))}</b>'
            f'<p>Generic ingredients: {text(ingredients, "Not available")}</p>'
            f'<p class="muted">{label_count} product labels available. '
            "Choose the matching formulation below.</p></div>"
        )
        items.append(card)

    return "".join(items)
