"""Presentation-only HTML; the OCR/knowledge services remain frontend-independent."""
from html import escape

DOCUMENT = '<svg width="38" height="44" viewBox="0 0 38 44" fill="none" aria-hidden="true"><path d="M8 2h15l9 9v29H8zM23 2v10h9M13 21h14M13 27h14M13 33h10" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>'
EMPTY_READING = f'<div class="reading-empty">{DOCUMENT}<h3>Your reading will appear here</h3><p>Upload an image, then select Read handwriting.</p></div>'
EMPTY_MATCHES = '<p class="muted">Confirm the spelling to see generic ingredients and source labels.</p>'


def text(value, fallback='Not found'):
    return escape(str(value)) if value is not None and str(value).strip() else fallback


def render_readings(report):
    cards=[]
    for line in report['lines']:
        def joined(key):return ', '.join(line.get(key) or [])
        fields=[('Possible medicine',line.get('medicine_candidate')),('Strength as written',joined('strength_as_written')),
                ('Schedule as written',joined('frequency_as_written')),('Explicit times/day',line.get('times_per_day_as_written'))]
        fields_html=''.join(f'<div><dt>{label}</dt><dd>{text(value)}</dd></div>' for label,value in fields)
        alternatives=line.get('ocr',{}).get('alternatives',[])
        alt=''.join(f'<li>{text(item["text"])}</li>' for item in alternatives)
        cards.append(f'<article class="reading"><div class="reading-meta"><span>Region {text(line["region_id"])}</span><span class="review-label">Review required</span></div><h3 class="reading-text">{text(line["raw_text"],"No readable text")}</h3><dl>{fields_html}</dl><details><summary>Alternative readings</summary><ul>{alt or "<li>None available</li>"}</ul><p class="muted">Alternatives are possible readings, not verified medicine names.</p></details></article>')
    return ''.join(cards) or EMPTY_READING


def render_matches(report):
    if not report.get('candidates'):
        return '<div class="lookup-note"><b>No exact match found.</b><p>Check the spelling and formulation. Regional brands may be absent from these sources.</p></div>'
    items=[]
    for c in report['candidates']:
        ingredients=', '.join(c.get('ingredients') or [])
        items.append(f'<div class="source-match"><b>{text(c.get("name"))}</b><p>Generic ingredients: {text(ingredients,"Not available")}</p><p class="muted">{len(c.get("labels",[]))} product labels available. Choose the matching formulation below.</p></div>')
    return ''.join(items)
