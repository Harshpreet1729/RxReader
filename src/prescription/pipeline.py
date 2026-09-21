from .layout import detect_regions
from .extraction import extract_line


def process_image(image, reader, mode='word', template=None):
    regions = detect_regions(image, mode, template)
    lines = []
    for region in regions:
        reading = reader.read(image.crop(region['box']))
        result = extract_line(reading['text'])
        result.update(region_id=region['id'], box=region['box'], ocr=reading)
        if reading.get('possibly_truncated'):
            result['review_reasons'].append('Generation reached its length limit; inspect the crop.')
        lines.append(result)
    return {'status': 'needs_review', 'mode': mode, 'lines': lines,
            'limitations': ['Candidate text is not a confirmed medicine.',
                           'Schedules are copied only from the same region; separate lines are not automatically linked.',
                           'Word-crop training data does not validate full-page accuracy.']}
