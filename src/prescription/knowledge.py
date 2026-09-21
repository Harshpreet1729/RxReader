"""Source retrieval after a user confirms spelling; no unsourced medical generation."""
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from urllib.parse import quote
import requests

RX = 'https://rxnav.nlm.nih.gov/REST'
DAILY = 'https://dailymed.nlm.nih.gov/dailymed/services/v2'


class MedicineKnowledge:
    def __init__(self, session=None):
        self.session = session or requests.Session()

    def _get(self, url, params=None):
        response = self.session.get(url, params=params, timeout=(5, 25))
        response.raise_for_status()
        return response

    def lookup(self, confirmed_name):
        name = confirmed_name.strip()
        if not name or len(name) > 200:
            raise ValueError('Enter a confirmed medicine/product name (1-200 characters).')
        data = self._get(RX + '/rxcui.json', {'name': name, 'search': 0}).json()
        ids = (data.get('idGroup') or {}).get('rxnormId', []) or []
        result = {'query': name, 'match_method': 'RxNorm exact name/synonym only', 'status': 'unresolved',
                  'retrieved_at': datetime.now(timezone.utc).isoformat(), 'candidates': [],
                  'coverage_note': 'US terminology and labeling; local brands may be absent. No fuzzy substitutions.'}
        for identifier in ids[:10]:
            if not str(identifier).isdigit():
                continue
            properties = self._get(f'{RX}/rxcui/{identifier}/properties.json').json().get('properties') or {}
            related = self._get(f'{RX}/rxcui/{identifier}/related.json', {'tty': 'IN'}).json()
            ingredients = [p['name'] for group in (related.get('relatedGroup') or {}).get('conceptGroup', []) for p in group.get('conceptProperties', [])]
            if properties.get('tty') == 'IN':
                ingredients = [properties['name']]
            spls = self._get(DAILY + '/spls.json', {'rxcui': identifier, 'pagesize': 10}).json()
            labels = [{'setid': s['setid'], 'title': s['title'], 'published_date': s.get('published_date'),
                       'url': 'https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=' + quote(s['setid'])} for s in spls.get('data', [])]
            result['candidates'].append({'rxcui': identifier, 'name': properties.get('name'), 'term_type': properties.get('tty'),
                                         'ingredients': ingredients, 'labels': labels,
                                         'source': f'https://rxnav.nlm.nih.gov/REST/rxcui/{identifier}/properties.json'})
        if result['candidates']:
            result['status'] = 'select_matching_product_label'
        return result

    def indications(self, confirmed_name, setid):
        # Require a label belonging to the exact lookup result; do not select a random formulation.
        if not re.fullmatch(r'[0-9a-fA-F-]{36}', setid):
            raise ValueError('Invalid DailyMed setid.')
        lookup = self.lookup(confirmed_name)
        label = next((label for c in lookup['candidates'] for label in c['labels'] if label['setid'] == setid), None)
        if not label:
            raise ValueError('Select a label returned for this confirmed product name.')
        xml = self._get(f'{DAILY}/spls/{setid}.xml').content
        if len(xml) > 15_000_000:
            raise ValueError('Label XML too large.')
        root = ET.fromstring(xml)
        ns = {'h': 'urn:hl7-org:v3'}
        excerpts = []
        for section in root.findall('.//h:section', ns):
            code = section.find('h:code', ns)
            # Prescription indications and OTC uses section codes.
            if code is not None and code.get('code') in {'34067-9', '34667-7'}:
                node = section.find('h:text', ns)
                if node is not None:
                    excerpts.append(' '.join(' '.join(node.itertext()).split()))
        return {'confirmed_name': confirmed_name, 'label': label, 'uses_as_published': excerpts,
                'status': 'source_text' if excerpts else 'uses_section_unavailable',
                'retrieved_at': datetime.now(timezone.utc).isoformat(),
                'note': 'General labeled uses, not a diagnosis, cure guarantee, or personal dosing recommendation.'}
