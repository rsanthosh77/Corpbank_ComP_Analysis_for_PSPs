"""Read-only presentation model for the saved competitor research.

No research agents, provider clients, credentials or network requests are imported.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
import csv
import io
import json
import re

ROOT = Path(__file__).resolve().parent
DEFAULT_REPORT = ROOT / 'output' / 'briefing.json'
TIER_NAMES = {'tier_1_global': 'Global transaction giants',
              'tier_2_regional': 'Pan-ASEAN champions', 'tier_3_local': 'Local powerhouses'}
TIER_SHORT = {'tier_1_global': 'Global', 'tier_2_regional': 'Regional', 'tier_3_local': 'Local'}
STATUS_NAMES = {'documented_match': 'Documented match', 'conditional_match': 'Conditional match',
                'not_found': 'Evidence gap', 'not_supported': 'Confirmed non-support'}
BRANDS = {
    'stripe': ('Stripe', 'stripe.com'), 'adyen': ('Adyen', 'adyen.com'),
    'paypal-braintree': ('PayPal / Braintree', 'paypal.com'),
    'global-payments-worldpay': ('Global Payments / Worldpay', 'worldpay.com'),
    '2c2p': ('2C2P', '2c2p.com'), 'xendit': ('Xendit', 'xendit.co'), 'doku': ('DOKU', 'doku.com'),
    'ghl-systems': ('GHL Systems', 'ghl.com'), 'omise-opn': ('Omise / Opn', 'omise.co'),
    'paymongo': ('PayMongo', 'paymongo.com'), 'citi': ('Citi', 'citigroup.com'),
    'standard-chartered': ('Standard Chartered', 'sc.com'), 'hsbc': ('HSBC', 'hsbc.com'),
    'jpmorgan': ('J.P. Morgan', 'jpmorgan.com'), 'deutsche-bank': ('Deutsche Bank', 'db.com'),
    'dbs': ('DBS', 'dbs.com'), 'ocbc': ('OCBC', 'ocbc.com'), 'uob': ('UOB', 'uobgroup.com'),
    'cimb': ('CIMB', 'cimb.com'), 'bca': ('BCA', 'bca.co.id'),
    'bdo-unibank': ('BDO Unibank', 'bdo.com.ph'), 'kbank': ('Kasikornbank', 'kasikornbank.com'),
    'vietcombank': ('Vietcombank', 'vietcombank.com.vn'),
}
MARKETS = ['Singapore', 'Indonesia', 'Malaysia', 'Thailand', 'Philippines', 'Vietnam',
           'Cambodia', 'Brunei', 'Lao PDR', 'Myanmar', 'Timor-Leste', 'United States',
           'United Kingdom', 'European Union', 'Hong Kong', 'China', 'Japan', 'Australia', 'India']
MARKET_ALIASES = {'Lao PDR': ['laos', 'lao pdr'], 'United States': ['united states', 'us', 'usa'],
                  'United Kingdom': ['united kingdom', 'uk'], 'European Union': ['eu', 'european union', 'eea']}

def safe_url(value):
    if not isinstance(value, str):
        return None
    parsed = urlparse(value)
    return value if parsed.scheme in {'https', 'http'} and parsed.netloc else None

def text_matches(query, value):
    """All query words must occur; punctuation and case do not affect matching."""
    text = json.dumps(value, ensure_ascii=False).casefold()
    return all(word in text for word in re.findall(r'\w+', query.casefold()))

def market_matches(value, market):
    text = ' '.join(value if isinstance(value, list) else [str(value)]).casefold()
    return any(re.search(r'(?<!\w)' + re.escape(alias) + r'(?!\w)', text)
               for alias in MARKET_ALIASES.get(market, [market.casefold()]))

def csv_bytes(rows):
    if not rows:
        return b''
    target = io.StringIO()
    writer = csv.DictWriter(target, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        # Neutralise spreadsheet formulas from untrusted retrieved text.
        writer.writerow({key: ("'" + str(v) if str(v).lstrip().startswith(('=', '+', '-', '@')) else v)
                         for key, v in row.items()})
    return ('\ufeff' + target.getvalue()).encode('utf-8')

@dataclass
class ResearchData:
    document: dict

    def __post_init__(self):
        self.briefing = self.document.get('briefing', self.document)
        b = self.briefing
        required = {'catalogs', 'comparison_matrix', 'source_catalog', 'statistics', 'as_of_date'}
        if not required <= b.keys():
            raise ValueError('Choose a completed research briefing with catalogs, matrix and evidence.')
        self.psps = {p['id']: p for p in b['catalogs']['psps']}
        self.banks = {p['id']: {**p, 'tier': tier} for tier, banks in b['catalogs']['banks'].items() for p in banks}
        self.needs = {p['id']: p for p in b['catalogs']['needs']}
        self.products = {p['id']: p for p in b['catalogs']['capabilities']}
        self.sources = b['source_catalog']
        if len(self.psps) != len(b['catalogs']['psps']) or len(self.needs) != len(b['catalogs']['needs']):
            raise ValueError('The briefing contains duplicate PSP or need identifiers.')
        self.assessments = {}
        for row in b['comparison_matrix']['need_rows']:
            for cell in row['cells']:
                key = (cell['bank_id'], cell['need_id'])
                if key in self.assessments or key[0] not in self.banks or key[1] not in self.needs:
                    raise ValueError('The comparison matrix contains duplicate or unknown references.')
                if cell['status'] not in STATUS_NAMES:
                    raise ValueError('Unknown assessment status.')
                self.assessments[key] = cell
        if len(self.assessments) != len(self.banks) * len(self.needs):
            raise ValueError('The saved briefing has an incomplete comparison matrix.')
        self.reviews = defaultdict(list)
        def walk(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    if key == 'source_reviews' and isinstance(child, list):
                        for review in child:
                            if review not in self.reviews[review['evidence_id']]:
                                self.reviews[review['evidence_id']].append(review)
                    else:
                        walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)
        walk(b.get('research_methods', {}))

    def name(self, identity):
        return BRANDS.get(identity, (self.banks.get(identity, self.psps.get(identity, {})).get('name', identity), ''))[0]

    def needs_filtered(self, query='', psps=(), markets=(), basis=(), themes=()):
        return [need for need in self.needs.values()
                if (not psps or set(psps) & set(need['psp_ids']))
                and (not markets or any(market_matches(need.get('jurisdictions', []), m) for m in markets))
                and (not basis or need['basis'] in basis)
                and (not themes or set(themes) & set(need['themes']))
                and text_matches(query, need)]

    def products_filtered(self, bank_id, query='', markets=(), availability=(), value_add=False):
        return [p for p in self.products.values() if p['bank_id'] == bank_id
                and (not markets or any(market_matches(p.get('jurisdictions', []), m) for m in markets))
                and (not availability or p['availability'] in availability)
                and (not value_add or p['is_value_add']) and text_matches(query, p)]

    def counts(self, bank_id=None, need_ids=None):
        allowed = set(need_ids) if need_ids is not None else set(self.needs)
        counts = Counter(c['status'] for (bank, need), c in self.assessments.items()
                         if (bank_id is None or bank == bank_id) and need in allowed)
        return {status: counts[status] for status in STATUS_NAMES}

    def source_grade(self, identity):
        kinds = {r['source_type'] for r in self.reviews.get(identity, [])}
        return next(iter(kinds)) if len(kinds) == 1 else ('mixed' if kinds else 'not_reviewed')

    def source_rows(self, query='', grades=(), topics=(), dates=()):
        return [s for identity, s in self.sources.items()
                if (not grades or self.source_grade(identity) in grades)
                and (not topics or s['topic'] in topics)
                and (not dates or s['date_status'] in dates)
                and text_matches(query, {'source': s, 'reviews': self.reviews.get(identity, [])})]

    def need_label(self, need):
        return ' / '.join(self.name(p) for p in need['psp_ids']) + ' · ' + need['description'][:100]

def read_report(path=DEFAULT_REPORT):
    return ResearchData(json.loads(Path(path).read_text(encoding='utf-8')))
