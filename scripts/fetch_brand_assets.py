"""Download official-site brand icons and open fonts for the local dashboard.

Run explicitly; never called by the dashboard. Origins are recorded in manifest.json.
"""
import asyncio
import io
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

import httpx
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SITES = {
    'stripe': 'https://stripe.com', 'adyen': 'https://www.adyen.com',
    'paypal-braintree': 'https://www.paypal.com', 'global-payments-worldpay': 'https://www.worldpay.com',
    '2c2p': 'https://2c2p.com', 'xendit': 'https://www.xendit.co', 'doku': 'https://www.doku.com',
    'ghl-systems': 'https://www.ghl.com', 'omise-opn': 'https://www.omise.co',
    'paymongo': 'https://www.paymongo.com', 'citi': 'https://www.citigroup.com',
    'standard-chartered': 'https://www.sc.com', 'hsbc': 'https://www.hsbc.com',
    'jpmorgan': 'https://www.jpmorgan.com', 'deutsche-bank': 'https://www.db.com',
    'dbs': 'https://www.dbs.com', 'ocbc': 'https://www.ocbc.com', 'uob': 'https://www.uobgroup.com',
    'cimb': 'https://www.cimb.com', 'bca': 'https://www.bca.co.id',
    'bdo-unibank': 'https://www.bdo.com.ph', 'kbank': 'https://www.kasikornbank.com',
    'vietcombank': 'https://www.vietcombank.com.vn',
}

class Icons(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'link' and 'icon' in a.get('rel', '').lower() and a.get('href'):
            priority = 0 if 'apple' in a.get('rel', '') else 1
            self.links.append((priority, a['href']))

async def main():
    assets = ROOT / 'dashboard_assets'
    assets.mkdir(exist_ok=True)
    manifest = {}
    async with httpx.AsyncClient(follow_redirects=True, timeout=18,
            headers={'User-Agent': 'Mozilla/5.0 (compatible; local-brand-asset-download)'}) as client:
        semaphore = asyncio.Semaphore(5)
        async def brand(key, site):
            async with semaphore:
                candidates = []
                try:
                    page = await client.get(site)
                    parser = Icons(); parser.feed(page.text)
                    candidates = [urljoin(str(page.url), href) for _, href in sorted(parser.links)]
                except httpx.HTTPError:
                    pass
                candidates += [site + '/apple-touch-icon.png', site + '/favicon.ico']
                if key == 'xendit':
                    candidates.append('https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/xendit.svg')
                if key == 'ghl-systems':
                    candidates.append('https://www.bworldonline.com/wp-content/uploads/2023/08/GHL-logo.jpg')
                for url in dict.fromkeys(candidates):
                    try:
                        response = await client.get(url); response.raise_for_status()
                        if len(response.content) > 2_000_000:
                            continue
                        if re.match(r'\s*(?:<\?xml[^>]+>\s*)?<svg\b',response.text) and not re.search(r'<(?:script|foreignObject)|<!DOCTYPE', response.text, re.I):
                            filename = key + '.svg'
                            (assets / filename).write_bytes(response.content)
                            manifest[key] = {'website': site, 'asset_url': str(response.url), 'file': filename,
                                'note': 'Brand mark for identification; trademark belongs to its owner.'}
                            print(key + ': saved vector brand mark', flush=True)
                            return
                        image = Image.open(io.BytesIO(response.content)).convert('RGBA')
                        if min(image.size) < 16:
                            continue
                        image.thumbnail((256, 256))
                        image.save(assets / (key + '.png'))
                        manifest[key] = {'website': site, 'asset_url': str(response.url), 'file': key + '.png'}
                        print(key + ': saved brand icon', flush=True)
                        return
                    except (httpx.HTTPError, OSError, ValueError):
                        continue
                manifest[key] = {'website': site, 'file': None, 'note': 'Official icon unavailable; use clearly labeled initials.'}
                print(key + ': initials fallback', flush=True)
        await asyncio.gather(*(brand(key, url) for key, url in SITES.items()))
        for family, name in [('inter', 'Inter[opsz,wght].ttf'), ('manrope', 'Manrope[wght].ttf')]:
            base = 'https://raw.githubusercontent.com/google/fonts/main/ofl/' + family + '/'
            for remote, local in [(name, family + '.ttf'), ('OFL.txt', family + '-LICENSE.txt')]:
                response = await client.get(base + remote); response.raise_for_status()
                (ROOT / 'static' / 'fonts' / local).write_bytes(response.content)
    (assets / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

if __name__ == '__main__':
    asyncio.run(main())
