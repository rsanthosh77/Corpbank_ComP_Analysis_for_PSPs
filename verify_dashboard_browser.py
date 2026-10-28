"""Real-browser checks. Start the dashboard first; uses installed Google Chrome."""
from pathlib import Path
import argparse
import json
from playwright.sync_api import sync_playwright

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8501')
    parser.add_argument('--channel', default='chrome')
    args = parser.parse_args()
    output = Path('output/dashboard-preview')
    output.mkdir(parents=True, exist_ok=True)
    checks = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.channel, headless=True)
        page = browser.new_page(viewport={'width':1512,'height':1800}, device_scale_factor=1)
        errors, external = [], []
        page.on('pageerror', lambda err: errors.append(str(err)))
        page.on('request', lambda req: external.append(req.url) if req.url.startswith(('https://','http://')) and not req.url.startswith(args.url) else None)
        page.goto(args.url)
        page.get_by_role('button',name='Explore PSP needs',exact=True).wait_for(timeout=45000)
        page.wait_for_timeout(1000)
        for label, filename in [('Overview','overview'),('PSP needs','needs'),
                                ('Bank comparison','comparison'),('Bank profiles','bank-profile'),
                                ('Evidence library','evidence')]:
            if label!='Overview':
                page.get_by_role('button',name=label,exact=True).click()
                page.wait_for_timeout(1000)
            assert page.locator('[data-testid="stException"]').count()==0, label
            assert not page.evaluate('document.documentElement.scrollWidth > innerWidth'), label
            broken=page.locator('img').evaluate_all('(imgs)=>imgs.filter(i=>!i.complete||i.naturalWidth===0).length')
            assert broken==0,(label,'broken images')
            if label=='Overview':
                assert page.locator('img[alt$=" logo"]').count()==23
                heights=page.locator('.tier-card').evaluate_all('(els)=>els.map(e=>e.getBoundingClientRect().height)')
                assert max(heights)-min(heights)<2,heights
            if label=='Bank comparison':
                heights=page.locator('.comparison-summary').evaluate_all('(els)=>els.map(e=>e.getBoundingClientRect().height)')
                assert len(heights)==3 and max(heights)-min(heights)<2,heights
            if label=='PSP needs':
                with page.expect_download() as download:
                    page.get_by_role('button',name='Export filtered needs',exact=True).click()
                assert download.value.suggested_filename=='psp-needs.csv'
                assert b'Requirement' in download.value.path().read_bytes()
            page.screenshot(path=str(output/(filename+'.png')),full_page=True)
            checks.append(label)
        page.get_by_role('button',name='Overview',exact=True).click()
        page.wait_for_timeout(700)
        page.set_viewport_size({'width':390,'height':844})
        page.wait_for_timeout(700)
        collapse=page.locator('[data-testid="stSidebarCollapseButton"] button')
        if collapse.count() and collapse.is_visible():
            collapse.click()
            page.wait_for_timeout(400)
        assert not page.evaluate('document.documentElement.scrollWidth > innerWidth')
        page.screenshot(path=str(output/'mobile.png'),full_page=True)
        assert page.evaluate('document.fonts.check("400 14px Inter")')
        assert page.evaluate('document.fonts.check("600 14px Manrope")')
        assert not errors,errors
        assert not external,external
        browser.close()
    (output/'checks.json').write_text(json.dumps({'views':checks,'mobile':'passed',
        'fonts':'local','logos':23,'browser_errors':errors,'external_requests':external},indent=2)+'\n')
    print('PASS: all five views, 23 logos, aligned competitor cards, CSV download, local fonts, mobile layout and zero external requests')

if __name__=='__main__':
    main()
