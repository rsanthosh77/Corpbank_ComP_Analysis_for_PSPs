"""Cash Atlas: a local, read-only workspace for the completed research briefing.

Start from the project directory: uv run streamlit run dashboard.py
Optional report override: CASH_RESEARCH_REPORT=/path/to/briefing.json
"""
from collections import Counter
from datetime import date
from functools import lru_cache
from html import escape
from pathlib import Path
from urllib.parse import urlparse
import base64
import json
import os

import pandas as pd
import streamlit as st

from dashboard_data import (
    BRANDS, DEFAULT_REPORT, MARKETS, ROOT, STATUS_NAMES, TIER_NAMES, TIER_SHORT,
    ResearchData, csv_bytes, safe_url,
)

st.set_page_config(page_title='Cash Atlas · Research workspace', page_icon=str(ROOT / 'dashboard_assets' / 'workspace.svg'),
                   layout='wide', initial_sidebar_state='expanded')
st.html(ROOT / 'dashboard_assets' / 'theme.css')

ICONS = {
    'layers': '<path d="m12 3 9 5-9 5-9-5 9-5Z"/><path d="m3 12 9 5 9-5M3 16l9 5 9-5"/>',
    'bank': '<path d="m3 9 9-6 9 6H3ZM3 21h18M5 10v8m5-8v8m4-8v8m5-8v8"/>',
    'companies': '<rect x="4" y="3" width="11" height="18" rx="1"/><path d="M15 9h5v12M7 7h5M7 11h5M7 15h5M8 21v-3h3v3"/>',
    'needs': '<path d="M8 5H5v16h14V5h-3M9 3h6v4H9zM8 12h8M8 16h6"/>',
    'matrix': '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M3 15h18M9 3v18M15 3v18"/>',
    'info': '<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10v1"/>',
}

def icon(name, size=18):
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="#4d9d8c" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>'
    encoded = base64.b64encode(svg.encode()).decode()
    return f'<img width="{size}" height="{size}" src="data:image/svg+xml;base64,{encoded}" alt="" aria-hidden="true">'

def html(value):
    st.html(value)

@lru_cache(maxsize=64)
def logo(identity, large=False):
    path = ROOT / 'dashboard_assets' / f'{identity}.png'
    if not path.is_file():
        path = ROOT / 'dashboard_assets' / f'{identity}.svg'
    name = BRANDS.get(identity, (identity, ''))[0]
    classes = 'logo large' if large else 'logo'
    if path.is_file():
        encoded = base64.b64encode(path.read_bytes()).decode()
        media = 'image/svg+xml' if path.suffix == '.svg' else 'image/png'
        return f'<img class="{classes}" src="data:{media};base64,{encoded}" alt="{escape(name)} logo">'
    initials = ''.join(word[0] for word in name.split()[:2]).upper()
    return f'<span class="{classes} initials" aria-label="{escape(name)} initials">{escape(initials)}</span>'

def badge(value, label=None):
    return f'<span class="badge {escape(value)}">{escape(label or STATUS_NAMES.get(value, value.replace("_", " ").title()))}</span>'

def heading(title, subtitle='', eyebrow='RESEARCH EXPLORER'):
    html(f'<div class="page-title"><div class="eyebrow">{escape(eyebrow)}</div><h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>')

def section(title, meta=''):
    html(f'<div class="section-heading"><h2>{escape(title)}</h2><span>{escape(meta)}</span></div>')

def empty(title='No matching research', detail='Try a broader search or remove a filter.'):
    html(f'<div class="empty-state"><strong>{escape(title)}</strong>{escape(detail)}</div>')

def caveat():
    html(f'<div class="note-card">{icon("info")}<div><strong>Evidence first. Conclusions second.</strong><p>Conditional matches require verification. An evidence gap means insufficient retrieved evidence, not proof that a bank lacks a service.</p></div></div>')

@st.cache_data(show_spinner=False)
def load_snapshot(path, modified, size):
    return json.loads(Path(path).read_text(encoding='utf-8'))

report_path = Path(os.environ.get('CASH_RESEARCH_REPORT', str(DEFAULT_REPORT))).expanduser().resolve()
try:
    stat = report_path.stat()
    document = load_snapshot(str(report_path), stat.st_mtime_ns, stat.st_size)
    data = ResearchData(document)
except (OSError, ValueError, KeyError, TypeError) as exc:
    st.error(f'The saved research briefing could not be opened: {exc}')
    st.info('Place a completed briefing at output/briefing.json, or set CASH_RESEARCH_REPORT to its local path.')
    st.stop()
b = data.briefing
snapshot_date = date.fromisoformat(b['as_of_date']).strftime('%d %b %Y')

def navigate(view, **selection):
    st.session_state['view'] = view
    st.session_state.update(selection)
    st.rerun()

def evidence(evidence_ids, key):
    ids = list(dict.fromkeys(evidence_ids))
    if not ids:
        st.caption('No source evidence is attached to this record. Treat the statement as an unresolved research gap.')
        return
    for index, identity in enumerate(ids):
        source = data.sources.get(identity)
        if not source:
            st.warning('A referenced source is unavailable in this snapshot.')
            continue
        url = safe_url(source.get('url'))
        title = source.get('title') or (urlparse(url).netloc if url else 'Retrieved source')
        grade = data.source_grade(identity).replace('_', ' ').title()
        publication = source.get('published_at') or 'Publication date unconfirmed'
        with st.expander(f'{index + 1}. {title}', expanded=False):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.caption(f'{grade} · {source.get("topic", "general").title()} · {publication}')
            with c2:
                if url:
                    st.link_button('Open source', url, icon=':material/open_in_new:', width='stretch')
            st.text(source.get('excerpt', 'No excerpt was captured.'))
            reviews = data.reviews.get(identity, [])
            if reviews:
                st.caption('Source review · model assessment, not independent verification')
                for review in reviews:
                    st.write(review['credibility_reason'])
                    st.caption('Supporting quote')
                    st.text(review['supporting_quote'])
            else:
                st.caption('This retrieved source has no retained credibility review. Retrieval alone does not establish credibility.')
            st.caption(f'Date status: {source.get("date_status", "unknown")} · Retrieved: {source.get("retrieved_at", "unknown")}')

def product_detail(product, key):
    html('<div class="badge-line">' + badge(product['availability']) + badge(product['psp_eligibility'])
         + (badge('explicit', 'Value-add') if product['is_value_add'] else '') + '</div>')
    st.write(product['description'])
    st.caption('Stated jurisdictions · product-level evidence, not headquarters')
    st.write(' · '.join(product.get('jurisdictions', [])) or 'Not established')
    st.caption('Legal entities')
    st.write(' · '.join(product.get('legal_entities', [])) or 'Not established')
    if product.get('value_add_rationale'):
        st.caption('Value-add rationale · does not establish market exclusivity')
        st.write(product['value_add_rationale'])
    if product['constraints']:
        st.caption('Conditions and limits')
        for condition in product['constraints']:
            st.write('• ' + condition)
    with st.expander(f'Supporting evidence · {len(product["evidence_ids"])} records'):
        evidence(product['evidence_ids'], key)

def need_detail(need, key):
    html('<div class="badge-line">' + badge(need['basis']) + ''.join(badge('theme', data.name(p)) for p in need['psp_ids']) + '</div>')
    html(f'<div class="detail-heading">{escape(need["description"])}</div>')
    html('<div class="badge-line">' + ''.join(badge('theme', t) for t in need['themes']) + '</div>')
    st.caption('Stated jurisdictions')
    st.write(' · '.join(need.get('jurisdictions', [])) or 'Not established')
    with st.expander('Why this is a need', expanded=False):
        st.write(need['rationale'])
    with st.expander(f'Supporting evidence · {len(need["evidence_ids"])} records'):
        evidence(need['evidence_ids'], key)

def filters(prefix, include_themes=True):
    c1, c2, c3 = st.columns([2, 1.4, 1.15])
    with c1:
        query = st.text_input('Search research', placeholder='Try “tokenised”, “virtual accounts” or “FX”…', key=prefix+'_q', icon=':material/search:')
    with c2:
        psps = st.multiselect('Payment providers', list(data.psps), format_func=data.name, key=prefix+'_psps', placeholder='All PSPs')
    with c3:
        markets = st.multiselect('Need jurisdictions', MARKETS, key=prefix+'_markets', placeholder='All jurisdictions')
    c1, c2 = st.columns([1, 3])
    with c1:
        basis = st.multiselect('Need basis', ['explicit', 'inferred'], format_func=str.title, key=prefix+'_basis', placeholder='Explicit + inferred')
    with c2:
        themes = st.multiselect('Research themes', sorted({t for n in data.needs.values() for t in n['themes']}),
                               key=prefix+'_themes', placeholder='All themes, including emerging technologies') if include_themes else []
    return data.needs_filtered(query, psps, markets, basis, themes)

def status_bars(counts):
    total = sum(counts.values())
    rows = ''
    for status in ['conditional_match', 'not_found', 'documented_match', 'not_supported']:
        count = counts[status]
        percent = 100 * count / total if total else 0
        rows += f'<div class="status-row"><span class="status-label"><i class="status-dot {status}"></i>{STATUS_NAMES[status]}</span><strong>{count:,}<span style="font-weight:400;color:#95a1ae;margin-left:8px">{percent:.0f}%</span></strong></div><div class="track"><div class="{status}" style="width:{percent:.2f}%"></div></div>'
    return rows

def overview():
    html(f'<div class="hero"><div><div class="hero-kicker">CORPORATE CASH INTELLIGENCE</div><h1>A clearer view of<br>the <em>cash landscape.</em></h1><p>Explore what payment providers need, what banks offer,<br>and the evidence connecting them.</p></div><div class="hero-art"><div class="orbit"></div><div class="orbit two"></div><div class="art-core">{icon("layers",34)}</div><div class="orbit-dot one"></div><div class="orbit-dot two"></div><div class="orbit-dot three"></div><span class="mini-label one">PSP NEEDS</span><span class="mini-label two">BANK CAPABILITIES</span></div></div>')
    statistics = b['statistics']
    cards = [('Payment providers', len(data.psps), 'Global + ASEAN discovery', 'companies'),
             ('Banks researched', len(data.banks), 'Across three competitor tiers', 'bank'),
             ('PSP needs', len(data.needs), f'{statistics["need_basis_counts"].get("explicit",0)} explicit · {statistics["need_basis_counts"].get("inferred",0)} inferred', 'needs'),
             ('Bank–need assessments', len(data.assessments), 'Complete comparison matrix', 'matrix')]
    html('<div class="metric-grid">' + ''.join(f'<div class="metric-card"><div class="metric-top">{title}<span class="metric-icon">{icon(symbol)}</span></div><div class="metric-number">{count:,}</div><div class="metric-foot">{escape(foot)}</div></div>' for title,count,foot,symbol in cards) + '</div>')
    caveat()
    left, right = st.columns([1.4, 1], gap='large')
    with left:
        section('Your research, connected', 'A practical place to start')
        html('<div class="panel">' + ''.join(f'<div class="research-step"><span class="step-number">0{i}</span><div><div class="step-title">{title}</div><div class="step-text">{description}</div></div></div>' for i,(title,description) in enumerate([
            ('Start with the PSP need', 'Find a provider, market or capability. Keep explicit demand separate from inferred requirements.'),
            ('Compare the banking response', 'Put two or three banks side by side. Read fit, eligibility and jurisdiction conditions together.'),
            ('Follow the evidence', 'Inspect the retrieved source, publication date and credibility review before drawing a conclusion.')],1)) + '</div>')
        c1,c2 = st.columns(2)
        with c1:
            if st.button('Explore PSP needs', icon=':material/manage_search:', width='stretch', type='primary', key='start_needs'):
                navigate('PSP needs')
        with c2:
            if st.button('Compare banks', icon=':material/compare_arrows:', width='stretch', key='start_compare'):
                navigate('Bank comparison')
    with right:
        section('What the evidence establishes', 'Whole snapshot')
        html('<div class="panel"><div class="panel-title">Assessment coverage</div>' + status_bars(data.counts()) + '<div class="panel-foot">Counts describe the researched sample. They are not market share, bank rankings or proof of product absence.</div></div>')
    section('The competitor landscape', f'{len(data.banks)} banks · three distinct lenses')
    tier_descriptions = {'tier_1_global': 'Global networks and cross-border infrastructure',
                         'tier_2_regional': 'Regional connectivity and ASEAN cash franchises',
                         'tier_3_local': 'Domestic rails and local cash ecosystems'}
    for index,(column,(tier,banks)) in enumerate(zip(st.columns(3,gap='medium'),b['catalogs']['banks'].items()),1):
        with column:
            rows = ''.join(f'<div class="brand-row">{logo(bank["id"])}<div><div class="brand-name">{escape(data.name(bank["id"]))}</div><div class="brand-meta">{escape(bank["headquarters_country"])}</div></div></div>' for bank in banks)
            html(f'<div class="tier-card"><div class="tier-eyebrow"><span>TIER 0{index}</span><span>{len(banks)} BANKS</span></div><h3>{TIER_NAMES[tier]}</h3><p>{tier_descriptions[tier]}</p><div class="brand-list">{rows}</div></div>')
    st.caption('Country labels above indicate headquarters. They do not establish product availability or full country coverage.')
    section('Payment providers in focus', 'Open the needs explorer to investigate')
    html('<div class="brand-tiles">' + ''.join(f'<div class="brand-tile"><div class="brand-row">{logo(p)}<div><div class="brand-name">{escape(data.name(p))}</div><div class="brand-meta">{sum(p in n["psp_ids"] for n in data.needs.values())} researched needs</div></div></div></div>' for p in data.psps) + '</div>')

def needs_explorer():
    heading('Start with the need.', 'Search every researched requirement, from daily cash operations to emerging financial infrastructure.')
    needs = filters('needs')
    section('PSP requirements', f'{len(needs)} of {len(data.needs)} needs')
    if not needs:
        empty();return
    rows = [{'PSP': ' / '.join(data.name(p) for p in n['psp_ids']), 'Requirement': n['description'],
             'Basis': n['basis'].title(), 'Jurisdictions': ' · '.join(n.get('jurisdictions', [])),
             'Sources': len(n['evidence_ids'])} for n in needs]
    event = st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True, height=340,
        on_select='rerun', selection_mode='single-row', key='needs_table',
        column_config={'PSP': st.column_config.TextColumn(width='small'),
                       'Requirement': st.column_config.TextColumn(width='large'),
                       'Jurisdictions': st.column_config.TextColumn(width='medium')})
    c1,c2 = st.columns([3,1])
    with c1:
        st.caption('Select a row to read the complete requirement and its supporting evidence.')
    with c2:
        st.download_button('Export filtered needs',csv_bytes(rows),'psp-needs.csv','text/csv',icon=':material/download:',width='stretch',key='needs_csv')
    positions = event.selection.rows
    selected = needs[positions[0]] if positions and positions[0] < len(needs) else needs[0]
    section('Requirement detail')
    with st.container(border=True,key='requirement_detail'):
        need_detail(selected, 'selected_need')
        st.divider()
        counts = data.counts(need_ids=[selected['id']])
        st.caption(f'{counts["conditional_match"]} conditional matches · {counts["not_found"]} evidence gaps across {len(data.banks)} banks')
        if st.button('Compare banks for this need', icon=':material/compare_arrows:', key='need_to_compare',type='primary'):
            navigate('Bank comparison', focused_need=selected['id'], compare_psps=selected['psp_ids'],
                     compare_markets=[], compare_q='', compare_basis=[])
    with st.expander('PSP discovery profiles & selection evidence'):
        psp_id = st.selectbox('Provider profile', list(data.psps), format_func=data.name, key='psp_profile')
        psp = data.psps[psp_id]
        html(f'<div class="comparison-brand">{logo(psp_id,True)}<div><div class="brand-name">{escape(data.name(psp_id))}</div><div class="brand-meta">Discovery profile</div></div></div>')
        st.write(psp['selection_rationale'])
        for criterion,review in psp.get('rubric',{}).items():
            st.markdown('**'+criterion.replace('_',' ').title()+' · '+review['rating'].title()+'**')
            st.write(review['assessment'])
            for gap in review.get('gaps',[]):st.caption('Open question: '+gap)
        evidence(psp['evidence_ids'],'psp_profile_sources')

def comparison():
    heading('Compare the banking response.', 'Select two or three banks. Explore the full matrix, then inspect the conditions behind an individual assessment.')
    bank_ids = st.multiselect('Banks to compare',list(data.banks), default=['citi','hsbc','bca'] if {'citi','hsbc','bca'} <= data.banks.keys() else list(data.banks)[:3],
                             format_func=data.name, max_selections=3,key='compare_banks')
    needs = filters('compare',include_themes=False)
    if not bank_ids or not needs:
        empty('Choose banks and matching needs', 'Select up to three banks and broaden your need filters.');return
    columns = st.columns(len(bank_ids), gap='medium')
    for column, bank_id in zip(columns,bank_ids):
        counts=data.counts(bank_id,[n['id'] for n in needs])
        with column:
            html(f'<div class="panel comparison-summary"><div class="comparison-brand">{logo(bank_id,True)}<div><div class="brand-name">{escape(data.name(bank_id))}</div><div class="brand-meta">{TIER_NAMES[data.banks[bank_id]["tier"]]}</div></div></div><div class="source-summary"><span class="badge conditional_match">{counts["conditional_match"]} conditional</span><span class="badge not_found">{counts["not_found"]} evidence gaps</span></div></div>')
    st.caption('Counts apply to the filtered needs below and describe research outcomes, not relative bank quality.')
    rows=[]
    for need in needs:
        rows.append({'PSP':' / '.join(data.name(p) for p in need['psp_ids']), 'Requirement':need['description'],
                     **{data.name(bank_id):STATUS_NAMES[data.assessments[bank_id,need['id']]['status']] for bank_id in bank_ids}})
    section('Comparison matrix',f'{len(needs)} needs × {len(bank_ids)} banks')
    event=st.dataframe(pd.DataFrame(rows),width='stretch',hide_index=True,height=330,on_select='rerun',
        selection_mode='single-row',key='comparison_table_'+st.session_state.get('focused_need','all'),column_config={
            'Requirement':st.column_config.TextColumn(width='large'),'PSP':st.column_config.TextColumn(width='small')})
    st.caption('Select a matrix row, or choose a requirement below, to inspect the full comparison.')
    positions=event.selection.rows
    focused=st.session_state.get('focused_need')
    selected_id=needs[positions[0]]['id'] if positions and positions[0]<len(needs) else (focused if focused in {n['id'] for n in needs} else needs[0]['id'])
    options=[n['id'] for n in needs]
    need_id=st.selectbox('Requirement to inspect',options,index=options.index(selected_id),
        format_func=lambda identity:data.need_label(data.needs[identity]),key='compare_need_'+selected_id[:18])
    need=data.needs[need_id]
    with st.container(border=True,key='comparison_need_detail'):need_detail(need,'comparison_need')
    for column,bank_id in zip(st.columns(len(bank_ids),gap='medium'),bank_ids):
        cell=data.assessments[bank_id,need_id]
        with column:
            with st.container(border=True,key='assessment_card_'+bank_id):
                html(f'<div class="comparison-brand">{logo(bank_id)}<div class="brand-name">{escape(data.name(bank_id))}</div></div>'+badge(cell['status']))
                html(f'<div class="compare-rationale">{escape(cell["rationale"])}</div>')
                with st.expander(f'Conditions & limits · {len(cell["constraints"])}',expanded=False):
                    if not cell['constraints']:st.caption('No conditions were captured. This does not establish eligibility.')
                    for condition in cell['constraints']:st.write('• '+condition)
                for product_id in cell['capability_ids']:
                    product=data.products[product_id]
                    with st.expander(product['product_name']):product_detail(product,'compare_'+product_id)
                with st.expander(f'Assessment evidence · {len(cell["evidence_ids"])}'):
                    evidence(cell['evidence_ids'],'compare_'+bank_id)
    st.download_button('Export this comparison',csv_bytes(rows),'bank-comparison.csv','text/csv',icon=':material/download:',key='comparison_csv')

def bank_profiles():
    heading('Know the competitor.', 'Explore each bank’s cash products, value-adds and the limits of the available evidence.')
    c1,c2=st.columns([1,2])
    with c1:
        tier=st.selectbox('Competitor tier',['all',*TIER_NAMES],format_func=lambda k:'All three tiers' if k=='all' else TIER_NAMES[k],key='profile_tier')
    options=[identity for identity,bank in data.banks.items() if tier=='all' or bank['tier']==tier]
    with c2:bank_id=st.selectbox('Bank',options,format_func=data.name,key='profile_bank_'+tier)
    bank=data.banks[bank_id];counts=data.counts(bank_id)
    html(f'<div class="panel"><div class="comparison-brand">{logo(bank_id,True)}<div><div class="brand-name">{escape(data.name(bank_id))}</div><div class="brand-meta">{TIER_NAMES[bank["tier"]]} · Headquarters: {escape(bank["headquarters_country"])}</div></div></div><p class="detail-body">{escape(bank["tier_rationale"])}</p></div>')
    left,right=st.columns([2,1],gap='large')
    with left:
        with st.expander('Discovery criteria & bank selection evidence'):
            st.write(bank['selection_rationale'])
            for criterion,review in bank.get('rubric',{}).items():
                st.markdown('**'+criterion.replace('_',' ').title()+' · '+review['rating'].title()+'**');st.write(review['assessment'])
                for gap in review.get('gaps',[]):st.caption('Open question: '+gap)
            evidence(bank['evidence_ids'],'bank_discovery')
        c1,c2=st.columns([2,1])
        with c1:query=st.text_input('Find a product',placeholder='API, liquidity, tokenised deposits…',icon=':material/search:',key='products_q')
        with c2:value_add=st.toggle('Value-adds only',key='value_add_only',help='Value-add means a differentiated offering in this research; it does not establish exclusivity.')
        c1,c2=st.columns(2)
        with c1:markets=st.multiselect('Product jurisdictions',MARKETS,key='product_markets',placeholder='All stated jurisdictions')
        with c2:availability=st.multiselect('Product maturity',['live','pilot','announced','unknown','discontinued'],format_func=str.title,key='product_maturity',placeholder='All maturity levels')
        products=data.products_filtered(bank_id,query,markets,availability,value_add)
        section('Cash capabilities',f'{len(products)} products')
        if not products:empty()
        for product in products:
            with st.expander(product['product_name']):product_detail(product,'profile_'+product['id'])
        with st.expander('Research method & remaining questions'):
            summary=next((s for s in b['research_methods'].get('capability_research',[]) if s['bank_id']==bank_id),{})
            st.write(summary.get('methodology','Method not captured.'))
            for note in summary.get('coverage_notes',[]):st.write('• '+note)
    with right:
        html('<div class="panel"><div class="panel-title">Fit against all PSP needs</div>'+status_bars(counts)+'<div class="panel-foot">PSP eligibility, live availability and country coverage must be verified at product level.</div></div>')
        if st.button('Compare this bank',width='stretch',icon=':material/compare_arrows:',key='profile_compare'):
            others=[x for x in data.banks if x!=bank_id][:1]
            navigate('Bank comparison',compare_banks=[bank_id,*others])
        st.caption('A bank’s headquarters or group footprint does not establish product availability in a particular market.')

def evidence_library():
    heading('Go back to the source.', 'Inspect retrieved evidence, dates and credibility reviews. No live searches are performed here.')
    c1,c2,c3=st.columns([2,1.3,1])
    with c1:query=st.text_input('Search sources & excerpts',placeholder='Publisher, company, product or keyword…',icon=':material/search:',key='evidence_q')
    with c2:grades=st.multiselect('Source review',sorted({data.source_grade(i) for i in data.sources}),format_func=lambda s:s.replace('_',' ').title(),key='source_grades',placeholder='All source types')
    with c3:topics=st.multiselect('Search channel',sorted({s['topic'] for s in data.sources.values()}),format_func=str.title,key='source_topics',placeholder='Web + news')
    dates=st.multiselect('Publication-date status',sorted({s['date_status'] for s in data.sources.values()}),format_func=lambda s:s.replace('_',' ').title(),key='source_dates',placeholder='All dates, including uncertain')
    sources=data.source_rows(query,grades,topics,dates)
    section('Evidence library',f'{len(sources):,} of {len(data.sources):,} retrieval records')
    st.caption('A retained review is a model assessment, not independent verification. Different retrievals may share a URL. Undated sources are not automatically recent.')
    if not sources:empty();return
    page_count=max(1,(len(sources)+11)//12)
    page=st.selectbox('Evidence page',list(range(1,page_count+1)),format_func=lambda n:f'Page {n} of {page_count}',key=f'evidence_page_{len(sources)}')
    for source in sources[(page-1)*12:page*12]:
        source_id=source['id'];url=safe_url(source['url']);domain=urlparse(url).netloc.removeprefix('www.') if url else 'Unavailable URL'
        with st.container(border=True,key='source_card_'+source_id):
            c1,c2=st.columns([5,1])
            with c1:
                html(f'<div class="source-domain">{escape(domain)}</div><div class="detail-heading">{escape(source["title"] or domain)}</div>')
            with c2:
                if url:st.link_button('Open source',url,icon=':material/open_in_new:',width='stretch')
            grade=data.source_grade(source_id)
            html('<div class="badge-line">'+badge(grade)+badge(source['topic'])+badge('date',source.get('published_at') or 'Date unconfirmed')+'</div>')
            with st.expander('Read excerpt & credibility review'):
                st.text(source['excerpt'])
                for review in data.reviews.get(source_id,[]):
                    st.caption(review['publisher']+' · '+review['source_type'].replace('_',' '));st.write(review['credibility_reason']);st.text(review['supporting_quote'])
                if not data.reviews.get(source_id):st.caption('No retained source review. Do not treat this retrieval as verified evidence.')
                st.caption('Retrieved: '+source.get('retrieved_at','Unknown'))
    export=[{'Title':s['title'],'URL':s['url'],'Publication date':s.get('published_at') or '',
             'Date status':s['date_status'],'Source review':data.source_grade(s['id']),'Channel':s['topic']} for s in sources]
    st.download_button('Export filtered source list',csv_bytes(export),'research-sources.csv','text/csv',icon=':material/download:',key='sources_csv')
    with st.expander('Geographic coverage & research limitations'):
        st.write('Coverage below records discovery review outcomes; it does not establish complete product coverage in every ASEAN market.')
        for stage,rows in b['scope'].get('reported_market_coverage',{}).items():
            st.markdown('**'+stage.replace('_',' ').title()+'**')
            st.dataframe(pd.DataFrame(rows),width='stretch',hide_index=True)
        for limitation in b['limitations']:st.write('• '+limitation)

VIEWS={'Overview':(':material/space_dashboard:',overview), 'PSP needs':(':material/manage_search:',needs_explorer),
       'Bank comparison':(':material/compare_arrows:',comparison), 'Bank profiles':(':material/account_balance:',bank_profiles),
       'Evidence library':(':material/library_books:',evidence_library)}
st.session_state.setdefault('view','Overview')
with st.sidebar:
    html(f'<div class="workspace-brand"><span class="brand-symbol">{icon("layers",23)}</span><div><div class="brand-title">Cash Atlas<span style="color:#83d8be">.</span></div><div class="brand-sub">THE RESEARCH WORKSPACE</div></div></div><div class="rail-label">EXPLORE YOUR RESEARCH</div>')
    for label,(symbol,_) in VIEWS.items():
        if st.button(label,icon=symbol,key='nav_'+label,width='stretch',type='primary' if st.session_state['view']==label else 'secondary'):
            navigate(label)
    html('<div class="rail-note"><strong>A research lens, not a leaderboard.</strong><p>Explore the conditions behind each fit. Evidence gaps are an invitation to investigate, not proof of absence.</p></div>')
    st.divider()
    st.caption('YOUR SAVED SNAPSHOT')
    st.caption(snapshot_date+' · All ASEAN research scope')
    st.download_button('Download complete briefing',report_path.read_bytes(),file_name='briefing.json',mime='application/json',icon=':material/download:',width='stretch',key='download_report')
    html('<div class="rail-status"><span></span>Local workspace</div>')
    st.caption('Reads your saved report. No model calls, live searches or account keys required.')

html(f'<div class="topline"><span>CORPORATE BANKING <span style="color:#c5ced6;margin:0 9px">/</span> CASH INTELLIGENCE</span><span class="snapshot">SNAPSHOT · {escape(snapshot_date.upper())}</span></div>')
VIEWS[st.session_state['view']][1]()
html(f'<div class="footer-line"><span>CASH ATLAS · A personal workspace for informed product research</span><span>As of {escape(snapshot_date)} · Saved evidence, with uncertainty retained</span></div>')
