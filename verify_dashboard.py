"""Offline dashboard checks: complete data, filters, comparisons and navigation."""
from copy import deepcopy
from streamlit.testing.v1 import AppTest
from dashboard_data import read_report, ResearchData, safe_url, csv_bytes, market_matches

def main():
    data = read_report()
    assert len(data.assessments) == len(data.banks) * len(data.needs)
    assert sum(data.counts().values()) == len(data.assessments)
    for psp_id in data.psps:
        needs = data.needs_filtered(psps=[psp_id])
        assert needs and all(psp_id in n['psp_ids'] for n in needs)
    inferred = data.needs_filtered(basis=['inferred'])
    assert len(inferred) == data.briefing['statistics']['need_basis_counts']['inferred']
    assert data.needs_filtered(query='stablecoin')
    assert not market_matches(['Global'], 'Indonesia')  # No invented geography.
    assert market_matches(['Indonesia (subject to onboarding)'], 'Indonesia')
    assert safe_url('javascript:alert(1)') is None
    assert safe_url('file:///etc/passwd') is None
    assert safe_url('https://example.com')
    assert "'=1+1" in csv_bytes([{'Requirement': '=1+1'}]).decode('utf-8-sig')
    broken = deepcopy(data.document)
    broken['briefing']['comparison_matrix']['need_rows'][0]['cells'].pop()
    try:
        ResearchData(broken)
    except ValueError:
        pass
    else:
        raise AssertionError('Incomplete research silently accepted')
    app = AppTest.from_file('dashboard.py', default_timeout=40).run()
    assert not app.exception
    app.button(key='nav_PSP needs').click().run()
    app.multiselect(key='needs_psps').set_value(['xendit']).run()
    assert not app.exception
    assert len(app.dataframe[0].value) == len(data.needs_filtered(psps=['xendit']))
    app.text_input(key='needs_q').set_value('payout').run()
    assert not app.exception
    assert len(app.dataframe[0].value) == len(data.needs_filtered(query='payout', psps=['xendit']))
    app.text_input(key='needs_q').set_value('nonexistent_query_984738').run()
    assert not app.exception and len(app.dataframe) == 0
    app.text_input(key='needs_q').set_value('').run()
    app.button(key='need_to_compare').click().run()
    assert not app.exception
    assert app.session_state['compare_psps'] == ['xendit']
    app.multiselect(key='compare_banks').set_value(['bca', 'kbank']).run()
    assert not app.exception
    assert set(app.dataframe[0].value.columns) == {'PSP', 'Requirement', 'BCA', 'Kasikornbank'}
    app.multiselect(key='compare_banks').set_value([]).run()
    assert not app.exception and len(app.dataframe) == 0
    app.button(key='nav_Bank profiles').click().run()
    app.selectbox(key='profile_tier').set_value('tier_3_local').run()
    assert not app.exception
    assert len(app.selectbox(key='profile_bank_tier_3_local').options) == 4
    app.selectbox(key='profile_bank_tier_3_local').set_value('kbank').run()
    app.toggle(key='value_add_only').set_value(True).run()
    assert not app.exception
    app.button(key='nav_Evidence library').click().run()
    app.multiselect(key='source_grades').set_value(['primary']).run()
    app.multiselect(key='source_dates').set_value(['uncertain']).run()
    assert not app.exception
    app.text_input(key='evidence_q').set_value('nonexistent_query_984738').run()
    assert not app.exception
    print('PASS: complete matrix, unfiltered themes, safe exports, scoped filters, need-to-comparison handoff, all five views and empty states')

if __name__ == '__main__':
    main()
