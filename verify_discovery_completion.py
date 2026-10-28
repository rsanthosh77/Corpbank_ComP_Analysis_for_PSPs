"""Offline legacy-checkpoint migration, incremental merging and completeness gate."""

import asyncio
import json
import tempfile
import warnings
from copy import deepcopy
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from bank_discovery import BankDiscoveryReport, TIERS, validated_bank_update
from discovery_completion import (
    DiscoveryIncompleteError, create_discovery_completion_node, require_complete_discovery, _merge_report,
)
from graph import build_graph, ResearchStageError
from psp_discovery import DiscoveryReport, validated_update
from research_store import ResearchStore, BudgetedSearchClient, configure_checkpointer
from state import initial_state
from verify_bank_discovery import fixture_report as bank_fixture
from verify_psp_discovery import ScriptedModel, FakeTavily, fixture_report as psp_fixture

MARKETS = ['Singapore', 'Indonesia', 'Philippines', 'Thailand', 'Viet Nam']


def coverage(evidence_id):
    return [{'market': market, 'findings': 'Synthetic country review', 'evidence_ids': [evidence_id], 'gaps': []}
            for market in MARKETS]


class CompletionModel(ScriptedModel):
    incomplete: bool = False

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        completed = [m for m in messages if isinstance(m, ToolMessage)]
        if len(completed) < 2:
            name = 'search_web' if not completed else 'search_news'
            agent = 'PSP' if 'Agent 1' in messages[0].content else 'bank'
            message = AIMessage(content='', tool_calls=[{'id': name, 'name': name,
                'args': {'query': agent + ' incremental completion'}, 'type': 'tool_call'}])
        else:
            message = AIMessage(content='Ready for structured report')
        return ChatResult(generations=[ChatGeneration(message=message)])

    def with_structured_output(self, schema, **kwargs):
        def respond(messages):
            first = next(m for m in messages if isinstance(m, ToolMessage))
            evidence_id = json.loads(first.content)['evidence'][0]['id']
            if schema is DiscoveryReport:
                report = psp_fixture(evidence_id)
                report['psps'] = report['psps'][:1]
                report['psps'][0].update(id='additional-psp', name='Additional PSP')
                if self.incomplete:
                    report['psps'] = []
                    report['research_gaps'] = ['No additional candidate could be established']
            else:
                assert schema is BankDiscoveryReport
                report = bank_fixture(evidence_id)
                report['banks']['tier_1_global'] = []
                report['banks']['tier_2_regional'] = []
                # Reproduce provider omissions: these are deterministic batch
                # metadata, not missing research on the already-retained banks.
                report['tier_gaps']['tier_1_global'] = []
                report['tier_gaps']['tier_2_regional'] = []
                report.pop('allocation_rationale')
                for bank, market in zip(report['banks']['tier_3_local'], MARKETS[1:]):
                    bank['countries'] = [market]
            report['market_coverage'] = coverage(evidence_id)
            return schema.model_validate(report)
        return RunnableLambda(respond)


def baseline():
    state = initial_state(as_of_date='2026-10-26', local_markets=MARKETS)
    reports = [psp_fixture('old-psp'), bank_fixture('old-bank')]
    reports[0]['psps'] = reports[0]['psps'][:9]
    reports[0]['research_gaps'] = ['Original shortlist shortfall']
    reports[1]['banks']['tier_3_local'] = []
    extra = deepcopy(reports[1]['banks']['tier_2_regional'][0])
    extra.update(id='regional-extra', group_id='regional-extra', name='Extra regional bank')
    reports[1]['banks']['tier_2_regional'].append(extra)
    reports[1]['tier_gaps']['tier_3_local'] = ['Original empty local tier']
    reports[1]['research_gaps'] = ['Original bank shortfall']
    updates = []
    for agent, report, evidence_id, validator in zip(
        ('psp_discovery', 'bank_discovery'), reports, ('old-psp', 'old-bank'),
        (validated_update, validated_bank_update)):
        report['market_coverage'] = coverage(evidence_id)
        record = {'id': evidence_id, 'agent': agent, 'topic': 'general',
                  'url': 'https://example.com/' + evidence_id, 'excerpt': 'Fixture evidence only'}
        messages = [ToolMessage(content='saved', name='search_web', tool_call_id='web', artifact={'evidence': [record]}),
                    ToolMessage(content='saved news', name='search_news', tool_call_id='news', artifact={'evidence': []})]
        update = validator({'structured_response': report, 'messages': messages}, state)
        updates.append(update)
        state.update({k: v for k, v in update.items() if k not in ('evidence', 'research_gaps')})
        state['evidence'].extend(update['evidence'])
        state['research_gaps'].extend(update['research_gaps'])
    return state, updates


async def main():
    state, updates = baseline()
    # Country spelling variants must not appear to be missing jurisdictions.
    from discovery_contracts import market_key
    assert market_key('Vietnam') == market_key('Viet Nam')
    assert market_key('Brunei') == market_key('Brunei Darussalam')
    assert market_key('Laos') == market_key('Lao PDR')
    invalid_psp = psp_fixture('old-psp')
    invalid_psp['psps'][0]['name'] = 'BDO Unibank, Inc.'
    try:
        DiscoveryReport.model_validate(invalid_psp)
    except ValueError as exc:
        assert 'commercial bank' in str(exc)
    else:
        raise AssertionError('Bank legal entity accepted as PSP')
    try:
        require_complete_discovery(state)
    except DiscoveryIncompleteError as exc:
        assert '9/10' in str(exc) and 'tier_3_local' in str(exc)
    else:
        raise AssertionError('Incomplete baseline passed the gate')
    needs_calls, discovery_calls = [], []

    async def psp(s):
        discovery_calls.append('psp')
        return updates[0]

    async def banks(s):
        discovery_calls.append('bank')
        return updates[1]

    async def needs(s):
        require_complete_discovery(s)
        needs_calls.append(True)
        return {}

    async def noop(s):
        return {}

    async def compile_report(s):
        return {'briefing': {'complete': True}}

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        store = ResearchStore(root / 'requests.sqlite', per_worker=1)
        fake = FakeTavily()
        async with AsyncSqliteSaver.from_conn_string(str(root / 'checkpoints.sqlite')) as saver:
            configure_checkpointer(saver)
            nodes = {'psp_discovery': psp, 'bank_discovery': banks, 'needs_gatherer': needs,
                     'capability_mapper': noop, 'compile_briefing': compile_report}
            # Seed a checkpoint produced by the old topology, already at needs.
            legacy = build_graph(nodes=nodes, checkpointer=saver)
            legacy.interrupt_before_nodes = ['needs_gatherer']
            config = {'configurable': {'thread_id': 'legacy-partial'}}
            await legacy.ainvoke(initial_state(as_of_date='2026-10-26', local_markets=MARKETS), config)
            assert not needs_calls
            completed = build_graph(nodes=nodes, checkpointer=saver, discovery_completion=
                create_discovery_completion_node(model=CompletionModel(), client=BudgetedSearchClient(fake, store)))
            completed.interrupt_after_nodes = ['complete_discovery']
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', DeprecationWarning)
                result = await completed.ainvoke(None, config)
            require_complete_discovery(result)
            inferred = deepcopy(result)
            inferred['local_markets'] = []
            inferred['psp_discovery_summary']['market_coverage'][-1]['market'] = 'Vietnam'
            require_complete_discovery(inferred)
            inferred['bank_discovery_summary']['market_coverage'] = inferred['bank_discovery_summary']['market_coverage'][:-1]
            try:
                require_complete_discovery(inferred)
            except DiscoveryIncompleteError as exc:
                assert 'missing market review' in str(exc)
            else:
                raise AssertionError('Unequal inferred ASEAN coverage accepted')
            assert len(result['psps']) == 10
            assert sum(len(v) for v in result['banks'].values()) == 13
            assert len(result['banks']['tier_3_local']) == 4
            assert len({b['countries'][0] for b in result['banks']['tier_3_local']}) == 4
            assert not needs_calls and len(discovery_calls) == 2
            assert len(result['evidence']) == len({e['id'] for e in result['evidence']})
            assert not any('Original shortlist shortfall' in g for g in result['research_gaps'])
            assert any('Original shortlist shortfall' in g for g in result['discovery_history'][0]['research_gaps'])
            assert (await completed.aget_state(config)).next == ('needs_gatherer',)
            normal = build_graph(nodes=nodes, checkpointer=saver, discovery_completion=
                create_discovery_completion_node(model=CompletionModel(), client=BudgetedSearchClient(fake, store)))
            final = await normal.ainvoke(None, config)
            assert final['briefing']['complete'] and len(needs_calls) == 1
            assert len(discovery_calls) == 2 and len(fake.topics) == 4
        # An insufficient addition must fail closed, not unlock needs gathering.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            try:
                await create_discovery_completion_node(model=CompletionModel(incomplete=True),
                    client=FakeTavily())(state)
            except DiscoveryIncompleteError:
                pass
            else:
                raise AssertionError('Insufficient completion was accepted')
        # Duplicate a retained PSP in a batch: merging must reject it.
        duplicate = psp_fixture('old-psp')
        duplicate['psps'] = duplicate['psps'][:1]
        duplicate['market_coverage'] = coverage('old-psp')
        try:
            _merge_report(state, 'psp_discovery', {'report': DiscoveryReport.model_validate(duplicate), 'evidence': []})
        except ValueError as exc:
            assert 'Duplicate PSPs' in str(exc)
        else:
            raise AssertionError('Repeated baseline entity accepted')
        store.close()
    print('PASS: legacy checkpoint gate, incremental completion, four local markets, unique entities/evidence, gap history and downstream resume')


if __name__ == '__main__':
    asyncio.run(main())
