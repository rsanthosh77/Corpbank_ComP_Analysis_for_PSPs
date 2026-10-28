"""Native numbered report, complete matrix, fair budget and SQLite resume checks."""

import asyncio
import hashlib
import json
import sqlite3
import tempfile
import warnings
from unittest.mock import patch
from pathlib import Path

import httpx
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from capability_mapper import (
    CapabilityReport, capability_payload_schema, canonical_capability_result,
    create_capability_mapper_node, plan_bank_searches, validated_capability_update,
    complete_capability_result,
    repair_capability_links,
)
from check_saved_discovery import require_complete_capabilities
from evidence_recovery import source_issues
from graph import build_graph, ResearchStageError
from nebius_runtime import RuntimeSettings, create_nebius_model
from research_store import ResearchStore, BudgetedSearchClient, BudgetExceeded, configure_checkpointer
from state import initial_state
from verify_bank_discovery import fixture_report as bank_fixture
from verify_capability_mapper import CapabilityModel, fixture_report
from verify_psp_discovery import FakeTavily, fixture_report as psp_fixture


async def main():
    state = initial_state(as_of_date='2026-10-26', local_markets=['Singapore'])
    state['psps'] = psp_fixture('discovery')['psps'][:1]
    bank = bank_fixture('discovery')['banks']['tier_1_global'][0]
    state['banks'] = {'tier_1_global': [bank], 'tier_2_regional': [], 'tier_3_local': []}
    state['needs'] = [{'id': 'need-' + hashlib.sha256(str(i).encode()).hexdigest(),
        'psp_ids': [state['psps'][0]['id']], 'description': f'Synthetic unfiltered need {i}',
        'themes': ['arbitrary advanced category'], 'basis': 'inferred', 'rationale': 'Synthetic only',
        'jurisdictions': ['Singapore'], 'evidence_ids': ['upstream']} for i in range(84)]
    task = {'bank': bank, 'tier': 'tier_1_global', 'needs': state['needs']}
    report = fixture_report(task, 'retrieved')
    wire = {**report, 'need_assessments': [{'need_index': i,
                **{k: v for k, v in item.items() if k != 'need_id'}} for i, item in enumerate(report['need_assessments'])]}
    payload_schema = capability_payload_schema(task['needs']).model_json_schema()
    assert state['needs'][0]['id'] not in json.dumps(payload_schema)
    requests = []

    def provider(request):
        requests.append(request)
        body = json.loads(request.content)
        name = body['tools'][0]['function']['name']
        assert name == 'BankCapabilityPayload'
        assert body['tool_choice']['function']['name'] == name
        return httpx.Response(200, json={'id': 'synthetic', 'object': 'chat.completion', 'created': 0,
            'model': 'synthetic', 'choices': [{'index': 0, 'finish_reason': 'tool_calls', 'message': {
                'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'report', 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(wire)}}]}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
        model = create_nebius_model(RuntimeSettings(nebius_api_key='synthetic', max_retries=0),
                                    http_client=None, http_async_client=client)
        raw = await model.with_structured_output(payload_schema).ainvoke('Synthetic report formatting only')
    assert isinstance(raw, dict) and len(requests) == 1
    canonical = canonical_capability_result({'structured_response': raw}, task)['structured_response']
    assert [a.need_id for a in canonical.need_assessments] == [n['id'] for n in state['needs']]
    from copy import deepcopy
    null_extra = deepcopy(wire)
    null_extra['need_assessments'][0]['reasoning'] = None
    assert canonical_capability_result({'structured_response': null_extra}, task)['structured_response'] == canonical
    null_extra['need_assessments'][0]['reasoning'] = 'Actual extra content must not be discarded'
    try:
        canonical_capability_result({'structured_response': null_extra}, task)
    except ValueError:
        pass
    else:
        raise AssertionError('Unknown assessment content silently discarded')
    optional_metadata = deepcopy(wire)
    optional_metadata['capabilities'][0].pop('value_add_rationale')
    optional_metadata['capabilities'][1].pop('legal_entities')
    optional_metadata['capabilities'][1]['local_entities_note'] = 'Possible group entities; not verified'
    normalized = canonical_capability_result({'structured_response': optional_metadata}, task)['structured_response']
    assert normalized.capabilities[0].value_add_rationale is None
    assert normalized.capabilities[1].legal_entities == []
    assert any('Possible group entities; not verified' in text for text in normalized.capabilities[1].constraints)
    missing_explanation = deepcopy(wire)
    missing_explanation['capabilities'][2].pop('value_add_rationale')
    try:
        canonical_capability_result({'structured_response': missing_explanation}, task)
    except ValueError as exc:
        assert 'value-add product requires' in str(exc)
    else:
        raise AssertionError('Unsupported value-add classification accepted')
    missing = {**wire, 'need_assessments': wire['need_assessments'][:-1]}
    try:
        canonical_capability_result({'structured_response': missing}, task)
    except ValueError as exc:
        assert 'need_assessments' in str(exc)
    else:
        raise AssertionError('Missing need assessment accepted')

    # Primary-source requirements apply to definitive assessments and to every
    # product those definitive assessments cite, not just bank-discovery fields.
    industry = CapabilityReport.model_validate(report)
    industry.source_reviews[0].source_type = 'industry_research'
    record = {'id': 'retrieved', 'agent': 'capability_mapper:' + bank['id'],
              'url': 'https://example.com', 'excerpt': 'Fixture evidence only'}
    artifacts = {'messages': [ToolMessage(content='web', name='search_web', tool_call_id='web',
                                        artifact={'evidence': [record]}),
                             ToolMessage(content='news', name='search_news', tool_call_id='news', artifact={})]}
    # Repair one genuinely omitted row through a bounded model call, preserving
    # every existing row. The old object format is supported for saved work.
    from langchain_core.runnables import RunnableLambda

    class MissingRowModel:
        calls = 0

        def with_structured_output(self, schema):
            def respond(messages):
                self.calls += 1
                request = json.loads(messages[-1][1])
                assert [n['need_index'] for n in request['needs_to_assess']] == [4]
                return {'need_assessments': [wire['need_assessments'][4]]}
            return RunnableLambda(respond)

    legacy = {**wire, 'need_assessments': {f'need_{i}': {k: v for k, v in item.items() if k != 'need_index'}
                                         for i, item in enumerate(wire['need_assessments']) if i != 4}}
    missing_model = MissingRowModel()
    completed = await complete_capability_result({**artifacts, 'structured_response': legacy}, task, model=missing_model)
    assert missing_model.calls == 1
    assert completed['structured_response'].model_dump() == canonical.model_dump()

    class LinkModel:
        def with_structured_output(self, schema):
            row = canonical.need_assessments[0]
            return RunnableLambda(lambda messages: {'assessment_0': {
                'status': row.status, 'rationale': row.rationale, 'constraints': row.constraints,
                'capability_indices': [0], 'evidence_indices': [0]}})

    broken_link = canonical.model_copy(deep=True)
    broken_link.need_assessments[0].capability_local_ids = ['unreported-product-reference']
    fixed_link = await repair_capability_links({**artifacts, 'structured_response': broken_link}, task, model=LinkModel())
    assert fixed_link['structured_response'].need_assessments == canonical.need_assessments
    assert fixed_link['structured_response'].capabilities == canonical.capabilities
    # Invalid positive/negative statuses need real reassessment, not automatic
    # downgrading or dropping product links to make the validator accept them.
    inconsistent = canonical.model_copy(deep=True)
    inconsistent.need_assessments[0].status = 'not_found'
    repaired_status = await repair_capability_links(
        {**artifacts, 'structured_response': inconsistent}, task, model=LinkModel())
    assert repaired_status['structured_response'].need_assessments == canonical.need_assessments
    assert repaired_status['structured_response'].capabilities == canonical.capabilities
    diagnostics = source_issues(artifacts, industry, agent_name=record['agent'])['diagnostics']
    assert any(d['path'] == 'report.need_assessments[0].evidence_ids' and d['primary_required'] for d in diagnostics)
    assert any(d['path'] == 'report.capabilities[0].evidence_ids' and d['primary_required'] for d in diagnostics)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        store = ResearchStore(root / 'budgets.sqlite')
        for i in range(135):
            store.reserve('search', f'prior-{i}')
        bounded = BudgetedSearchClient(FakeTavily(), store)
        bank_ids = [f'bank-{i}' for i in range(13)]
        quotas = plan_bank_searches(bounded, bank_ids)
        assert set(quotas.values()) == {5} and sum(quotas.values()) == 65
        for _ in range(5):
            store.reserve('search', 'capability_mapper:bank-0', per_worker_limit=5)
        try:
            store.reserve('search', 'capability_mapper:bank-0', per_worker_limit=5)
        except BudgetExceeded:
            pass
        else:
            raise AssertionError('A bank consumed another bank search allowance')
        store.close()
        store = ResearchStore(root / 'budgets.sqlite')
        assert plan_bank_searches(BudgetedSearchClient(FakeTavily(), store), bank_ids) == quotas
        store.close()

        mapper = create_capability_mapper_node(model=CapabilityModel(), client=FakeTavily())

        async def noop(s):
            return {}

        async def compiler(s):
            require_complete_capabilities(s)
            return {'briefing': {'synthetic': True}}

        config = {'configurable': {'thread_id': 'numbered-wire'}}
        nodes = {'psp_discovery': noop, 'bank_discovery': noop, 'needs_gatherer': noop,
                 'capability_mapper': mapper, 'compile_briefing': compiler}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            async with AsyncSqliteSaver.from_conn_string(str(root / 'checkpoints.sqlite')) as saver:
                configure_checkpointer(saver)
                graph = build_graph(nodes=nodes, checkpointer=saver)
                graph.interrupt_before_nodes = ['compile_briefing']
                result = await graph.ainvoke(state, config)
                require_complete_capabilities(result)
                assert len(result['need_assessments']) == 84
                assert (await graph.aget_state(config)).next == ('compile_briefing',)
            async with AsyncSqliteSaver.from_conn_string(str(root / 'checkpoints.sqlite')) as saver:
                configure_checkpointer(saver)
                result = await build_graph(nodes=nodes, checkpointer=saver).ainvoke(None, config)
                assert result['briefing']['synthetic']

        # One rejected bank must not cancel peers or repeat their research on
        # resume. The rejected bank reuses its exact saved report and sources.
        isolation_store = ResearchStore(root / 'isolation-requests.sqlite')
        isolation_client = FakeTavily()
        several = {**state, 'banks': {'tier_1_global': bank_fixture('prior')['banks']['tier_1_global'][:3],
                                      'tier_2_regional': [], 'tier_3_local': []}}
        failing_id = several['banks']['tier_1_global'][0]['id']
        reject = True
        real_validate = validated_capability_update

        def controlled_validate(result, task):
            if reject and task['bank']['id'] == failing_id:
                raise ValueError('Synthetic single-bank validation failure')
            return real_validate(result, task)

        with patch('capability_mapper.validated_capability_update', side_effect=controlled_validate):
            for retry in (False, True):
                reject = not retry
                async with AsyncSqliteSaver.from_conn_string(str(root / 'isolation-checkpoints.sqlite')) as saver:
                    configure_checkpointer(saver)
                    node = create_capability_mapper_node(model=CapabilityModel(),
                        client=BudgetedSearchClient(isolation_client, isolation_store))
                    graph = build_graph(checkpointer=saver, nodes={
                        'psp_discovery': noop, 'bank_discovery': noop, 'needs_gatherer': noop,
                        'capability_mapper': node, 'compile_briefing': compiler})
                    config = {'max_concurrency': 2, 'configurable': {'thread_id': 'isolated-failure'}}
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', DeprecationWarning)
                        if not retry:
                            try:
                                await graph.ainvoke(several, config)
                            except ResearchStageError as exc:
                                assert 'Synthetic single-bank validation failure' in str(exc)
                            else:
                                raise AssertionError('Failed bank was accepted')
                            assert len(isolation_client.topics) == 6  # Every peer finished.
                        else:
                            result = await graph.ainvoke(None, config)
                            require_complete_capabilities(result)
                            assert len(result['need_assessments']) == 3 * 84
                            assert len(isolation_client.topics) == 6  # No repeated searches.
                            assert len(result['capability_research_summaries']) == 3
        isolation_store.close()
        # A hard provider/budget interruption occurs before the fan-out joins.
        # Successful peer writes must survive a fresh graph/process instance.
        from collections import Counter
        calls = Counter()
        blocked_id = several['banks']['tier_1_global'][-1]['id']
        blocked = True
        def interrupted_validate(result, task):
            bank_id = task['bank']['id']
            calls[bank_id] += 1
            if blocked and bank_id == blocked_id:
                raise RuntimeError('Synthetic provider budget interruption')
            return real_validate(result, task)
        with patch('capability_mapper.validated_capability_update', side_effect=interrupted_validate):
            for retry in (False, True):
                blocked = not retry
                async with AsyncSqliteSaver.from_conn_string(str(root / 'interrupted.sqlite')) as saver:
                    configure_checkpointer(saver)
                    node = create_capability_mapper_node(model=CapabilityModel(), client=FakeTavily())
                    graph = build_graph(checkpointer=saver, nodes={
                        'psp_discovery': noop, 'bank_discovery': noop, 'needs_gatherer': noop,
                        'capability_mapper': node, 'compile_briefing': compiler})
                    config = {'max_concurrency': 1, 'configurable': {'thread_id': 'provider-interruption'}}
                    if not retry:
                        try:
                            await graph.ainvoke(several, config)
                        except ResearchStageError as exc:
                            assert 'Synthetic provider budget interruption' in str(exc)
                        else:
                            raise AssertionError('Provider interruption was swallowed')
                    else:
                        result = await graph.ainvoke(None, config)
                        require_complete_capabilities(result)
            assert calls[blocked_id] == 2
            assert all(count == 1 for bank_id, count in calls.items() if bank_id != blocked_id)
    print('PASS: native numbered 84-need payload, missing-cell rejection, primary evidence rules, fair durable searches and SQLite resume')


if __name__ == '__main__':
    asyncio.run(main())
