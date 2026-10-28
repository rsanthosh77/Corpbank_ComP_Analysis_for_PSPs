"""Offline recovery from rejected citations using the real prebuilt ReAct worker."""

import asyncio
import json
import tempfile
import warnings
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from bank_discovery import BankDiscoveryReport, bank_recovery_targets, recover_bank_evidence
from discovery_evidence import ResearchReportError, validate_with_citation_correction
from bank_discovery import BANK_PROMPT, validated_bank_update
from research_store import BudgetedSearchClient, ResearchStore, BudgetExceeded
from state import initial_state
from verify_bank_discovery import BankModel, fixture_report
from verify_psp_discovery import FakeTavily


class RecoveryModel(BankModel):
    reject: bool = False

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        new = [m for m in messages if isinstance(m, ToolMessage) and m.name == 'targeted_official_search']
        message = (AIMessage(content='Recovery complete') if new else
                   AIMessage(content='', tool_calls=[{'id': 'target-0', 'name': 'targeted_official_search',
                                                      'args': {'target_index': 0}, 'type': 'tool_call'}]))
        return ChatResult(generations=[ChatGeneration(message=message)])

    def with_structured_output(self, schema, **kwargs):
        def response(messages):
            if schema.__name__ == 'CitationFieldSelections':
                request = json.loads(messages[-1][1])
                return schema.model_validate({field['selection_key']: [] for field in request['failed_fields']})
            new = next(m for m in messages if isinstance(m, ToolMessage) and m.name == 'targeted_official_search')
            evidence_id = json.loads(new.content)['evidence'][0]['id']
            report = fixture_report(evidence_id)
            if self.reject:
                report['source_reviews'][0]['source_type'] = 'unverified'
            return schema.model_validate(report)
        return RunnableLambda(response)


async def main():
    evidence = {'id': 'saved', 'agent': 'bank_discovery', 'url': 'https://example.com/saved',
                'title': 'Fixture', 'topic': 'general', 'excerpt': 'Fixture evidence only',
                'date_status': 'uncertain', 'published_at': None, 'raw_published_at': None,
                'retrieved_at': '2026-10-26T00:00:00+00:00'}
    report = fixture_report('saved')
    report['scope_evidence_ids'] = ['saved']
    report['source_reviews'].append({**report['source_reviews'][0], 'evidence_id': 'industry',
                                     'source_type': 'industry_research'})
    report['banks']['tier_1_global'][0]['rubric']['corporate_cash_management']['evidence_ids'] = ['industry']
    parsed = BankDiscoveryReport.model_validate(report)
    assert len(bank_recovery_targets(parsed)) == 1
    messages = [ToolMessage(content='Saved web', name='search_web', tool_call_id='saved-web',
                            artifact={'evidence': [evidence, {**evidence, 'id': 'industry'}]}),
                ToolMessage(content='Saved news check', name='search_news', tool_call_id='saved-news',
                            artifact={'evidence': []})]
    result = {'structured_response': parsed, 'messages': messages}
    state = initial_state(as_of_date='2026-10-26', local_markets=['Singapore'])

    class CitationModel:
        def __init__(self, reclassify=False):
            self.calls, self.reclassify = 0, reclassify

        def with_structured_output(self, schema):
            def response(prompt):
                assert schema.__name__ == 'CitationFieldSelections'  # Never regenerate the large report.
                self.calls += 1
                request = json.loads(prompt[-1][1])
                fields = request['failed_fields']
                return schema.model_validate({field['selection_key']: [999 if self.reclassify else 0]
                                              for field in fields})
            return RunnableLambda(response)

    patched = await validate_with_citation_correction(result, model=CitationModel(), schema=BankDiscoveryReport,
        prompt=BANK_PROMPT, validate=lambda value: validated_bank_update(value, state), agent_name='bank_discovery')
    assert patched['banks']['tier_1_global'][0]['rubric']['corporate_cash_management']['evidence_ids'] == ['saved']
    assert result['structured_response'].banks.tier_1_global[0].rubric.corporate_cash_management.evidence_ids == ['industry']
    try:
        await validate_with_citation_correction(result, model=CitationModel(reclassify=True), schema=BankDiscoveryReport,
            prompt=BANK_PROMPT, validate=lambda value: validated_bank_update(value, state), agent_name='bank_discovery')
    except ResearchReportError as exc:
        assert 'literal_error' in str(exc)
    else:
        raise AssertionError('Source relabelling accepted')
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'requests.sqlite'
        store = ResearchStore(path, per_worker=1)
        # Discovery is exhausted, but recovery has a separate explicit allowance.
        store.reserve('search', 'bank_discovery')
        fake = FakeTavily()
        client = BudgetedSearchClient(fake, store)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            update = await recover_bank_evidence(result, state=state, model=RecoveryModel(), client=client)
            assert len(update['banks']['tier_1_global']) == 4
            assert len(update['evidence']) == 3  # Saved and newly retrieved artifacts.
            assert fake.topics == ['general']  # No discovery or news replay.
            await recover_bank_evidence(result, state=state, model=RecoveryModel(), client=client)
            assert fake.topics == ['general']  # Persistent cache replay.
            try:
                await recover_bank_evidence(result, state=state, model=RecoveryModel(reject=True), client=client)
            except ResearchReportError:
                pass
            else:
                raise AssertionError('Unverified recovery citations accepted')
        for _ in range(7):
            store.reserve('search', 'bank_discovery_recovery', per_worker_limit=8)
        store.close()
        store = ResearchStore(path)
        try:
            store.reserve('search', 'bank_discovery_recovery', per_worker_limit=8)
        except BudgetExceeded:
            pass
        else:
            raise AssertionError('Recovery limit reset after reopening the saved run')
        assert store.usage()['search']['attempts'] == 9
        store.close()
    print('PASS: targeted ReAct recovery, retained evidence, strict sources, cached searches and durable eight-attempt cap')


if __name__ == '__main__':
    asyncio.run(main())
