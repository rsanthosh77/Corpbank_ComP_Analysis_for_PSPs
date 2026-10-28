"""Legacy empty checkpoint -> quote failure -> targeted ReAct recovery -> resume."""

import asyncio
import json
import tempfile
import warnings
from pathlib import Path
from copy import deepcopy

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import PrivateAttr

from evidence_recovery import run_source_recovery
from graph import build_graph, ResearchStageError
from psp_discovery import DiscoveryReport, PSP_PROMPT, create_psp_discovery_node, validated_update
from research_store import ResearchStore, BudgetedSearchClient, configure_checkpointer
from state import initial_state
from verify_psp_discovery import FakeTavily, ScriptedModel, fixture_report


class LegacyRecoveryModel(ScriptedModel):
    _fail_once: bool = PrivateAttr(default=True)
    _legacy_formatted: bool = PrivateAttr(default=False)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if "TARGETED EVIDENCE RECOVERY" not in messages[0].content:
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        completed = [m for m in messages if isinstance(m, ToolMessage) and m.name == 'targeted_official_search']
        if not completed:
            # Duplicates in a parallel batch must share one search, not two.
            message = AIMessage(content='', tool_calls=[
                {'id': f'recovery-{i}', 'name': 'targeted_official_search',
                 'args': {'target_index': 0}, 'type': 'tool_call'} for i in range(2)])
        else:
            message = AIMessage(content='Recovery research complete')
        return ChatResult(generations=[ChatGeneration(message=message)])

    def with_structured_output(self, schema, **kwargs):
        def respond(messages):
            if schema.__name__ == 'ExactQuoteSelections':
                request = json.loads(messages[-1][1])
                return schema.model_validate({source['selection_key']: {'segment_index': None}
                                              for source in request['sources']})
            recovery = [m for m in messages if isinstance(m, ToolMessage)
                        and m.name == 'targeted_official_search']
            if recovery:
                if self._fail_once:
                    self._fail_once = False
                    raise RuntimeError('Synthetic interruption at recovery formatter')
                evidence_id = json.loads(recovery[0].content)['evidence'][0]['id']
                return schema.model_validate(fixture_report(evidence_id))
            if not self._legacy_formatted:
                self._legacy_formatted = True
                return None  # Reproduce an old completed worker checkpoint.
            web = next(m for m in messages if isinstance(m, ToolMessage) and m.name == 'search_web')
            evidence_id = json.loads(web.content)['evidence'][0]['id']
            report = fixture_report(evidence_id)
            report['scope_evidence_ids'] = [evidence_id]
            report['source_reviews'][0]['supporting_quote'] = 'Licence statement absent from the excerpt'
            return schema.model_validate(report)
        return RunnableLambda(respond)


async def main():
    model, fake = LegacyRecoveryModel(), FakeTavily()
    bank_runs = []

    async def bank(state):
        bank_runs.append(True)
        return {'research_gaps': ['Synthetic successful sibling retained']}

    async def noop(state):
        return {}

    async def compile_report(state):
        assert len(state['psps']) == 10
        assert len(state['evidence']) == len({e['id'] for e in state['evidence']})
        return {'briefing': {'recovered': True}}

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        for resumed in (False, True):
            store = ResearchStore(path / 'requests.sqlite')
            async with AsyncSqliteSaver.from_conn_string(str(path / 'checkpoints.sqlite')) as saver:
                configure_checkpointer(saver)
                graph = build_graph(checkpointer=saver, nodes={
                    'psp_discovery': create_psp_discovery_node(model=model, client=BudgetedSearchClient(fake, store)),
                    'bank_discovery': bank, 'needs_gatherer': noop,
                    'capability_mapper': noop, 'compile_briefing': compile_report})
                config = {'configurable': {'thread_id': 'legacy-recovery'}}
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', DeprecationWarning)
                    if not resumed:
                        try:
                            await graph.ainvoke(initial_state(as_of_date='2026-10-26',
                                                             local_markets=['Singapore']), config)
                        except ResearchStageError as exc:
                            assert 'Synthetic interruption' in str(exc)
                        else:
                            raise AssertionError('Formatter interruption hidden')
                        assert fake.topics == ['general', 'news', 'general']
                    else:
                        result = await graph.ainvoke(None, config)
                        assert result['briefing']['recovered']
                        assert fake.topics == ['general', 'news', 'general']
                        assert len(bank_runs) == 1
                        assert store.usage()['search']['attempts'] == 3
                        assert len(result['evidence']) == 2  # Same exact fixture excerpt deduplicated.
            store.close()
    # Resume a failed tool while the recomputed plan has changed order. Tool
    # indices must still resolve against the plan stored in the nested worker.
    class TwoStepModel(LegacyRecoveryModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            completed = [m for m in messages if isinstance(m, ToolMessage)
                         and m.name == 'targeted_official_search']
            if len(completed) < 2:
                index = len(completed)
                message = AIMessage(content='', tool_calls=[{'id': f'step-{index}',
                    'name': 'targeted_official_search', 'args': {'target_index': index}, 'type': 'tool_call'}])
            else:
                message = AIMessage(content='Recovery complete')
            return ChatResult(generations=[ChatGeneration(message=message)])

    class FlakySearch(FakeTavily):
        def __init__(self):
            super().__init__()
            self.queries, self.fail = [], True

        async def search(self, query, **kwargs):
            self.queries.append(query)
            if query == 'second saved query' and self.fail:
                self.fail = False
                raise RuntimeError('Synthetic interrupted second target')
            return await super().search(query, **kwargs)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        model = TwoStepModel()
        model._fail_once = False
        fake = FlakySearch()
        old = {'id': 'saved', 'agent': 'psp_discovery', 'topic': 'general',
               'url': 'https://example.com/saved', 'excerpt': 'Fixture evidence only'}
        saved_result = {'structured_response': fixture_report('saved'), 'messages': [
            ToolMessage(content='Saved web', name='search_web', tool_call_id='web', artifact={'evidence': [old]}),
            ToolMessage(content='Saved news', name='search_news', tool_call_id='news', artifact={'evidence': []})]}
        for resumed in (False, True):
            store = ResearchStore(root / 'requests.sqlite')
            async with AsyncSqliteSaver.from_conn_string(str(root / 'checkpoints.sqlite')) as saver:
                configure_checkpointer(saver)

                async def recovering(state, config=None):
                    targets = ['first saved query', 'second saved query']
                    if resumed:
                        targets.reverse()
                    return await run_source_recovery(saved_result, state=state, model=model,
                        client=BudgetedSearchClient(fake, store), schema=DiscoveryReport,
                        prompt=PSP_PROMPT, agent_name='psp_discovery', worker_name='saved_plan_worker',
                        targets=targets, validate=lambda r: validated_update(r, state), config=config)

                graph = build_graph(checkpointer=saver, nodes={
                    'psp_discovery': recovering, 'bank_discovery': noop, 'needs_gatherer': noop,
                    'capability_mapper': noop, 'compile_briefing': compile_report})
                config = {'configurable': {'thread_id': 'changed-plan'}}
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', DeprecationWarning)
                    if not resumed:
                        try:
                            await graph.ainvoke(initial_state(as_of_date='2026-10-26', local_markets=['Singapore']), config)
                        except ResearchStageError as exc:
                            assert 'interrupted second target' in str(exc)
                        else:
                            raise AssertionError('Search failure hidden')
                    else:
                        result = await graph.ainvoke(None, config)
                        assert result['briefing']['recovered']
                        assert fake.queries == ['first saved query', 'second saved query', 'second saved query']
            store.close()
    print('PASS: legacy empty report, unsupported quote, parallel duplicates, interrupted recovery, changed-plan resume and sibling retention')


if __name__ == '__main__':
    asyncio.run(main())
