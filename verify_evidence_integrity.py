"""Regression checks for final prompts and citation-only corrections."""

import asyncio
import json
import warnings
from copy import deepcopy

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from discovery_evidence import (
    REPORT_FORMAT_POLICY, SOURCE_POLICY, ResearchReportError, validate_with_citation_correction,
    _review_unreferenced_evidence,
)
from psp_discovery import DiscoveryReport, PSP_PROMPT, create_psp_discovery_node, validated_update
from state import initial_state
from verify_psp_discovery import FakeTavily, ScriptedModel, fixture_report


class PromptCheckingModel(ScriptedModel):
    def with_structured_output(self, schema, **kwargs):
        original = super().with_structured_output(schema, **kwargs)

        def inspect_prompt(messages):
            assert isinstance(messages[0], SystemMessage)
            assert SOURCE_POLICY in messages[0].content
            assert REPORT_FORMAT_POLICY in messages[0].content
            return original.invoke(messages)
        return RunnableLambda(inspect_prompt)


async def main():
    state = initial_state(as_of_date="2026-10-25", local_markets=["Singapore"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = await create_psp_discovery_node(model=PromptCheckingModel(), client=FakeTavily())(state)
    assert len(result["psps"]) == 10

    evidence = result["evidence"][0]
    valid = fixture_report(evidence["id"])
    broken = deepcopy(valid)
    broken["source_reviews"][0]["evidence_id"] = "invented-reference"
    messages = [ToolMessage(content=json.dumps({"evidence": [evidence]}), name="search_web",
                            tool_call_id="web", artifact={"evidence": [evidence]}),
                ToolMessage(content="No news results", name="search_news", tool_call_id="news",
                            artifact={"evidence": []})]

    class CorrectionModel:
        def __init__(self, response):
            self.response = response
            self.calls = 0

        def with_structured_output(self, schema):
            def respond(messages):
                self.calls += 1
                request = json.loads(messages[-1][1])
                if schema.__name__ == 'UniqueRetrievedSourceReviews':
                    return schema.model_validate({f'source_{i}': None for i in range(len(request['retrieved_candidates']))})
                if schema.__name__ == 'CitationFieldSelections':
                    return schema.model_validate({field['selection_key']: [] for field in request['failed_fields']})
                assert {e["id"] for e in request["allowed_evidence_manifest"]} == {evidence["id"]}
                assert request["report_to_correct"]["psps"] == broken["psps"]
                return schema.model_validate(self.response)
            return RunnableLambda(respond)

    async def correct(model, report=broken):
        return await validate_with_citation_correction(
            {"structured_response": report, "messages": messages}, model=model,
            schema=DiscoveryReport, prompt=PSP_PROMPT,
            validate=lambda result: validated_update(result, state), agent_name="psp_discovery",
        )

    model = CorrectionModel(valid)
    corrected = await correct(model)
    assert model.calls == 1 and len(corrected["psps"]) == 10
    assert corrected["psps"] == valid["psps"]

    # An already valid report must not trigger any additional model calls.
    unused = CorrectionModel(valid)
    await correct(unused, valid)
    assert unused.calls == 0

    # An uncited review cannot affect findings. Preserve the evidence artifacts
    # and remove only the unused review, without spending another model call.
    uncited = deepcopy(valid)
    uncited['source_reviews'].append({**uncited['source_reviews'][0], 'evidence_id': 'uncited-reference',
                                    'supporting_quote': 'Not a retrieved quote'})
    no_formatting = CorrectionModel(valid)
    cleaned = await correct(no_formatting, uncited)
    assert no_formatting.calls == 0 and cleaned['psps'] == valid['psps']

    # A legacy completed worker checkpoint can contain no structured report.
    # Recover from saved artifacts without rerunning the ReAct search loop.
    class RecoveryModel:
        def __init__(self, response):
            self.response, self.calls = response, 0

        def with_structured_output(self, schema):
            def respond(saved_messages):
                self.calls += 1
                request = json.loads(saved_messages[-1][1])
                assert request["allowed_evidence_manifest"] == [evidence]
                assert saved_messages[1:-1] == messages
                return self.response
            return RunnableLambda(respond)

    recovery = RecoveryModel(DiscoveryReport.model_validate(valid))
    recovered = await correct(recovery, None)
    assert recovery.calls == 1 and recovered["psps"] == valid["psps"]
    empty_recovery = RecoveryModel(None)
    try:
        await correct(empty_recovery, None)
    except ResearchReportError as exc:
        assert "one formatting attempt" in str(exc)
    else:
        raise AssertionError("Empty checkpoint recovery silently accepted")
    assert empty_recovery.calls == 1

    class QuoteModel:
        def __init__(self, index=0):
            self.calls, self.index = 0, index

        def with_structured_output(self, schema):
            def respond(prompt):
                self.calls += 1
                assert schema.__name__ == "ExactQuoteSelections"
                request = json.loads(prompt[-1][1])
                assert len(request["sources"]) == 1
                assert "report_to_correct" not in request
                return schema.model_validate({"source_0": {"segment_index": self.index}})
            return RunnableLambda(respond)

    paraphrased = deepcopy(valid)
    paraphrased["source_reviews"][0]["supporting_quote"] = "A paraphrase, not the saved excerpt"
    quote_model = QuoteModel()
    exact = await correct(quote_model, paraphrased)
    assert quote_model.calls == 1 and exact["psps"] == valid["psps"]
    assert paraphrased["source_reviews"][0]["supporting_quote"] == "A paraphrase, not the saved excerpt"
    for index in (None, -1, 999):
        failing_quote = QuoteModel(index)
        try:
            await correct(failing_quote, paraphrased)
        except ResearchReportError:
            pass
        else:
            raise AssertionError("Unsupported or invalid quote selection accepted")
        assert failing_quote.calls == 1

    # A report may have multiple independent defects: fixing a quote can expose
    # invented reference IDs. Resolve them only by explicit choices among real
    # retrieved records, with a new source review and exact segment quote.
    mixed = deepcopy(paraphrased)
    mixed['psps'][0]['evidence_ids'] = ['unknown-reference']
    mixed['source_reviews'].append({**valid['source_reviews'][0], 'evidence_id': 'unknown-reference'})
    new_record = {**evidence, 'id': 'new-primary', 'url': 'https://example.com/new',
                  'excerpt': 'New fixture evidence only'}
    mixed_messages = [ToolMessage(content='Saved web', name='search_web', tool_call_id='web',
                                  artifact={'evidence': [evidence, new_record]}), messages[1]]

    class MixedRepairModel:
        def __init__(self):
            self.schemas = []

        def with_structured_output(self, schema):
            def respond(prompt):
                self.schemas.append(schema.__name__)
                request = json.loads(prompt[-1][1])
                if schema.__name__ == 'ExactQuoteSelections':
                    data = {'source_0': {'segment_index': 0}}
                elif schema.__name__ == 'UniqueRetrievedSourceReviews':
                    assert len(request['retrieved_candidates']) == 1
                    data = {'source_0': {'source_type': 'primary',
                        'publisher': 'New primary publisher', 'credibility_reason': 'Synthetic official fixture',
                        'segment_index': 0}}
                elif schema.__name__ == 'CitationFieldSelections':
                    data = {field['selection_key']: [next(c['index'] for c in field['candidates']
                            if c['publisher'] == 'New primary publisher')] for field in request['failed_fields']}
                else:
                    raise AssertionError('Mixed defects must not regenerate the full findings report')
                return schema.model_validate(data)
            return RunnableLambda(respond)

    mixed_model = MixedRepairModel()
    grounded = await validate_with_citation_correction(
        {'structured_response': mixed, 'messages': mixed_messages}, model=mixed_model,
        schema=DiscoveryReport, prompt=PSP_PROMPT, validate=lambda r: validated_update(r, state),
        agent_name='psp_discovery')
    assert grounded['psps'][0]['evidence_ids'] == ['new-primary']
    assert grounded['psps'][0]['name'] == valid['psps'][0]['name']
    assert mixed_model.schemas == ['ExactQuoteSelections', 'UniqueRetrievedSourceReviews', 'CitationFieldSelections']

    class SpanModel:
        def with_structured_output(self, schema):
            return RunnableLambda(lambda prompt: schema.model_validate({
                'source_0': {'segment_index': 0, 'end_segment_index': 1}}))

    multiline = {**evidence, 'excerpt': 'Licence class\nPayment institution. Jurisdiction Singapore.'}
    span_messages = [ToolMessage(content='Saved table', name='search_web', tool_call_id='web',
                                 artifact={'evidence': [multiline]}), messages[1]]
    span = await validate_with_citation_correction(
        {'structured_response': paraphrased, 'messages': span_messages}, model=SpanModel(), schema=DiscoveryReport,
        prompt=PSP_PROMPT, validate=lambda r: validated_update(r, state), agent_name='psp_discovery')
    review = span['psp_discovery_summary']['source_reviews'][0]
    assert review['supporting_quote'] == 'Licence class\nPayment institution.'

    class RepeatedReviewModel:
        def __init__(self, conflict=False):
            self.conflict = conflict

        def with_structured_output(self, schema):
            def respond(prompt):
                if self.conflict:
                    return schema.model_validate({'reviews': []})
                return schema.model_validate({'source_0': {
                    'source_type': 'primary', 'publisher': 'New publisher',
                    'credibility_reason': 'Both cited facts', 'segment_index': 0,
                    'end_segment_index': 1}})
            return RunnableLambda(respond)

    repeated = {**new_record, 'excerpt': 'First cited fact.\nSecond cited fact.'}
    reviewed = await _review_unreferenced_evidence(DiscoveryReport.model_validate(valid),
        {evidence['id']: evidence, repeated['id']: repeated}, model=RepeatedReviewModel())
    assert len(reviewed.source_reviews) == 2
    assert reviewed.source_reviews[-1].supporting_quote == repeated['excerpt']
    try:
        await _review_unreferenced_evidence(DiscoveryReport.model_validate(valid),
            {evidence['id']: evidence, repeated['id']: repeated}, model=RepeatedReviewModel(conflict=True))
    except ValueError as exc:
        assert 'validation error' in str(exc)
    else:
        raise AssertionError('Conflicting source classifications accepted')

    changed = deepcopy(valid)
    changed["psps"].pop()
    changed["research_gaps"] = ["Attempted to discard a finding"]
    for response in (broken, changed):
        failing = CorrectionModel(response)
        try:
            await correct(failing)
        except ResearchReportError as exc:
            assert "psp_discovery" in str(exc)
        else:
            raise AssertionError("Invalid citations or deleted findings were accepted")
        assert failing.calls == (3 if response is broken else 1)

    unsupported = deepcopy(valid)
    unsupported["psps"][0]["rubric"]["annual_tpv"]["rating"] = "strong"
    no_retry = CorrectionModel(valid)
    try:
        await correct(no_retry, unsupported)
    except ResearchReportError:
        pass
    else:
        raise AssertionError("A non-citation validity problem was accepted")
    assert no_retry.calls == 0
    print("PASS: final system prompt retained, bounded citation correction, strict provenance and unchanged findings")


if __name__ == "__main__":
    asyncio.run(main())
