"""Offline hybrid-compiler validation and the fully assembled research graph."""

import asyncio
import json
import warnings
from copy import deepcopy

from langchain_core.runnables import RunnableLambda

from bank_discovery import BankDiscoveryReport
from briefing import BriefingNarrative, assemble_briefing, create_briefing_node, validate_narrative
from capability_mapper import CapabilityReport
from graph import build_research_graph
from needs_gatherer import NeedsReport
from psp_discovery import DiscoveryReport
from state import initial_state
from verify_bank_discovery import BankModel
from verify_capability_mapper import CapabilityModel
from verify_needs_gatherer import NeedsModel, task_from_messages
from verify_psp_discovery import FakeTavily, ScriptedModel


def narrative_fixture(context):
    sections = {key: [] for key in BriefingNarrative.model_fields}
    for ref, record in context["records"].items():
        if ref.startswith("need:") and record.get("evidence_ids"):
            sections["executive_summary"] = [{
                "kind": "observation", "text": "Fixture: a company-specific need was researched.",
                "record_refs": [ref], "evidence_ids": record["evidence_ids"][:1],
            }]
            break
    for ref, record in context["records"].items():
        if record.get("status") == "conditional_match":
            sections["competitive_implications"] = [{
                "kind": "interpretation", "text": "Fixture: this conditional match needs validation.",
                "record_refs": [ref], "evidence_ids": record["evidence_ids"][:1],
            }]
            break
    for ref, record in context["records"].items():
        if record.get("is_value_add"):
            sections["product_opportunities"] = [{
                "kind": "recommendation", "text": "Fixture: investigate this evidenced value-add workflow.",
                "record_refs": [ref], "evidence_ids": record["evidence_ids"][:1],
            }]
            break
    for ref in context["records"]:
        if ref.startswith("gap:"):
            sections["follow_up_questions"] = [{
                "kind": "recommendation", "text": "Fixture: resolve this documented research gap.",
                "record_refs": [ref], "evidence_ids": [],
            }]
            break
    return sections


class SynthesisModel:
    def with_structured_output(self, schema, **kwargs):
        def respond(messages):
            context = json.loads(messages[-1][1])
            data = narrative_fixture(context)
            point_schema = None if isinstance(schema, dict) else schema.model_fields['executive_summary'].annotation.__args__[0]
            if point_schema is None or 'evidence_ids' not in point_schema.model_fields:
                for points in data.values():
                    for point in points:
                        point.pop('evidence_ids', None)
            return data if isinstance(schema, dict) else schema.model_validate(data)
        return RunnableLambda(respond)


class UnifiedModel(ScriptedModel):
    """Route scripted behavior by stage while testing the single-model assembly."""
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        task = task_from_messages(messages)
        implementation = (CapabilityModel if "bank" in task else
                          NeedsModel if "psp" in task else ScriptedModel)
        return implementation._generate(self, messages, stop, run_manager, **kwargs)

    def with_structured_output(self, schema, **kwargs):
        if getattr(schema, '__name__', '') == 'ReferencedBriefingNarrative' or isinstance(schema, dict) and schema.get('title') == 'ReferencedBriefingNarrative':
            return SynthesisModel.with_structured_output(self, schema, **kwargs)
        if isinstance(schema, dict):
            return CapabilityModel.with_structured_output(self, schema, **kwargs)
        implementation = {
            DiscoveryReport: ScriptedModel, BankDiscoveryReport: BankModel,
            NeedsReport: NeedsModel, CapabilityReport: CapabilityModel,
            BriefingNarrative: SynthesisModel,
        }[schema]
        return implementation.with_structured_output(self, schema, **kwargs)


async def main():
    state = initial_state(as_of_date="2026-10-25", local_markets=["Singapore"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = await build_research_graph(model=UnifiedModel(), client=FakeTavily()).ainvoke(
            state, config={"max_concurrency": 2})
    report = result["briefing"]
    assert report["statistics"]["psp_count"] == 10
    assert report["statistics"]["bank_count"] == 12
    assert report["statistics"]["need_count"] == 110
    assert report["statistics"]["assessment_count"] == 1320
    assert report["statistics"]["capability_count"] == 36
    assert report["statistics"]["evidence_count"] == 48
    assert len(report["comparison_matrix"]["need_rows"]) == 110
    assert len(report["comparison_matrix"]["bank_columns"]) == 12
    assert all(len(row["cells"]) == 12 for row in report["comparison_matrix"]["need_rows"])
    assert report["catalogs"]["needs"] == result["needs"]
    assert report["catalogs"]["banks"] == result["banks"]
    assert report["catalogs"]["capabilities"] == result["capabilities"]
    assert len(report["value_add_capability_ids"]) == 12
    assert sum(report["statistics"]["assessment_status_counts"].values()) == 1320
    for section in report["narrative"].values():
        for point in section:
            assert all(source["url"] == report["source_catalog"][source["evidence_id"]]["url"]
                       for source in point["sources"])
    assert json.loads(json.dumps(report))["statistics"] == report["statistics"]

    prior = deepcopy(result)
    rebuilt = assemble_briefing(result)
    assert result == prior  # Assembly does not mutate central state.
    assert rebuilt == assemble_briefing(result)
    class CapturedSynthesis:
        def __init__(self):
            self.batches = []
        def with_structured_output(self, schema):
            def respond(messages):
                context = json.loads(messages[-1][1])
                if 'candidate_narratives' not in context:
                    self.batches.append(set(context['records']))
                data = narrative_fixture(context)
                point_schema = None if isinstance(schema, dict) else schema.model_fields['executive_summary'].annotation.__args__[0]
                if point_schema is None or 'evidence_ids' not in point_schema.model_fields:
                    for points in data.values():
                        for point in points:
                            point.pop('evidence_ids', None)
                return data if isinstance(schema, dict) else schema.model_validate(data)
            return RunnableLambda(respond)
    captured = CapturedSynthesis()
    batched_state = deepcopy(result)
    batched_state['research_gaps'].extend(('Synthetic sizing gap ' + str(i) + ' x' * 10_000) for i in range(30))
    batched = (await create_briefing_node(model=captured)(batched_state))['briefing']
    assert len(captured.batches) > 1
    processed = [set(batch['record_refs']) for batch in batched['narrative_batches']]
    assert sum(map(len, processed)) == len(set().union(*processed)) == len(batched['records'])
    assert batched['catalogs'] == rebuilt['catalogs']
    assert batched['comparison_matrix'] == rebuilt['comparison_matrix']
    assert batched['source_catalog'] == rebuilt['source_catalog']
    narrative = narrative_fixture(rebuilt)
    invalid = []
    bad_record = deepcopy(narrative)
    bad_record["executive_summary"][0]["record_refs"] = ["need:invented"]
    invalid.append(bad_record)
    invented = deepcopy(narrative)
    invented["executive_summary"][0]["evidence_ids"] = ["fabricated"]
    invalid.append(invented)
    unrelated = deepcopy(narrative)
    related = set(rebuilt["records"][narrative["executive_summary"][0]["record_refs"][0]]["evidence_ids"])
    unrelated["executive_summary"][0]["evidence_ids"] = [
        next(evidence_id for evidence_id in rebuilt["source_catalog"] if evidence_id not in related)]
    invalid.append(unrelated)
    unsupported = deepcopy(narrative)
    unsupported["executive_summary"][0]["evidence_ids"] = []
    invalid.append(unsupported)
    raw_url = deepcopy(narrative)
    raw_url["executive_summary"][0]["text"] = "Invented link https://example.com/invented"
    invalid.append(raw_url)
    for output in invalid:
        try:
            validate_narrative(output, rebuilt)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid narrative references accepted")

    corrupt_states = []
    missing_cell = deepcopy(result)
    missing_cell["need_assessments"].pop()
    corrupt_states.append(missing_cell)
    duplicate_cell = deepcopy(result)
    duplicate_cell["need_assessments"].append(duplicate_cell["need_assessments"][0])
    corrupt_states.append(duplicate_cell)
    missing_source = deepcopy(result)
    missing_source["evidence"] = []
    corrupt_states.append(missing_source)
    wrong_bank = deepcopy(result)
    wrong_bank["capabilities"][0]["bank_id"] = "unknown-bank"
    corrupt_states.append(wrong_bank)
    nested_source = deepcopy(result)
    nested_source["psps"][0]["rubric"]["annual_tpv"]["evidence_ids"] = ["unknown-source"]
    corrupt_states.append(nested_source)
    for corrupted in corrupt_states:
        try:
            assemble_briefing(corrupted)
        except ValueError:
            pass
        else:
            raise AssertionError("Corrupt matrix/source data accepted")

    empty = await create_briefing_node(model=SynthesisModel())(state)
    assert empty["briefing"]["statistics"]["assessment_count"] == 0
    assert empty["briefing"]["research_gaps"]
    assert not empty["briefing"]["narrative"]["executive_summary"]
    class UnsupportedDraftModel:
        def with_structured_output(self, schema):
            def respond(messages):
                context = json.loads(messages[-1][1])
                ref = next(ref for ref, row in context['records'].items() if not row.get('evidence_ids'))
                data = {key: [] for key in BriefingNarrative.model_fields}
                data['executive_summary'] = [{'kind': 'observation', 'text': 'Unsupported absence claim', 'record_refs': [ref]}]
                data['follow_up_questions'] = [{'kind': 'recommendation', 'text': 'Validate the recorded coverage gap.', 'record_refs': [ref]}]
                return data if isinstance(schema, dict) else schema.model_validate(data)
            return RunnableLambda(respond)
    rejected_draft = (await create_briefing_node(model=UnsupportedDraftModel())(result))['briefing']
    assert not rejected_draft['narrative']['executive_summary']
    assert rejected_draft['narrative_omissions']
    assert rejected_draft['narrative']['follow_up_questions']
    assert rejected_draft['records'] == rebuilt['records']

    class FailingModel:
        def with_structured_output(self, schema):
            async def fail(messages):
                raise TimeoutError("Fixture synthesis timeout")
            return RunnableLambda(fail)

    try:
        await create_briefing_node(model=FailingModel())(result)
    except TimeoutError:
        pass
    else:
        raise AssertionError("Synthesis failure was hidden")
    print("PASS: complete assembled pipeline, deterministic 1,320-cell matrix, citation validation and failure checks")


if __name__ == "__main__":
    asyncio.run(main())
