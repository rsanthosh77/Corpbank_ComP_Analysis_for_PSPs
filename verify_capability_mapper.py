"""Offline per-bank ReAct mapping, matrix completeness and evidence checks."""

import asyncio
import json
import warnings
from copy import deepcopy

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from capability_mapper import TIERS, create_capability_mapper_node, validated_capability_update
from bank_discovery import create_bank_discovery_node
from graph import build_graph
from needs_gatherer import create_needs_gatherer_node
from psp_discovery import create_psp_discovery_node
from state import initial_state
from verify_bank_discovery import BankModel, fixture_report as bank_fixture
from verify_needs_gatherer import NeedsModel, ParallelClient, task_from_messages
from verify_psp_discovery import FakeTavily, ScriptedModel, fixture_report as psp_fixture


def fixture_report(task, evidence_id):
    product = {
        "local_id": "live-clearing", "product_name": "Fixture live clearing",
        "description": "Fixture product only", "availability": "live",
        "jurisdictions": ["Singapore"], "legal_entities": ["Fixture Bank SG"],
        "psp_eligibility": "psp_explicit", "constraints": [],
        "evidence_ids": [evidence_id], "is_value_add": False, "value_add_rationale": None,
    }
    pilot = {**product, "local_id": "pilot-money", "product_name": "Fixture programmable money",
             "availability": "pilot", "constraints": ["Pilot access only"]}
    extra = {**product, "local_id": "extra-liquidity", "product_name": "Fixture liquidity value-add",
             "is_value_add": True, "value_add_rationale": "Fixture differentiating workflow"}
    assessments = []
    for i, need in enumerate(task["needs"]):
        status = ["documented_match", "conditional_match", "not_supported", "not_found"][i % 4]
        assessments.append({
            "need_id": need["id"], "status": status,
            "capability_local_ids": (["live-clearing"] if status == "documented_match" else
                                     ["pilot-money"] if status == "conditional_match" else []),
            "rationale": "Fixture only: documented fit, conditional fit, limitation or evidence gap",
            "constraints": ["Pilot access only"] if status == "conditional_match" else [],
            "evidence_ids": [] if status == "not_found" else [evidence_id],
        })
    return {
        "bank_id": task["bank"]["id"], "tier": task["tier"],
        "capabilities": [product, pilot, extra], "need_assessments": assessments,
        "methodology": "Offline fixture, not factual research",
        "coverage_notes": ["Fixture: every need plus extra value-add research"],
        "source_reviews": [{"evidence_id": evidence_id, "publisher": "Fixture publisher",
                            "source_type": "primary", "credibility_reason": "Offline fixture",
                            "supporting_quote": "Fixture evidence only"}],
        "research_gaps": [],
    }


class CapabilityModel(ScriptedModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        task = task_from_messages(messages)
        bank_id = task["bank"]["id"]
        completed = [m for m in messages if isinstance(m, ToolMessage)]
        for message in completed:
            assert message.artifact["evidence"][0]["agent"] == f"capability_mapper:{bank_id}"
        if len(completed) < 2:
            has_news = any(message.name == 'search_news' for message in completed)
            message = AIMessage(content="", tool_calls=[{
                "id": f"{bank_id}-{len(completed)}",
                "name": "search_web" if not completed or has_news else "search_news",
                "args": {"query": bank_id}, "type": "tool_call",
            }])
        else:
            message = AIMessage(content="Fixture mapping complete")
        return ChatResult(generations=[ChatGeneration(message=message)])

    def with_structured_output(self, schema, **kwargs):
        def format_report(messages):
            task = task_from_messages(messages)
            message = next(m for m in messages if isinstance(m, ToolMessage))
            evidence_id = json.loads(message.content)["evidence"][0]["id"]
            if isinstance(schema, dict):
                indexed = {**task, "needs": [{**need, "id": str(need.get("need_index", i))}
                                             for i, need in enumerate(task["needs"])]}
                report = fixture_report(indexed, evidence_id)
                report["need_assessments"] = [{"need_index": int(item["need_id"]),
                    **{k: v for k, v in item.items() if k != "need_id"}} for item in report["need_assessments"]]
                return report
            return schema.model_validate(fixture_report(task, evidence_id))
        return RunnableLambda(format_report)


async def main():
    state = initial_state(as_of_date="2026-10-25", local_markets=["Singapore"])
    state["psps"] = psp_fixture("psp-evidence")["psps"][:1]
    state["banks"] = bank_fixture("bank-evidence")["banks"]
    state["needs"] = [{
        "id": f"need-{i}", "psp_ids": [state["psps"][0]["id"]], "description": description,
        "themes": [description], "basis": "explicit" if i % 2 == 0 else "inferred",
        "rationale": "Fixture only", "jurisdictions": ["Singapore"], "evidence_ids": ["upstream"],
    } for i, description in enumerate(("instant clearing", "programmable money", "FX", "future category"))]
    state["capabilities"] = [{"id": "existing", "bank_id": "prior", "description": "Prior capability",
                              "matched_need_ids": [], "availability": "unknown", "jurisdictions": [],
                              "evidence_ids": []}]
    client = ParallelClient()
    node = create_capability_mapper_node(model=CapabilityModel(), client=client)

    async def passthrough(state):
        return {}

    compiler_calls = []

    async def briefing(state):
        compiler_calls.append(len(state["capability_research_summaries"]))
        assert len(state["capability_research_summaries"]) == 12
        return {"briefing": {"fixture_only": True}}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = await build_graph(nodes={
            "psp_discovery": passthrough, "bank_discovery": passthrough,
            "needs_gatherer": passthrough, "capability_mapper": node, "compile_briefing": briefing,
        }).ainvoke(state, config={"max_concurrency": 2})
    assert client.peak == 2
    assert len(client.calls) == 24
    assert compiler_calls == [12]
    assert len(result["capabilities"]) == 37
    assert len(result["need_assessments"]) == 48
    assert len(result["evidence"]) == 24
    assert len({c["id"] for c in result["capabilities"]}) == 37
    for tier in TIERS:
        for bank in state["banks"][tier]:
            rows = [a for a in result["need_assessments"] if a["bank_id"] == bank["id"]]
            assert {a["need_id"] for a in rows} == {n["id"] for n in state["needs"]}
            assert {a["status"] for a in rows} == {
                "documented_match", "conditional_match", "not_supported", "not_found"}
            assert all(a["tier"] == tier for a in rows)
    extras = [c for c in result["capabilities"] if c.get("is_value_add")]
    assert len(extras) == 12 and all(not c["matched_need_ids"] for c in extras)
    assert state["needs"] == result["needs"]  # Never rewrite or prune needs.

    bank = state["banks"][TIERS[0]][0]
    task = {"bank": bank, "tier": TIERS[0], **{k: state[k] for k in (
        "psps", "needs", "research_question", "local_scope", "local_markets", "as_of_date")}}
    evidence = next(e for e in result["evidence"] if e["agent"] == f"capability_mapper:{bank['id']}")
    messages = [ToolMessage(content="fixture", tool_call_id="web", name="search_web",
                            artifact={"evidence": [evidence]}),
                ToolMessage(content="fixture", tool_call_id="news", name="search_news",
                            artifact={"evidence": []})]
    base = fixture_report(task, evidence["id"])

    def validate(report, input_task=task):
        return validated_capability_update({"structured_response": report, "messages": messages}, input_task)

    invalid = []
    for key, value in (("bank_id", "wrong-bank"), ("tier", TIERS[1])):
        wrong = deepcopy(base)
        wrong[key] = value
        invalid.append(wrong)
    missing_need = deepcopy(base)
    missing_need["need_assessments"].pop()
    invalid.append(missing_need)
    duplicate_need = deepcopy(base)
    duplicate_need["need_assessments"].append(duplicate_need["need_assessments"][0])
    invalid.append(duplicate_need)
    foreign_need = deepcopy(base)
    foreign_need["need_assessments"][0]["need_id"] = "not-an-input-need"
    invalid.append(foreign_need)
    forged = deepcopy(base)
    forged["capabilities"][0]["evidence_ids"] = ["invented"]
    invalid.append(forged)
    quote = deepcopy(base)
    quote["source_reviews"][0]["supporting_quote"] = "Invented quotation"
    invalid.append(quote)
    for key, value in (("availability", "pilot"), ("psp_eligibility", "unclear"),
                       ("jurisdictions", ["Malaysia"]), ("legal_entities", [])):
        overclaim = deepcopy(base)
        overclaim["capabilities"][0][key] = value
        invalid.append(overclaim)
    no_negative_proof = deepcopy(base)
    no_negative_proof["need_assessments"][2]["evidence_ids"] = []
    invalid.append(no_negative_proof)
    unknown_product = deepcopy(base)
    unknown_product["need_assessments"][0]["capability_local_ids"] = ["invented-product"]
    invalid.append(unknown_product)
    no_conditions = deepcopy(base)
    no_conditions["need_assessments"][1]["constraints"] = []
    invalid.append(no_conditions)
    for report in invalid:
        try:
            validate(report)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid capability report accepted")

    unknown_scope = deepcopy(task)
    unknown_scope["needs"][0]["jurisdictions"] = []
    try:
        validate(base, unknown_scope)
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown scope was accepted as a documented match")
    skipped = await node({**state, "banks": {tier: [] for tier in TIERS}})
    assert not skipped["capabilities"] and "skipped" in skipped["research_gaps"][0]

    class SerialClient:
        async def search(self, query, **kwargs):
            return {"results": [{"url": "https://example.com/fixture", "content": "Fixture evidence only"}]}

    single_bank = {tier: ([bank] if tier == TIERS[0] else []) for tier in TIERS}
    zero_needs_node = create_capability_mapper_node(model=CapabilityModel(), client=SerialClient())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        extra_only = await zero_needs_node({**state, "banks": single_bank, "needs": []})
    assert len(extra_only["capabilities"]) == 3 and not extra_only["need_assessments"]

    class FailingClient:
        async def search(self, query, **kwargs):
            raise TimeoutError("Fixture API timeout")

    failing = create_capability_mapper_node(model=CapabilityModel(), client=FailingClient())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            await failing({**state, "banks": single_bank})
        except TimeoutError:
            pass
        else:
            raise AssertionError("Bank research failure was hidden")

    # All four actual worker factories together; only the future briefing is a stub.
    async def full_briefing(state):
        assert len(state["psps"]) == 10
        assert sum(len(state["banks"][tier]) for tier in TIERS) == 12
        assert len(state["needs"]) == 110
        assert len(state["need_assessments"]) == 1320
        assert len(state["capabilities"]) == 36
        assert len(state["evidence"]) == 48
        return {"briefing": {"fixture_only": True, "all_research_stages_completed": True}}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        pipeline = build_graph(nodes={
            "psp_discovery": create_psp_discovery_node(model=ScriptedModel(), client=FakeTavily()),
            "bank_discovery": create_bank_discovery_node(model=BankModel(), client=FakeTavily()),
            "needs_gatherer": create_needs_gatherer_node(model=NeedsModel(), client=SerialClient()),
            "capability_mapper": create_capability_mapper_node(model=CapabilityModel(), client=SerialClient()),
            "compile_briefing": full_briefing,
        })
        complete = await pipeline.ainvoke(
            initial_state(as_of_date="2026-10-25", local_markets=["Singapore"]),
            config={"max_concurrency": 2},
        )
    assert complete["briefing"]["all_research_stages_completed"]
    print("PASS: 12 isolated bank ReAct runs, full need matrix, tiers, unmatched value-adds and overclaim checks")


if __name__ == "__main__":
    asyncio.run(main())
