"""Exercise the real prebuilt ReAct graph with a scripted model and Tavily fake."""

import asyncio
import json
import warnings
from copy import deepcopy

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from graph import build_graph
from psp_discovery import DiscoveryReport, create_psp_discovery_node, validated_update
from state import initial_state


def fixture_report(evidence_id, quote="Fixture evidence only"):
    unknown = {"rating": "unknown", "assessment": "Fixture: not disclosed",
               "evidence_ids": [], "gaps": ["Fixture disclosure gap"]}
    known = {"rating": "moderate", "assessment": "Fixture: regional provider",
             "evidence_ids": [evidence_id], "gaps": []}
    return {
        "psps": [{
            "id": f"fixture-{i}", "name": f"Fixture PSP {i}",
            "countries": ["Singapore"], "selection_rationale": "Fixture comparison",
            "evidence_ids": [evidence_id],
            "rubric": {"annual_tpv": unknown, "developer_ecosystem": known,
                       "market_position": known, "funding_and_valuation": unknown,
                       "licensing": unknown},
        } for i in range(10)],
        "methodology": "Fixture only: test transport and validation, not research quality",
        "scope_evidence_ids": [],
        "market_coverage": [{"market": "Singapore", "findings": "Fixture coverage",
                             "evidence_ids": [evidence_id], "gaps": []}],
        "excluded_candidates": [], "research_gaps": [],
        "source_reviews": [{"evidence_id": evidence_id, "publisher": "Fixture publisher",
                            "source_type": "primary", "credibility_reason": "Offline fixture",
                            "supporting_quote": quote}],
    }


class ScriptedModel(BaseChatModel):
    @property
    def _llm_type(self):
        return "offline-psp-fixture"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        completed = [m for m in messages if isinstance(m, ToolMessage)]
        if len(completed) < 2:
            tool = "search_web" if not completed else "search_news"
            message = AIMessage(content="", tool_calls=[{
                "id": f"search-{len(completed)}", "name": tool,
                "args": {"query": "Fixture PSP research"}, "type": "tool_call",
            }])
        else:
            message = AIMessage(content="Fixture research complete")
        return ChatResult(generations=[ChatGeneration(message=message)])

    def with_structured_output(self, schema, **kwargs):
        def format_report(messages):
            tool_message = next(m for m in messages if isinstance(m, ToolMessage))
            evidence_id = json.loads(tool_message.content)["evidence"][0]["id"]
            return schema.model_validate(fixture_report(evidence_id))
        return RunnableLambda(format_report)


class FakeTavily:
    def __init__(self):
        self.topics = []

    async def search(self, query, **kwargs):
        self.topics.append(kwargs["topic"])
        return {"results": [{"url": "https://example.com/fixture-psps",
                             "title": "Fixture", "content": "Fixture evidence only"}]}


async def main():
    state = initial_state(as_of_date="2026-10-25", local_markets=["Singapore"])
    client = FakeTavily()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        node = create_psp_discovery_node(model=ScriptedModel(), client=client)

    async def banks(state):
        return {"banks": {"tier_1_global": [], "tier_2_regional": [], "tier_3_local": []}}

    async def needs(state):
        assert len(state["psps"]) == 10
        assert state["psps"][0]["rubric"]["annual_tpv"]["rating"] == "unknown"
        assert len(state["evidence"]) == 2  # Preserve separate web/news provenance.
        return {"needs": []}

    async def capabilities(state):
        return {"capabilities": []}

    async def briefing(state):
        return {"briefing": {"fixture_only": True}}

    result = await build_graph(nodes={
        "psp_discovery": node, "bank_discovery": banks, "needs_gatherer": needs,
        "capability_mapper": capabilities, "compile_briefing": briefing,
    }).ainvoke(state)
    assert client.topics == ["general", "news"]
    assert result["psp_discovery_summary"]["selected_count"] == 10
    assert len(result["research_gaps"]) == 31  # Includes undated news warning.
    assert result["briefing"]["fixture_only"]

    evidence = result["evidence"][0]
    base = fixture_report(evidence["id"])
    messages = [ToolMessage(content="fixture", tool_call_id="fixture", name="search_web",
                            artifact={"evidence": [evidence]}),
                ToolMessage(content="fixture news", tool_call_id="news", name="search_news",
                            artifact={"evidence": []})]

    def validate(report, input_state=state):
        return validated_update({"structured_response": report, "messages": messages}, input_state)

    invalid = []
    forged = deepcopy(base)
    forged["psps"][0]["rubric"]["licensing"] = {
        "rating": "strong", "assessment": "Invented", "evidence_ids": ["fabricated"], "gaps": []}
    invalid.append(forged)
    missing = deepcopy(base)
    del missing["psps"][0]["rubric"]["licensing"]
    invalid.append(missing)
    duplicate = deepcopy(base)
    duplicate["psps"][1] = duplicate["psps"][0]
    invalid.append(duplicate)
    incomplete = deepcopy(base)
    incomplete["psps"] = incomplete["psps"][:9]
    invalid.append(incomplete)
    unknown = deepcopy(base)
    unknown["psps"][0]["rubric"]["licensing"]["gaps"] = []
    invalid.append(unknown)
    fabricated_quote = deepcopy(base)
    fabricated_quote["source_reviews"][0]["supporting_quote"] = "Invented quotation"
    invalid.append(fabricated_quote)
    unverified = deepcopy(base)
    unverified["source_reviews"][0]["source_type"] = "unverified"
    invalid.append(unverified)
    missing_review = deepcopy(base)
    missing_review["source_reviews"] = []
    invalid.append(missing_review)
    for report in invalid:
        try:
            validate(report)
        except (ValueError, ValidationError):
            pass
        else:
            raise AssertionError("An invalid discovery report was accepted")

    incomplete["research_gaps"] = ["Fixture: only nine defensible candidates"]
    assert len(validate(incomplete)["psps"]) == 9
    broader_state = {**state, "local_markets": ["Singapore", "Malaysia"]}
    try:
        validate(base, broader_state)
    except ValueError:
        pass
    else:
        raise AssertionError("Missing requested market was accepted")
    print("PASS: iterative ReAct tools, structured report, parent graph integration, provenance and rubric validation")


if __name__ == "__main__":
    asyncio.run(main())
