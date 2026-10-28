"""Offline bank contract checks and parallel execution of both real ReAct workers."""

import asyncio
import json
import warnings
from copy import deepcopy

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda

from bank_discovery import (
    TIERS, BankDiscoveryReport, create_bank_discovery_node, validated_bank_update,
)
from graph import build_graph
from psp_discovery import create_psp_discovery_node
from state import initial_state
from verify_psp_discovery import FakeTavily, ScriptedModel


def fixture_report(evidence_id):
    assessment = {"rating": "moderate", "assessment": "Fixture corporate business",
                  "evidence_ids": [evidence_id], "gaps": []}
    banks = {tier: [{
        "id": f"{tier}-{i}", "name": f"Fixture {tier} bank {i}",
        "group_id": f"{tier}-group-{i}", "headquarters_country": "Singapore",
        "countries": ["Singapore"], "tier_rationale": "Fixture tier assignment",
        "selection_rationale": "Fixture supported comparison",
        "evidence_ids": [evidence_id],
        "rubric": {key: deepcopy(assessment) for key in (
            "corporate_cash_management", "corporate_apis", "clearing_connectivity")},
    } for i in range(4)] for tier in TIERS}
    return {
        "banks": banks, "methodology": "Offline transport fixture only",
        "allocation_rationale": "Fixture: four in each tier",
        "expansion_rationale": "Fixture: distinct coverage warrants 12 institutions",
        "tier_gaps": {tier: [] for tier in TIERS},
        "scope_evidence_ids": [],
        "market_coverage": [{"market": "Singapore", "findings": "Fixture coverage",
                             "evidence_ids": [evidence_id], "gaps": []}],
        "excluded_candidates": [], "research_gaps": [],
        "source_reviews": [{"evidence_id": evidence_id, "publisher": "Fixture publisher",
                            "source_type": "primary", "credibility_reason": "Offline fixture",
                            "supporting_quote": "Fixture evidence only"}],
    }


class BankModel(ScriptedModel):
    def with_structured_output(self, schema, **kwargs):
        def format_report(messages):
            message = next(m for m in messages if isinstance(m, ToolMessage))
            evidence_id = json.loads(message.content)["evidence"][0]["id"]
            return schema.model_validate(fixture_report(evidence_id))
        return RunnableLambda(format_report)


async def main():
    state = initial_state(as_of_date="2026-10-25", local_markets=["Singapore"])
    bank_client, psp_client = FakeTavily(), FakeTavily()
    started = {key: asyncio.Event() for key in ("bank", "psp")}

    def parallel_check(name, other, node):
        async def run(state, config=None):
            started[name].set()
            await asyncio.wait_for(started[other].wait(), timeout=5)
            return await node(state, config)
        return run

    bank_node = create_bank_discovery_node(model=BankModel(), client=bank_client)
    psp_node = create_psp_discovery_node(model=ScriptedModel(), client=psp_client)

    async def needs(state):
        assert len(state["psps"]) == 10
        assert sum(len(state["banks"][tier]) for tier in TIERS) == 12
        assert len(state["evidence"]) == 4  # Web/news for each agent.
        return {"needs": []}

    async def capabilities(state):
        return {"capabilities": []}

    async def briefing(state):
        return {"briefing": {"fixture_only": True}}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = await build_graph(nodes={
            "psp_discovery": parallel_check("psp", "bank", psp_node),
            "bank_discovery": parallel_check("bank", "psp", bank_node),
            "needs_gatherer": needs, "capability_mapper": capabilities,
            "compile_briefing": briefing,
        }).ainvoke(state)
    assert bank_client.topics == psp_client.topics == ["general", "news"]
    assert result["bank_discovery_summary"]["selected_count"] == 12
    assert result["bank_discovery_summary"]["tier_counts"] == {tier: 4 for tier in TIERS}
    evidence = next(e for e in result["evidence"] if e["agent"] == "bank_discovery")
    base = fixture_report(evidence["id"])
    messages = [ToolMessage(content="fixture", tool_call_id="web", name="search_web",
                            artifact={"evidence": [evidence]}),
                ToolMessage(content="fixture", tool_call_id="news", name="search_news",
                            artifact={"evidence": []})]

    def validate(report):
        return validated_bank_update({"structured_response": report, "messages": messages}, state)

    invalid = []
    no_expansion_reason = deepcopy(base)
    no_expansion_reason["expansion_rationale"] = None
    invalid.append(no_expansion_reason)
    duplicate_group = deepcopy(base)
    duplicate_group["banks"][TIERS[1]][0]["group_id"] = base["banks"][TIERS[0]][0]["group_id"]
    invalid.append(duplicate_group)
    no_cash_business = deepcopy(base)
    no_cash_business["banks"][TIERS[0]][0]["rubric"]["corporate_cash_management"] = {
        "rating": "unknown", "assessment": "Not established", "evidence_ids": [], "gaps": ["Unknown"]}
    invalid.append(no_cash_business)
    empty_tier = deepcopy(base)
    empty_tier["banks"][TIERS[2]] = []
    empty_tier["research_gaps"] = ["Fixture incomplete shortlist"]
    invalid.append(empty_tier)
    forged = deepcopy(base)
    forged["banks"][TIERS[0]][0]["evidence_ids"] = ["not-retrieved"]
    invalid.append(forged)
    secondary_cash = deepcopy(base)
    secondary_cash["source_reviews"][0]["source_type"] = "reputable_news"
    invalid.append(secondary_cash)
    missing_market = deepcopy(base)
    missing_market["market_coverage"][0]["market"] = "Malaysia"
    invalid.append(missing_market)
    missing_quote = deepcopy(base)
    missing_quote["source_reviews"][0]["supporting_quote"] = "Fabricated"
    invalid.append(missing_quote)
    for report in invalid:
        try:
            validate(report)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid bank report accepted")
    empty_tier["tier_gaps"][TIERS[2]] = ["No defensible local bank found in fixture"]
    assert not validate(empty_tier)["banks"][TIERS[2]]
    print("PASS: both ReAct workers in parallel, tiered bank state, justified expansion, source and eligibility checks")


if __name__ == "__main__":
    asyncio.run(main())
