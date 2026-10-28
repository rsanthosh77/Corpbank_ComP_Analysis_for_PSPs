"""Offline verification of real graph scheduling; no model or Tavily calls."""

import asyncio
from langgraph.checkpoint.memory import InMemorySaver

from graph import build_graph
from state import initial_state


async def main():
    started = {name: asyncio.Event() for name in ("psp", "bank")}

    def discovery(name, other, field):
        async def run(state):
            started[name].set()
            # A sequential graph would time out: both branches must start.
            await asyncio.wait_for(started[other].wait(), timeout=5)
            entity = {"id": name, "name": f"Fixture {name}", "countries": [],
                      "selection_rationale": "Scheduling fixture only",
                      "evidence_ids": [name]}
            if field == "psps":
                entity["rubric"] = {
                    criterion: {"rating": "unknown", "assessment": "Scheduling fixture",
                                "evidence_ids": [], "gaps": ["Not researched"]}
                    for criterion in ("annual_tpv", "developer_ecosystem", "market_position",
                                      "funding_and_valuation", "licensing")
                }
            else:
                entity.update(group_id="fixture-bank-group", headquarters_country="Singapore",
                              tier_rationale="Scheduling fixture", rubric={
                    criterion: {"rating": "moderate", "assessment": "Scheduling fixture",
                                "evidence_ids": [name], "gaps": []}
                    for criterion in ("corporate_cash_management", "corporate_apis", "clearing_connectivity")
                })
            return {
                field: ([entity] if field == "psps" else {
                    "tier_1_global": [entity], "tier_2_regional": [], "tier_3_local": []}),
                "evidence": [{"id": name, "agent": name,
                              "url": f"https://example.com/{name}",
                              "title": "Fixture", "published_at": None,
                              "retrieved_at": "2026-10-25", "excerpt": "Fixture"}],
            }
        return run

    async def needs(state):
        assert len(state["psps"]) == len(state["banks"]["tier_1_global"]) == 1
        assert {e["id"] for e in state["evidence"]} == {"psp", "bank"}
        return {"needs": [{"id": "n1", "psp_ids": ["psp"],
                           "description": "Fixture programmable money requirement",
                           "themes": ["arbitrary emerging category"],
                           "basis": "explicit", "evidence_ids": ["psp"]}]}

    async def capabilities(state):
        assert state["needs"][0]["id"] == "n1"
        return {"capabilities": []}

    async def briefing(state):
        return {"briefing": {"fixture_only": True, "need_count": len(state["needs"])}}

    graph = build_graph(nodes={
        "psp_discovery": discovery("psp", "bank", "psps"),
        "bank_discovery": discovery("bank", "psp", "banks"),
        "needs_gatherer": needs,
        "capability_mapper": capabilities,
        "compile_briefing": briefing,
    })
    result = await graph.ainvoke(initial_state(as_of_date="2026-10-25", local_markets=[]))
    assert result["briefing"] == {"fixture_only": True, "need_count": 1}
    assert len(result["evidence"]) == 2
    # The paid discovery-only diagnostic must stop before any needs work, and
    # its checkpoint must remain resumable by the normal assembled graph.
    nodes = {
        "psp_discovery": discovery("psp", "bank", "psps"),
        "bank_discovery": discovery("bank", "psp", "banks"),
        "needs_gatherer": needs, "capability_mapper": capabilities,
        "compile_briefing": briefing,
    }
    saver = InMemorySaver()
    diagnostic = build_graph(nodes=nodes, checkpointer=saver)
    diagnostic.interrupt_before_nodes = ["needs_gatherer"]
    config = {"configurable": {"thread_id": "discovery-only"}}
    stopped = await diagnostic.ainvoke(initial_state(as_of_date="2026-10-25", local_markets=[]), config)
    assert not stopped.get("needs") and not stopped.get("briefing")
    assert (await diagnostic.aget_state(config)).next == ("needs_gatherer",)
    normal = build_graph(nodes=nodes, checkpointer=saver)
    resumed = await normal.ainvoke(None, config)
    assert resumed["briefing"]["need_count"] == 1
    assert len(resumed["evidence"]) == 2
    print("PASS: parallel discovery, join barrier, evidence merge, and downstream flow")


if __name__ == "__main__":
    asyncio.run(main())
