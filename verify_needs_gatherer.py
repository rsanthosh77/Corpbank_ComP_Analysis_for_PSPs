"""Offline map/reduce and per-PSP ReAct verification, using no live APIs."""

import asyncio
import json
import warnings
from copy import deepcopy

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from graph import build_graph
from needs_gatherer import create_needs_gatherer_node, validated_needs_update
from state import initial_state
from verify_psp_discovery import ScriptedModel, fixture_report as psp_fixture
from check_saved_discovery import require_complete_needs
from research_store import configure_checkpointer


def fixture_report(psp_id, evidence_id):
    # More than ten needs and an arbitrary future category prove there is no cap
    # or category enum; these are transport fixtures, not actual market claims.
    categories = ["APIs", "instant clearing", "FX", "tokenised deposits", "digital assets",
                  "programmable money", "AI liquidity forecasting", "safeguarding",
                  "reconciliation", "virtual accounts", "unanticipated future category"]
    return {
        "psp_id": psp_id,
        "needs": [{"description": f"Fixture {category} requirement", "themes": [category],
                   "basis": "explicit" if i % 2 == 0 else "inferred",
                   "rationale": "Fixture only: example of an attributed fact or inference",
                   "jurisdictions": ["Singapore"], "evidence_ids": [evidence_id]}
                  for i, category in enumerate(categories)],
        "methodology": "Offline fixture only",
        "coverage_notes": ["Fixture conventional, emerging and open-ended follow-up searches"],
        "source_reviews": [{"evidence_id": evidence_id, "publisher": "Fixture publisher",
                            "source_type": "primary", "credibility_reason": "Offline fixture",
                            "supporting_quote": "Fixture evidence only"}],
        "research_gaps": [],
    }


def task_from_messages(messages):
    return json.loads(next(m.content for m in messages if isinstance(m, HumanMessage)))


class NeedsModel(ScriptedModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        task = task_from_messages(messages)
        psp_id = task["psp"]["id"]
        completed = [m for m in messages if isinstance(m, ToolMessage)]
        # Tool messages from other PSP invocations must never enter this history.
        for message in completed:
            assert message.artifact["evidence"][0]["agent"] == f"needs_gatherer:{psp_id}"
        if len(completed) < 2:
            name = "search_web" if not completed else "search_news"
            message = AIMessage(content="", tool_calls=[{
                "id": f"{psp_id}-{len(completed)}", "name": name,
                "args": {"query": psp_id}, "type": "tool_call",
            }])
        else:
            message = AIMessage(content="Fixture research complete")
        return ChatResult(generations=[ChatGeneration(message=message)])

    def with_structured_output(self, schema, **kwargs):
        def format_report(messages):
            task = task_from_messages(messages)
            message = next(m for m in messages if isinstance(m, ToolMessage))
            evidence_id = json.loads(message.content)["evidence"][0]["id"]
            return schema.model_validate(fixture_report(task["psp"]["id"], evidence_id))
        return RunnableLambda(format_report)


class ParallelClient:
    def __init__(self):
        self.calls = []
        self.first_pair_started = asyncio.Event()
        self.first_webs = set()
        self.active = 0
        self.peak = 0

    async def search(self, query, **kwargs):
        self.calls.append((query, kwargs["topic"]))
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            if kwargs["topic"] == "general":
                self.first_webs.add(query)
                if len(self.first_webs) >= 2:
                    self.first_pair_started.set()
                # Sequential execution would time out here.
                await asyncio.wait_for(self.first_pair_started.wait(), timeout=5)
            await asyncio.sleep(0)
            return {"results": [{"url": f"https://example.com/{query}",
                                 "title": "Fixture", "content": "Fixture evidence only",
                                 "published_date": None}]}
        finally:
            self.active -= 1


async def main():
    state = initial_state(as_of_date="2026-10-25", local_markets=["Singapore"])
    state["psps"] = psp_fixture("discovery-evidence")["psps"]
    client = ParallelClient()
    node = create_needs_gatherer_node(model=NeedsModel(), client=client)
    # Existing central findings must survive once, rather than get appended twice.
    state["needs"] = [{"id": "existing", "psp_ids": ["prior"], "description": "Prior need",
                       "themes": [], "basis": "inferred", "evidence_ids": []}]
    state["evidence"] = [{"id": "existing", "agent": "prior", "url": "https://example.com/prior",
                          "title": "Prior", "published_at": None,
                          "retrieved_at": "2026-10-25", "excerpt": "Prior evidence"}]

    async def discovery(state):
        return {}

    mapper_calls = []

    async def mapper(state):
        mapper_calls.append(len(state["needs_research_summaries"]))
        assert len(state["needs_research_summaries"]) == 10
        assert len(state["needs"]) == 111
        assert len(state["evidence"]) == 21
        return {"capabilities": []}

    async def briefing(state):
        return {"briefing": {"fixture_only": True}}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = await build_graph(nodes={
            "psp_discovery": discovery, "bank_discovery": discovery,
            "needs_gatherer": node, "capability_mapper": mapper, "compile_briefing": briefing,
        }).ainvoke(state, config={"max_concurrency": 2})
    assert client.peak == 2
    assert len(client.calls) == 20
    assert mapper_calls == [10]  # Join runs once after every PSP completes.
    assert len({need["id"] for need in result["needs"]}) == 111
    assert {summary["psp_id"] for summary in result["needs_research_summaries"]} == {
        psp["id"] for psp in state["psps"]}
    assert any("unanticipated future category" in n["themes"] for n in result["needs"])
    assert len(result["research_gaps"]) == 10  # Undated news flagged for each PSP.

    # The live diagnostic boundary must save all PSP results before Agent 4.
    saver = configure_checkpointer(InMemorySaver())
    boundary = build_graph(checkpointer=saver, nodes={
        "psp_discovery": discovery, "bank_discovery": discovery,
        "needs_gatherer": create_needs_gatherer_node(model=NeedsModel(), client=ParallelClient()),
        "capability_mapper": mapper, "compile_briefing": briefing,
    })
    boundary.interrupt_before_nodes = ['capability_mapper']
    clean = {**state, 'needs': [], 'evidence': [], 'needs_research_summaries': []}
    config = {'max_concurrency': 2, 'configurable': {'thread_id': 'needs-only-check'}}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        stopped = await boundary.ainvoke(clean, config)
    require_complete_needs(stopped)
    assert len(stopped['needs']) == 110
    assert (await boundary.aget_state(config)).next == ('capability_mapper',)
    assert mapper_calls == [10]  # This diagnostic did not invoke the mapper.
    try:
        require_complete_needs({**stopped, 'needs_research_summaries': stopped['needs_research_summaries'][:-1]})
    except ValueError:
        pass
    else:
        raise AssertionError('Incomplete saved Agent 3 batch accepted')

    psp = state["psps"][0]
    task = {"psp": psp, "as_of_date": state["as_of_date"], "local_scope": state["local_scope"],
            "local_markets": state["local_markets"], "research_question": state["research_question"]}
    evidence = next(e for e in result["evidence"] if e["agent"] == f"needs_gatherer:{psp['id']}")
    messages = [ToolMessage(content="fixture", tool_call_id="web", name="search_web",
                            artifact={"evidence": [evidence]}),
                ToolMessage(content="fixture", tool_call_id="news", name="search_news",
                            artifact={"evidence": []})]
    base = fixture_report(psp["id"], evidence["id"])

    def validate(report):
        return validated_needs_update({"structured_response": report, "messages": messages}, task)

    invalid = []
    wrong_psp = deepcopy(base)
    wrong_psp["psp_id"] = "another-psp"
    invalid.append(wrong_psp)
    invented_ref = deepcopy(base)
    invented_ref["needs"][0]["evidence_ids"] = ["invented"]
    invalid.append(invented_ref)
    fabricated_quote = deepcopy(base)
    fabricated_quote["source_reviews"][0]["supporting_quote"] = "Invented quote"
    invalid.append(fabricated_quote)
    missing_basis = deepcopy(base)
    del missing_basis["needs"][0]["basis"]
    invalid.append(missing_basis)
    duplicate = deepcopy(base)
    duplicate["needs"].append(duplicate["needs"][0])
    invalid.append(duplicate)
    empty = deepcopy(base)
    empty["needs"] = []
    invalid.append(empty)
    unverified = deepcopy(base)
    unverified["source_reviews"][0]["source_type"] = "unverified"
    invalid.append(unverified)
    for report in invalid:
        try:
            validate(report)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid needs report accepted")
    empty["research_gaps"] = ["Fixture: no company-specific needs supported"]
    assert validate(empty)["needs"] == []
    skipped = await node({**state, "psps": []})
    assert skipped["needs"] == skipped["needs_research_summaries"] == []
    assert "skipped" in skipped["research_gaps"][0]
    try:
        await node({**state, "psps": [psp, psp]})
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate PSP tasks accepted")

    class FailingClient:
        async def search(self, query, **kwargs):
            raise TimeoutError("Fixture API timeout")

    failing = create_needs_gatherer_node(model=NeedsModel(), client=FailingClient())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            await failing({**state, "psps": [psp]})
        except TimeoutError:
            pass
        else:
            raise AssertionError("Failed PSP research reported as complete")
    print("PASS: ten isolated ReAct runs, bounded fan-out/join, unfiltered needs, delta merges and failures")


if __name__ == "__main__":
    asyncio.run(main())
