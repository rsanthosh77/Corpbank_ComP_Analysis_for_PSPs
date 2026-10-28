"""Offline budgets, request replay and real SQLite failure/resume regression."""

import asyncio
import json
import tempfile
import warnings
from pathlib import Path

import httpx
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from graph import build_graph, ResearchStageError
from nebius_runtime import RuntimeSettings, create_nebius_model
from psp_discovery import create_psp_discovery_node
from research_store import ResearchStore, BudgetedSearchClient, CachedModelTransport, BudgetExceeded, configure_checkpointer
from search_tools import build_search_tools
from state import initial_state
from verify_psp_discovery import FakeTavily, fixture_report
from needs_gatherer import create_needs_gatherer_node
from verify_needs_gatherer import NeedsModel


async def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        calls = []

        class Search:
            async def search(self, query, **kwargs):
                calls.append(query)
                return {"results": [{"url": "https://example.com", "content": "Synthetic"}]}

        store = ResearchStore(root / "limits.sqlite", per_worker=1, searches=2, models=1)
        # Legacy run migration adds reasoning effort to an existing metadata key.
        legacy = {"model": "synthetic", "max_output_tokens": 32768}
        store.metadata("run", legacy)
        migrated = {**store.metadata("run"), "reasoning_effort": "low"}
        assert store.metadata("run", migrated) == migrated
        assert store.metadata("run", migrated) == migrated  # Repeated migration is safe.
        search = BudgetedSearchClient(Search(), store)
        tools = build_search_tools(client=search, agent_name="psp_discovery", as_of_date="2026-10-26")
        first = await tools[0].ainvoke({"query": "same query"})
        again = await tools[0].ainvoke({"query": "same query"})
        assert first == again  # Includes stable retrieval date on replay.
        assert len(calls) == 1
        exhausted = await tools[1].ainvoke({"query": "different"})
        assert exhausted["budget_exhausted"] and not exhausted["evidence"]
        assert len(calls) == 1
        assert store.usage()["search"]["attempts"] == 1
        assert store.usage()["search"]["cache_hits"] == 1
        store.reserve("model", "test")
        try:
            store.reserve("model", "test")
        except BudgetExceeded:
            pass
        else:
            raise AssertionError("Global model request budget was bypassed")
        store.close()

        # Reopen the persistent cache, not an in-memory copy.
        store = ResearchStore(root / "limits.sqlite", per_worker=1)
        assert store.metadata("run") == migrated
        reopened = build_search_tools(client=BudgetedSearchClient(Search(), store),
                                      agent_name="psp_discovery", as_of_date="2026-10-26")
        assert await reopened[0].ainvoke({"query": "same query"}) == first
        assert len(calls) == 1
        store.close()

        provider_calls = []
        allow_report = False
        bank_runs = []
        tavily = FakeTavily()
        settings = RuntimeSettings(nebius_api_key="synthetic-secret", max_retries=0)

        def provider(request):
            nonlocal allow_report
            provider_calls.append(request)
            data = json.loads(request.content)
            tool = data["tools"][0]["function"]["name"]
            if tool == "DiscoveryReport":
                if not allow_report:
                    return httpx.Response(402, json={"error": {"message": "Synthetic exhausted quota"}})
                tool_message = next(m for m in data["messages"] if m["role"] == "tool")
                evidence_id = json.loads(tool_message["content"])["evidence"][0]["id"]
                name, args = tool, fixture_report(evidence_id)
            else:
                completed = [m for m in data["messages"] if m["role"] == "tool"]
                if len(completed) >= 2:
                    return httpx.Response(200, json={
                        "id": "synthetic", "object": "chat.completion", "created": 0,
                        "model": settings.model_id, "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                        "choices": [{"index": 0, "finish_reason": "stop",
                                     "message": {"role": "assistant", "content": "Synthetic research complete"}}]})
                name = "search_web" if not completed else "search_news"
                args = {"query": "Synthetic research"}
            return httpx.Response(200, json={
                "id": "synthetic", "object": "chat.completion", "created": 0, "model": settings.model_id,
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                    "role": "assistant", "content": None, "tool_calls": [{"id": name + "-call",
                    "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}}]})

        async def banks(state):
            bank_runs.append(True)
            return {"research_gaps": ["Synthetic completed bank discovery"]}

        async def noop(state):
            return {}

        async def compile_report(state):
            return {"briefing": {"psp_count": len(state["psps"])}}

        for resumed in (False, True):
            store = ResearchStore(root / "replay.sqlite")
            with httpx.Client(transport=httpx.MockTransport(provider)) as sync_http:
                async with (httpx.AsyncClient(transport=CachedModelTransport(store, httpx.MockTransport(provider))) as async_http,
                            AsyncSqliteSaver.from_conn_string(str(root / "graph.sqlite")) as checkpointer):
                    configure_checkpointer(checkpointer)
                    model = create_nebius_model(settings, http_client=sync_http, http_async_client=async_http)
                    graph = build_graph(checkpointer=checkpointer, nodes={
                        "psp_discovery": create_psp_discovery_node(
                            model=model, client=BudgetedSearchClient(tavily, store)),
                        "bank_discovery": banks, "needs_gatherer": noop,
                        "capability_mapper": noop, "compile_briefing": compile_report,
                    })
                    config = {"configurable": {"thread_id": "synthetic-resume"}}
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", DeprecationWarning)
                        if not resumed:
                            try:
                                await graph.ainvoke(initial_state(as_of_date="2026-10-26",
                                                                 local_markets=["Singapore"]), config)
                            except ResearchStageError:
                                pass
                            else:
                                raise AssertionError("Synthetic quota error was hidden")
                            assert len(provider_calls) == 4
                            allow_report = True
                        else:
                            result = await graph.ainvoke(None, config)
                            assert result["briefing"]["psp_count"] == 10
                            assert len(provider_calls) == 5  # Checkpoint resumes at the failed formatter.
                            assert len(tavily.topics) == 2  # Neither search repeated remotely.
                            assert len(bank_runs) == 1  # Pending successful sibling was retained.
                            # Nested ReAct checkpoints already retain completed turns.
                            assert store.usage()["model"]["cache_hits"] == 0
                            previous = provider_calls[0]
                            replayed = await async_http.post(str(previous.url), content=previous.content,
                                                             headers={"authorization": "Bearer synthetic-secret"})
                            assert replayed.status_code == 200 and len(provider_calls) == 5
                            assert store.usage()["model"]["cache_hits"] == 1
                            assert store.usage()["model"]["input_tokens"] == 40
                            assert store.usage()["model"]["output_tokens"] == 20
                            assert len(result["evidence"]) == 2
            store.close()
        raw = (root / "replay.sqlite").read_bytes()
        assert b"synthetic-secret" not in raw

        # Failed per-PSP fan-out must not duplicate a successful sibling's deltas.
        fail_once = True
        search_calls = []

        class FlakySearch:
            async def search(self, query, **kwargs):
                nonlocal fail_once
                search_calls.append((query, kwargs["topic"]))
                if query == "fixture-1" and kwargs["topic"] == "news" and fail_once:
                    fail_once = False
                    raise RuntimeError("Synthetic search failure")
                return {"results": [{"url": "https://example.com/" + query,
                                     "content": "Fixture evidence only"}]}

        state = initial_state(as_of_date="2026-10-26", local_markets=["Singapore"])
        state["psps"] = fixture_report("discovery")["psps"][:2]

        async def final_needs(state):
            assert len(state["needs"]) == 22
            assert len(state["needs_research_summaries"]) == 2
            assert len(state["evidence"]) == 4
            return {"briefing": {"resumed_needs": True}}

        for resumed in (False, True):
            store = ResearchStore(root / "fanout-cache.sqlite")
            async with AsyncSqliteSaver.from_conn_string(str(root / "fanout.sqlite")) as checkpointer:
                configure_checkpointer(checkpointer)
                graph = build_graph(checkpointer=checkpointer, nodes={
                    "psp_discovery": noop, "bank_discovery": noop,
                    "needs_gatherer": create_needs_gatherer_node(
                        model=NeedsModel(), client=BudgetedSearchClient(FlakySearch(), store)),
                    "capability_mapper": noop, "compile_briefing": final_needs,
                })
                config = {"configurable": {"thread_id": "fanout"}, "max_concurrency": 2}
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    if not resumed:
                        try:
                            await graph.ainvoke(state, config)
                        except ResearchStageError:
                            pass
                        else:
                            raise AssertionError("Synthetic per-PSP failure was hidden")
                    else:
                        outcome = await graph.ainvoke(None, config)
                        assert outcome["briefing"]["resumed_needs"]
                        assert search_calls.count(("fixture-0", "general")) == 1
                        assert search_calls.count(("fixture-0", "news")) == 1
            store.close()
    print("PASS: search/model budgets, durable cache replay, usage totals, SQLite resume and no duplicate paid work")


if __name__ == "__main__":
    asyncio.run(main())
