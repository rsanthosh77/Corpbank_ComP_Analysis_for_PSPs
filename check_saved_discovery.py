"""Paid, bounded saved-run check; stop at a selected research-stage boundary.

Usage: .venv/bin/python check_saved_discovery.py RUN_ID
       .venv/bin/python check_saved_discovery.py RUN_ID --through needs
       .venv/bin/python check_saved_discovery.py RUN_ID --through capabilities
Uses the existing run's checkpoints, cache and request budgets, never resets them.
"""

import asyncio
import argparse
import fcntl
import json
import sys
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from tavily import AsyncTavilyClient

from graph import build_research_graph
from discovery_completion import discovery_shortfalls, require_complete_discovery
from nebius_runtime import RuntimeSettings, nebius_session, safe_error_detail
from progress import ResearchProgress
from research_store import ResearchStore, BudgetedSearchClient, configure_checkpointer


def require_complete_needs(state):
    expected = {psp["id"] for psp in state["psps"]}
    summaries = state.get("needs_research_summaries", [])
    actual = [summary["psp_id"] for summary in summaries if summary.get("status") == "completed"]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Agent 3 must have exactly one completed research summary per selected PSP")
    needs = state.get("needs", [])
    if len({need["id"] for need in needs}) != len(needs):
        raise ValueError("Agent 3 produced duplicate need IDs")
    if any(not set(need["psp_ids"]) <= expected for need in needs):
        raise ValueError("Agent 3 produced a need for an unselected PSP")
    for summary in summaries:
        count = sum(summary["psp_id"] in need["psp_ids"] for need in needs)
        if summary["need_count"] != count:
            raise ValueError("Agent 3 summary count does not match saved needs")


def require_complete_capabilities(state):
    banks = {bank["id"]: tier for tier, rows in state["banks"].items() for bank in rows}
    summaries = state.get("capability_research_summaries", [])
    completed = [s["bank_id"] for s in summaries if s.get("status") == "completed"]
    if len(completed) != len(banks) or set(completed) != set(banks):
        raise ValueError("Agent 4 needs one completed research summary per bank")
    expected = {(bank_id, need["id"]) for bank_id in banks for need in state["needs"]}
    rows = state.get("need_assessments", [])
    pairs = [(row["bank_id"], row["need_id"]) for row in rows]
    if len(pairs) != len(expected) or set(pairs) != expected:
        raise ValueError("Agent 4 must assess every bank-need pair exactly once")
    capabilities = state.get("capabilities", [])
    if len({c["id"] for c in capabilities}) != len(capabilities):
        raise ValueError("Agent 4 produced duplicate capability IDs")
    if any(banks.get(c["bank_id"]) != c["tier"] for c in capabilities):
        raise ValueError("Agent 4 changed a bank's tier")
    for summary in summaries:
        if summary["tier"] != banks[summary["bank_id"]] or summary["assessed_need_count"] != len(state["needs"]):
            raise ValueError("Agent 4 summary has an incorrect tier or matrix size")
        if summary["capability_count"] != sum(c["bank_id"] == summary["bank_id"] for c in capabilities):
            raise ValueError("Agent 4 summary capability count does not match saved results")


async def verify(run_id, *, through="discovery"):
    if through not in {"discovery", "needs", "capabilities"}:
        raise ValueError("Unknown verification stage")
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("Supply a saved run ID, not a path")
    folder = Path(__file__).resolve().parent / ".research" / run_id
    if not (folder / "requests.sqlite").is_file() or not (folder / "checkpoints.sqlite").is_file():
        raise ValueError("Saved run databases are missing")
    settings = RuntimeSettings.from_env(require_tavily=True)
    with (folder / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        store = ResearchStore(folder / "requests.sqlite", per_worker=settings.max_searches_per_worker,
                              searches=settings.max_search_requests, models=settings.max_model_requests)
        try:
            saved = store.metadata("run")
            if not saved or any(saved.get(k) != v for k, v in {
                "model": settings.model_id, "base_url": settings.base_url,
                "max_output_tokens": settings.max_output_tokens,
                "reasoning_effort": settings.reasoning_effort,
            }.items()):
                raise ValueError("Saved run settings do not match; use the normal resume command")
            async with (nebius_session(settings, store=store) as (model, _),
                        AsyncSqliteSaver.from_conn_string(str(folder / "checkpoints.sqlite")) as saver,
                        AsyncTavilyClient(api_key=settings.tavily_api_key.get_secret_value()) as tavily):
                configure_checkpointer(saver)
                graph = build_research_graph(model=model, client=BudgetedSearchClient(tavily, store), checkpointer=saver)
                # This is an execution boundary, not a research-node change.
                graph.interrupt_before_nodes = ["compile_briefing" if through == "capabilities" else "capability_mapper"]
                graph.interrupt_after_nodes = ["complete_discovery"] if through == "discovery" else []
                progress = ResearchProgress(usage_reader=store.usage)
                config = {"configurable": {"thread_id": run_id,
                          "stage_timeout_seconds": settings.stage_timeout_seconds},
                          "recursion_limit": settings.recursion_limit,
                          "max_concurrency": settings.concurrency, "callbacks": [progress]}
                before = await graph.aget_state(config)
                if not before.values:
                    raise ValueError("Saved graph state is missing")
                if through == "capabilities":
                    require_complete_discovery(before.values)
                    require_complete_needs(before.values)
                    if before.values.get("capability_research_summaries"):
                        require_complete_capabilities(before.values)
                        print("PASS: Agent 4 results are already saved; no model or search calls made")
                        return
                    if before.next != ("capability_mapper",):
                        raise ValueError("Saved run must be ready at capability_mapper for this check")
                elif through == "needs":
                    require_complete_discovery(before.values)
                    if before.values.get("needs_research_summaries"):
                        require_complete_needs(before.values)
                        print("PASS: Agent 3 results are already saved; no model or search calls made")
                        return
                    if set(before.next) != {"needs_gatherer"}:
                        raise ValueError("Saved run must be ready at needs_gatherer for this check")
                elif not discovery_shortfalls(before.values):
                    print("PASS: complete validated discovery results are already saved")
                    return
                if through == "discovery" and set(before.next) - {"psp_discovery", "bank_discovery", "complete_discovery", "needs_gatherer"}:
                    raise ValueError("Saved graph is beyond discovery; no later stages were started")
                async with progress.heartbeat():
                    async for update in graph.astream(None, config, stream_mode="updates"):
                        for stage, delta in update.items():
                            if stage == "needs_gatherer" and delta is None:
                                print("Discovery completeness gate: routing to completion", flush=True)
                            elif stage != "__interrupt__":
                                print("Completed:", stage, flush=True)
                snapshot = await graph.aget_state(config)
                require_complete_discovery(snapshot.values)
                if through == "capabilities":
                    require_complete_capabilities(snapshot.values)
                    if snapshot.next != ("compile_briefing",):
                        raise ValueError("Agent 4 check did not stop before the briefing compiler")
                    print("PASS: live Agent 4 validated and saved; stopped before briefing compilation")
                    print("Bank-need assessments:", len(snapshot.values["need_assessments"]),
                          "Capabilities:", len(snapshot.values["capabilities"]))
                elif through == "needs":
                    require_complete_needs(snapshot.values)
                    if snapshot.next != ("capability_mapper",):
                        raise ValueError("Agent 3 check did not stop at the capability-mapping boundary")
                    print("PASS: live Agent 3 research validated and saved; stopped before Agent 4")
                    print("Completed PSPs:", len(snapshot.values["needs_research_summaries"]),
                          "Needs:", len(snapshot.values["needs"]))
                else:
                    print("PASS: live discovery validated and saved; stopped before needs gathering")
                print("PSPs:", len(snapshot.values.get("psps", [])), "Banks:",
                      sum(len(b) for b in snapshot.values.get("banks", {}).values()))
        finally:
            usage = store.usage()
            (folder / "usage.json").write_text(json.dumps(usage, indent=2) + "\n")
            print("Cumulative usage:", usage["model"]["attempts"], "model attempts,",
                  usage["search"]["attempts"], "search attempts", flush=True)
            store.close()


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("run_id")
        parser.add_argument("--through", choices=("discovery", "needs", "capabilities"), default="discovery")
        args = parser.parse_args()
        asyncio.run(verify(args.run_id, through=args.through))
    except Exception as exc:
        print("FAIL:", type(exc).__name__, safe_error_detail(exc))
        raise SystemExit(1)
