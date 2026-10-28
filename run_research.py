"""CLI: an explicit compatibility check or a complete live research run."""

import argparse
import asyncio
import json
import sys
import re
import fcntl
import os
import shlex
from uuid import uuid4
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from pydantic import ValidationError
from tavily import AsyncTavilyClient

from compatibility_check import check_provider_contracts
from graph import build_research_graph
from nebius_runtime import (
    ConfigurationError, RuntimeSettings, check_model_access, nebius_session, safe_error_detail,
)
from state import initial_state
from progress import ResearchProgress
from research_store import ResearchStore, BudgetedSearchClient, configure_checkpointer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Live model access, ReAct and five schema checks; no Tavily calls")
    research = commands.add_parser("run", help="Run the full live research pipeline")
    research.add_argument("--as-of")
    research.add_argument("--market", action="append", default=[],
                          help="Explicit ASEAN market; repeat for the full market list. Omit for researched ASEAN scope.")
    research.add_argument("--output", type=Path, help="New JSON report path; existing files are never overwritten")
    identifiers = research.add_mutually_exclusive_group()
    identifiers.add_argument("--run-id", help="Identifier for a new saved run")
    identifiers.add_argument("--resume", help="Resume a previously saved run ID")
    research.add_argument("--state-dir", type=Path, default=Path(__file__).resolve().parent / ".research")
    return parser.parse_args(argv)


async def execute(args):
    settings = RuntimeSettings.from_env(require_tavily=args.command == "run")
    if args.command == "check":
        async with nebius_session(settings) as (model, http_client):
            await check_model_access(settings, http_client)
            print(f"Model access confirmed: {settings.model_id}", flush=True)
            await check_provider_contracts(model, model_id=settings.model_id,
                                           progress=lambda message: print(message, flush=True))
            print("Compatibility passed using synthetic data; live research accuracy is not established.")
        return
    run_id = args.resume or args.run_id or uuid4().hex
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id):
        raise ConfigurationError("Run IDs may contain only letters, digits, underscores and hyphens")
    folder = args.state_dir / run_id
    if args.resume and not (folder / "requests.sqlite").exists():
        raise ConfigurationError("No saved run exists with that ID")
    if args.as_of:
        date.fromisoformat(args.as_of)
    if any(not market.strip() for market in args.market):
        raise ConfigurationError("--market values must not be empty")
    if args.output and args.output.exists():
        raise FileExistsError("Output already exists; choose a new --output path")
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (folder / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ConfigurationError("This run is already active in another process") from exc
        store = ResearchStore(folder / "requests.sqlite", per_worker=settings.max_searches_per_worker,
                              searches=settings.max_search_requests, models=settings.max_model_requests)
        try:
            saved = store.metadata("run")
            if saved and not args.resume:
                raise ConfigurationError("Run ID already exists; use --resume or a new --run-id")
            if args.resume and saved is None:
                raise ConfigurationError("Saved run metadata is missing")
            if saved:
                if (saved["model"] != settings.model_id or saved["base_url"] != settings.base_url
                        or saved["max_output_tokens"] != settings.max_output_tokens):
                    raise ConfigurationError("Resume requires the same model, endpoint and output-token setting")
                # Allow old runs to adopt explicit reasoning control once. The
                # request cache keys include the body, so changed requests miss.
                if saved.get("reasoning_effort", settings.reasoning_effort) != settings.reasoning_effort:
                    raise ConfigurationError("Resume requires the same reasoning-effort setting")
                if "reasoning_effort" not in saved:
                    saved["reasoning_effort"] = settings.reasoning_effort
                    store.metadata("run", saved)
                if args.as_of and args.as_of != saved["as_of"]:
                    raise ConfigurationError("Resume cannot change the research date")
                if args.market and [m.strip() for m in args.market] != saved["markets"]:
                    raise ConfigurationError("Resume cannot change the market scope")
            else:
                saved = {"model": settings.model_id, "base_url": settings.base_url,
                         "max_output_tokens": settings.max_output_tokens,
                         "reasoning_effort": settings.reasoning_effort,
                         "as_of": args.as_of or datetime.now(ZoneInfo("Asia/Singapore")).date().isoformat(),
                         "markets": [m.strip() for m in args.market]}
                store.metadata("run", saved)
            output = args.output or Path("output") / f"briefing-{saved['as_of']}.json"
            if output.exists():
                raise FileExistsError("Output already exists; choose a new --output path")
            output.parent.mkdir(parents=True, exist_ok=True)
            state_arg = (" --state-dir " + shlex.quote(str(args.state_dir))
                         if args.state_dir != Path(__file__).resolve().parent / ".research" else "")
            print(f"Saved run ID: {run_id}\nResume command: uv run python run_research.py run --resume {run_id}{state_arg}", flush=True)
            await execute_saved(args, settings, store, folder, saved, run_id, output)
        finally:
            usage = store.usage()
            (folder / "usage.json").write_text(json.dumps(usage, indent=2) + "\n")
            print(f"Usage: {usage['model']['attempts']} model attempts, "
                  f"{usage['model']['input_tokens']} input / {usage['model']['output_tokens']} output tokens, "
                  f"{usage['search']['attempts']} search attempts; usage saved in {folder / 'usage.json'}", flush=True)
            store.close()


async def execute_saved(args, settings, store, folder, saved, run_id, output):
    async with (nebius_session(settings, store=store) as (model, http_client),
                AsyncSqliteSaver.from_conn_string(str(folder / "checkpoints.sqlite")) as checkpointer):
        os.chmod(folder / "checkpoints.sqlite", 0o600)
        configure_checkpointer(checkpointer)
        async with AsyncTavilyClient(api_key=settings.tavily_api_key.get_secret_value()) as tavily:
            graph = build_research_graph(model=model, client=BudgetedSearchClient(tavily, store),
                                         checkpointer=checkpointer)
            state = initial_state(as_of_date=saved["as_of"], local_markets=saved["markets"])
            progress = ResearchProgress(usage_reader=store.usage)
            config = {"max_concurrency": settings.concurrency, "recursion_limit": settings.recursion_limit,
                      "configurable": {"stage_timeout_seconds": settings.stage_timeout_seconds, "thread_id": run_id},
                      "callbacks": [progress]}
            snapshot = await graph.aget_state(config)
            finished = bool(snapshot.values) and not snapshot.next and snapshot.values.get("briefing")
            if not finished:
                await check_model_access(settings, http_client)
                print(f"Model access confirmed: {settings.model_id}", flush=True)
            print(f"Request read timeout: {settings.timeout_seconds:g}s; "
                  f"SDK retries: {settings.max_retries}; stage deadline: {settings.stage_timeout_seconds:g}s", flush=True)
            print(f"Budgets: {settings.max_searches_per_worker} searches/worker, "
                  f"{settings.max_search_requests} total searches, {settings.max_model_requests} model attempts", flush=True)
            briefing = snapshot.values.get("briefing") if finished else None
            async with progress.heartbeat():
                if not finished:
                    inputs = None if args.resume and snapshot.values else state
                    async for update in graph.astream(inputs, config=config, stream_mode="updates"):
                        for stage, delta in update.items():
                            if stage == "needs_gatherer" and delta is None:
                                print("Discovery completeness gate: routing to completion", flush=True)
                            else:
                                print(f"Completed: {stage}", flush=True)
                            if stage == "compile_briefing":
                                briefing = delta["briefing"]
            if briefing is None:
                raise ValueError("Pipeline ended without a validated briefing")
            document = {
                "run": {"model": settings.model_id, "provider": "Nebius Token Factory",
                        "run_id": run_id, "usage": store.usage(),
                        "completed_at": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
                        "concurrency": settings.concurrency, "recursion_limit": settings.recursion_limit},
                "briefing": briefing,
            }
            # Serialize before opening the final file; never truncate an existing report.
            encoded = json.dumps(document, ensure_ascii=False, indent=2)
            with output.open("x", encoding="utf-8") as handle:
                handle.write(encoded + "\n")
            print(f"Report saved: {output}")


def main(argv=None):
    args = parse_args(argv)
    try:
        asyncio.run(execute(args))
    except (ConfigurationError, FileExistsError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ValidationError as exc:
        # Omit input values (including credentials and long research excerpts).
        locations = [".".join(map(str, error["loc"])) for error in exc.errors(include_input=False)]
        print("Validation failed in: " + ", ".join(locations[:5]), file=sys.stderr)
        return 1
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        detail = safe_error_detail(exc)
        print(f"Run failed: {type(exc).__name__}" + (f" (HTTP {status})" if status else "") +
              (f": {detail}" if detail else "") + ". No successful report was written.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
