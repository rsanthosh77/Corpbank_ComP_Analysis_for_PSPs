"""Outer orchestration with injectable nodes and explicit worker placeholders."""

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import merge_configs

from state import ResearchState
from research_store import MODEL_OWNER
from langgraph.types import Command

Node = Callable[[ResearchState], Awaitable[dict[str, Any]]]
ROLES = ("psp_discovery", "bank_discovery", "needs_gatherer", "capability_mapper")


class ResearchStageError(RuntimeError):
    """Preserve the failing stage and original error without returning partial success."""


class ResearchStageTimeout(TimeoutError):
    pass


def observed_stage(name, node):
    accepts_config = "config" in inspect.signature(node).parameters

    async def run(state: ResearchState, config: RunnableConfig):
        configured = merge_configs(config, {"metadata": {"research_stage": name}})
        deadline = configured.get("configurable", {}).get("stage_timeout_seconds")
        await adispatch_custom_event("research_stage_started", {"stage": name}, config=configured)
        owner_token = MODEL_OWNER.set(name)
        try:
            async with asyncio.timeout(deadline):
                return await node(state, configured) if accepts_config else await node(state)
        except TimeoutError as exc:
            raise ResearchStageTimeout(f"{name}: stage or request timed out; "
                                       "check NEBIUS_TIMEOUT_SECONDS and RESEARCH_STAGE_TIMEOUT_SECONDS") from exc
        except Exception as exc:
            cause = exc
            while cause.__cause__ is not None:
                cause = cause.__cause__
            if type(cause).__name__ == "BudgetExceeded":
                raise ResearchStageError(f"{name}: {cause}") from exc
            hint = (" Increase NEBIUS_TIMEOUT_SECONDS if generation needs longer; "
                    "provider-side limits may also apply." if "Timeout" in type(exc).__name__ else "")
            raise ResearchStageError(f"{name}: {type(exc).__name__}: {exc}.{hint}") from exc
        finally:
            MODEL_OWNER.reset(owner_token)

    return run


def create_worker_placeholder(*, model: Any, tools: list[Any], prompt: str):
    """Factory boundary; called only after each worker's design is approved."""
    return create_react_agent(model=model, tools=tools, prompt=prompt)


def pending_node(role: str) -> Node:
    async def run(state: ResearchState) -> dict[str, Any]:
        # Future adapter: select inputs -> invoke worker -> validate -> return delta.
        raise NotImplementedError(f"{role}: worker contract not implemented yet")

    return run


def build_graph(*, nodes: dict[str, Node] | None = None, checkpointer=None, discovery_completion=None):
    """Inject approved workers, e.g. nodes={"psp_discovery": agent_1_node}.

    Discovery factories live in psp_discovery.create_psp_discovery_node and
    bank_discovery.create_bank_discovery_node. Missing workers fail explicitly.
    needs_gatherer.create_needs_gatherer_node supplies the per-PSP research stage.
    capability_mapper.create_capability_mapper_node supplies per-bank mapping.
    briefing.create_briefing_node supplies the hybrid briefing compiler.
    """
    supplied = nodes or {}
    names = (*ROLES, "compile_briefing")
    unknown = supplied.keys() - set(names)
    if unknown:
        raise ValueError(f"Unknown nodes: {sorted(unknown)}")

    builder = StateGraph(ResearchState)
    for name in names:
        node = supplied.get(name, pending_node(name))
        if name == "needs_gatherer" and discovery_completion is not None:
            original_needs = node

            async def gated_needs(state, config):
                from discovery_completion import discovery_shortfalls
                # Old checkpoints may already have needs_gatherer scheduled.
                # Command routes those through completion before any LLM call.
                if discovery_shortfalls(state):
                    return Command(goto="complete_discovery")
                update = (await original_needs(state, config) if "config" in inspect.signature(original_needs).parameters
                          else await original_needs(state))
                return Command(update=update, goto="capability_mapper")

            node = gated_needs
        builder.add_node(name, observed_stage(name, node))

    builder.add_edge(START, "psp_discovery")
    builder.add_edge(START, "bank_discovery")
    if discovery_completion is None:
        builder.add_edge(["psp_discovery", "bank_discovery"], "needs_gatherer")
        builder.add_edge("needs_gatherer", "capability_mapper")
    else:
        builder.add_node("complete_discovery", observed_stage("complete_discovery", discovery_completion))
        # Preserve the legacy join channel so already-scheduled checkpoints
        # remain executable. The guarded needs node routes partial discovery.
        builder.add_edge(["psp_discovery", "bank_discovery"], "needs_gatherer")
        builder.add_edge("complete_discovery", "needs_gatherer")
    builder.add_edge("capability_mapper", "compile_briefing")
    builder.add_edge("compile_briefing", END)
    return builder.compile(checkpointer=checkpointer)


def build_research_graph(*, model: Any, client: Any, checkpointer=None):
    """Assemble all approved stages with injected model/client; no API calls here.

    For different models per stage, use build_graph(nodes=...) and the factories.
    The caller owns credentials, the async Tavily client and execution config.
    """
    from bank_discovery import create_bank_discovery_node
    from briefing import create_briefing_node
    from capability_mapper import create_capability_mapper_node
    from needs_gatherer import create_needs_gatherer_node
    from psp_discovery import create_psp_discovery_node
    from discovery_completion import create_discovery_completion_node

    return build_graph(checkpointer=checkpointer,
        discovery_completion=create_discovery_completion_node(model=model, client=client), nodes={
        "psp_discovery": create_psp_discovery_node(model=model, client=client),
        "bank_discovery": create_bank_discovery_node(model=model, client=client),
        "needs_gatherer": create_needs_gatherer_node(model=model, client=client),
        "capability_mapper": create_capability_mapper_node(model=model, client=client),
        "compile_briefing": create_briefing_node(model=model),
    })

