"""Offline request timeout, stage deadline and credential-safe progress checks."""

import asyncio
from uuid import uuid4

import httpx
from langchain_core.messages import HumanMessage

from graph import ResearchStageError, ResearchStageTimeout, build_graph
from nebius_runtime import RuntimeSettings, create_nebius_model, request_timeout
from progress import ResearchProgress
from state import initial_state


async def main():
    settings = RuntimeSettings(nebius_api_key="synthetic-private-key", max_retries=0)
    assert request_timeout(settings).read == 600
    assert request_timeout(settings).connect == 15
    assert request_timeout(settings).write == request_timeout(settings).pool == 60
    emitted = []
    progress = ResearchProgress(emit=emitted.append)
    requests = []

    def timeout_response(request):
        requests.append(request)
        raise httpx.ReadTimeout("Synthetic read timed out", request=request)

    async def no_op(state):
        return {}

    with httpx.Client(transport=httpx.MockTransport(timeout_response)) as sync_http:
        async with httpx.AsyncClient(transport=httpx.MockTransport(timeout_response)) as async_http:
            model = create_nebius_model(settings, http_client=sync_http, http_async_client=async_http)

            async def research(state, config):
                await model.ainvoke("Synthetic input, never log this prompt", config=config)
                return {}

            graph = build_graph(nodes={"psp_discovery": research, "bank_discovery": no_op})
            try:
                await graph.ainvoke(initial_state(as_of_date="2026-10-25", local_markets=[]),
                                    config={"callbacks": [progress]})
            except ResearchStageError as exc:
                assert "psp_discovery" in str(exc) and "Timeout" in str(exc)
                assert "NEBIUS_TIMEOUT_SECONDS" in str(exc)
                assert "Timeout" in type(exc.__cause__).__name__
            else:
                raise AssertionError("Provider timeout was hidden")
    assert len(requests) == 1  # max_retries=0 is honored.
    assert not progress.active
    assert any("Started: psp_discovery" in line for line in emitted)
    assert any("psp_discovery: model request started" in line for line in emitted)

    cancelled = []

    async def blocked(state):
        try:
            await asyncio.sleep(5)
        finally:
            cancelled.append(True)
        return {}

    try:
        await build_graph(nodes={"psp_discovery": blocked, "bank_discovery": no_op}).ainvoke(
            initial_state(as_of_date="2026-10-25", local_markets=[]),
            config={"configurable": {"stage_timeout_seconds": 0.01}},
        )
    except ResearchStageTimeout as exc:
        assert "psp_discovery" in str(exc)
    else:
        raise AssertionError("Stage deadline did not fire")
    assert cancelled == [True]

    identity = uuid4()
    await progress.on_chat_model_start({}, [[HumanMessage(content="Never print private prompt")]],
                                       run_id=identity, metadata={"research_stage": "bank_discovery"})
    async with progress.heartbeat(interval=0.01):
        await asyncio.sleep(0.025)
    await progress.on_llm_end(None, run_id=identity)
    assert any("Still working: bank_discovery" in line for line in emitted)
    assert not progress.active
    text = "\n".join(emitted)
    assert "Never print private prompt" not in text and "synthetic-private-key" not in text
    assert "Synthetic input" not in text
    print("PASS: native request timeouts, stage attribution/deadlines, cancellation and safe heartbeat progress")


if __name__ == "__main__":
    asyncio.run(main())
