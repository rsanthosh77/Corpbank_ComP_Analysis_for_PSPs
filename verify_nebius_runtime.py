"""Offline HTTP-level checks for the actual native Nebius adapter and CLI."""

import asyncio
import contextlib
import io
import json
import os
import tempfile
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

from compatibility_check import check_provider_contracts, contract_samples, validate_sample_markers
from nebius_runtime import MODEL_ID, ModelOutputError, RuntimeSettings, check_model_access, create_nebius_model, safe_error_detail
from run_research import execute, main, parse_args
from verify_briefing import UnifiedModel
from verify_psp_discovery import FakeTavily


async def main_async():
    requests = []
    bank_attempts = 0
    samples = {schema.__name__: sample for schema, sample in contract_samples()}
    settings = RuntimeSettings(nebius_api_key="synthetic-nebius-key", tavily_api_key="synthetic-tavily-key")

    def handle(request):
        nonlocal bank_attempts
        requests.append(request)
        assert request.headers["authorization"] == "Bearer synthetic-nebius-key"
        if request.url.path.endswith("/models"):
            assert request.extensions["timeout"]["read"] == 30
            return httpx.Response(200, json={"data": [{"id": MODEL_ID}]})
        assert request.url.path == "/v1/chat/completions"
        assert request.extensions["timeout"]["read"] == 600
        assert request.extensions["timeout"]["connect"] == 15
        payload = json.loads(request.content)
        assert payload["model"] == MODEL_ID
        assert payload["max_tokens"] == 32768
        assert payload["reasoning_effort"] == "low"
        tools = payload.get("tools", [])
        assert tools
        name = tools[0]["function"]["name"]
        if name in samples:
            assert payload["tool_choice"] == {"type": "function", "function": {"name": name}}
            assert "parallel_tool_calls" not in payload
            arguments = json.loads(json.dumps(samples[name]))
            if name == "BankDiscoveryReport":
                bank_attempts += 1
                if bank_attempts == 1:
                    # Reproduce the live malformed nesting; the check must retry.
                    bank = arguments["banks"]["tier_1_global"][0]
                    bank.update(bank.pop("rubric"))
        elif payload["messages"][-1]["role"] == "tool":
            return httpx.Response(200, json={
                "id": "synthetic", "object": "chat.completion", "created": 0, "model": MODEL_ID,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "Synthetic check complete"}}],
            })
        else:
            assert name == "compatibility_echo"
            arguments = {"value": "ok"}
        return httpx.Response(200, json={
            "id": "synthetic", "object": "chat.completion", "created": 0, "model": MODEL_ID,
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "tool_calls": [{
                    "id": "synthetic-call", "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }],
            }}],
        })

    transport = httpx.MockTransport(handle)
    with httpx.Client(transport=transport) as sync_client:
        async with httpx.AsyncClient(transport=transport) as async_client:
            model = create_nebius_model(settings, http_client=sync_client, http_async_client=async_client)
            assert model.model_name == MODEL_ID
            assert "synthetic-nebius-key" not in repr(settings)
            await check_model_access(settings, async_client)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                await check_provider_contracts(model, model_id=MODEL_ID, progress=lambda message: None)
    assert len(requests) == 9  # Catalog + two ReAct requests + five schemas + one repair.
    assert bank_attempts == 2

    # Provider replies may use the full allowance without producing a report.
    # Both truncation and a missing forced tool must fail explicitly, once.
    for finish_reason in ("length", "stop"):
        attempts = []

        def unusable(request):
            attempts.append(request)
            return httpx.Response(200, json={
                "id": "synthetic", "object": "chat.completion", "created": 0, "model": MODEL_ID,
                "metadata": {"trace": [{"version": "default", "start": 0, "end": 1}]},
                "choices": [{"index": 0, "finish_reason": finish_reason,
                             "message": {"role": "assistant", "content": ""}}],
            })

        async with httpx.AsyncClient(transport=httpx.MockTransport(unusable)) as client:
            model = create_nebius_model(settings, http_client=None, http_async_client=client)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", UserWarning)
                    await model.with_structured_output(next(iter(contract_samples()))[0]).ainvoke("Synthetic")
            except ModelOutputError as exc:
                assert ("truncated" if finish_reason == "length" else "required report tool") in str(exc)
            else:
                raise AssertionError("Unusable report accepted")
            assert len(attempts) == 1

    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"data": []}))) as http_client:
        try:
            await check_model_access(settings, http_client)
        except ValueError:
            pass
        else:
            raise AssertionError("Unavailable model was silently accepted")

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "report.json"
        output.write_text("existing", encoding="utf-8")
        with patch.dict(os.environ, {"NEBIUS_API_KEY": "synthetic", "TAVILY_API_KEY": "synthetic"}):
            try:
                await execute(parse_args(["run", "--output", str(output)]))
            except FileExistsError:
                pass
            else:
                raise AssertionError("Existing report was not protected before API calls")
        assert output.read_text(encoding="utf-8") == "existing"

        @asynccontextmanager
        async def fake_session(settings, **kwargs):
            async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"data": [{"id": MODEL_ID}]}))) as http_client:
                yield UnifiedModel(), http_client

        class FakeTavilySession(FakeTavily):
            def __init__(self, **kwargs):
                super().__init__()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        completed = Path(directory) / "completed.json"
        environment = {"NEBIUS_API_KEY": "synthetic-nebius-key",
                       "TAVILY_API_KEY": "synthetic-tavily-key"}
        with (patch.dict(os.environ, environment, clear=True),
              patch("run_research.nebius_session", fake_session),
              patch("run_research.AsyncTavilyClient", FakeTavilySession),
              contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings()):
            warnings.simplefilter("ignore", DeprecationWarning)
            await execute(parse_args(["run", "--as-of", "2026-10-25", "--market", "Singapore",
                                      "--state-dir", str(Path(directory) / "state"),
                                      "--output", str(completed)]))
        document = json.loads(completed.read_text(encoding="utf-8"))
        assert document["run"]["model"] == MODEL_ID
        assert document["briefing"]["statistics"]["assessment_count"] == 1320
        assert "synthetic-nebius-key" not in completed.read_text(encoding="utf-8")
        assert "synthetic-tavily-key" not in completed.read_text(encoding="utf-8")
        resumed = Path(directory) / "resumed.json"
        with (patch.dict(os.environ, environment, clear=True),
              patch("run_research.nebius_session", fake_session),
              patch("run_research.AsyncTavilyClient", FakeTavilySession),
              contextlib.redirect_stdout(io.StringIO())):
            await execute(parse_args(["run", "--resume", document["run"]["run_id"],
                                      "--state-dir", str(Path(directory) / "state"), "--output", str(resumed)]))
        resumed_doc = json.loads(resumed.read_text())
        assert resumed_doc["briefing"] == document["briefing"]
        assert resumed_doc["run"]["usage"] == document["run"]["usage"]

        class FailingGraph:
            async def aget_state(self, config):
                from types import SimpleNamespace
                return SimpleNamespace(values={}, next=())

            async def astream(self, *args, **kwargs):
                if False:
                    yield {}
                raise TimeoutError("Synthetic stage failure")

        failed = Path(directory) / "failed.json"
        with (patch.dict(os.environ, environment, clear=True),
              patch("run_research.nebius_session", fake_session),
              patch("run_research.AsyncTavilyClient", FakeTavilySession),
              patch("run_research.build_research_graph", return_value=FailingGraph()),
              contextlib.redirect_stdout(io.StringIO())):
            try:
                await execute(parse_args(["run", "--output", str(failed),
                                          "--state-dir", str(Path(directory) / "state")]))
            except TimeoutError:
                pass
            else:
                raise AssertionError("Failed research did not propagate its error")
        assert not failed.exists()


def runtime_checks():
    with patch.dict(os.environ, {}, clear=True):
        error = io.StringIO()
        with contextlib.redirect_stderr(error):
            assert main(["check"]) == 1
        assert "NEBIUS_API_KEY" in error.getvalue()
        assert "TAVILY_API_KEY" not in error.getvalue()
    with patch.dict(os.environ, {"NEBIUS_API_KEY": "synthetic"}, clear=True):
        assert RuntimeSettings.from_env(require_tavily=False).model_id == MODEL_ID
        try:
            RuntimeSettings.from_env(require_tavily=True)
        except ValueError:
            pass
        else:
            raise AssertionError("Full research accepted missing Tavily credentials")
    with patch.dict(os.environ, {"NEBIUS_API_KEY": "do-not-print-this-key"}, clear=True):
        detail = safe_error_detail(ValueError("Useful diagnostic; key=do-not-print-this-key; Bearer another-secret"))
        assert "Useful diagnostic" in detail
        assert "do-not-print-this-key" not in detail and "another-secret" not in detail
        error = io.StringIO()
        with (patch("run_research.execute", AsyncMock(side_effect=ValueError("Specific failure detail"))),
              contextlib.redirect_stderr(error)):
            assert main(["check"]) == 1
        assert "Specific failure detail" in error.getvalue()
    for schema, sample in contract_samples():
        changed = json.loads(json.dumps(sample))
        if "methodology" in changed:
            changed["methodology"] = "Different wording, identical schema and markers"
        validate_sample_markers(schema, changed, sample)
    from needs_gatherer import NeedsReport
    needs_sample = next(sample for schema, sample in contract_samples() if schema is NeedsReport)
    changed = json.loads(json.dumps(needs_sample))
    changed["needs"][0]["basis"] = "explicit"
    try:
        validate_sample_markers(NeedsReport, changed, needs_sample)
    except ValueError:
        pass
    else:
        raise AssertionError("An inferred need was promoted to explicit during compatibility checks")
    asyncio.run(main_async())
    print("PASS: native Nebius HTTP requests, ReAct loop, five real schemas, model access and CLI checks")


def main_checks():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        # Offline checks must never load the user's actual project credentials.
        with patch("nebius_runtime.PROJECT_ENV_FILE", root / "absent.env"):
            runtime_checks()
        env_file = root / ".env"
        env_file.write_text('NEBIUS_API_KEY="file-only-key"\nTAVILY_API_KEY=file-tavily\n'
                            'NEBIUS_MODEL=zai-org/GLM-5.3\n', encoding="utf-8")
        with patch.dict(os.environ, {}, clear=True):
            settings = RuntimeSettings.from_env(require_tavily=True, env_file=env_file)
            assert settings.nebius_api_key.get_secret_value() == "file-only-key"
            assert settings.tavily_api_key.get_secret_value() == "file-tavily"
            assert "NEBIUS_API_KEY" not in os.environ
            with patch("nebius_runtime.PROJECT_ENV_FILE", env_file):
                # Default discovery is tied to the module's project, not cwd.
                elsewhere = root / "elsewhere"
                elsewhere.mkdir()
                previous = Path.cwd()
                try:
                    os.chdir(elsewhere)
                    assert RuntimeSettings.from_env(require_tavily=False).model_id == MODEL_ID
                finally:
                    os.chdir(previous)
        with patch.dict(os.environ, {"NEBIUS_API_KEY": "exported-key"}, clear=True):
            settings = RuntimeSettings.from_env(require_tavily=True, env_file=env_file)
            assert settings.nebius_api_key.get_secret_value() == "exported-key"
        env_file.write_text("Nebius_API_Key=wrong-case\n", encoding="utf-8")
        with patch.dict(os.environ, {}, clear=True):
            try:
                RuntimeSettings.from_env(require_tavily=False, env_file=env_file)
            except ValueError as exc:
                assert "NEBIUS_API_KEY" in str(exc) and "wrong-case" not in str(exc)
            else:
                raise AssertionError("Incorrectly cased variable was accepted")
    print("PASS: .env loading, exported precedence, working-directory independence and secret-safe errors")


if __name__ == "__main__":
    main_checks()
