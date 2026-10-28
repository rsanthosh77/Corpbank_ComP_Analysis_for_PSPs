"""Native Nebius configuration; credentials never enter graph state or reports."""

import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

import httpx
from dotenv import dotenv_values
from langchain_nebius import ChatNebius
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, ConfigDict, Field, SecretStr

MODEL_ID = "zai-org/GLM-5.3"
BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
PROJECT_ENV_FILE = Path(__file__).resolve().with_name(".env")


class ConfigurationError(ValueError):
    pass


class ModelOutputError(ValueError):
    """A completed HTTP request did not produce a usable model response."""


class ResearchChatNebius(ChatNebius):
    """Keep provider metadata available until generation and parsing are checked."""

    def _create_chat_result(self, response, generation_info=None):
        # Inspect SDK objects directly. Dumping the whole response introduces
        # serialization warnings for Nebius-specific metadata; the native
        # adapter already handles those extras when constructing messages.
        choices = response.get("choices") if isinstance(response, dict) else getattr(response, "choices", None)
        if any((choice.get("finish_reason") if isinstance(choice, dict)
                else getattr(choice, "finish_reason", None)) == "length" for choice in choices or []):
            raise ModelOutputError(
                "Nebius response truncated (finish_reason=length): output-token limit reached. "
                "No partial report accepted. Reduce reasoning effort or revise the report size."
            )
        return super()._create_chat_result(response, generation_info)

    def with_structured_output(self, schema=None, *, include_raw=False, **kwargs):
        # The native adapter already forces this schema's named tool. Retain the
        # raw reply so a missing tool call cannot silently become a None report.
        runnable = super().with_structured_output(schema, include_raw=True, **kwargs)

        def checked(result):
            if result["parsing_error"] is not None:
                raise result["parsing_error"]
            if result["parsed"] is None:
                reason = result["raw"].response_metadata.get("finish_reason", "unknown")
                raise ModelOutputError(
                    f"Nebius did not submit the required report tool (finish_reason={reason}). "
                    "No report accepted; no automatic paid retry was made."
                )
            return result if include_raw else result["parsed"]

        return runnable | RunnableLambda(checked)


def safe_error_detail(exc: Exception) -> str:
    """Retain diagnostic messages while redacting configured secret values."""
    message = str(exc)
    # Include both file and exported values: either could occur in an error.
    values = list(dotenv_values(PROJECT_ENV_FILE, interpolate=False).items()) + list(os.environ.items())
    secrets = {value for name, value in values if value and
               any(marker in name.upper() for marker in ("KEY", "TOKEN", "SECRET", "PASSWORD"))}
    for value in sorted(secrets, key=len, reverse=True):
        message = message.replace(value, "[REDACTED]")
    message = re.sub(r"(?i)(Bearer\s+)\S+", r"\1[REDACTED]", message)
    return " ".join(message.split())[:600]


class RuntimeSettings(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)
    nebius_api_key: SecretStr
    tavily_api_key: SecretStr | None = None
    model_id: str = Field(default=MODEL_ID, min_length=1)
    base_url: str = BASE_URL
    timeout_seconds: Annotated[float, Field(gt=0)] = 600
    connect_timeout_seconds: Annotated[float, Field(gt=0)] = 15
    max_retries: Annotated[int, Field(ge=0)] = 1
    stage_timeout_seconds: Annotated[float, Field(gt=0)] = 3600
    max_output_tokens: Annotated[int, Field(gt=0)] = 32768
    reasoning_effort: Literal["low", "high", "max"] = "low"
    concurrency: Annotated[int, Field(gt=0)] = 2
    recursion_limit: Annotated[int, Field(ge=8)] = 100
    max_searches_per_worker: Annotated[int, Field(ge=1)] = 12
    max_search_requests: Annotated[int, Field(ge=1)] = 200
    max_model_requests: Annotated[int, Field(ge=1)] = 150

    @classmethod
    def from_env(cls, *, require_tavily: bool, env_file: Path | None = None):
        # Resolve beside this module, so IDE launches work from any directory.
        # Do not mutate os.environ or override explicitly exported values.
        path = PROJECT_ENV_FILE if env_file is None else env_file
        environment = {**dotenv_values(path, interpolate=False), **os.environ}
        required = ["NEBIUS_API_KEY"] + (["TAVILY_API_KEY"] if require_tavily else [])
        missing = [name for name in required if not (environment.get(name) or "").strip()]
        if missing:
            raise ConfigurationError("Set these environment variables: " + ", ".join(missing))
        options = {
            "model_id": "NEBIUS_MODEL", "base_url": "NEBIUS_API_BASE",
            "timeout_seconds": "NEBIUS_TIMEOUT_SECONDS", "max_output_tokens": "NEBIUS_MAX_TOKENS",
            "reasoning_effort": "NEBIUS_REASONING_EFFORT",
            "connect_timeout_seconds": "NEBIUS_CONNECT_TIMEOUT_SECONDS", "max_retries": "NEBIUS_MAX_RETRIES",
            "stage_timeout_seconds": "RESEARCH_STAGE_TIMEOUT_SECONDS",
            "concurrency": "RESEARCH_CONCURRENCY", "recursion_limit": "RESEARCH_RECURSION_LIMIT",
            "max_searches_per_worker": "RESEARCH_SEARCHES_PER_WORKER",
            "max_search_requests": "RESEARCH_MAX_SEARCH_REQUESTS",
            "max_model_requests": "RESEARCH_MAX_MODEL_REQUESTS",
        }
        values = {key: environment[name].strip() for key, name in options.items()
                  if (environment.get(name) or "").strip()}
        settings = cls(nebius_api_key=environment["NEBIUS_API_KEY"].strip(),
                       tavily_api_key=(environment.get("TAVILY_API_KEY") or "").strip() or None, **values)
        url = urlsplit(settings.base_url)
        if (url.scheme != "https" or not url.hostname or url.username or url.password
                or url.query or url.fragment):
            raise ConfigurationError("NEBIUS_API_BASE must be an HTTPS URL without credentials or query parameters")
        return settings


def create_nebius_model(settings: RuntimeSettings, *, http_client, http_async_client):
    return ResearchChatNebius(
        api_key=settings.nebius_api_key, model=settings.model_id,
        base_url=settings.base_url, timeout=request_timeout(settings), max_retries=settings.max_retries,
        max_tokens=settings.max_output_tokens, use_responses_api=False,
        reasoning_effort=settings.reasoning_effort,
        disabled_params={"parallel_tool_calls": None},
        http_client=http_client, http_async_client=http_async_client,
    )


def request_timeout(settings: RuntimeSettings) -> httpx.Timeout:
    """Allow long model generation while bounding connect, write and pool waits."""
    return httpx.Timeout(read=settings.timeout_seconds, connect=settings.connect_timeout_seconds,
                         write=60, pool=60)


@asynccontextmanager
async def nebius_session(settings: RuntimeSettings, *, store=None):
    """Own HTTP client lifetimes explicitly, including exceptional exits."""
    with httpx.Client(timeout=request_timeout(settings)) as sync_http:
        transport = None
        if store is not None:
            from research_store import CachedModelTransport
            transport = CachedModelTransport(store, httpx.AsyncHTTPTransport())
        async with httpx.AsyncClient(timeout=request_timeout(settings), transport=transport) as async_http:
            yield create_nebius_model(settings, http_client=sync_http,
                                      http_async_client=async_http), async_http


async def check_model_access(settings: RuntimeSettings, http_client: httpx.AsyncClient):
    response = await http_client.get(
        settings.base_url.rstrip("/") + "/models",
        headers={"Authorization": "Bearer " + settings.nebius_api_key.get_secret_value()},
        timeout=httpx.Timeout(30, connect=settings.connect_timeout_seconds),
    )
    response.raise_for_status()
    ids = {model["id"] for model in response.json()["data"]}
    if settings.model_id not in ids:
        raise ConfigurationError("The configured NEBIUS_MODEL is not listed for this API key; no fallback model was selected")
