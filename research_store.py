"""Per-run durable request replay, search budgets and usage accounting."""

import asyncio
import contextvars
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

import httpx

SEARCH_OWNER = contextvars.ContextVar("research_search_owner", default="unscoped")
MODEL_OWNER = contextvars.ContextVar("research_model_owner", default="unscoped")


class BudgetExceeded(RuntimeError):
    pass


def configure_checkpointer(checkpointer):
    """Explicitly allow only our known Pydantic report types in checkpoints."""
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    checkpointer.serde = JsonPlusSerializer(allowed_msgpack_modules=[
        ("psp_discovery", "DiscoveryReport"), ("bank_discovery", "BankDiscoveryReport"),
        ("needs_gatherer", "NeedsReport"), ("capability_mapper", "CapabilityReport"),
        ("briefing", "BriefingNarrative"),
    ])
    return checkpointer


class ResearchStore:
    """Caches contain research text, never API keys or HTTP auth headers.

    Successful identical requests are replayed within this run. Failed requests
    are not cached. Attempt budgets count failures too; failed token billing is
    unknown. SQLite reservations are transactional across concurrent tasks.
    """
    def __init__(self, path: Path, *, per_worker=12, searches=200, models=150):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        os.chmod(path, 0o600)
        self.per_worker, self.search_limit, self.model_limit = per_worker, searches, models
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS cache (kind TEXT, key TEXT, body TEXT,
                PRIMARY KEY(kind,key));
            CREATE TABLE IF NOT EXISTS attempts (id INTEGER PRIMARY KEY, kind TEXT,
                owner TEXT, status TEXT, input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0);
            CREATE TABLE IF NOT EXISTS hits (kind TEXT, count INTEGER DEFAULT 0);
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT);
        """)
        self.locks = {}

    def close(self):
        self.db.close()

    def lock(self, key):
        return self.locks.setdefault(key, asyncio.Lock())

    def key(self, value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def cached(self, kind, key):
        row = self.db.execute("SELECT body FROM cache WHERE kind=? AND key=?", (kind, key)).fetchone()
        if row:
            with self.db:
                self.db.execute("INSERT INTO hits(kind,count) VALUES (?,1)", (kind,))
            return json.loads(row[0])
        return None

    def reserve(self, kind, owner, *, per_worker_limit=None):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            total = self.db.execute("SELECT COUNT(*) FROM attempts WHERE kind=?", (kind,)).fetchone()[0]
            limit = self.model_limit if kind == "model" else self.search_limit
            worker = self.db.execute("SELECT COUNT(*) FROM attempts WHERE kind=? AND owner=?",
                                     (kind, owner)).fetchone()[0]
            worker_limit = self.per_worker if per_worker_limit is None else per_worker_limit
            if total >= limit or (kind == "search" and worker >= worker_limit):
                raise BudgetExceeded(f"{kind} request budget reached for {owner}; resume after explicitly increasing the budget")
            cursor = self.db.execute("INSERT INTO attempts(kind,owner,status) VALUES (?,?,'started')", (kind, owner))
            self.db.commit()
            return cursor.lastrowid
        except BaseException:
            self.db.rollback()
            raise

    def succeeded(self, attempt, kind, key, body, usage=None):
        usage = usage or {}
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?)", (kind, key, json.dumps(body)))
            self.db.execute("UPDATE attempts SET status='succeeded', input_tokens=?, output_tokens=? WHERE id=?",
                            (usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), attempt))

    def failed(self, attempt):
        with self.db:
            self.db.execute("UPDATE attempts SET status='failed' WHERE id=?", (attempt,))

    def metadata(self, key, value=None):
        if value is not None:
            with self.db:
                self.db.execute(
                    "INSERT INTO metadata(key,value) VALUES (?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (key, json.dumps(value)),
                )
        row = self.db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def usage(self):
        result = {}
        for kind in ("model", "search"):
            rows = self.db.execute("SELECT status,COUNT(*),SUM(input_tokens),SUM(output_tokens) "
                                   "FROM attempts WHERE kind=? GROUP BY status", (kind,)).fetchall()
            result[kind] = {"attempts": sum(r[1] for r in rows),
                            "succeeded": sum(r[1] for r in rows if r[0] == "succeeded"),
                            "failed_or_interrupted": sum(r[1] for r in rows if r[0] != "succeeded"),
                            "input_tokens": sum(r[2] or 0 for r in rows),
                            "output_tokens": sum(r[3] or 0 for r in rows),
                            "cache_hits": self.db.execute("SELECT COALESCE(SUM(count),0) FROM hits WHERE kind=?",
                                                          (kind,)).fetchone()[0]}
        result["billing_note"] = "Successful response usage only; failed requests may incur unreported charges. Tavily credits are not estimated."
        result["by_worker"] = [dict(zip(("kind", "worker", "attempts", "input_tokens", "output_tokens"), row))
                               for row in self.db.execute("SELECT kind,owner,COUNT(*),SUM(input_tokens),SUM(output_tokens) "
                                                          "FROM attempts GROUP BY kind,owner")]
        return result


class BudgetedSearchClient:
    def __init__(self, client, store, *, owner=None, per_worker_limit=None):
        self.client, self.store = client, store
        self.owner, self.per_worker_limit = owner, per_worker_limit

    async def search(self, query, **kwargs):
        # Search tools attach the actual worker identity via a context variable.
        owner = self.owner or SEARCH_OWNER.get()
        key = self.store.key({"query": " ".join(query.split()), **kwargs})
        async with self.store.lock(("search", key)):
            cached = self.store.cached("search", key)
            if cached is not None:
                return cached
            attempt = self.store.reserve("search", owner, per_worker_limit=self.per_worker_limit)
            try:
                response = await self.client.search(query=query, **kwargs)
                response = {**response, "_research_retrieved_at": datetime.now(timezone.utc).isoformat()}
                self.store.succeeded(attempt, "search", key, response)
                return response
            except BaseException:
                self.store.failed(attempt)
                raise


class CachedModelTransport(httpx.AsyncBaseTransport):
    def __init__(self, store, transport):
        self.store, self.transport = store, transport

    async def handle_async_request(self, request):
        if request.method != "POST" or not request.url.path.endswith("/chat/completions"):
            return await self.transport.handle_async_request(request)
        payload = json.loads(request.content)
        key = self.store.key({"url": str(request.url), "payload": payload})
        async with self.store.lock(("model", key)):
            cached = self.store.cached("model", key)
            if cached is not None:
                return httpx.Response(200, json=cached, request=request)
            attempt = self.store.reserve("model", MODEL_OWNER.get())
            try:
                response = await self.transport.handle_async_request(request)
                body = await response.aread()
                status = response.status_code
                await response.aclose()
                if status == 200:
                    parsed = json.loads(body)
                    self.store.succeeded(attempt, "model", key, parsed, parsed.get("usage"))
                else:
                    self.store.failed(attempt)
                return httpx.Response(status, content=body,
                                      headers={"content-type": "application/json"}, request=request)
            except BaseException:
                self.store.failed(attempt)
                raise

    async def aclose(self):
        await self.transport.aclose()
