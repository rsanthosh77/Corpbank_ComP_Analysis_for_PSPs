# Corporate Banking Cash Competitor Research

Python 3.11+ and LangGraph, using Nebius Token Factory (`zai-org/GLM-5.3`) and Tavily.
The four research workers use `langgraph.prebuilt.create_react_agent`; the outer
`StateGraph` coordinates discovery, per-PSP needs research, per-bank mapping and a
hybrid briefing compiler. This prebuilt agent API is deprecated upstream but is
retained here as explicitly requested; dependencies are locked in `uv.lock`.

## Local research dashboard

To explore an existing `output/briefing.json` without running any research:

```sh
uv sync
uv run streamlit run dashboard.py
```

Open http://127.0.0.1:8501. **Cash Atlas** provides an overview, searchable PSP
needs, side-by-side bank comparisons, product profiles and a source library.
Filters preserve explicit/inferred distinctions and product-level conditions.
The complete needs dataset is available; no capability taxonomy is excluded.
Filtered needs, comparisons and source lists can be exported as CSV.

The dashboard uses saved files only, binds to localhost, and does not load `.env`
or call Nebius/Tavily. Fonts and entity logos are bundled locally. To choose a
different completed report, set `CASH_RESEARCH_REPORT` to its local JSON path.
Research dates remain those of the saved report; opening the app does not refresh
the underlying evidence. Source links open externally only when selected.

Verify the app offline with `uv run python verify_dashboard.py`. For optional
browser/layout checks, run the app first, then `uv run python
verify_dashboard_browser.py` (uses the installed Google Chrome and the dev group).
Asset provenance and font licenses are recorded in `dashboard_assets/` and
`static/fonts/`. To stop the server, press Ctrl+C in its terminal.

## Research pipeline setup

```sh
uv sync --python 3.11
cp .env.example .env
```

Fill in `NEBIUS_API_KEY` and `TAVILY_API_KEY` in `.env`, using these exact uppercase
names. The application loads the project-root `.env` automatically, including
when launched from an IDE or another working directory. Exported environment
variables take precedence over `.env` values. No shell export step is required.
Values are read literally (without `${VARIABLE}` expansion).

Keep credentials local. `.env` and generated `output/` reports are ignored by Git.
Credentials are not passed to graph state, prompts or report metadata.

## Check provider compatibility first

```sh
uv run python run_research.py check
```

This makes live Nebius requests: it checks that the exact configured model is
listed for your key, executes a small ReAct tool roundtrip, and tests nested
synthetic examples against all five report schemas. It uses function calling
plus local Pydantic validation for structured reports. Schema probes allow one
corrective attempt and check identifiers and uncertainty markers rather than
verbatim descriptive wording. A second failure identifies the affected contract.
It does not call Tavily
or establish real research accuracy, large-context reliability, or report completeness.
Only `NEBIUS_API_KEY` is required for this command.

## Run live research

```sh
uv run python run_research.py run --output output/briefing.json
```

This invokes paid Nebius and Tavily APIs. Omitting `--market` instructs discovery
to research ASEAN membership and market coverage. The research date defaults to
the current date in Asia/Singapore. Use `--as-of YYYY-MM-DD` to override it; this
sets the news window and research instructions, not a guaranteed historical snapshot.
`--market` can be repeated to supply explicit market names; for the agreed
all-ASEAN scope, supply the complete list or let the agents establish it.

The runner prints completed stages and writes JSON only after the compiler
accepts the briefing. Existing reports are never overwritten; choose a new path.
It also prints stage starts and model/tool timings, with a heartbeat every 30
seconds while requests are active. Prompts, tool inputs and credentials are not
included in progress messages. Failures identify the top-level stage.
The report retains all needs, bank tiers, capability assessments, value-adds,
source records, methods, uncertainties and a validated narrative.
For large datasets, the compiler processes every research record in bounded
narrative batches, then consolidates their findings. Short identifiers are used
only in model requests. Python restores the original record identifiers and
attaches sources directly from the referenced records. The final JSON retains
the complete matrix and original source excerpts; narrative batching does not
truncate the needs or capability catalogs.
Draft observations or interpretations citing only records without source
evidence are excluded from the narrative and logged in `narrative_omissions`.
The underlying gap records and every bank/need assessment remain in the report.

## Budgets, saved runs and resume

Every full research run prints a run ID and saves checkpoints, successful request
responses and a usage ledger under `.research/RUN_ID/`. To give a new run a
recognizable ID:

```sh
uv run python run_research.py run --run-id asean-pilot --output output/asean-pilot.json
```

After a failure, resume that SAME run instead of starting over:

```sh
uv run python run_research.py run --resume asean-pilot --output output/asean-pilot.json
```

LangGraph retains completed stages and nested worker progress. Successful identical
model/search requests can also be replayed from the per-run cache. Failed responses
are not cached. Replays do not increase the reported paid-token totals. Exact
request replay requires unchanged prompts and generation settings; changed inputs
can require new API calls. A request that was charged but failed before its response
was saved cannot be recovered from cache. Existing runs cannot be resumed concurrently.

The model, endpoint, output-token setting, research date and market scope must
remain the same when resuming. Budgets may be increased explicitly through the
environment, but their counters persist: resuming never resets them. `--state-dir`
selects another local storage directory. A new run ID starts fresh research.

The initial budgets are 12 paid searches per worker, 200 search attempts per run,
and 150 model HTTP attempts per run (including SDK retries). Cached searches do
not consume the search budget. A denied search tells the agent to finalize its
supported findings and records an incomplete-coverage gap; it does not remove
capability categories or impose a top-N cap on needs. Model-budget exhaustion
stops the run so it can be resumed after an explicit budget change.

These are effort/request limits, not a guaranteed dollar spending cap. The usage
ledger records successful responses' input/output tokens, request counts, cache
hits and worker attribution. Failed/interrupted requests may have charges that
the provider did not report. Reasoning tokens are not added again to completion
tokens. Tavily credit consumption and exact dollar cost are not estimated.
Usage appears during model progress, in `.research/RUN_ID/usage.json` on exit,
and in the successful report's run metadata.

`.research/` is Git-ignored and contains locally saved research text, prompts and
model responses. API authorization headers and configured credentials are not
saved to caches or graph state. The old failed run made before checkpoints were
implemented cannot be recovered from its console log alone. Add Nebius funds
before attempting another live run; no automatic model substitution occurs.

## Runtime settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `NEBIUS_MODEL` | `zai-org/GLM-5.3` | Exact model ID; no automatic fallback |
| `NEBIUS_API_BASE` | `https://api.tokenfactory.nebius.com/v1/` | Explicit Token Factory endpoint |
| `NEBIUS_MAX_TOKENS` | `32768` | Requested output allowance per model call; endpoint limits still apply |
| `NEBIUS_REASONING_EFFORT` | `low` | GLM-5.3 reasoning effort: `low`, `high`, or `max`; lower effort leaves more room for the report |
| `NEBIUS_TIMEOUT_SECONDS` | `600` | Model request read timeout in seconds |
| `NEBIUS_CONNECT_TIMEOUT_SECONDS` | `15` | Connection timeout |
| `NEBIUS_MAX_RETRIES` | `1` | SDK retries per failed API request, after the initial attempt |
| `RESEARCH_STAGE_TIMEOUT_SECONDS` | `3600` | Wall-clock limit for each top-level stage, including retries |
| `RESEARCH_CONCURRENCY` | `2` | Concurrent graph tasks, not a global HTTP request limit |
| `RESEARCH_RECURSION_LIMIT` | `100` | LangGraph step allowance, including ReAct loops |
| `RESEARCH_SEARCHES_PER_WORKER` | `12` | Paid search attempts per worker identity |
| `RESEARCH_MAX_SEARCH_REQUESTS` | `200` | Paid search attempts across the saved run |
| `RESEARCH_MAX_MODEL_REQUESTS` | `150` | Model HTTP attempts across the saved run |

Both discovery workers can run one targeted ReAct recovery pass when source or
quote validation fails. They use fixed queries for rejected rubric and market
claims, with at most eight search attempts per recovery identity
(`bank_discovery_recovery` or `psp_discovery_recovery`) and a 16-step graph limit.
Each search allowance persists across resume and still
counts toward the global search limit; model requests count toward the global
model limit. Saved evidence and cached responses are reused. New sources must
pass the same provenance and credibility checks. Unresolved claims must be
qualified or excluded with explicit gaps; recovery is not guaranteed to succeed.

Before Agent 3 can make a research call, the production graph requires exactly
the configured PSP target (10 by default), at least the baseline bank count, a
nonempty selection in each of the three bank tiers, and matching market-review
coverage across both discoveries. Country-name aliases such as Vietnam/Viet Nam
are treated as the same jurisdiction. Unknown individual metrics and capabilities
remain explicit evidence gaps; they are not filled to satisfy the gate.

Incomplete saved discoveries route through `complete_discovery`, including old
checkpoints that already scheduled Agent 3. The original validated selections
are retained and revalidated with incremental additions. Completion has separate
durable allowances of 12 PSP and 24 bank search attempts, counted within the
existing global run limits. These allowances do not reset on resume. New local
banks are researched across ASEAN with priority on missing country coverage.
If completion remains insufficient or fails evidence validation, downstream
research stays blocked. Resolved shortlist warnings and previous summaries are
retained in `discovery_history`, while active gaps reflect the completed lists.

The request read timeout is separate from the full-stage deadline. A 600-second
read allowance with one SDK retry can take longer than 600 seconds overall; the
stage deadline bounds all calls and retries within that stage. Longer client
timeouts cannot override provider-side limits. A failed research run is not
automatically restarted. No needs or comparison records are removed to shorten
generation time.

Lists of needs and bank comparisons are not truncated to fit token limits.
Large contexts or output limits can cause a run to fail; passing the small
compatibility check does not guarantee a full run will fit. Research-stage errors
fail explicitly rather than being reported as successful partial research.
Checkpoint/resume is available for saved runs. The four
research workers explicitly pass source instructions to the final formatting
call. Citation-validation failures use a bounded repair sequence against that
worker's actual evidence manifest, without rerunning searches. Quote repair
selects exact excerpt segments; source-policy repair patches only failed citation
fields, preserving existing source classifications and all findings. Other
provenance failures allow one full citation-formatting correction. An unresolved
failure rejects the report. Discovery workers can then attempt the bounded
source recovery described above, with explicit gaps for any revised or excluded
unsupported findings. Quote-only repair selects indexed segments of the original
excerpt; replacement quote prose cannot be supplied by the model. Unsupported
sources are never silently remapped or reclassified.

News tools retain clearly dated results within three calendar months before the
research date and preserve uncertain dates with a label. Older authoritative web
reference documents remain usable. Source reviews and interpretation include model
judgment; quote/reference checks establish traceability, not freedom from hallucinations.

## Offline verification

Run the `verify_*.py` scripts using `uv run python SCRIPT.py`. All use synthetic
fixtures; none require credentials or make live API requests. For example:

```sh
uv run python verify_briefing.py
uv run python verify_nebius_runtime.py
```

The first checks the entire assembled pipeline and deterministic comparison
matrix. The second checks the real native Nebius adapter with mocked HTTP,
schema parsing, the CLI, output protection and failure behavior.

To test a saved run's discovery stages against live APIs without starting needs
gathering, use `uv run python check_saved_discovery.py RUN_ID`. This is a **paid**
check: it uses the saved run's cache and budgets and stops before the next stage.
Validated discoveries remain in its checkpoint for the normal resume command.

To test Agent 3 after discovery is complete, use
`uv run python check_saved_discovery.py RUN_ID --through needs`. It resumes the
saved run, validates one completed needs report per selected PSP, and stops before
Agent 4. The check verifies unique need IDs and per-PSP summary counts. Repeating
the check after successful completion reads saved results without API calls.
Request counters and limits remain cumulative across all stages and retries.

Provider references: [native adapter](https://github.com/nebius/langchain-nebius),
[Token Factory API](https://docs.tokenfactory.nebius.com/api-reference/introduction),
[function calling](https://docs.tokenfactory.nebius.com/ai-models-inference/function-calling).
