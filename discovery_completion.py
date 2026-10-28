"""Incremental, evidence-validated discovery before downstream research."""

import asyncio
import json

from langchain_core.messages import ToolMessage
from langgraph.prebuilt import create_react_agent
from langgraph.types import Overwrite

from bank_discovery import BANK_PROMPT, TIERS, BankDiscoveryReport, validated_bank_update
from psp_discovery import PSP_PROMPT, DiscoveryReport, validated_update
from discovery_evidence import REPORT_FORMAT_POLICY, validate_with_citation_correction
from research_store import BudgetedSearchClient
from search_tools import build_search_tools
from discovery_contracts import market_key


class DiscoveryIncompleteError(ValueError):
    pass


def _reviewed_markets(state, summary):
    return {market_key(m["market"]) for m in state.get(summary, {}).get("market_coverage", [])}


def _expected_markets(state):
    return ({market_key(m) for m in state["local_markets"]} or
            _reviewed_markets(state, "psp_discovery_summary") | _reviewed_markets(state, "bank_discovery_summary"))


def discovery_shortfalls(state):
    gaps = []
    if len(state.get("psps", [])) != state["psp_target_count"]:
        gaps.append(f"PSPs: {len(state.get('psps', []))}/{state['psp_target_count']}")
    banks = state.get("banks", {})
    count = sum(len(banks.get(tier, [])) for tier in TIERS)
    if count < state["bank_target_count"]:
        gaps.append(f"banks: {count}/{state['bank_target_count']}")
    for tier in TIERS:
        if not banks.get(tier):
            gaps.append(f"empty bank bucket: {tier}")
    for summary in ("psp_discovery_summary", "bank_discovery_summary"):
        covered = _reviewed_markets(state, summary)
        expected = _expected_markets(state)
        if not covered or not expected <= covered:
            gaps.append(f"missing market review: {summary}")
    return gaps


def require_complete_discovery(state):
    gaps = discovery_shortfalls(state)
    if gaps:
        raise DiscoveryIncompleteError("Discovery must be completed before needs gathering: " + "; ".join(gaps))


def _saved_report(state, agent):
    schema, field, summary = ((DiscoveryReport, "psps", "psp_discovery_summary") if agent == "psp_discovery"
                              else (BankDiscoveryReport, "banks", "bank_discovery_summary"))
    data = {key: value for key, value in state[summary].items() if key in schema.model_fields}
    data.update({field: state[field], "research_gaps": []})
    return schema.model_validate(data)


async def _research_additions(state, *, agent, model, client, config):
    psp = agent == "psp_discovery"
    summary = state["psp_discovery_summary" if psp else "bank_discovery_summary"]
    markets = state["local_markets"] or list({market_key(m["market"]): m["market"]
        for key in ("psp_discovery_summary", "bank_discovery_summary")
        for m in state[key]["market_coverage"]}.values())
    missing = max(0, state["psp_target_count"] - len(state["psps"])) if psp else max(
        0, state["bank_target_count"] - sum(len(v) for v in state["banks"].values()))
    if not psp and not state["banks"]["tier_3_local"]:
        missing = max(missing, 4)
    working = {**state, "local_markets": markets,
               "psp_target_count" if psp else "bank_target_count": missing}
    schema, base_prompt = (DiscoveryReport, PSP_PROMPT) if psp else (BankDiscoveryReport, BANK_PROMPT)
    limit = 12 if psp else 24
    bounded = (BudgetedSearchClient(client.client, client.store, owner=agent + "_completion",
                                   per_worker_limit=limit) if isinstance(client, BudgetedSearchClient) else client)
    remaining = limit
    if isinstance(client, BudgetedSearchClient):
        used = client.store.db.execute("SELECT COUNT(*) FROM attempts WHERE kind='search' AND owner=?",
                                       (agent + "_completion",)).fetchone()[0]
        remaining = max(0, limit - used)
    role_instructions = ("""
PSP COMPLETION ONLY: select a standalone payment technology company or payment
gateway, never a commercial bank, bank legal entity, or corporate bank division.
Bank merchant acquiring alone does not make a bank eligible for this shortlist.
Compare credible PayMongo, VNPAY, Antom/Alipay+ or other independent PSP candidates
against all five rubric criteria. Examples are leads, not guaranteed selections.
Return exactly additional_target_count eligible additions, or explicit gaps.
""" if psp else """
BANK COMPLETION ONLY: prioritize missing bank tiers, especially local corporate
cash banks in Indonesia, Philippines, Thailand and Vietnam. Review the other
ASEAN markets too. Search leads include BCA, Permata, BDO, KBank, Vietcombank,
Baiduri, ACLEDA, BCEL, KBZ and BNCTL; verify eligibility rather than assuming it.
Dedicated corporate cash services require official evidence. Seek geographic
diversity; do not repeat retained bank groups through subsidiaries. Empty buckets
in this incremental report need tier_gaps explaining baseline selections remain.
""")
    prompt = base_prompt + REPORT_FORMAT_POLICY + role_instructions + """
INCREMENTAL DISCOVERY COMPLETION (overrides the original selection count)
Return ONLY new selections, never repeat a previously selected company or bank
group. The input target is the number of ADDITIONS, not the final total. Preserve
all required rubric fields. Unknown metrics need explicit gaps; never fabricate
support to fill counts. Use both web and recent-news tools. Cite only this
invocation's retrieved evidence, not IDs in the baseline context.
Review every supplied ASEAN market, including those where no addition qualifies.
Return one market_coverage entry PER supplied market. Never put comma-separated
markets into a single entry. Do not omit markets, including Vietnam/Viet Nam.
Existing validated selections are retained by the orchestrator; do not rewrite
them. Mark baseline-only market facts as context, not newly verified claims.
Keep prose concise and quotes short and verbatim. Do not write a full report as
ordinary assistant prose before the framework's report tool becomes available.
Report evidence gaps honestly; an incomplete report will not unlock downstream
research. No limits are imposed on downstream needs or emerging technologies.
"""
    request = {"as_of_date": state["as_of_date"], "local_markets": markets,
               "additional_target_count": missing, "research_question": state["research_question"],
               "retained_selections": state["psps"] if psp else state["banks"],
               "remaining_uncached_search_attempts": remaining,
               "missing_bank_tiers": [t for t in TIERS if not state["banks"][t]]}
    worker = create_react_agent(model=model, tools=build_search_tools(client=bounded, agent_name=agent,
        as_of_date=state["as_of_date"]), prompt=prompt, response_format=(prompt, schema),
        name=agent + "_completion_worker")
    worker_config = config
    if psp and config and config.get("configurable", {}).get("checkpoint_ns"):
        # The first completion implementation produced an invalid bank-as-PSP
        # report. Keep that checkpoint for audit, and start the corrected PSP
        # contract in a versioned namespace; the bank worker keeps its namespace.
        worker_config = {**config, "configurable": {**config["configurable"],
            "checkpoint_ns": config["configurable"]["checkpoint_ns"] + "|psp_role_v2"}}
    result = await worker.ainvoke({"messages": [("user", json.dumps(request))]}, config=worker_config)
    validator = validated_update if psp else validated_bank_update
    # Capture a corrected report as well as its update; merging uses the exact
    # report that passed source, quote, entity and scope validation.
    accepted = {}

    def validate(value):
        if not psp:
            report = schema.model_validate(value["structured_response"]).model_copy(deep=True)
            for tier in TIERS:
                if not getattr(report.banks, tier) and not report.tier_gaps.get(tier) and state["banks"][tier]:
                    report.tier_gaps[tier] = [
                        f"Incremental batch: {len(state['banks'][tier])} previously validated banks retained in baseline."
                    ]
            value = {**value, "structured_response": report}
        update = validator(value, working)
        accepted["report"] = schema.model_validate(value["structured_response"])
        accepted["evidence"] = update["evidence"]
        return update

    await validate_with_citation_correction(result, model=model, schema=schema, prompt=prompt,
        validate=validate, agent_name=agent, config=config)
    return accepted


def _merge_report(state, agent, addition):
    psp = agent == "psp_discovery"
    base = _saved_report(state, agent).model_dump()
    extra = addition["report"].model_dump()
    if psp:
        base["psps"].extend(extra["psps"])
        selected = base["psps"]
    else:
        for tier in TIERS:
            base["banks"][tier].extend(extra["banks"][tier])
            if extra["banks"][tier]:
                base["tier_gaps"][tier] = extra["tier_gaps"][tier]
        base["allocation_rationale"] = "Completed evidenced bank allocation: " + ", ".join(
            f"{tier}={len(base['banks'][tier])}" for tier in TIERS) + ". Initial allocation is retained in discovery history."
        base["expansion_rationale"] = "Additional evidenced in-country banks complete tier and ASEAN comparison coverage."
        selected = [b for tier in TIERS for b in base["banks"][tier]]
    names = {item["name"].strip().casefold() for item in selected}
    base["excluded_candidates"] = [item for item in base["excluded_candidates"] + extra["excluded_candidates"]
                                   if item["name"].strip().casefold() not in names]
    # New market reviews supersede incomplete baseline reviews. The baseline
    # findings and original gaps remain available in discovery_history.
    base["market_coverage"] = extra["market_coverage"]
    base["research_gaps"] = extra["research_gaps"]
    base["methodology"] += " Incremental completion: " + extra["methodology"]
    reviews = {r["evidence_id"]: r for r in base["source_reviews"]}
    for review in extra["source_reviews"]:
        if review["evidence_id"] in reviews and reviews[review["evidence_id"]]["source_type"] != review["source_type"]:
            raise ValueError("Completion attempted to change an existing source classification")
        reviews.setdefault(review["evidence_id"], review)
    base["source_reviews"] = list(reviews.values())
    evidence = {e["id"]: e for e in state["evidence"] if e["agent"] == agent}
    evidence.update({e["id"]: e for e in addition["evidence"]})
    messages = [ToolMessage(content="Saved and incremental evidence", name="search_web", tool_call_id="merged-web",
                            artifact={"evidence": list(evidence.values())}),
                ToolMessage(content="Prior and incremental news checks completed", name="search_news",
                            tool_call_id="merged-news", artifact={"evidence": []})]
    return (validated_update if psp else validated_bank_update)(
        {"structured_response": base, "messages": messages}, state)


def create_discovery_completion_node(*, model, client):
    async def run(state, config=None):
        if not discovery_shortfalls(state):
            return {}
        jobs = []
        if (len(state["psps"]) < state["psp_target_count"] or
                not _expected_markets(state) <= _reviewed_markets(state, "psp_discovery_summary")):
            jobs.append("psp_discovery")
        if (sum(len(v) for v in state["banks"].values()) < state["bank_target_count"]
                or any(not state["banks"][tier] for tier in TIERS)
                or not _expected_markets(state) <= _reviewed_markets(state, "bank_discovery_summary")):
            jobs.append("bank_discovery")
        if not jobs:
            require_complete_discovery(state)
        # Let both requests finish before raising, so a sibling's paid research
        # is checkpointed and reusable if the other worker fails validation.
        results = await asyncio.gather(*[_research_additions(state, agent=agent, model=model, client=client,
                                                            config=config) for agent in jobs], return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result
        candidate = dict(state)
        updates = []
        for agent, result in zip(jobs, results):
            update = _merge_report(state, agent, result)
            candidate.update({key: value for key, value in update.items() if key not in {"evidence", "research_gaps"}})
            updates.append(update)
        require_complete_discovery(candidate)
        prior_ids = {e["id"] for e in state["evidence"]}
        new_evidence = {e["id"]: e for u in updates for e in u["evidence"] if e["id"] not in prior_ids}
        gaps = [g for g in state["research_gaps"] if not any(g.startswith(prefix) for prefix in
                ("Agent 1:", "Agent 2:"))]
        # Include unchanged discovery's existing gaps if only one agent ran.
        for agent, prefix in (("psp_discovery", "Agent 1:"), ("bank_discovery", "Agent 2:")):
            if agent not in jobs:
                gaps.extend(g for g in state["research_gaps"] if g.startswith(prefix))
        gaps.extend(g for update in updates for g in update["research_gaps"])
        history = list(state.get("discovery_history", [])) + [{
            "psp_count": len(state["psps"]), "tier_counts": {t: len(state["banks"][t]) for t in TIERS},
            "research_gaps": state["research_gaps"],
            "psp_summary": state["psp_discovery_summary"], "bank_summary": state["bank_discovery_summary"]}]
        delta = {key: value for update in updates for key, value in update.items()
                 if key not in {"evidence", "research_gaps"}}
        return {**delta, "evidence": list(new_evidence.values()), "research_gaps": Overwrite(list(dict.fromkeys(gaps))),
                "discovery_history": history}
    return run
