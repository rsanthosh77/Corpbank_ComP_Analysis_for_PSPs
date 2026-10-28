"""Small live provider checks using synthetic data; these are not market research."""

import json
from typing import Literal

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from pydantic import ValidationError

from bank_discovery import BankDiscoveryReport
from briefing import BriefingNarrative
from capability_mapper import CapabilityReport
from needs_gatherer import NeedsReport
from psp_discovery import DiscoveryReport


class CompatibilityError(ValueError):
    """A named provider contract check failed, with no credential-bearing payload."""


def validate_sample_markers(schema, output, sample):
    """Check identities and qualifications, not verbatim descriptive prose."""
    output = schema.model_validate(output)
    if schema is DiscoveryReport:
        valid = [p.id for p in output.psps] == [p["id"] for p in sample["psps"]]
    elif schema is BankDiscoveryReport:
        valid = ([b.id for b in output.banks.tier_1_global] == ["synthetic-bank"]
                 and not output.banks.tier_2_regional and not output.banks.tier_3_local
                 and output.banks.tier_1_global[0].rubric.corporate_cash_management.rating == "moderate"
                 and output.banks.tier_1_global[0].rubric.corporate_apis.rating == "unknown"
                 and output.banks.tier_1_global[0].rubric.clearing_connectivity.rating == "unknown")
    elif schema is NeedsReport:
        valid = (output.psp_id == "synthetic-psp" and len(output.needs) == 1
                 and output.needs[0].basis == "inferred"
                 and {"programmable money", "arbitrary new theme"} <= set(output.needs[0].themes))
    elif schema is CapabilityReport:
        valid = (output.bank_id == "synthetic-bank" and output.tier == "tier_1_global"
                 and len(output.capabilities) == len(output.need_assessments) == 1
                 and output.capabilities[0].availability == "pilot"
                 and output.capabilities[0].psp_eligibility == "unclear"
                 and output.need_assessments[0].status == "conditional_match"
                 and bool(output.need_assessments[0].constraints)
                 and output.need_assessments[0].capability_local_ids == [output.capabilities[0].local_id])
    else:
        valid = (len(output.executive_summary) == 1
                 and output.executive_summary[0].kind == "interpretation"
                 and output.executive_summary[0].record_refs == ["need:synthetic-need"]
                 and output.executive_summary[0].evidence_ids == ["synthetic-source"])
    if not valid:
        raise CompatibilityError("Synthetic identifiers or uncertainty markers were not preserved")
    return output


def contract_samples():
    """Include nested records so parsing exercises more than empty lists."""
    unknown = {"rating": "unknown", "assessment": "Synthetic: not researched",
               "evidence_ids": [], "gaps": ["Synthetic fixture only"]}
    known = {"rating": "moderate", "assessment": "Synthetic corporate business",
             "evidence_ids": ["synthetic-source"], "gaps": []}
    review = {"evidence_id": "synthetic-source", "publisher": "Synthetic publisher",
              "source_type": "primary", "credibility_reason": "Synthetic test only",
              "supporting_quote": "Synthetic excerpt"}
    market = {"market": "Singapore", "findings": "Synthetic coverage",
              "evidence_ids": [], "gaps": ["Not researched"]}
    psp = {"id": "synthetic-psp", "name": "Synthetic PSP", "countries": ["Singapore"],
           "selection_rationale": "Synthetic only", "evidence_ids": ["synthetic-source"],
           "rubric": {key: unknown for key in ("annual_tpv", "developer_ecosystem", "market_position",
                                              "funding_and_valuation", "licensing")}}
    bank = {"id": "synthetic-bank", "name": "Synthetic Bank", "group_id": "synthetic-group",
            "headquarters_country": "Singapore", "countries": ["Singapore"],
            "tier_rationale": "Synthetic only", "selection_rationale": "Synthetic only",
            "evidence_ids": ["synthetic-source"],
            "rubric": {"corporate_cash_management": known, "corporate_apis": unknown,
                       "clearing_connectivity": unknown}}
    common = {"methodology": "Synthetic compatibility check, not real research",
              "source_reviews": [review], "research_gaps": ["Synthetic data only"]}
    return [
        (DiscoveryReport, {**common, "psps": [psp], "scope_evidence_ids": [],
                           "market_coverage": [market], "excluded_candidates": []}),
        (BankDiscoveryReport, {**common, "banks": {"tier_1_global": [bank], "tier_2_regional": [],
                                                   "tier_3_local": []},
                               "allocation_rationale": "Synthetic only", "expansion_rationale": None,
                               "tier_gaps": {tier: ["Synthetic coverage gap"] for tier in (
                                   "tier_1_global", "tier_2_regional", "tier_3_local")},
                               "scope_evidence_ids": [], "market_coverage": [market],
                               "excluded_candidates": []}),
        (NeedsReport, {**common, "psp_id": "synthetic-psp", "coverage_notes": ["Synthetic advanced need"],
                       "needs": [{"description": "Synthetic programmable money requirement",
                                  "themes": ["programmable money", "arbitrary new theme"], "basis": "inferred",
                                  "rationale": "Synthetic hypothesis only", "jurisdictions": [],
                                  "evidence_ids": ["synthetic-source"]}]}),
        (CapabilityReport, {**common, "bank_id": "synthetic-bank", "tier": "tier_1_global",
                            "coverage_notes": ["Synthetic pilot"], "capabilities": [{
                                "local_id": "synthetic-product", "product_name": "Synthetic Product",
                                "description": "Synthetic pilot", "availability": "pilot",
                                "jurisdictions": ["Singapore"], "legal_entities": ["Synthetic Bank SG"],
                                "psp_eligibility": "unclear", "constraints": ["Synthetic pilot access"],
                                "evidence_ids": ["synthetic-source"], "is_value_add": True,
                                "value_add_rationale": "Synthetic workflow"}],
                            "need_assessments": [{"need_id": "synthetic-need", "status": "conditional_match",
                                                  "capability_local_ids": ["synthetic-product"],
                                                  "rationale": "Synthetic only", "constraints": ["Pilot"],
                                                  "evidence_ids": ["synthetic-source"]}]}),
        (BriefingNarrative, {"executive_summary": [{"kind": "interpretation", "text": "Synthetic observation",
                                                   "record_refs": ["need:synthetic-need"],
                                                   "evidence_ids": ["synthetic-source"]}],
                            "competitive_implications": [], "product_opportunities": [], "follow_up_questions": []}),
    ]


async def check_provider_contracts(model, *, model_id: str, progress=print):
    calls = []

    @tool
    def compatibility_echo(value: Literal["ok"]) -> str:
        """Return the supplied synthetic compatibility marker."""
        calls.append(value)
        return value

    worker = create_react_agent(
        model=model, tools=[compatibility_echo],
        prompt="Call compatibility_echo exactly once with value='ok', then finish. This is a synthetic API check.",
    )
    result = await worker.ainvoke({"messages": [("user", "Run the synthetic check now.")]},
                                 config={"recursion_limit": 8})
    if calls != ["ok"] or not any(isinstance(m, ToolMessage) for m in result["messages"]):
        raise ValueError("Provider did not complete the synthetic ReAct tool roundtrip")
    for message in result["messages"]:
        if isinstance(message, AIMessage):
            served = message.response_metadata.get("model_name")
            if served != model_id:
                raise ValueError("Provider response did not identify the configured model")
    progress("PASS: native Nebius ReAct tool roundtrip")

    for schema, sample in contract_samples():
        schema.model_validate(sample)
        messages = [
            ("system", "Synthetic schema test. Use the required output function. Preserve the supplied IDs, "
             "ratings, uncertainty markers and nesting. Each bank's criterion assessments MUST be inside "
             "its rubric object, never flattened onto the bank. Do not research or invent data. "
             "Follow this JSON Schema:\n" + json.dumps(schema.model_json_schema())),
            ("user", json.dumps(sample)),
        ]
        for attempt in range(2):
            try:
                output = await model.with_structured_output(schema).ainvoke(messages)
                validate_sample_markers(schema, output, sample)
                break
            except (ValidationError, CompatibilityError) as exc:
                if isinstance(exc, ValidationError):
                    issues = "; ".join(
                        ".".join(map(str, error["loc"])) + " (" + error["type"] + ")"
                        for error in exc.errors(include_input=False)[:8])
                else:
                    issues = str(exc)
                if attempt == 1:
                    raise CompatibilityError(f"{schema.__name__} failed after two attempts: {issues}") from exc
                progress(f"RETRY: {schema.__name__} did not satisfy its schema/markers")
                messages.append(("user", "Correct the response. Validation issues: " + issues +
                                 ". Return the supplied sample with the exact nesting and preserved markers."))
        progress(f"PASS: {schema.__name__}")
