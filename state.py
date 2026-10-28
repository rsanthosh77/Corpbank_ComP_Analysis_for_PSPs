"""Central research contracts; worker conversations remain worker-local."""

from operator import add
from typing import Annotated, Literal, NotRequired, TypedDict


class Entity(TypedDict):
    id: str
    name: str
    countries: list[str]
    selection_rationale: str
    evidence_ids: list[str]


class Evidence(TypedDict):
    id: str
    agent: str
    url: str
    title: str
    published_at: str | None
    retrieved_at: str
    excerpt: str
    topic: NotRequired[str]
    date_status: NotRequired[Literal["dated", "uncertain"]]
    raw_published_at: NotRequired[str | None]


class CriterionAssessment(TypedDict):
    rating: Literal["strong", "moderate", "weak", "unknown"]
    assessment: str
    evidence_ids: list[str]
    gaps: list[str]


class PSPRubric(TypedDict):
    annual_tpv: CriterionAssessment
    developer_ecosystem: CriterionAssessment
    market_position: CriterionAssessment
    funding_and_valuation: CriterionAssessment
    licensing: CriterionAssessment


class PSP(Entity):
    rubric: PSPRubric


class BankRubric(TypedDict):
    corporate_cash_management: CriterionAssessment
    corporate_apis: CriterionAssessment
    clearing_connectivity: CriterionAssessment


class Bank(Entity):
    group_id: str
    headquarters_country: str
    tier_rationale: str
    rubric: BankRubric


class BankBuckets(TypedDict):
    tier_1_global: list[Bank]
    tier_2_regional: list[Bank]
    tier_3_local: list[Bank]


class Need(TypedDict):
    id: str
    psp_ids: list[str]
    description: str
    themes: list[str]  # Open vocabulary: conventional and emerging needs.
    basis: Literal["explicit", "inferred"]
    evidence_ids: list[str]
    rationale: NotRequired[str]
    jurisdictions: NotRequired[list[str]]


class Capability(TypedDict):
    id: str
    bank_id: str
    description: str
    matched_need_ids: list[str]
    availability: str
    jurisdictions: list[str]
    evidence_ids: list[str]
    tier: NotRequired[str]
    product_name: NotRequired[str]
    legal_entities: NotRequired[list[str]]
    psp_eligibility: NotRequired[str]
    constraints: NotRequired[list[str]]
    is_value_add: NotRequired[bool]
    value_add_rationale: NotRequired[str | None]


class BankNeedAssessment(TypedDict):
    bank_id: str
    tier: str
    need_id: str
    psp_ids: list[str]
    status: Literal["documented_match", "conditional_match", "not_supported", "not_found"]
    capability_ids: list[str]
    rationale: str
    constraints: list[str]
    evidence_ids: list[str]


class ResearchState(TypedDict):
    discovery_history: NotRequired[list[dict]]
    research_question: str
    local_scope: str
    local_markets: list[str]  # Explicit market names supplied at invocation.
    as_of_date: str
    psp_target_count: int
    bank_target_count: int
    psps: list[PSP]
    psp_discovery_summary: dict[str, object]
    banks: BankBuckets
    bank_discovery_summary: dict[str, object]
    needs: Annotated[list[Need], add]
    needs_research_summaries: Annotated[list[dict[str, object]], add]
    capabilities: Annotated[list[Capability], add]
    need_assessments: Annotated[list[BankNeedAssessment], add]
    capability_research_summaries: Annotated[list[dict[str, object]], add]
    evidence: Annotated[list[Evidence], add]
    research_gaps: Annotated[list[str], add]
    briefing: dict[str, object]  # Deterministic catalogs/matrix plus validated narrative.


def initial_state(*, as_of_date: str, local_markets: list[str]) -> ResearchState:
    return {
        "research_question": "Which corporate bank cash capabilities meet PSP needs?",
        "local_scope": "All ASEAN markets",
        "local_markets": list(local_markets),
        "as_of_date": as_of_date,
        "psp_target_count": 10,
        "bank_target_count": 10,
        "psps": [],
        "psp_discovery_summary": {},
        "banks": {"tier_1_global": [], "tier_2_regional": [], "tier_3_local": []},
        "bank_discovery_summary": {},
        "needs": [],
        "needs_research_summaries": [],
        "capabilities": [],
        "need_assessments": [],
        "capability_research_summaries": [],
        "evidence": [],
        "research_gaps": [],
        "briefing": {},
    }
