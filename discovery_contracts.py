"""Common runtime-validated contracts for the two discovery workers."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Text = Annotated[str, Field(min_length=1, pattern=r"\S")]


def market_key(value):
    """Compare country-name aliases without changing displayed source wording."""
    key = " ".join(value.casefold().replace("-", " ").split())
    return {"viet nam": "vietnam", "brunei darussalam": "brunei", "lao pdr": "laos",
            "lao people's democratic republic": "laos", "east timor": "timor leste"}.get(key, key)


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Assessment(Contract):
    rating: Literal["strong", "moderate", "weak", "unknown"]
    assessment: Text
    evidence_ids: list[Text]
    gaps: list[Text]

    @model_validator(mode="after")
    def require_support(self):
        if self.rating != "unknown" and not self.evidence_ids:
            raise ValueError("A known rating needs retrieved evidence")
        if self.rating == "unknown" and not self.gaps:
            raise ValueError("An unknown rating needs an explicit research gap")
        return self


class MarketCoverage(Contract):
    market: Text
    findings: Text
    evidence_ids: list[Text]
    gaps: list[Text]

    @model_validator(mode="after")
    def require_support_or_gap(self):
        if not self.evidence_ids and not self.gaps:
            raise ValueError("Market coverage needs evidence or a research gap")
        return self


class ExcludedCandidate(Contract):
    name: Text
    reason: Text
    evidence_ids: list[Text] = Field(min_length=1)
