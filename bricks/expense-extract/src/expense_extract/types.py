"""Shared result type for this brick."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import Decimal


@dataclass
class ExpenseLine:
    source_file: str
    vendor: str
    date: str
    amount: Decimal | None
    category: str
    raw_text: str
    error: str | None = None
    currency: str | None = None
    review_reasons: list[str] = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        return bool(self.error or self.review_reasons or self.amount is None or self.currency is None)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["amount"] = str(self.amount) if self.amount is not None else None
        payload["needs_review"] = self.needs_review
        return payload


def totals_by_currency(lines: list[ExpenseLine]) -> dict[str, str]:
    totals: dict[str, Decimal] = {}
    for line in lines:
        if not line.needs_review and line.amount is not None and line.currency:
            totals[line.currency] = totals.get(line.currency, Decimal(0)) + line.amount
    return {currency: format(amount, "f") for currency, amount in sorted(totals.items())}
