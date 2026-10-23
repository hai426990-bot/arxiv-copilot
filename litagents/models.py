"""文献 Agent 的共享数据模型。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from quant_analysis import PaperMetrics, PaperRecord


@dataclass
class AgentEvent:
    agent: str
    action: str
    detail: str
    elapsed_ms: int = 0
    at: str = ""

    def __post_init__(self) -> None:
        if not self.at:
            self.at = datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class PaperCard:
    arxiv_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    published: str = ""
    categories: list[str] = field(default_factory=list)
    problem: str = ""
    method: str = ""
    findings: str = ""
    limitations: str = ""
    keywords: list[str] = field(default_factory=list)
    evidence: str = ""
    source: str = "abstract"  # abstract / llm

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Theme:
    label: str
    keywords: list[str]
    paper_ids: list[str]
    summary: str = ""
    mean_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LiteratureState:
    query: str
    max_results: int = 30
    top_k: int = 10
    min_relevance: float = 0.02
    subqueries: list[str] = field(default_factory=list)
    candidates: list[PaperRecord] = field(default_factory=list)
    shortlist: list[PaperRecord] = field(default_factory=list)
    metrics: dict[str, PaperMetrics] = field(default_factory=dict)
    cards: list[PaperCard] = field(default_factory=list)
    themes: list[Theme] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    reading_order: list[str] = field(default_factory=list)
    narrative: str = ""
    warnings: list[str] = field(default_factory=list)
    trace: list[AgentEvent] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def paper(self, arxiv_id: str) -> PaperRecord | None:
        for record in self.shortlist:
            if record.arxiv_id == arxiv_id:
                return record
        return None

    def card(self, arxiv_id: str) -> PaperCard | None:
        for card in self.cards:
            if card.arxiv_id == arxiv_id:
                return card
        return None

    def score(self, arxiv_id: str) -> PaperMetrics | None:
        return self.metrics.get(arxiv_id)

    def theme_of(self, arxiv_id: str) -> Theme | None:
        for theme in self.themes:
            if arxiv_id in theme.paper_ids:
                return theme
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "generated_at": self.generated_at,
            "max_results": self.max_results,
            "top_k": self.top_k,
            "min_relevance": self.min_relevance,
            "subqueries": self.subqueries,
            "stats": self.stats,
            "warnings": self.warnings,
            "narrative": self.narrative,
            "gaps": self.gaps,
            "reading_order": self.reading_order,
            "papers": [
                {
                    "paper": record.to_dict(),
                    "metrics": self.metrics[record.arxiv_id].to_dict()
                    if record.arxiv_id in self.metrics
                    else None,
                    "card": self.card(record.arxiv_id).to_dict()
                    if self.card(record.arxiv_id)
                    else None,
                    "theme": self.theme_of(record.arxiv_id).label
                    if self.theme_of(record.arxiv_id)
                    else None,
                }
                for record in self.shortlist
            ],
            "themes": [theme.to_dict() for theme in self.themes],
            "trace": [asdict(event) for event in self.trace],
        }
