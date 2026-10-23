from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

import pytest

from litagents import LiteratureAgentSystem
from litagents.agents import (
    CriticAgent,
    ReadingAgent,
    ScreeningAgent,
    SearchAgent,
    SynthesisAgent,
)
from litagents.llm import LLMError
from litagents.models import LiteratureState
from quant_analysis import PaperRecord, compute_metrics

NOW = datetime(2026, 10, 23, tzinfo=timezone.utc)


def make_paper(pid: str, **overrides) -> PaperRecord:
    data = {
        "arxiv_id": pid,
        "title": "Causal inference with instrumental variables",
        "abstract": (
            "Causal inference is challenging with observational data. "
            "We propose an instrumental variable estimator for heterogeneous effects. "
            "Results show improved coverage and lower bias. "
            "A limitation is that weak instruments remain challenging."
        ),
        "authors": ["Alice Smith", "Bob Jones"],
        "categories": ["stat.ME", "cs.LG"],
        "primary_category": "stat.ME",
        "published": (NOW.replace(month=9, day=1)).isoformat(),
        "citation_count": 40,
        "influential_citation_count": 10,
        "author_h_index": 20,
        "author_h_index_max": 35,
    }
    data.update(overrides)
    return PaperRecord(**data)


def fixture_papers() -> list[PaperRecord]:
    return [
        make_paper("2401.00001"),
        make_paper(
            "2401.00002",
            title="Causal discovery in time series",
            abstract=(
                "We propose a causal discovery method for nonstationary time series. "
                "Experiments show it outperforms baselines on synthetic data. "
                "Future work will address nonlinear feedback loops."
            ),
            categories=["stat.ML", "cs.LG"],
            primary_category="stat.ML",
            citation_count=18,
            author_h_index=12,
            author_h_index_max=20,
        ),
        make_paper(
            "2401.00003",
            title="A survey of causal inference methods",
            abstract=(
                "This survey reviews causal inference methods for observational studies. "
                "We categorize existing approaches and compare their assumptions. "
                "Limitations of current benchmarks remain an open problem."
            ),
            categories=["cs.LG"],
            primary_category="cs.LG",
            citation_count=120,
            influential_citation_count=30,
            author_h_index=30,
            author_h_index_max=45,
        ),
        make_paper(
            "2401.00004",
            title="Protein folding with transformers",
            abstract=(
                "We develop a transformer for protein structure prediction. "
                "Results show high accuracy on standard benchmarks. "
                "Limitations include data scarcity for rare folds."
            ),
            categories=["q-bio.BM"],
            primary_category="q-bio.BM",
            citation_count=5,
            author_h_index=6,
            author_h_index_max=8,
        ),
    ]


def fake_search(query: str, limit: int) -> list[PaperRecord]:
    return fixture_papers()[:limit]


def state_with_shortlist(query: str = "causal inference", top_k: int = 4) -> LiteratureState:
    papers = fixture_papers()
    state = LiteratureState(query=query, max_results=10, top_k=top_k)
    state.candidates = papers
    metrics = compute_metrics(papers, query, now=NOW)
    state.metrics = {paper.arxiv_id: metric for paper, metric in zip(papers, metrics)}
    state.shortlist = papers[:3]
    return state


def test_query_planning():
    queries = SearchAgent.plan_queries("causal inference")
    assert queries[0] == 'all:"causal inference"'
    assert any("survey" in query for query in queries)
    assert len(SearchAgent.plan_queries("causal inference", expand=False)) == 2
    assert SearchAgent.plan_queries('ti:"causal"') == ['ti:"causal"']
    with pytest.raises(ValueError):
        SearchAgent.plan_queries("   ")


def test_search_dedupes_and_caps():
    state = LiteratureState(query="causal inference", max_results=3)
    agent = SearchAgent(
        max_results=3,
        expand=False,
        delay=0,
        search_fn=lambda query, limit: fixture_papers(),
    )
    agent.run(state)
    assert len(state.candidates) == 3
    assert len({paper.arxiv_id for paper in state.candidates}) == 3
    assert state.stats["search"]["raw_results"] == 8
    assert state.stats["search"]["unique_results"] == 4


def test_search_raises_when_empty():
    agent = SearchAgent(search_fn=lambda query, limit: [], delay=0)
    with pytest.raises(RuntimeError):
        agent.run(LiteratureState(query="nothing"))


def test_reading_extracts_structured_fields():
    state = LiteratureState(query="causal inference")
    state.shortlist = [make_paper("2401.00001")]
    state.metrics = {
        "2401.00001": compute_metrics(state.shortlist, "causal inference", now=NOW)[0]
    }
    ReadingAgent().run(state)

    card = state.cards[0]
    assert "observational data" in card.problem
    assert "instrumental variable" in card.method
    assert "improved coverage" in card.findings
    assert "limitation" in card.limitations.lower()
    assert card.keywords
    assert card.source == "abstract"


def test_reading_does_not_treat_challenge_as_limitation():
    paper = make_paper(
        "2401.00005",
        abstract=(
            "Unmeasured confounding challenges causal inference in longitudinal studies. "
            "We propose a proxy variable estimator. Results show lower bias."
        ),
    )
    state = LiteratureState(query="causal inference")
    state.shortlist = [paper]
    state.metrics = {"2401.00005": compute_metrics(state.shortlist, "causal inference", now=NOW)[0]}
    ReadingAgent().run(state)

    assert state.cards[0].limitations == "摘要未明确说明局限"


def test_screening_filters_off_topic_and_falls_back():
    state = LiteratureState(query="causal inference", top_k=4)
    state.candidates = fixture_papers()
    ScreeningAgent(top_k=4, min_relevance=0.02, use_corpus=False).run(state)
    assert "2401.00004" not in [paper.arxiv_id for paper in state.shortlist]
    assert len(state.shortlist) == 3

    fallback = LiteratureState(query="causal inference", top_k=2)
    fallback.candidates = fixture_papers()
    ScreeningAgent(top_k=2, min_relevance=0.99, use_corpus=False).run(fallback)
    assert fallback.shortlist
    assert any("回退" in warning for warning in fallback.warnings)


def test_synthesis_builds_themes_gaps_and_reading_order():
    state = state_with_shortlist()
    ReadingAgent().run(state)
    SynthesisAgent(cluster_threshold=0.12).run(state)

    covered = {paper_id for theme in state.themes for paper_id in theme.paper_ids}
    assert covered == {paper.arxiv_id for paper in state.shortlist}
    assert state.themes
    assert state.gaps
    assert state.reading_order[0] == "2401.00003"  # survey first
    assert state.narrative.startswith("### 核心脉络")


def test_critic_removes_invalid_citations():
    state = state_with_shortlist()
    state.narrative = "有效 [arXiv:2401.00001]，无效 [arXiv:9999.99999]。"
    CriticAgent().run(state)
    assert "[arXiv:2401.00001]" in state.narrative
    assert "9999.99999" not in state.narrative
    assert "[citation removed]" in state.narrative
    assert state.stats["critic"]["invalid_citations"] == 1
    assert any("引用" in warning and "移除" in warning for warning in state.warnings)


class FakeLLM:
    def __init__(self, payload=None, text=None, error=None):
        self.payload = payload or {}
        self.text = text or ""
        self.error = error

    def chat_json(self, system, user):
        if self.error:
            raise self.error
        return self.payload

    def chat(self, system, user, temperature=0.2):
        if self.error:
            raise self.error
        return self.text


def test_llm_reading_populates_card():
    state = LiteratureState(query="causal inference")
    state.shortlist = [make_paper("2401.00001")]
    state.metrics = {
        "2401.00001": compute_metrics(state.shortlist, "causal inference", now=NOW)[0]
    }
    llm = FakeLLM(
        payload={
            "problem": "Observational causal inference is biased.",
            "method": "Instrumental variables.",
            "findings": "Lower bias.",
            "limitations": "Weak instruments.",
            "keywords": ["causal inference", "IV"],
        }
    )
    ReadingAgent(llm=llm, use_llm=True).run(state)
    card = state.cards[0]
    assert card.source == "llm"
    assert card.problem.startswith("Observational")


def test_llm_failure_falls_back_to_abstract():
    state = LiteratureState(query="causal inference")
    state.shortlist = [make_paper("2401.00001")]
    state.metrics = {
        "2401.00001": compute_metrics(state.shortlist, "causal inference", now=NOW)[0]
    }
    ReadingAgent(llm=FakeLLM(error=LLMError("boom")), use_llm=True).run(state)
    assert state.cards[0].source == "abstract"
    assert any("回退" in warning for warning in state.warnings)


def test_end_to_end_reports(tmp_path):
    system = LiteratureAgentSystem(
        max_results=10,
        top_k=4,
        offline=True,
        search_fn=fake_search,
        out_dir=str(tmp_path),
        use_corpus=False,
        delay=0,
    )
    state, paths = system.run("causal inference")

    assert len(state.shortlist) == 3
    assert len(state.cards) == 3
    assert state.themes
    assert [event.agent for event in state.trace] == [
        "search", "enrichment", "screening", "reading", "synthesis", "critic"
    ]
    assert all(event.action == "done" for event in state.trace)

    for path in paths.values():
        assert (tmp_path / path.rsplit("/", 1)[-1]).exists()

    with open(paths["json"], encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["query"] == "causal inference"
    assert len(payload["papers"]) == 3
    assert payload["themes"]
    assert payload["trace"]

    with open(paths["csv"], encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert rows[0]["rank"] == "1"

    with open(paths["markdown"], encoding="utf-8") as f:
        markdown = f.read()
    assert "文献 Agent 分析报告" in markdown
    assert "主题图谱" in markdown
    assert "阅读顺序" in markdown
    assert "论文卡片" in markdown
    assert "Agent 执行轨迹" in markdown
    assert "A survey of causal inference methods" in markdown
