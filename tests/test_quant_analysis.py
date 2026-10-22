import csv
import json
from datetime import datetime, timedelta, timezone

import pytest

from quant_analysis import (
    DEFAULT_WEIGHTS,
    PaperRecord,
    _normalize_weights,
    analyze_records,
    apply_enrichment,
    clean_query,
    compute_metrics,
    enrich_papers,
    load_corpus,
    parse_weight_spec,
    render_markdown,
    tokenize,
    update_corpus,
)

NOW = datetime(2026, 10, 22, tzinfo=timezone.utc)


def make_paper(pid="2401.00001", **overrides):
    data = {
        "arxiv_id": pid,
        "title": "Causal inference with instrumental variables",
        "abstract": "We study causal inference using instrumental variables and prove consistency.",
        "authors": ["Alice Smith", "Bob Jones"],
        "categories": ["stat.ME", "cs.LG"],
        "primary_category": "stat.ME",
        "published": (NOW - timedelta(days=30)).isoformat(),
    }
    data.update(overrides)
    return PaperRecord(**data)


def test_clean_query_and_tokenize():
    assert "causal" in tokenize(clean_query('ti:"causal" AND abs:inference'))
    assert "AND" not in tokenize(clean_query("causal AND inference"))
    assert "the" not in tokenize("The causal inference")
    assert "novel" not in tokenize("a novel method")


def test_relevance_orders_matching_paper_first():
    matching = make_paper("2401.00001")
    unrelated = make_paper(
        "2401.00002",
        title="Protein folding with deep learning",
        abstract="A neural network predicts protein structures.",
    )
    metrics = compute_metrics(
        [unrelated, matching],
        "causal inference",
        weights={"impact": 0, "author": 0, "novelty": 0},
        now=NOW,
    )
    by_id = dict(zip(["2401.00002", "2401.00001"], metrics))
    assert by_id["2401.00001"].relevance > by_id["2401.00002"].relevance
    assert by_id["2401.00001"].relevance > 0.5


def test_missing_metrics_are_imputed_without_boosting():
    paper = make_paper()
    metrics = compute_metrics([paper], "causal inference", now=NOW)[0]
    assert metrics.impact is None
    assert metrics.author is None
    assert metrics.novelty is None
    # 可用：相关性 .30 + 新鲜度 .13 + 渠道 .10 + 跨学科 .05
    assert metrics.confidence == pytest.approx(0.58)
    expected = (
        0.30 * metrics.relevance
        + 0.13 * metrics.freshness
        + 0.05 * metrics.interdisciplinarity
    )
    assert metrics.composite == pytest.approx(expected, abs=1e-6)
    assert set(metrics.raw["imputed_metrics"]) == {"impact", "author", "novelty"}


def test_impact_and_author_drive_composite():
    low = make_paper("2401.00001", citation_count=0, author_h_index=2, author_h_index_max=3)
    high = make_paper(
        "2401.00002",
        citation_count=120,
        influential_citation_count=40,
        author_h_index=25,
        author_h_index_max=45,
    )
    metrics = compute_metrics([low, high], "causal inference", now=NOW)
    assert metrics[0].impact < metrics[1].impact
    assert metrics[0].author < metrics[1].author
    assert metrics[0].composite < metrics[1].composite
    assert metrics[1].raw["citations_per_year"] > 100


def test_venue_score():
    journal = make_paper("2401.00001", journal_ref="Journal of Causal Inference")
    accepted = make_paper("2401.00002", comment="Accepted at NeurIPS 2026")
    preprint = make_paper("2401.00003", comment="Submitted to ICML")
    metrics = compute_metrics([journal, accepted, preprint], "causal", now=NOW)
    assert metrics[0].venue == 1.0
    assert metrics[1].venue == 0.6
    assert metrics[2].venue == 0.0


def test_novelty_uses_history_corpus():
    prior = {
        "2401.99999": {
            "title": "Causal inference with instrumental variables",
            "abstract": "We study causal inference using instrumental variables and prove consistency.",
        }
    }
    same = make_paper("2401.00001")
    different = make_paper(
        "2401.00002",
        title="Protein structure prediction",
        abstract="A transformer predicts protein folding from sequence.",
    )
    metrics = compute_metrics([same, different], "causal inference", corpus=prior, now=NOW)
    assert metrics[0].novelty < 0.1
    assert metrics[1].novelty > 0.8


def test_no_corpus_means_no_novelty():
    metrics = compute_metrics([make_paper()], "causal", corpus={}, now=NOW)
    assert metrics[0].novelty is None


def test_novelty_ignores_only_the_paper_itself():
    paper = make_paper("2401.00001")
    corpus = {
        "2401.00001": {"title": paper.title, "abstract": paper.abstract},
        "2401.00002": {"title": "Unrelated protein folding study", "abstract": "Proteins."},
    }
    metrics = compute_metrics([paper], "causal", corpus=corpus, now=NOW)[0]
    assert metrics.novelty is not None
    assert metrics.novelty > 0.8
    only_self = {"2401.00001": {"title": paper.title, "abstract": paper.abstract}}
    assert compute_metrics([paper], "causal", corpus=only_self, now=NOW)[0].novelty is None


def test_percentiles_and_ranking():
    low = make_paper("2401.00001", citation_count=0)
    high = make_paper("2401.00002", citation_count=200, author_h_index=40, author_h_index_max=60)
    metrics = compute_metrics([low, high], "causal inference", now=NOW)
    assert metrics[1].percentile > metrics[0].percentile
    assert metrics[1].composite > metrics[0].composite


def test_weight_normalization_and_validation():
    weights = _normalize_weights({"relevance": 3, "impact": 0})
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["relevance"] > DEFAULT_WEIGHTS["relevance"]
    assert weights["impact"] == 0
    with pytest.raises(ValueError):
        _normalize_weights({"unknown": 1})
    with pytest.raises(ValueError):
        _normalize_weights({key: 0 for key in DEFAULT_WEIGHTS})
    assert parse_weight_spec("relevance=0.5,impact=0.1") == {
        "relevance": 0.5,
        "impact": 0.1,
    }


def test_apply_enrichment():
    paper = make_paper()
    data = {
        "2401.00001": {
            "citationCount": 12,
            "influentialCitationCount": 3,
            "referenceCount": 40,
            "authors": [{"hIndex": 10}, {"hIndex": 30}, {"name": "no h"}],
            "venue": "NeurIPS",
        }
    }
    assert apply_enrichment([paper], data) == 1
    assert paper.citation_count == 12
    assert paper.author_h_index == 20.0
    assert paper.author_h_index_max == 30.0
    assert paper.venue == "NeurIPS"


def test_enrich_offline_degrades_gracefully():
    data, warnings, stats = enrich_papers([make_paper()], offline=True)
    assert data == {}
    assert warnings
    assert stats["requested"] == 0


def test_corpus_roundtrip(tmp_path):
    path = str(tmp_path / "corpus.json")
    paper = make_paper()
    assert update_corpus([paper], path) == 1
    corpus = load_corpus(path)
    assert corpus["2401.00001"]["title"] == paper.title
    # 重复更新不会产生重复条目
    assert update_corpus([paper], path) == 1


def test_reports_written(tmp_path):
    papers = [
        make_paper("2401.00001", citation_count=10, author_h_index=8, author_h_index_max=12),
        make_paper(
            "2401.00002",
            title="Causal inference with graph neural networks",
            abstract="We benchmark causal inference estimators against message passing networks.",
            categories=["cs.LG"],
            citation_count=200,
            influential_citation_count=80,
            author_h_index=40,
            author_h_index_max=60,
            comment="Accepted at ICML 2026",
        ),
    ]
    result, paths, warnings = analyze_records(
        papers,
        "causal inference",
        enrich=False,
        use_corpus=False,
        out_dir=str(tmp_path),
        now=NOW,
    )

    assert result.papers[0]["paper"]["arxiv_id"] == "2401.00002"
    assert set(paths) == {"json", "markdown", "csv"}
    for path in paths.values():
        assert (tmp_path / path.rsplit("/", 1)[-1]).exists()

    with open(paths["json"], encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["summary"]["sample_size"] == 2
    assert len(payload["papers"]) == 2
    assert payload["weights"]["relevance"] == pytest.approx(0.30)

    with open(paths["csv"], encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["rank"] == "1"

    markdown = render_markdown(result)
    assert "综合排名" in markdown
    assert papers[1].title in markdown
    assert "指标与权重" in markdown
