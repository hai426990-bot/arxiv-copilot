#!/usr/bin/env python3
"""arXiv 论文量化分析。

为每篇论文计算多维指标，生成可排序、可复现的 Markdown / JSON / CSV 报告：

- relevance           查询词与标题+摘要的 BM25 相关性
- impact              引用速度（每年引用数），并用有影响力引用占比修正
- author              作者 h-index（Semantic Scholar，缺失时自动降级）
- freshness           按 180 天半衰期衰减的发表时间得分
- venue               期刊 / DOI / 会议接收信息
- interdisciplinarity 跨 arXiv 分类的程度
- novelty             与历史已分析论文的最大相似度取反

外部指标缺失时（离线、论文尚未被 Semantic Scholar 索引），缺失项按 0 计，
同时输出置信度与指标覆盖率，避免用缺失值假装精确。

独立使用：
    python quant_analysis.py --query "causal inference" --max-results 25
    python quant_analysis.py --from-json analysis/quant_report_xxx.json --offline
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

ANALYSIS_DIR = "analysis"
CORPUS_FILE = os.path.join(ANALYSIS_DIR, "corpus.json")
CACHE_FILE = os.path.join(ANALYSIS_DIR, "semantic_scholar_cache.json")

S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_FIELDS = (
    "title,citationCount,influentialCitationCount,referenceCount,"
    "venue,year,publicationDate,authors.name,authors.hIndex,externalIds"
)

HALF_LIFE_DAYS = 180.0
IMPACT_SCALE = 10.0     # 每年 10 次引用约达到 63% 的影响力得分
AUTHOR_SCALE = 30.0     # h-index 30 约达到 63% 的作者得分
BM25_K1 = 1.5
BM25_B = 0.75
BM25_NORM = 1.0         # score/(score+1)，把 BM25 压缩到 0~1
CACHE_TTL_DAYS = 7
CORPUS_LIMIT = 5000

DEFAULT_WEIGHTS: dict[str, float] = {
    "relevance": 0.30,
    "impact": 0.22,
    "author": 0.15,
    "freshness": 0.13,
    "venue": 0.10,
    "interdisciplinarity": 0.05,
    "novelty": 0.05,
}

METRIC_LABELS = {
    "relevance": "相关性",
    "impact": "影响力",
    "author": "作者",
    "freshness": "新鲜度",
    "venue": "渠道",
    "interdisciplinarity": "跨学科",
    "novelty": "新颖度",
}

METRIC_FORMULAS = {
    "relevance": "BM25（标题权重×2，k1=1.5，b=0.75），压缩到 0~1",
    "impact": "1-exp(-引用/年/10)，再按有影响力引用占比修正",
    "author": "作者 h-index 的 0.6×最大值 + 0.4×平均值，饱和变换",
    "freshness": "0.5^(发表天数/180)",
    "venue": "期刊/DOI/正式 venue=1，仅会议接收信息=0.6，预印本=0",
    "interdisciplinarity": "min(1, log2(分类数)/2)",
    "novelty": "1 - 与历史语料的最大余弦相似度",
}

# 常见英文停用词，仅用于相关性/新颖度计算
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "also", "am",
    "an", "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "could", "did",
    "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "him", "his", "how", "however", "i", "if", "in", "into", "is", "it",
    "its", "itself", "just", "may", "me", "might", "more", "most", "much",
    "my", "no", "nor", "not", "now", "of", "off", "on", "once", "only",
    "or", "other", "our", "ours", "out", "over", "own", "paper", "papers",
    "same", "she", "should", "so", "some", "such", "than", "that", "the",
    "their", "theirs", "them", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "will", "with", "would", "you", "your", "yours", "using", "based",
    "study", "results", "show", "shows", "propose", "proposed", "approach",
    "method", "methods", "novel", "new",
}

TOKEN_RE = re.compile(r"[a-z][a-z0-9\-]{1,}")
FIELD_PREFIX_RE = re.compile(r"\b(?:ti|abs|all|au|cat|co|jr|rn|id|doi)\s*:", re.I)
BOOLEAN_RE = re.compile(r"\b(?:AND|OR|ANDNOT|NOT)\b")
VENUE_HINT_RE = re.compile(
    r"\b(?:accepted|to appear|published|camera[- ]ready|proceedings|"
    r"conference|workshop|journal)\b",
    re.I,
)


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


@dataclass
class PaperRecord:
    """论文元数据 + 外部富化字段。"""

    arxiv_id: str
    title: str
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    primary_category: str = ""
    published: str = ""
    updated: str = ""
    comment: str = ""
    journal_ref: str = ""
    doi: str = ""
    pdf_path: str = ""
    # Semantic Scholar 富化结果
    citation_count: int | None = None
    influential_citation_count: int | None = None
    reference_count: int | None = None
    author_h_index: float | None = None
    author_h_index_max: float | None = None
    venue: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PaperRecord":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PaperMetrics:
    relevance: float | None = None
    impact: float | None = None
    author: float | None = None
    freshness: float | None = None
    venue: float | None = None
    interdisciplinarity: float | None = None
    novelty: float | None = None
    composite: float = 0.0
    confidence: float = 0.0
    percentile: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    def scores(self) -> dict[str, float | None]:
        return {key: getattr(self, key) for key in DEFAULT_WEIGHTS}

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["scores"] = self.scores()
        return data


@dataclass
class AnalysisResult:
    query: str
    generated_at: str
    weights: dict[str, float]
    papers: list[dict[str, Any]]
    summary: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# 文本处理、BM25 与 TF-IDF（历史语料相似度）
# ---------------------------------------------------------------------------


def clean_query(query: str) -> str:
    """去掉 arXiv 检索语法，保留可比较的自然语言词。"""
    query = FIELD_PREFIX_RE.sub(" ", query)
    query = BOOLEAN_RE.sub(" ", query)
    return query.replace('"', " ").replace("(", " ").replace(")", " ")


def tokenize(text: str) -> list[str]:
    tokens = TOKEN_RE.findall((text or "").lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def _tfidf_vectors(docs: Sequence[Sequence[str]]) -> list[dict[str, float]]:
    n = len(docs)
    if n == 0:
        return []
    df: Counter[str] = Counter()
    for tokens in docs:
        df.update(set(tokens))
    vectors: list[dict[str, float]] = []
    for tokens in docs:
        counts = Counter(tokens)
        vec = {
            term: (1.0 + math.log(count))
            * (math.log((1.0 + n) / (1.0 + df[term])) + 1.0)
            for term, count in counts.items()
        }
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        vectors.append({k: v / norm for k, v in vec.items()})
    return vectors


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(value * b.get(term, 0.0) for term, value in a.items())


def tfidf_vectors(docs: Sequence[Sequence[str]]) -> list[dict[str, float]]:
    """公开接口：为一组词序列构建 L2 归一化的 TF-IDF 向量。"""
    return _tfidf_vectors(docs)


def cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    """公开接口：两个稀疏向量的余弦相似度。"""
    return _cosine(a, b)


def _relevance_scores(records: Sequence[PaperRecord], query: str) -> list[float]:
    """BM25 相关性：标题权重×2，带长度归一化，最后压缩到 0~1。"""
    query_terms = sorted(set(tokenize(clean_query(query))))
    docs = [tokenize(r.title) * 2 + tokenize(r.abstract) for r in records]
    if not query_terms or not any(docs):
        return [0.0] * len(records)

    n = len(docs)
    df: Counter[str] = Counter()
    for tokens in docs:
        df.update(set(tokens))
    avgdl = statistics.fmean(len(tokens) for tokens in docs) or 1.0
    idf = {
        term: math.log(1.0 + (n - df[term] + 0.5) / (df[term] + 0.5))
        for term in query_terms
    }

    scores: list[float] = []
    for tokens in docs:
        counts = Counter(tokens)
        dl = len(tokens)
        score = 0.0
        for term in query_terms:
            tf = counts.get(term, 0)
            if not tf:
                continue
            denom = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / avgdl)
            score += idf[term] * tf * (BM25_K1 + 1.0) / denom
        scores.append(round(score / (score + BM25_NORM), 6))
    return scores


def _novelty_scores(
    records: Sequence[PaperRecord],
    corpus: dict[str, dict[str, Any]],
) -> list[float | None]:
    """当前论文与历史语料的最大相似度取反；没有历史时返回 None。"""
    prior_ids = [
        pid
        for pid, entry in corpus.items()
        if entry.get("title") or entry.get("abstract")
    ]
    if not prior_ids:
        return [None] * len(records)

    prior_docs = [
        tokenize(corpus[pid].get("title", "")) * 2
        + tokenize(corpus[pid].get("abstract", ""))
        for pid in prior_ids
    ]
    current_docs = [tokenize(r.title) * 2 + tokenize(r.abstract) for r in records]
    vectors = _tfidf_vectors(prior_docs + current_docs)

    prior_vectors = vectors[: len(prior_docs)]
    scores: list[float | None] = []
    for offset, record in enumerate(records):
        current = vectors[len(prior_docs) + offset]
        # 只排除当前论文自身；历史语料中的其他论文仍应参与比较
        candidates = [
            vector
            for pid, vector in zip(prior_ids, prior_vectors)
            if pid != record.arxiv_id
        ]
        if not current or not candidates:
            scores.append(None)
            continue
        similarity = max((_cosine(current, pv) for pv in candidates), default=0.0)
        scores.append(round(min(1.0, max(0.0, 1.0 - similarity)), 6))
    return scores


# ---------------------------------------------------------------------------
# 外部富化：Semantic Scholar
# ---------------------------------------------------------------------------


def _load_json(path: str, default: Any) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def _save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _base_arxiv_id(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id)


def _extract_author_stats(authors: Any) -> tuple[float | None, float | None]:
    if not isinstance(authors, list):
        return None, None
    values = [
        float(a["hIndex"])
        for a in authors
        if isinstance(a, dict) and isinstance(a.get("hIndex"), (int, float))
    ]
    if not values:
        return None, None
    return round(statistics.fmean(values), 2), max(values)


def enrich_papers(
    records: Sequence[PaperRecord],
    cache_path: str = CACHE_FILE,
    ttl_days: int = CACHE_TTL_DAYS,
    timeout: float = 20.0,
    offline: bool = False,
) -> tuple[dict[str, dict[str, Any]], list[str], dict[str, Any]]:
    """批量查询 Semantic Scholar，带磁盘缓存、重试与离线降级。

    返回 (id -> 富化数据, 警告, 统计信息)。
    """
    stats = {"requested": 0, "cache_hits": 0, "fetched": 0, "missing": 0}
    warnings: list[str] = []
    if offline:
        warnings.append("已启用离线模式：未查询 Semantic Scholar，引用与作者指标不可用。")
        return {}, warnings, stats

    ids = [_base_arxiv_id(r.arxiv_id) for r in records]
    stats["requested"] = len(ids)
    if not ids:
        return {}, warnings, stats

    cache = _load_json(cache_path, {})
    now = time.time()
    ttl = ttl_days * 86400
    result: dict[str, dict[str, Any]] = {}
    pending: list[str] = []

    for arxiv_id in ids:
        entry = cache.get(arxiv_id)
        if isinstance(entry, dict) and now - entry.get("fetched_at", 0) < ttl:
            stats["cache_hits"] += 1
            if entry.get("data"):
                result[arxiv_id] = entry["data"]
        elif arxiv_id not in pending:
            pending.append(arxiv_id)

    if pending:
        try:
            import requests
        except ImportError:
            warnings.append("未安装 requests，无法查询 Semantic Scholar。")
            return result, warnings, stats

        headers = {}
        api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        if api_key:
            headers["x-api-key"] = api_key

        for start in range(0, len(pending), 500):
            chunk = pending[start : start + 500]
            data, error = _fetch_s2_batch(chunk, headers, timeout, requests)
            if error:
                warnings.append(error)
                continue
            for arxiv_id, item in zip(chunk, data):
                stats["fetched"] += 1
                cache[arxiv_id] = {"fetched_at": now, "data": item}
                if item:
                    result[arxiv_id] = item
                else:
                    stats["missing"] += 1

        try:
            _save_json(cache_path, cache)
        except OSError as exc:
            warnings.append(f"Semantic Scholar 缓存写入失败：{exc}")

    return result, warnings, stats


def _fetch_s2_batch(
    ids: Sequence[str],
    headers: dict[str, str],
    timeout: float,
    requests: Any,
    attempts: int = 3,
) -> tuple[list[Any], str | None]:
    payload = {"ids": [f"ARXIV:{arxiv_id}" for arxiv_id in ids]}
    last_error = ""
    for attempt in range(attempts):
        try:
            response = requests.post(
                S2_BATCH_URL,
                params={"fields": S2_FIELDS},
                json=payload,
                headers=headers,
                timeout=timeout,
            )
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    return data, None
                return [], "Semantic Scholar 返回了非预期格式。"
            last_error = f"Semantic Scholar HTTP {response.status_code}"
            if response.status_code not in (429, 500, 502, 503, 504):
                break
        except Exception as exc:  # 网络异常统一降级，不阻塞主流程
            last_error = f"Semantic Scholar 请求失败：{exc}"
        if attempt < attempts - 1:
            time.sleep(2**attempt)

    return [], f"{last_error}（已跳过外部指标，使用本地指标）。"


def apply_enrichment(
    records: Sequence[PaperRecord],
    data: dict[str, dict[str, Any]],
) -> int:
    applied = 0
    for record in records:
        item = data.get(_base_arxiv_id(record.arxiv_id))
        if not item:
            continue
        record.citation_count = item.get("citationCount")
        record.influential_citation_count = item.get("influentialCitationCount")
        record.reference_count = item.get("referenceCount")
        mean_h, max_h = _extract_author_stats(item.get("authors"))
        record.author_h_index = mean_h
        record.author_h_index_max = max_h
        if item.get("venue"):
            record.venue = str(item["venue"])
        applied += 1
    return applied


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value.strip():
        text = value.strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            try:
                dt = datetime.strptime(text[:10], "%Y-%m-%d")
            except ValueError:
                return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _saturate(value: float, scale: float) -> float:
    return 1.0 - math.exp(-max(0.0, value) / scale)


def _venue_score(record: PaperRecord) -> float:
    if record.journal_ref or record.doi:
        return 1.0
    if record.venue and record.venue.strip().lower() not in {"arxiv", "corr"}:
        return 1.0
    if record.comment and VENUE_HINT_RE.search(record.comment):
        return 0.6
    return 0.0


def _author_score(record: PaperRecord) -> float | None:
    if record.author_h_index is None and record.author_h_index_max is None:
        return None
    mean_h = record.author_h_index or 0.0
    max_h = record.author_h_index_max or 0.0
    blend = 0.6 * max_h + 0.4 * mean_h
    return round(_saturate(blend, AUTHOR_SCALE), 6)


def _impact_score(record: PaperRecord, age_days: float) -> float | None:
    if record.citation_count is None:
        return None
    age_years = max(age_days / 365.25, 0.25)  # 新论文至少按 3 个月计，避免除零放大
    citations_per_year = max(0, record.citation_count) / age_years
    score = _saturate(citations_per_year, IMPACT_SCALE)
    if record.influential_citation_count and record.citation_count > 0:
        ratio = min(1.0, record.influential_citation_count / record.citation_count)
        score *= 0.8 + 0.2 * ratio
    return round(score, 6)


def _normalize_weights(weights: dict[str, float] | None) -> dict[str, float]:
    merged = dict(DEFAULT_WEIGHTS)
    if weights:
        unknown = set(weights) - set(DEFAULT_WEIGHTS)
        if unknown:
            raise ValueError(f"未知指标：{', '.join(sorted(unknown))}")
        merged.update(weights)
    merged = {k: max(0.0, float(v)) for k, v in merged.items()}
    total = sum(merged.values())
    if total <= 0:
        raise ValueError("权重之和必须大于 0")
    return {k: v / total for k, v in merged.items()}


def compute_metrics(
    records: Sequence[PaperRecord],
    query: str,
    weights: dict[str, float] | None = None,
    corpus: dict[str, dict[str, Any]] | None = None,
    now: datetime | None = None,
) -> list[PaperMetrics]:
    weights = _normalize_weights(weights)
    now = now or datetime.now(timezone.utc)
    relevance = _relevance_scores(records, query)
    novelty = _novelty_scores(records, corpus or {})

    rows: list[tuple[PaperRecord, dict[str, float | None], float, float, float]] = []
    for record, rel, nov in zip(records, relevance, novelty):
        published = _parse_datetime(record.published) or now
        age_days = max(0.0, (now - published).total_seconds() / 86400)
        freshness = round(0.5 ** (age_days / HALF_LIFE_DAYS), 6)
        categories = record.categories or ([record.primary_category] if record.primary_category else [])
        interdisc = round(min(1.0, math.log2(max(1, len(categories))) / 2), 6)
        scores: dict[str, float | None] = {
            "relevance": round(rel, 6),
            "impact": _impact_score(record, age_days),
            "author": _author_score(record),
            "freshness": freshness,
            "venue": _venue_score(record),
            "interdisciplinarity": interdisc,
            "novelty": nov,
        }
        rows.append((record, scores, rel, nov, age_days))

    # 缺失指标按 0 计，固定权重分母：没有证据就不加分，避免缺项论文
    # 因“只比优势指标”而排名虚高；实际覆盖情况由 confidence 反映。
    results: list[PaperMetrics] = []
    for record, scores, rel, nov, age_days in rows:
        imputed = [key for key in DEFAULT_WEIGHTS if scores[key] is None]
        composite = sum(
            weights[key] * (scores[key] if scores[key] is not None else 0.0)
            for key in DEFAULT_WEIGHTS
        )
        confidence = sum(weights[key] for key in DEFAULT_WEIGHTS if scores[key] is not None)
        citations_per_year = None
        if record.citation_count is not None:
            citations_per_year = round(
                max(0, record.citation_count) / max(age_days / 365.25, 0.25), 3
            )

        results.append(
            PaperMetrics(
                relevance=scores["relevance"],
                impact=scores["impact"],
                author=scores["author"],
                freshness=scores["freshness"],
                venue=scores["venue"],
                interdisciplinarity=scores["interdisciplinarity"],
                novelty=scores["novelty"],
                composite=round(composite, 6),
                confidence=round(confidence, 6),
                raw={
                    "age_days": round(age_days, 1),
                    "citation_count": record.citation_count,
                    "influential_citation_count": record.influential_citation_count,
                    "reference_count": record.reference_count,
                    "citations_per_year": citations_per_year,
                    "author_h_index": record.author_h_index,
                    "author_h_index_max": record.author_h_index_max,
                    "category_count": len(
                        record.categories
                        or ([record.primary_category] if record.primary_category else [])
                    ),
                    "relevance_bm25": round(rel, 6),
                    "novelty_similarity": (
                        round(1.0 - nov, 6) if nov is not None else None
                    ),
                    "imputed_metrics": imputed,
                },
            )
        )

    _assign_percentiles(results)
    return results


def _assign_percentiles(metrics: Sequence[PaperMetrics]) -> None:
    values = [m.composite for m in metrics]
    n = len(values)
    if n == 0:
        return
    for metric in metrics:
        less = sum(1 for v in values if v < metric.composite)
        equal = sum(1 for v in values if v == metric.composite)
        metric.percentile = round((less + 0.5 * equal) / n, 6)


# ---------------------------------------------------------------------------
# 汇总与报告
# ---------------------------------------------------------------------------


def _metric_summary(metrics: Sequence[PaperMetrics], weights: dict[str, float]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in DEFAULT_WEIGHTS:
        values = [getattr(m, key) for m in metrics if getattr(m, key) is not None]
        summary[key] = {
            "weight": round(weights[key], 4),
            "coverage": round(len(values) / len(metrics), 4) if metrics else 0.0,
            "mean": round(statistics.fmean(values), 4) if values else None,
            "std": round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
        }
    return summary


def _build_summary(
    records: Sequence[PaperRecord],
    metrics: Sequence[PaperMetrics],
    weights: dict[str, float],
) -> dict[str, Any]:
    dates = [d for d in (_parse_datetime(r.published) for r in records) if d]
    categories = Counter(
        c for r in records for c in (r.categories or ([r.primary_category] if r.primary_category else []))
    )
    authors = Counter(a for r in records for a in r.authors)
    composites = [m.composite for m in metrics]
    return {
        "sample_size": len(records),
        "date_range": {
            "start": min(dates).date().isoformat() if dates else None,
            "end": max(dates).date().isoformat() if dates else None,
        },
        "composite_mean": round(statistics.fmean(composites), 4) if composites else 0.0,
        "composite_median": round(statistics.median(composites), 4) if composites else 0.0,
        "mean_confidence": round(statistics.fmean(m.confidence for m in metrics), 4)
        if metrics
        else 0.0,
        "top_categories": categories.most_common(8),
        "top_authors": authors.most_common(8),
        "metrics": _metric_summary(metrics, weights),
    }


def build_result(
    records: Sequence[PaperRecord],
    metrics: Sequence[PaperMetrics],
    query: str,
    weights: dict[str, float],
    warnings: Sequence[str] | None = None,
) -> AnalysisResult:
    order = sorted(range(len(records)), key=lambda i: metrics[i].composite, reverse=True)
    papers = []
    for rank, idx in enumerate(order, 1):
        papers.append(
            {
                "rank": rank,
                "paper": records[idx].to_dict(),
                "metrics": metrics[idx].to_dict(),
            }
        )

    warning_list = list(warnings or [])
    summary = _build_summary(records, metrics, weights)
    impact = summary["metrics"]["impact"]
    if impact["coverage"] == 0:
        warning_list.append(
            "本批论文没有可用引用数据（新预印本通常尚未被索引），综合得分实际由其余指标决定。"
        )
    elif impact["std"] == 0:
        warning_list.append("本批论文的影响力得分没有差异，该指标不参与区分排序。")

    return AnalysisResult(
        query=query,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        weights={k: round(v, 4) for k, v in weights.items()},
        papers=papers,
        summary=summary,
        warnings=warning_list,
    )


def _fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _md_escape(text: str) -> str:
    return (text or "").replace("|", "\\|").replace("\n", " ")


def render_markdown(result: AnalysisResult) -> str:
    lines: list[str] = []
    summary = result.summary
    weights = result.weights

    lines.append("# arXiv 论文量化分析报告")
    lines.append("")
    lines.append(f"- 生成时间：{result.generated_at}")
    lines.append(f"- 检索式：`{result.query}`")
    lines.append(f"- 样本量：{summary['sample_size']} 篇")
    date_range = summary["date_range"]
    if date_range["start"]:
        lines.append(f"- 发表时间：{date_range['start']} ~ {date_range['end']}")
    lines.append(
        f"- 综合得分：均值 {summary['composite_mean']:.3f}，"
        f"中位数 {summary['composite_median']:.3f}，"
        f"平均置信度 {summary['mean_confidence']:.3f}"
    )
    if summary["top_categories"]:
        lines.append(
            "- 主要分类："
            + "，".join(f"{name}（{count}）" for name, count in summary["top_categories"][:5])
        )
    lines.append("")

    if result.warnings:
        lines.append("> **注意**")
        for warning in result.warnings:
            lines.append(f"> - {warning}")
        lines.append("")

    lines.append("## 指标与权重")
    lines.append("")
    lines.append("| 指标 | 权重 | 覆盖率 | 均值 | 标准差 | 计算方式 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for key in DEFAULT_WEIGHTS:
        item = summary["metrics"][key]
        lines.append(
            f"| {METRIC_LABELS[key]} | {weights[key]:.2f} | {item['coverage']:.0%} | "
            f"{_fmt(item['mean'])} | {_fmt(item['std'])} | {METRIC_FORMULAS[key]} |"
        )
    lines.append("")

    lines.append("## 综合排名")
    lines.append("")
    header = ["#", "论文", "综合", "置信度"] + [METRIC_LABELS[k] for k in DEFAULT_WEIGHTS]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| ---: | --- | ---: | ---: |" + " ---: |" * len(DEFAULT_WEIGHTS))
    for item in result.papers:
        paper = item["paper"]
        metrics = item["metrics"]
        title = _md_escape(paper["title"])
        link = f"[{title}](https://arxiv.org/abs/{paper['arxiv_id']})"
        row = [
            str(item["rank"]),
            link,
            f"{metrics['composite']:.3f}",
            f"{metrics['confidence']:.2f}",
        ] + [_fmt(metrics["scores"][k]) for k in DEFAULT_WEIGHTS]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## 论文详情")
    lines.append("")
    for item in result.papers:
        paper = item["paper"]
        metrics = item["metrics"]
        raw = metrics["raw"]
        lines.append(f"### {item['rank']}. {paper['title']}")
        lines.append("")
        lines.append(f"- arXiv：`{paper['arxiv_id']}`")
        if paper["authors"]:
            authors = paper["authors"]
            shown = "，".join(authors[:8]) + (" 等" if len(authors) > 8 else "")
            lines.append(f"- 作者：{shown}")
        categories = paper["categories"] or [paper["primary_category"]]
        if categories:
            lines.append(f"- 分类：{', '.join(c for c in categories if c)}")
        if paper["published"]:
            lines.append(f"- 发表：{paper['published'][:10]}")
        if paper["journal_ref"]:
            lines.append(f"- 期刊：{paper['journal_ref']}")
        lines.append(
            "- 原始指标："
            f"引用 {raw['citation_count'] if raw['citation_count'] is not None else '—'}，"
            f"引用/年 {raw['citations_per_year'] if raw['citations_per_year'] is not None else '—'}，"
            f"作者 h-index {raw['author_h_index_max'] if raw['author_h_index_max'] is not None else '—'}，"
            f"分类数 {raw['category_count']}"
        )
        lines.append(
            "- 得分："
            f"综合 {metrics['composite']:.3f}（批次百分位 {metrics['percentile']:.0%}）"
        )
        imputed = raw.get("imputed_metrics") or []
        if imputed:
            lines.append(
                "- 缺失指标："
                + "、".join(METRIC_LABELS[key] for key in imputed)
                + "（按 0 计，不参与加分）"
            )
        if paper["abstract"]:
            abstract = paper["abstract"].strip().replace("\n", " ")
            if len(abstract) > 700:
                abstract = abstract[:700].rstrip() + "…"
            lines.append("")
            lines.append(f"> {abstract}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "说明：各指标均映射到 0~1；缺失指标按 0 计，不会因缺项而只比较优势指标，"
        "置信度即实际覆盖权重。"
    )
    lines.append("")
    return "\n".join(lines)


def write_reports(
    result: AnalysisResult,
    out_dir: str = ANALYSIS_DIR,
    timestamp: datetime | None = None,
) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    stamp = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    base = os.path.join(out_dir, f"quant_report_{stamp}")

    json_path = f"{base}.json"
    md_path = f"{base}.md"
    csv_path = f"{base}.csv"

    _save_json(json_path, result.to_dict())
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(result))

    fields = [
        "rank", "arxiv_id", "title", "authors", "published", "primary_category",
        "composite", "confidence", "percentile", *DEFAULT_WEIGHTS,
        "citation_count", "citations_per_year", "author_h_index_max", "category_count",
        "imputed_metrics",
    ]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in result.papers:
            paper = item["paper"]
            metrics = item["metrics"]
            row = {
                "rank": item["rank"],
                "arxiv_id": paper["arxiv_id"],
                "title": paper["title"],
                "authors": "; ".join(paper["authors"]),
                "published": paper["published"],
                "primary_category": paper["primary_category"],
                "composite": metrics["composite"],
                "confidence": metrics["confidence"],
                "percentile": metrics["percentile"],
                "citation_count": metrics["raw"]["citation_count"],
                "citations_per_year": metrics["raw"]["citations_per_year"],
                "author_h_index_max": metrics["raw"]["author_h_index_max"],
                "category_count": metrics["raw"]["category_count"],
                "imputed_metrics": "; ".join(metrics["raw"]["imputed_metrics"]),
            }
            row.update({key: metrics["scores"][key] for key in DEFAULT_WEIGHTS})
            writer.writerow(row)

    return {"json": json_path, "markdown": md_path, "csv": csv_path}


# ---------------------------------------------------------------------------
# 历史语料（用于相对新颖度）
# ---------------------------------------------------------------------------


def load_corpus(path: str = CORPUS_FILE) -> dict[str, dict[str, Any]]:
    data = _load_json(path, {})
    papers = data.get("papers", data) if isinstance(data, dict) else {}
    return papers if isinstance(papers, dict) else {}


def update_corpus(
    records: Sequence[PaperRecord],
    path: str = CORPUS_FILE,
    limit: int = CORPUS_LIMIT,
) -> int:
    corpus = load_corpus(path)
    now = time.time()
    for record in records:
        corpus[record.arxiv_id] = {
            "title": record.title,
            "abstract": record.abstract,
            "updated_at": now,
        }
    if len(corpus) > limit:
        ordered = sorted(
            corpus.items(),
            key=lambda item: item[1].get("updated_at", 0),
            reverse=True,
        )[:limit]
        corpus = dict(ordered)
    _save_json(path, {"updated_at": now, "papers": corpus})
    return len(corpus)


# ---------------------------------------------------------------------------
# 一站式入口
# ---------------------------------------------------------------------------


def analyze_records(
    records: Sequence[PaperRecord],
    query: str,
    weights: dict[str, float] | None = None,
    enrich: bool = True,
    use_corpus: bool = True,
    out_dir: str = ANALYSIS_DIR,
    cache_path: str | None = None,
    corpus_path: str | None = None,
    now: datetime | None = None,
) -> tuple[AnalysisResult, dict[str, str], list[str]]:
    if not records:
        raise ValueError("没有可分析的论文")

    weights = _normalize_weights(weights)
    warnings: list[str] = []
    cache_path = cache_path or os.path.join(out_dir, "semantic_scholar_cache.json")
    corpus_path = corpus_path or os.path.join(out_dir, "corpus.json")
    corpus = load_corpus(corpus_path) if use_corpus else {}

    if enrich:
        data, enrich_warnings, stats = enrich_papers(records, cache_path=cache_path)
        warnings.extend(enrich_warnings)
        applied = apply_enrichment(records, data)
        if stats["requested"] and applied == 0:
            warnings.append("没有论文获得外部引用数据，本次仅使用本地指标。")
    else:
        warnings.append("未查询外部指标（离线模式），引用与作者指标按缺失处理。")

    metrics = compute_metrics(records, query, weights, corpus=corpus, now=now)
    result = build_result(records, metrics, query, weights, warnings)
    paths = write_reports(result, out_dir=out_dir)
    if use_corpus:
        update_corpus(records, corpus_path)
    return result, paths, warnings


def fetch_arxiv_records(query: str, max_results: int = 25) -> list[PaperRecord]:
    try:
        import arxiv
    except ImportError as exc:
        raise RuntimeError("缺少 arxiv 依赖，请先 pip install -r requirements.txt") from exc

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    records: list[PaperRecord] = []
    for item in arxiv.Client().results(search):
        primary = item.primary_category
        records.append(
            PaperRecord(
                arxiv_id=item.get_short_id(),
                title=item.title.strip(),
                abstract=item.summary.strip(),
                authors=[a.name for a in item.authors],
                categories=list(item.categories),
                primary_category=primary.category if hasattr(primary, "category") else primary,
                published=item.published.isoformat(),
                updated=item.updated.isoformat(),
                comment=item.comment or "",
                journal_ref=getattr(item, "journal_ref", "") or "",
                doi=getattr(item, "doi", "") or "",
            )
        )
    return records


def _load_records_from_json(path: str) -> list[PaperRecord]:
    data = _load_json(path, None)
    if data is None:
        raise ValueError(f"无法读取 JSON：{path}")
    if isinstance(data, dict) and "papers" in data:
        items = [entry.get("paper", entry) for entry in data["papers"]]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError("JSON 需为论文数组或 quant_report.json")
    return [PaperRecord.from_dict(item) for item in items if isinstance(item, dict)]


def parse_weight_spec(spec: str | None) -> dict[str, float] | None:
    if not spec:
        return None
    weights: dict[str, float] = {}
    for part in spec.split(","):
        if not part.strip():
            continue
        key, sep, value = part.partition("=")
        if not sep:
            raise ValueError(f"权重格式应为 key=value：{part}")
        key = key.strip()
        if key not in DEFAULT_WEIGHTS:
            raise ValueError(f"未知指标：{key}")
        weights[key] = float(value)
    return weights


def _print_console(result: AnalysisResult, paths: dict[str, str], top: int = 10) -> None:
    print("\n" + "=" * 78)
    print(f"量化分析：{result.summary['sample_size']} 篇 | 平均综合得分 "
          f"{result.summary['composite_mean']:.3f} | 平均置信度 "
          f"{result.summary['mean_confidence']:.2f}")
    print("=" * 78)
    print(f"{'#':>3}  {'综合':>6}  {'置信':>5}  {'相关':>5}  {'影响':>5}  {'作者':>5}  论文")
    for item in result.papers[:top]:
        paper, metrics = item["paper"], item["metrics"]
        scores = metrics["scores"]
        title = paper["title"] if len(paper["title"]) <= 44 else paper["title"][:43] + "…"
        print(
            f"{item['rank']:>3}  {metrics['composite']:>6.3f}  {metrics['confidence']:>5.2f}  "
            f"{_fmt(scores['relevance']):>5}  {_fmt(scores['impact']):>5}  "
            f"{_fmt(scores['author']):>5}  {title}"
        )
    print("-" * 78)
    print(f"Markdown: {paths['markdown']}")
    print(f"JSON    : {paths['json']}")
    print(f"CSV     : {paths['csv']}\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="arXiv 论文多维量化分析")
    parser.add_argument("--query", default="causal", help="arXiv 检索式（默认 causal）")
    parser.add_argument("--max-results", type=int, default=25, help="最多分析多少篇")
    parser.add_argument("--from-json", help="从已有报告或论文数组 JSON 读取，不重新抓取")
    parser.add_argument("--offline", action="store_true", help="不查询 Semantic Scholar")
    parser.add_argument("--no-corpus", action="store_true", help="不使用/更新历史语料")
    parser.add_argument("--out", default=ANALYSIS_DIR, help="报告输出目录")
    parser.add_argument("--weights", help="覆盖权重，如 relevance=0.4,impact=0.2")
    parser.add_argument("--top", type=int, default=10, help="终端展示前 N 篇")
    args = parser.parse_args(argv)

    try:
        weights = parse_weight_spec(args.weights)
        if args.from_json:
            records = _load_records_from_json(args.from_json)
        else:
            print(f"[▶] 抓取 arXiv：{args.query}（最多 {args.max_results} 篇）")
            records = fetch_arxiv_records(args.query, args.max_results)
        if not records:
            print("[ℹ] 没有找到论文")
            return 1
        result, paths, warnings = analyze_records(
            records,
            args.query,
            weights=weights,
            enrich=not args.offline,
            use_corpus=not args.no_corpus,
            out_dir=args.out,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"[❌] {exc}", file=sys.stderr)
        return 2

    for warning in warnings:
        print(f"[⚠] {warning}")
    _print_console(result, paths, top=args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
