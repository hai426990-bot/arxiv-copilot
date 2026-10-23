"""文献 Agent 的具体实现。

每个 Agent 只负责一个明确步骤，读写共享的 LiteratureState，便于单独测试和替换。
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from typing import Callable, Sequence

from quant_analysis import (
    PaperRecord,
    apply_enrichment,
    compute_metrics,
    cosine_similarity,
    enrich_papers,
    fetch_arxiv_records,
    load_corpus,
    tfidf_vectors,
    tokenize,
)

from .llm import LLMClient, LLMError
from .models import LiteratureState, PaperCard, Theme

FIELD_PREFIX_RE = re.compile(r"\b(?:ti|abs|all|au|cat|co|jr|rn|id|doi)\s*:", re.I)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
SURVEY_RE = re.compile(r"\b(?:survey|review|overview|tutorial|benchmark)\b", re.I)
CITATION_RE = re.compile(r"\[arXiv:([^\]]+)\]")

METHOD_CUES = (
    "we propose", "we present", "we introduce", "we develop", "we design",
    "we build", "we use", "we apply", "our method", "our approach", "our framework",
    "this paper presents", "this work proposes", "based on",
)
FINDING_CUES = (
    "results show", "results suggest", "we show", "we find", "we demonstrate",
    "experiments show", "achieves", "outperforms", "improves", "reduces",
    "suggest that", "indicate that", "evidence that",
)
LIMITATION_CUES = (
    "limitation", "future work", "future research", "open question",
    "open problem", "remains an open", "remains unclear", "not addressed",
    "we leave", "fails to",
)


class Agent:
    name = "agent"

    def run(self, state: LiteratureState) -> None:  # pragma: no cover - 接口
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1. 检索
# ---------------------------------------------------------------------------


class SearchAgent(Agent):
    name = "search"

    def __init__(
        self,
        max_results: int = 30,
        expand: bool = True,
        delay: float = 3.0,
        search_fn: Callable[[str, int], list[PaperRecord]] = fetch_arxiv_records,
    ) -> None:
        self.max_results = max(1, max_results)
        self.expand = expand
        self.delay = max(0.0, delay)
        self.search_fn = search_fn

    @staticmethod
    def plan_queries(query: str, expand: bool = True) -> list[str]:
        text = (query or "").strip()
        if not text:
            raise ValueError("检索词不能为空")
        if FIELD_PREFIX_RE.search(text) or text.startswith('"'):
            return [text]

        phrase = text.strip().strip('"').strip()
        queries = [f'all:"{phrase}"', f'ti:"{phrase}"']
        if expand:
            queries.append(f'all:"{phrase}" AND all:survey')
            queries.append(f'all:"{phrase}" AND all:benchmark')

        unique: list[str] = []
        for item in queries:
            if item not in unique:
                unique.append(item)
        return unique

    def run(self, state: LiteratureState) -> None:
        state.subqueries = self.plan_queries(state.query, self.expand)
        per_query = max(3, math.ceil(self.max_results / len(state.subqueries)) + 3)
        raw = 0
        seen_ids: set[str] = set()
        seen_titles: set[str] = set()
        unique: list[PaperRecord] = []

        for index, subquery in enumerate(state.subqueries):
            try:
                records = self.search_fn(subquery, per_query)
            except Exception as exc:
                state.warnings.append(f"检索失败 `{subquery}`：{exc}")
                records = []
            raw += len(records)
            for record in records:
                base_id = re.sub(r"v\d+$", "", record.arxiv_id)
                title_key = re.sub(r"[^a-z0-9]", "", record.title.lower())[:200]
                if base_id in seen_ids or title_key in seen_titles:
                    continue
                seen_ids.add(base_id)
                seen_titles.add(title_key)
                unique.append(record)
            if self.delay and index < len(state.subqueries) - 1:
                time.sleep(self.delay)

        state.candidates = unique[: self.max_results]
        state.stats["search"] = {
            "queries": len(state.subqueries),
            "raw_results": raw,
            "unique_results": len(unique),
            "kept": len(state.candidates),
        }
        if not state.candidates:
            raise RuntimeError("没有检索到论文，请检查检索词或网络")


# ---------------------------------------------------------------------------
# 2. 外部指标富化
# ---------------------------------------------------------------------------


class EnrichmentAgent(Agent):
    name = "enrichment"

    def __init__(self, offline: bool = False, cache_path: str | None = None) -> None:
        self.offline = offline
        self.cache_path = cache_path

    def run(self, state: LiteratureState) -> None:
        if self.offline:
            state.warnings.append("离线模式：跳过 Semantic Scholar，引用与作者指标按缺失处理。")
            state.stats["enrichment"] = {"applied": 0, "offline": True}
            return

        kwargs = {}
        if self.cache_path:
            kwargs["cache_path"] = self.cache_path
        data, warnings, stats = enrich_papers(state.candidates, **kwargs)
        state.warnings.extend(warnings)
        applied = apply_enrichment(state.candidates, data)
        state.stats["enrichment"] = {
            "applied": applied,
            "cache_hits": stats.get("cache_hits", 0),
            "missing": stats.get("missing", 0),
            "offline": False,
        }


# ---------------------------------------------------------------------------
# 3. 量化筛选
# ---------------------------------------------------------------------------


class ScreeningAgent(Agent):
    name = "screening"

    def __init__(
        self,
        top_k: int = 10,
        min_relevance: float = 0.02,
        weights: dict[str, float] | None = None,
        corpus_path: str | None = None,
        use_corpus: bool = True,
    ) -> None:
        self.top_k = max(1, top_k)
        self.min_relevance = min_relevance
        self.weights = weights
        self.corpus_path = corpus_path
        self.use_corpus = use_corpus

    def run(self, state: LiteratureState) -> None:
        corpus = (
            load_corpus(self.corpus_path)
            if self.use_corpus and self.corpus_path
            else (load_corpus() if self.use_corpus else {})
        )
        metrics = compute_metrics(
            state.candidates,
            state.query,
            weights=self.weights,
            corpus=corpus,
        )
        state.metrics = {paper.arxiv_id: metric for paper, metric in zip(state.candidates, metrics)}

        pairs = list(zip(state.candidates, metrics))
        eligible = [(p, m) for p, m in pairs if m.relevance >= self.min_relevance]
        if not eligible:
            state.warnings.append(
                f"没有论文达到相关性阈值 {self.min_relevance:.2f}，按相关性回退选取。"
            )
            eligible = sorted(pairs, key=lambda pair: pair[1].relevance, reverse=True)

        eligible.sort(key=lambda pair: pair[1].composite, reverse=True)
        state.shortlist = [paper for paper, _ in eligible[: self.top_k]]
        state.stats["screening"] = {
            "candidates": len(state.candidates),
            "eligible": len(eligible),
            "shortlisted": len(state.shortlist),
            "mean_score": round(
                sum(m.composite for p, m in eligible[: self.top_k]) / max(1, len(state.shortlist)),
                4,
            ),
        }
        if len(state.shortlist) < self.top_k:
            state.warnings.append(
                f"仅 {len(state.shortlist)} 篇通过相关性阈值，少于目标 {self.top_k} 篇。"
            )


# ---------------------------------------------------------------------------
# 4. 逐篇阅读
# ---------------------------------------------------------------------------


class ReadingAgent(Agent):
    name = "reading"

    def __init__(self, llm: LLMClient | None = None, use_llm: bool = False) -> None:
        self.llm = llm
        self.use_llm = use_llm

    def run(self, state: LiteratureState) -> None:
        docs = [
            tokenize(paper.title) * 2 + tokenize(paper.abstract)
            for paper in state.shortlist
        ]
        vectors = tfidf_vectors(docs)
        llm_failed = False

        cards: list[PaperCard] = []
        for index, paper in enumerate(state.shortlist):
            card = self._extract_card(paper, vectors[index] if index < len(vectors) else {})
            if self.use_llm and self.llm:
                try:
                    card = self._llm_card(paper, card)
                except LLMError as exc:
                    if not llm_failed:
                        state.warnings.append(f"LLM 阅读失败，已回退到摘要抽取：{exc}")
                        llm_failed = True
            cards.append(card)

        state.cards = cards
        state.stats["reading"] = {
            "cards": len(cards),
            "llm_cards": sum(1 for card in cards if card.source == "llm"),
        }

    def _extract_card(
        self,
        paper: PaperRecord,
        vector: dict[str, float],
    ) -> PaperCard:
        sentences = [
            part.strip()
            for part in SENTENCE_RE.split((paper.abstract or "").strip())
            if len(part.strip()) > 20
        ]
        problem = self._truncate(" ".join(sentences[:2]) or paper.abstract)
        method = self._pick(sentences, METHOD_CUES) or "摘要未明确说明方法"
        findings = self._pick(sentences, FINDING_CUES) or (sentences[-1] if sentences else "")
        findings = self._truncate(findings) or "摘要未明确说明结论"
        limitations = self._pick(sentences, LIMITATION_CUES) or "摘要未明确说明局限"
        keywords = [
            term.replace("-", " ").title()
            for term, _ in sorted(vector.items(), key=lambda item: item[1], reverse=True)[:5]
        ]
        if not keywords:
            keywords = list(dict.fromkeys(tokenize(paper.title)))[:5]

        return PaperCard(
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            authors=paper.authors,
            published=paper.published,
            categories=paper.categories,
            problem=problem,
            method=self._truncate(method),
            findings=findings,
            limitations=self._truncate(limitations),
            keywords=keywords,
            evidence=self._truncate(paper.abstract, 500),
            source="abstract",
        )

    def _llm_card(self, paper: PaperRecord, fallback: PaperCard) -> PaperCard:
        system = (
            "你是严谨的文献阅读助手。只能依据给出的标题和摘要提取信息，不得编造。"
            "输出 JSON 对象，字段为 problem、method、findings、limitations、keywords。"
            "keywords 是 3-6 个英文关键词。limitations 若摘要未提及，写“摘要未明确说明局限”。"
        )
        user = (
            f"标题：{paper.title}\n作者：{', '.join(paper.authors[:8])}\n"
            f"摘要：{paper.abstract}\n\n只输出 JSON。"
        )
        data = self.llm.chat_json(system, user) if self.llm else None
        if not data:
            return fallback

        def field(name: str, default: str) -> str:
            value = data.get(name)
            return self._truncate(str(value).strip()) if isinstance(value, str) and value.strip() else default

        keywords = data.get("keywords")
        if not isinstance(keywords, list):
            keywords = fallback.keywords
        keywords = [str(item).strip() for item in keywords if str(item).strip()][:6]

        used = sum(
            1
            for name in ("problem", "method", "findings", "limitations")
            if isinstance(data.get(name), str) and str(data.get(name)).strip()
        )
        return PaperCard(
            arxiv_id=fallback.arxiv_id,
            title=fallback.title,
            authors=fallback.authors,
            published=fallback.published,
            categories=fallback.categories,
            problem=field("problem", fallback.problem),
            method=field("method", fallback.method),
            findings=field("findings", fallback.findings),
            limitations=field("limitations", fallback.limitations),
            keywords=keywords or fallback.keywords,
            evidence=fallback.evidence,
            source="llm" if used >= 2 else "abstract",
        )

    @staticmethod
    def _pick(sentences: Sequence[str], cues: Sequence[str]) -> str:
        for sentence in sentences:
            lowered = sentence.lower()
            if any(cue in lowered for cue in cues):
                return sentence
        return ""

    @staticmethod
    def _truncate(text: str, limit: int = 420) -> str:
        text = " ".join((text or "").split())
        return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


# ---------------------------------------------------------------------------
# 5. 主题综合
# ---------------------------------------------------------------------------


class SynthesisAgent(Agent):
    name = "synthesis"

    def __init__(
        self,
        cluster_threshold: float = 0.12,
        llm: LLMClient | None = None,
        use_llm: bool = False,
    ) -> None:
        self.cluster_threshold = cluster_threshold
        self.llm = llm
        self.use_llm = use_llm

    def run(self, state: LiteratureState) -> None:
        if not state.shortlist:
            state.warnings.append("没有可综合的论文。")
            return

        state.themes = self._cluster(state)
        state.gaps = self._gaps(state)
        state.reading_order = self._reading_order(state)
        state.narrative = self._narrative(state)
        state.stats["synthesis"] = {
            "themes": len(state.themes),
            "gaps": len(state.gaps),
            "llm_narrative": bool(self.use_llm and self.llm),
        }

    def _cluster(self, state: LiteratureState) -> list[Theme]:
        papers = state.shortlist
        docs = [tokenize(p.title) * 2 + tokenize(p.abstract) for p in papers]
        vectors = tfidf_vectors(docs)
        order = sorted(
            range(len(papers)),
            key=lambda i: state.metrics.get(papers[i].arxiv_id).composite
            if state.metrics.get(papers[i].arxiv_id)
            else 0.0,
            reverse=True,
        )

        clusters: list[list[int]] = []
        for index in order:
            best_cluster = -1
            best_similarity = 0.0
            for cluster_index, members in enumerate(clusters):
                similarity = max(
                    (cosine_similarity(vectors[index], vectors[member]) for member in members),
                    default=0.0,
                )
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster = cluster_index
            if best_cluster >= 0 and best_similarity >= self.cluster_threshold:
                clusters[best_cluster].append(index)
            else:
                clusters.append([index])

        themes: list[Theme] = []
        for members in clusters:
            keyword_scores: Counter[str] = Counter()
            for member in members:
                keyword_scores.update(vectors[member])
            keywords = [
                term.replace("-", " ").title()
                for term, _ in keyword_scores.most_common(5)
            ]
            representative = sorted(
                members,
                key=lambda i: state.metrics.get(papers[i].arxiv_id).composite
                if state.metrics.get(papers[i].arxiv_id)
                else 0.0,
                reverse=True,
            )
            titles = [papers[i].title for i in representative[:2]]
            scores = [
                state.metrics[papers[i].arxiv_id].composite
                for i in members
                if papers[i].arxiv_id in state.metrics
            ]
            label = " / ".join(keywords[:3]) or f"主题 {len(themes) + 1}"
            summary = (
                f"该主题包含 {len(members)} 篇，聚焦 {', '.join(keywords[:4]) or '未提取到关键词'}。"
                f"代表工作：{'；'.join(titles)}。"
            )
            themes.append(
                Theme(
                    label=label,
                    keywords=keywords,
                    paper_ids=[papers[i].arxiv_id for i in members],
                    summary=summary,
                    mean_score=round(sum(scores) / len(scores), 4) if scores else 0.0,
                )
            )

        themes.sort(key=lambda theme: theme.mean_score, reverse=True)
        return themes

    def _gaps(self, state: LiteratureState) -> list[str]:
        gaps: list[str] = []
        for theme in state.themes:
            if len(theme.paper_ids) == 1:
                gaps.append(
                    f"主题「{theme.label}」仅 1 篇，尚缺乏交叉验证，结论需谨慎。"
                )

        confidence = [
            state.metrics[paper.arxiv_id].confidence
            for paper in state.shortlist
            if paper.arxiv_id in state.metrics
        ]
        low_confidence = sum(1 for value in confidence if value < 0.6)
        if low_confidence:
            gaps.append(
                f"{low_confidence}/{len(state.shortlist)} 篇外部指标覆盖不足，"
                "引用与作者维度需要补查。"
            )

        if state.shortlist and all(not paper.journal_ref and not paper.doi for paper in state.shortlist):
            gaps.append("本批论文均未见期刊/DOI 信息，发表状态可能仍为预印本。")

        categories = Counter(
            category
            for paper in state.shortlist
            for category in (paper.categories or [paper.primary_category])
            if category
        )
        if categories and categories.most_common(1)[0][1] / len(state.shortlist) >= 0.7:
            gaps.append(
                f"分类高度集中于 {categories.most_common(1)[0][0]}，"
                "跨领域视角可能不足。"
            )

        missing_limitations = sum(
            1
            for card in state.cards
            if "未明确" in card.limitations
        )
        if missing_limitations >= max(1, len(state.cards) // 2):
            gaps.append("多数摘要未明确说明局限，建议阅读全文核实适用范围。")
        return gaps[:5]

    def _reading_order(self, state: LiteratureState) -> list[str]:
        papers = list(state.shortlist)
        papers.sort(
            key=lambda paper: (
                0 if SURVEY_RE.search(paper.title) else 1,
                -(state.metrics[paper.arxiv_id].composite
                  if paper.arxiv_id in state.metrics
                  else 0.0),
            )
        )
        return [paper.arxiv_id for paper in papers]

    def _narrative(self, state: LiteratureState) -> str:
        if self.use_llm and self.llm:
            try:
                text = self.llm.chat(
                    "你是严谨的文献综述作者。只能使用给定论文卡片中的信息，"
                    "每个结论后必须用 [arXiv:id] 标注来源，不得引用列表外的论文。"
                    "用中文输出 Markdown，使用三级标题（###）组织：核心脉络、主题对比、"
                    "研究空白、阅读建议。",
                    self._prompt(state),
                )
                if text and text.strip():
                    return text.strip()
            except LLMError as exc:
                state.warnings.append(f"LLM 综述失败，已回退到模板综述：{exc}")
        return self._template_narrative(state)

    def _prompt(self, state: LiteratureState) -> str:
        lines = [f"检索主题：{state.query}", f"候选论文数：{len(state.candidates)}", ""]
        for card in state.cards:
            lines.append(
                f"[arXiv:{card.arxiv_id}] {card.title}\n"
                f"问题：{card.problem}\n方法：{card.method}\n"
                f"发现：{card.findings}\n局限：{card.limitations}\n"
            )
        if state.gaps:
            lines.append("已识别的空白：")
            lines.extend(f"- {gap}" for gap in state.gaps)
        return "\n".join(lines)

    @staticmethod
    def _template_narrative(state: LiteratureState) -> str:
        themes = "；".join(
            f"{theme.label}（{len(theme.paper_ids)} 篇）" for theme in state.themes
        ) or "未形成明显主题"
        lines = [
            "### 核心脉络",
            "",
            f"围绕「{state.query}」检索并去重后得到 {len(state.candidates)} 篇候选论文，"
            f"按相关性与综合得分筛选出 {len(state.shortlist)} 篇。"
            f"这些工作可归纳为：{themes}。",
            "",
            "### 阅读建议",
            "",
            "建议先读综述/基准类工作建立框架，再按主题阅读代表论文；"
            "具体顺序见下方阅读清单。",
        ]
        if state.gaps:
            lines += ["", "### 研究空白", ""]
            lines.extend(f"- {gap}" for gap in state.gaps)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. 审稿与引用校验
# ---------------------------------------------------------------------------


class CriticAgent(Agent):
    name = "critic"

    def run(self, state: LiteratureState) -> None:
        valid_ids = {
            re.sub(r"v\d+$", "", paper.arxiv_id) for paper in state.shortlist
        }
        invalid = 0

        def replace(match: re.Match[str]) -> str:
            nonlocal invalid
            cited = re.sub(r"v\d+$", "", match.group(1).strip())
            if cited in valid_ids:
                return match.group(0)
            invalid += 1
            return "[citation removed]"

        cleaned = CITATION_RE.sub(replace, state.narrative)
        if invalid:
            state.narrative = cleaned
            state.warnings.append(
                f"综述中有 {invalid} 处引用不在本次论文列表内，已移除。"
            )

        missing_cards = len(state.shortlist) - len(state.cards)
        if missing_cards:
            state.warnings.append(f"{missing_cards} 篇论文缺少阅读卡片。")

        covered = {paper_id for theme in state.themes for paper_id in theme.paper_ids}
        uncovered = [paper.arxiv_id for paper in state.shortlist if paper.arxiv_id not in covered]
        if uncovered:
            state.warnings.append(f"{len(uncovered)} 篇论文未归入任何主题。")

        empty_fields = sum(
            1
            for card in state.cards
            if not card.problem or not card.method or not card.findings
        )
        if empty_fields:
            state.warnings.append(f"{empty_fields} 张阅读卡片存在空字段。")

        confidence = [
            state.metrics[paper.arxiv_id].confidence
            for paper in state.shortlist
            if paper.arxiv_id in state.metrics
        ]
        mean_confidence = sum(confidence) / len(confidence) if confidence else 0.0
        if len(state.shortlist) < 5:
            state.warnings.append(
                f"样本仅 {len(state.shortlist)} 篇，主题结论的稳定性有限。"
            )

        state.stats["critic"] = {
            "invalid_citations": invalid,
            "mean_confidence": round(mean_confidence, 4),
            "uncovered_papers": len(uncovered),
            "empty_cards": empty_fields,
        }
