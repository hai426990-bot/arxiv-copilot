"""文献 Agent 编排器与报告输出。"""

from __future__ import annotations

import csv
import json
import os
import re
import time
from datetime import datetime
from typing import Callable

from quant_analysis import (
    ANALYSIS_DIR,
    PaperRecord,
    fetch_arxiv_records,
    update_corpus,
)

from .agents import (
    Agent,
    CriticAgent,
    EnrichmentAgent,
    ReadingAgent,
    ScreeningAgent,
    SearchAgent,
    SynthesisAgent,
)
from .llm import LLMClient
from .models import AgentEvent, LiteratureState

SUMMARY_LABELS = {
    "search": "检索",
    "enrichment": "富化",
    "screening": "筛选",
    "reading": "阅读",
    "synthesis": "综合",
    "critic": "校验",
}


class LiteratureAgentSystem:
    def __init__(
        self,
        max_results: int = 30,
        top_k: int = 10,
        offline: bool = False,
        expand: bool = True,
        use_llm: bool = False,
        llm: LLMClient | None = None,
        weights: dict[str, float] | None = None,
        min_relevance: float = 0.02,
        cluster_threshold: float = 0.12,
        out_dir: str = ANALYSIS_DIR,
        use_corpus: bool = True,
        search_fn: Callable[[str, int], list[PaperRecord]] = fetch_arxiv_records,
        delay: float = 3.0,
    ) -> None:
        self.out_dir = out_dir
        self.use_corpus = use_corpus
        self.corpus_path = os.path.join(out_dir, "corpus.json")
        self.cache_path = os.path.join(out_dir, "semantic_scholar_cache.json")
        self.use_llm = use_llm
        self.llm = llm if llm is not None else (LLMClient.from_env() if use_llm else None)
        llm_enabled = bool(use_llm and self.llm)

        self.agents: list[Agent] = [
            SearchAgent(
                max_results=max_results,
                expand=expand,
                delay=delay,
                search_fn=search_fn,
            ),
            EnrichmentAgent(offline=offline, cache_path=self.cache_path),
            ScreeningAgent(
                top_k=top_k,
                min_relevance=min_relevance,
                weights=weights,
                corpus_path=self.corpus_path,
                use_corpus=use_corpus,
            ),
            ReadingAgent(llm=self.llm, use_llm=llm_enabled),
            SynthesisAgent(
                cluster_threshold=cluster_threshold,
                llm=self.llm,
                use_llm=llm_enabled,
            ),
            CriticAgent(),
        ]

    def run(self, query: str) -> tuple[LiteratureState, dict[str, str]]:
        state = LiteratureState(
            query=query,
            max_results=self.agents[0].max_results,
            top_k=self.agents[2].top_k,
        )
        if self.use_llm and not self.llm:
            state.warnings.append(
                "已请求 LLM 模式，但未配置 LITAGENTS_API_KEY/OPENAI_API_KEY，"
                "本次使用摘要抽取与模板综述。"
            )

        started = time.perf_counter()
        for agent in self.agents:
            agent_started = time.perf_counter()
            try:
                agent.run(state)
            except Exception as exc:
                elapsed = int((time.perf_counter() - agent_started) * 1000)
                state.trace.append(
                    AgentEvent(
                        agent=agent.name,
                        action="failed",
                        detail=str(exc),
                        elapsed_ms=elapsed,
                    )
                )
                raise
            elapsed = int((time.perf_counter() - agent_started) * 1000)
            state.trace.append(
                AgentEvent(
                    agent=agent.name,
                    action="done",
                    detail=self._detail(agent.name, state),
                    elapsed_ms=elapsed,
                )
            )

        state.stats["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
        state.warnings = list(dict.fromkeys(state.warnings))
        if self.use_corpus and state.shortlist:
            update_corpus(state.shortlist, self.corpus_path)
        paths = write_reports(state, out_dir=self.out_dir)
        return state, paths

    @staticmethod
    def _detail(name: str, state: LiteratureState) -> str:
        if name == "search":
            return f"{len(state.candidates)} 篇候选，{len(state.subqueries)} 个检索式"
        if name == "enrichment":
            applied = state.stats.get("enrichment", {}).get("applied", 0)
            return f"{applied} 篇获得外部指标"
        if name == "screening":
            return f"筛出 {len(state.shortlist)} 篇"
        if name == "reading":
            return f"生成 {len(state.cards)} 张阅读卡片"
        if name == "synthesis":
            return f"归纳 {len(state.themes)} 个主题"
        if name == "critic":
            invalid = state.stats.get("critic", {}).get("invalid_citations", 0)
            return f"校验完成，移除 {invalid} 处无效引用"
        return ""


# ---------------------------------------------------------------------------
# 报告渲染
# ---------------------------------------------------------------------------


def _fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _md(text: str) -> str:
    return (text or "").replace("|", "\\|").replace("\n", " ")


def _truncate(text: str, limit: int = 220) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _link(paper: PaperRecord) -> str:
    return f"[{_md(paper.title)}](https://arxiv.org/abs/{paper.arxiv_id})"


def _mean_confidence(state: LiteratureState) -> float:
    values = [
        state.metrics[paper.arxiv_id].confidence
        for paper in state.shortlist
        if paper.arxiv_id in state.metrics
    ]
    return sum(values) / len(values) if values else 0.0


def render_markdown(state: LiteratureState) -> str:
    stats = state.stats
    search = stats.get("search", {})
    reading = stats.get("reading", {})
    lines: list[str] = []

    lines.append("# 文献 Agent 分析报告")
    lines.append("")
    lines.append(f"- 生成时间：{state.generated_at}")
    lines.append(f"- 检索主题：`{state.query}`")
    unique = search.get("unique_results", len(state.candidates))
    candidate_line = f"- 候选论文：{len(state.candidates)} 篇"
    if unique and unique > len(state.candidates):
        candidate_line += f"（去重后 {unique} 篇，按上限截断）"
    lines.append(candidate_line)
    lines.append(
        f"- 筛选结果：{len(state.shortlist)} 篇（目标 {state.top_k} 篇，"
        f"相关性阈值 {state.min_relevance:.2f}）"
    )
    lines.append(f"- 主题数量：{len(state.themes)}")
    lines.append(f"- 阅读卡片：{len(state.cards)} 张（LLM 生成 {reading.get('llm_cards', 0)} 张）")
    lines.append(f"- 平均置信度：{_mean_confidence(state):.3f}")
    lines.append(f"- 端到端耗时：{stats.get('elapsed_ms', 0) / 1000:.1f} 秒")
    lines.append("")

    if state.subqueries:
        lines.append("**检索式**")
        lines.append("")
        for subquery in state.subqueries:
            lines.append(f"- `{subquery}`")
        lines.append("")

    if state.warnings:
        lines.append("> **质量提示**")
        for warning in state.warnings:
            lines.append(f"> - {warning}")
        lines.append("")

    lines.append("## 综述")
    lines.append("")
    lines.append(state.narrative.strip() or "未生成综述。")
    lines.append("")

    lines.append("## 主题图谱")
    lines.append("")
    for number, theme in enumerate(state.themes, 1):
        lines.append(
            f"### {number}. {theme.label}（{len(theme.paper_ids)} 篇，"
            f"平均分 {theme.mean_score:.3f}）"
        )
        lines.append("")
        if theme.keywords:
            lines.append(f"- 关键词：{', '.join(theme.keywords)}")
        lines.append(f"- {theme.summary}")
        lines.append("")
        for paper_id in theme.paper_ids:
            paper = state.paper(paper_id)
            metric = state.score(paper_id)
            if paper:
                lines.append(
                    f"  - {_link(paper)}（综合 {metric.composite:.3f}）"
                    if metric
                    else f"  - {_link(paper)}"
                )
        lines.append("")

    lines.append("## 阅读顺序")
    lines.append("")
    lines.append("| # | 论文 | 主题 | 综合 | 相关性 | 建议 |")
    lines.append("| ---: | --- | --- | ---: | ---: | --- |")
    for number, paper_id in enumerate(state.reading_order, 1):
        paper = state.paper(paper_id)
        metric = state.score(paper_id)
        theme = state.theme_of(paper_id)
        if not paper:
            continue
        reason = (
            "综述/基准类，先建立框架"
            if re.search(r"\b(?:survey|review|overview|tutorial|benchmark)\b", paper.title, re.I)
            else "综合得分优先"
        )
        lines.append(
            f"| {number} | {_link(paper)} | {_md(theme.label) if theme else '—'} | "
            f"{_fmt(metric.composite if metric else None)} | "
            f"{_fmt(metric.relevance if metric else None)} | {reason} |"
        )
    lines.append("")

    if state.gaps:
        lines.append("## 研究空白与风险")
        lines.append("")
        for gap in state.gaps:
            lines.append(f"- {gap}")
        lines.append("")

    lines.append("## 论文卡片")
    lines.append("")
    for number, card in enumerate(state.cards, 1):
        metric = state.score(card.arxiv_id)
        theme = state.theme_of(card.arxiv_id)
        lines.append(f"### {number}. {card.title}")
        lines.append("")
        lines.append(f"- arXiv：`{card.arxiv_id}`")
        if card.authors:
            authors = "，".join(card.authors[:8]) + (" 等" if len(card.authors) > 8 else "")
            lines.append(f"- 作者：{authors}")
        if card.published:
            lines.append(f"- 发表：{card.published[:10]}")
        if card.categories:
            lines.append(f"- 分类：{', '.join(card.categories)}")
        if theme:
            lines.append(f"- 主题：{theme.label}")
        if metric:
            lines.append(
                f"- 量化：综合 {metric.composite:.3f}，相关性 {metric.relevance:.3f}，"
                f"置信度 {metric.confidence:.2f}"
            )
        lines.append(f"- **问题**：{card.problem}")
        lines.append(f"- **方法**：{card.method}")
        lines.append(f"- **发现**：{card.findings}")
        lines.append(f"- **局限**：{card.limitations}")
        if card.keywords:
            lines.append(f"- **关键词**：{', '.join(card.keywords)}")
        if card.evidence:
            lines.append(f"- **原文依据**：{card.evidence}")
        lines.append("")

    lines.append("## Agent 执行轨迹")
    lines.append("")
    lines.append("| Agent | 动作 | 耗时 | 说明 |")
    lines.append("| --- | --- | ---: | --- |")
    for event in state.trace:
        lines.append(
            f"| {SUMMARY_LABELS.get(event.agent, event.agent)} | {event.action} | "
            f"{event.elapsed_ms} ms | {_md(event.detail)} |"
        )
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "说明：筛选与排序复用项目的量化指标；阅读卡片默认从摘要确定性抽取，"
        "配置 LLM 后可获得更强的结构化阅读；综述中的引用会经过 Critic Agent 校验。"
    )
    lines.append("")
    return "\n".join(lines)


def write_reports(
    state: LiteratureState,
    out_dir: str = ANALYSIS_DIR,
    timestamp: datetime | None = None,
) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    stamp = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    base = os.path.join(out_dir, f"litagents_report_{stamp}")

    json_path = f"{base}.json"
    md_path = f"{base}.md"
    csv_path = f"{base}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(state))

    fields = [
        "rank", "arxiv_id", "title", "theme", "composite", "relevance",
        "confidence", "published", "categories", "keywords",
        "problem", "method", "findings", "limitations",
    ]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rank, paper_id in enumerate(state.reading_order, 1):
            paper = state.paper(paper_id)
            card = state.card(paper_id)
            metric = state.score(paper_id)
            theme = state.theme_of(paper_id)
            if not paper:
                continue
            writer.writerow(
                {
                    "rank": rank,
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "theme": theme.label if theme else "",
                    "composite": metric.composite if metric else "",
                    "relevance": metric.relevance if metric else "",
                    "confidence": metric.confidence if metric else "",
                    "published": paper.published,
                    "categories": "; ".join(paper.categories),
                    "keywords": "; ".join(card.keywords) if card else "",
                    "problem": card.problem if card else "",
                    "method": card.method if card else "",
                    "findings": card.findings if card else "",
                    "limitations": card.limitations if card else "",
                }
            )

    return {"json": json_path, "markdown": md_path, "csv": csv_path}
