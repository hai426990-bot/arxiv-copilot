#!/usr/bin/env python3
"""文献 Agent 命令行入口。

示例：
    python literature_agents.py --query "causal inference" --max-results 30 --top-k 10
    python literature_agents.py --query "graph neural networks" --offline
    python literature_agents.py --query "causal inference" --use-llm
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from litagents import LiteratureAgentSystem
from quant_analysis import ANALYSIS_DIR, parse_weight_spec


def _print_summary(state, paths: dict[str, str], top: int = 8) -> None:
    print("\n" + "=" * 80)
    print(
        f"文献 Agent：候选 {len(state.candidates)} 篇 → 筛选 {len(state.shortlist)} 篇 | "
        f"主题 {len(state.themes)} 个 | 平均置信度 "
        f"{state.stats.get('critic', {}).get('mean_confidence', 0):.2f}"
    )
    print("=" * 80)

    if state.themes:
        print("主题：")
        for index, theme in enumerate(state.themes, 1):
            print(
                f"  {index}. {theme.label}  "
                f"({len(theme.paper_ids)} 篇, 均分 {theme.mean_score:.3f})"
            )

    print("\n阅读顺序：")
    for index, paper_id in enumerate(state.reading_order[:top], 1):
        paper = state.paper(paper_id)
        metric = state.score(paper_id)
        theme = state.theme_of(paper_id)
        title = paper.title if len(paper.title) <= 54 else paper.title[:53] + "…"
        print(
            f"  {index}. {title}\n"
            f"     {paper_id} | 综合 {metric.composite:.3f} | {theme.label if theme else '—'}"
        )

    print("-" * 80)
    print(f"Markdown: {paths['markdown']}")
    print(f"JSON    : {paths['json']}")
    print(f"CSV     : {paths['csv']}\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="文献 Agent：检索、筛选、阅读与综述")
    parser.add_argument("--query", default="causal inference", help="研究主题")
    parser.add_argument("--max-results", type=int, default=30, help="候选论文上限")
    parser.add_argument("--top-k", type=int, default=10, help="最终精读论文数")
    parser.add_argument("--offline", action="store_true", help="不查询 Semantic Scholar")
    parser.add_argument("--no-expand", action="store_true", help="不扩展 survey/benchmark 检索式")
    parser.add_argument("--no-corpus", action="store_true", help="不使用/更新历史语料")
    parser.add_argument("--use-llm", action="store_true", help="使用 OpenAI 兼容 LLM 阅读与综述")
    parser.add_argument("--out", default=ANALYSIS_DIR, help="报告输出目录")
    parser.add_argument("--weights", help="覆盖量化权重，如 relevance=0.4,impact=0.2")
    parser.add_argument("--min-relevance", type=float, default=0.02, help="相关性筛选阈值")
    parser.add_argument("--cluster-threshold", type=float, default=0.12, help="主题聚类相似度阈值")
    parser.add_argument("--delay", type=float, default=3.0, help="arXiv 子查询间隔（秒）")
    parser.add_argument("--top", type=int, default=8, help="终端展示的阅读顺序条数")
    args = parser.parse_args(argv)

    try:
        weights = parse_weight_spec(args.weights)
        system = LiteratureAgentSystem(
            max_results=args.max_results,
            top_k=args.top_k,
            offline=args.offline,
            expand=not args.no_expand,
            use_llm=args.use_llm,
            weights=weights,
            min_relevance=args.min_relevance,
            cluster_threshold=args.cluster_threshold,
            out_dir=args.out,
            use_corpus=not args.no_corpus,
            delay=args.delay,
        )
        state, paths = system.run(args.query)
    except (ValueError, RuntimeError) as exc:
        print(f"[❌] {exc}", file=sys.stderr)
        return 2

    for warning in state.warnings:
        print(f"[⚠] {warning}")
    _print_summary(state, paths, top=args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
