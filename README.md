# arXiv → Copilot 自动摘要与量化分析

一键抓取 arXiv 最新论文 → 上传至 Microsoft Copilot 生成中文摘要 → 计算多维量化指标并输出排序报告；另附一条命令可完成检索、筛选、精读与主题综述的文献 Agent。

## 功能

- 按关键词抓取 arXiv 最新论文，自动下载 PDF，带指数退避防限流
- 通过已登录的 Microsoft Copilot 逐篇生成中文摘要（Markdown）
- 对论文做文献计量与内容量化分析，输出 Markdown / JSON / CSV 报告
- 外部指标（引用、作者 h-index）来自 Semantic Scholar，带本地缓存与离线降级
- 缺失指标按 0 计（无证据不加分），不会因缺项获得虚高排名，并给出置信度
- 六个协作式文献 Agent：检索规划、富化、筛选、精读、主题综合、引用与质量校验

## 安装

```bash
pip install -r requirements.txt
playwright install chromium
```

## 首次登录（仅一次）

```bash
python main.py --init-auth
```

浏览器会打开 [Microsoft Copilot](https://copilot.microsoft.com)，手动完成登录后回到终端按回车。

## 日常使用

```bash
python main.py                 # 抓取 → 摘要 → 量化报告
python main.py --offline       # 跳过 Semantic Scholar，只用本地指标
python main.py --no-quant      # 只生成摘要，不做量化分析
```

每次运行后，量化报告位于 `analysis/quant_report_<时间戳>.{md,json,csv}`。
首次抓取新论文时会自动建立 `analysis/corpus.json`，用于计算“相对历史阅读的新颖度”。

## 量化指标

| 指标 | 默认权重 | 计算方式 |
| --- | ---: | --- |
| 相关性 relevance | 0.30 | BM25（标题权重×2，k1=1.5，b=0.75），压缩到 0~1 |
| 影响力 impact | 0.22 | `1-exp(-引用/年/10)`，并按有影响力引用占比修正 |
| 作者 author | 0.15 | 作者 h-index 的 `0.6×最大值 + 0.4×平均值`，再做饱和变换 |
| 新鲜度 freshness | 0.13 | `0.5^(发表天数/180)` |
| 渠道 venue | 0.10 | 期刊/DOI/正式 venue = 1，仅会议接收信息 = 0.6，纯预印本 = 0 |
| 跨学科 interdisciplinarity | 0.05 | `min(1, log2(分类数)/2)` |
| 新颖度 novelty | 0.05 | `1 - 与历史语料的最大余弦相似度` |

综合得分为各指标（均映射到 0~1）的加权平均，权重分母固定。某项指标缺失时按 0 计，不参与加分；这样缺项论文不会因为“只比较优势指标”而排名虚高，`confidence` 等于实际覆盖权重，用来提示结果可信度。

外部指标由 Semantic Scholar 批量接口提供，结果缓存 7 天。设置环境变量 `SEMANTIC_SCHOLAR_API_KEY` 可提高配额。新预印本通常尚无引用，此时报告会提示影响力指标暂不具备区分度。

## 独立运行量化分析

```bash
# 不下载 PDF，只抓元数据并分析
python quant_analysis.py --query "causal inference" --max-results 25

# 离线 / 自定义权重 / 指定输出目录
python quant_analysis.py --query "causal inference" --offline \
    --weights "relevance=0.4,impact=0.2,author=0.1" --out analysis

# 复用已有报告重新计算（例如换权重）
python quant_analysis.py --from-json analysis/quant_report_xxx.json --offline
```

权重可只覆盖部分指标，未指定的沿用默认值，最终会自动归一化。终端会打印前 N 名，完整结果写入报告。

### 报告结构

- **指标与权重**：覆盖率、均值、标准差，可直接看出哪些指标在本批次没有区分度
- **综合排名**：综合分、置信度、各分项得分
- **论文详情**：作者、分类、原始引用指标、批次百分位、摘要
- JSON 保留论文元数据与全部指标，CSV 适合直接导入表格工具

## 文献 Agent（自动综述）

一条命令完成“检索规划 → 检索去重 → 指标富化 → 筛选 → 精读 → 主题综合 → 质量校验”，直接产出可读的文献综述。

```bash
# 默认：arXiv + Semantic Scholar，确定性输出
python literature_agents.py --query "causal inference"

# 离线，只依赖 arXiv 元数据
python literature_agents.py --query "graph neural networks" --offline --no-expand

# 启用 LLM 阅读与综述（需要 OpenAI 兼容接口）
export LITAGENTS_API_KEY="sk-..."
python literature_agents.py --query "causal inference" --use-llm
```

| Agent | 职责 |
| --- | --- |
| 检索规划 SearchAgent | 生成 `all:` / `ti:` / survey / benchmark 子查询，按 arXiv ID 与标题去重 |
| 指标富化 EnrichmentAgent | 批量补充引用、影响力引用、作者 h-index、venue；离线时跳过 |
| 筛选 ScreeningAgent | 复用本项目的量化指标，按相关性阈值过滤后取综合分 Top-K |
| 精读 ReadingAgent | 抽取问题、方法、发现、局限、关键词，生成可追溯的阅读卡片 |
| 综合 SynthesisAgent | TF-IDF 聚类主题、识别研究空白、生成综述式阅读顺序与正文 |
| 审稿 CriticAgent | 清洗不在清单内的引用，检查卡片覆盖、空字段与样本量，写入质量告警 |

LLM 为可选增强，键名兼容 `OPENAI_API_KEY`，还可配置 `LITAGENTS_MODEL`（默认 `gpt-4o-mini`）与 `LITAGENTS_BASE_URL`。任何 LLM 调用失败都会自动回退到确定性的摘要抽取与模板综述，不会中断流程。

报告写入 `analysis/litagents_report_<时间戳>.{md,json,csv}`。Markdown 包含综述、主题图谱、阅读顺序、研究空白与风险、论文卡片、Agent 执行轨迹；JSON 保留完整状态，便于二次处理。

常用参数：`--max-results`（候选上限，默认 30）、`--top-k`（精读数，默认 10）、`--min-relevance`（相关性阈值，默认 0.02）、`--cluster-threshold`（主题聚类阈值，默认 0.12）、`--weights`（与量化分析相同的权重语法）、`--no-corpus`（不读写历史语料）。

## 配置

`main.py` 头部：

| 变量 | 说明 | 默认 |
| --- | --- | --- |
| `SEARCH_QUERY` | arXiv 检索式 | `"causal"` |
| `MAX_RESULTS` | 每次抓取篇数 | `2` |
| `PROMPT_TEMPLATE` | 发给 Copilot 的提示 | 见脚本 |
| `PAPERS_DIR` | PDF 目录 | `"papers"` |
| `SUMMARIES_DIR` | 摘要目录 | `"summaries"` |

## 目录结构

```
.
├── main.py                 # 抓取 + Copilot 摘要 + 量化分析入口
├── quant_analysis.py       # 量化指标、富化与报告生成
├── literature_agents.py    # 文献 Agent 命令行入口
├── litagents/              # 六个 Agent、共享状态、LLM 客户端与编排器
├── tests/                  # 指标、报告与 Agent 单元测试
├── auth_state.json         # 登录会话（自动生成）
├── history.json            # 已处理论文记录（自动生成）
├── papers/                 # PDF
├── summaries/              # Markdown 摘要
└── analysis/               # 量化/综述报告、语料与缓存
    ├── quant_report_*.md
    ├── quant_report_*.json
    ├── quant_report_*.csv
    ├── litagents_report_*.md
    ├── litagents_report_*.json
    ├── litagents_report_*.csv
    ├── corpus.json
    └── semantic_scholar_cache.json
```

## 测试

```bash
pip install pytest
python -m pytest tests -q
```

## 常见问题

| 问题 | 解决 |
| --- | --- |
| Copilot 等待超时 | 增大 `summarize_one_pdf` 中的 `max_wait` |
| 登录失效 | 删除 `auth_state.json` 后重新 `--init-auth` |
| 想重新抓取旧论文 | 删除 `history.json` |
| 引用指标全部为空 | 新论文尚未被 Semantic Scholar 索引，或使用 `--offline`；属正常降级 |
| 想换排序偏好 | 用 `--weights` 调整指标权重，或修改 `DEFAULT_WEIGHTS` |
| 新颖度始终为空 | 历史语料尚未建立，运行过一次分析后即可生效 |
| 文献 Agent 是否必须用 LLM | 不需要。默认使用确定性的摘要抽取与模板综述，`--use-llm` 仅作增强 |
| Agent 检索很慢或被 arXiv 限流 | 增大 `--delay`，减少 `--max-results`，或加 `--no-expand` 减少子查询 |
