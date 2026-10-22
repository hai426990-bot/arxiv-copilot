# arXiv → Copilot 自动摘要与量化分析

一键抓取 arXiv 最新论文 → 上传至 Microsoft Copilot 生成中文摘要 → 计算多维量化指标并输出排序报告。

## 功能

- 按关键词抓取 arXiv 最新论文，自动下载 PDF，带指数退避防限流
- 通过已登录的 Microsoft Copilot 逐篇生成中文摘要（Markdown）
- 对论文做文献计量与内容量化分析，输出 Markdown / JSON / CSV 报告
- 外部指标（引用、作者 h-index）来自 Semantic Scholar，带本地缓存与离线降级
- 缺失指标按 0 计（无证据不加分），不会因缺项获得虚高排名，并给出置信度

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
├── tests/                  # 指标与报告单元测试
├── auth_state.json         # 登录会话（自动生成）
├── history.json            # 已处理论文记录（自动生成）
├── papers/                 # PDF
├── summaries/              # Markdown 摘要
└── analysis/               # 量化报告、语料与缓存
    ├── quant_report_*.md
    ├── quant_report_*.json
    ├── quant_report_*.csv
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
