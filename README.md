使用说明：arXiv→Copilot 自动化摘要工具

本工具可自动完成以下流程：  
1. 从 arXiv 抓取最新论文（PDF）。  
2. 将 PDF 上传给 Microsoft Copilot，并提示其生成中文摘要。  
3. 将摘要保存为 Markdown 文件到本地。

---

1. 环境准备

- Python ≥ 3.8  
- 系统依赖  
  - 安装 Google Chrome（或已安装 Chrome 的浏览器）。  
  - 安装 Python 依赖：
    
```bash
    pip install -r requirements.txt
    ```

    若尚未生成 `requirements.txt`，可直接安装：
    
```bash
    pip install arxiv playwright pyperclip
    playwright install chromium
    ```

---

2. 首次运行：登录 Microsoft Copilot

Copilot 需要登录才能使用文件上传功能，仅需首次执行一次：

```bash
python main.py --init-auth
```

操作步骤：
1. 脚本会自动打开浏览器并跳转到 [https://copilot.microsoft.com](https://copilot.microsoft.com)。  
2. 手动完成登录（Microsoft 账户）。  
3. 登录成功并看到对话界面后，回到终端按回车。  
4. 登录凭据会被保存到本地 `auth_state.json`，后续运行无需再次登录。

---

3. 日常运行：抓取并生成摘要

```bash
python main.py
```

运行流程：
1. 根据 `SEARCH_QUERY`（默认 `"causal"`）抓取最新 2 篇 arXiv 论文。  
2. 每篇论文启动一次独立浏览器会话，上传 PDF 至 Copilot。  
3. 自动发送提示：
   
> 请总结这篇论文的题目，作者机构，贡献与方法，不用公式表达。只需要回答不需要其他内容。  
4. 等待 Copilot 生成回答，检测到内容不再变化后保存。  
5. 摘要保存为：
   
```
   summaries/
   └── <paper_id>_<时间戳>.md
   ```

---

4. 自定义配置

在脚本顶部可修改以下常量：

常量名	说明	默认值	
SEARCH_QUERY	arXiv 搜索关键词	`"causal"`	
MAX_RESULTS	每次运行的最大新论文数	`2`	
PROMPT_TEMPLATE	发给 Copilot 的提示语	见脚本	
PAPERS_DIR	PDF 下载目录	`"papers"`	
SUMMARIES_DIR	摘要保存目录	`"summaries"`	

---

5. 文件结构

```
.
├── main.py              # 主脚本
├── auth_state.json      # 登录会话（自动生成）
├── history.json         # 已处理论文 ID（自动生成）
├── papers/              # 下载的 PDF
├── summaries/           # 生成的 Markdown 摘要
└── requirements.txt     # Python 依赖（可选）
```

---

6. 常见问题

问题	解决	
`playwright._impl._api_types.TimeoutError`	网络或 Copilot 响应慢，可增大脚本中的 `max_wait` 时间。	
上传失败或找不到元素	确保浏览器已更新到最新版；若 Copilot 页面改版，需调整选择器。	
摘要中出现公式	脚本会优先通过「复制按钮」获取富文本，仍可能包含公式，可手动调整 `PROMPT_TEMPLATE` 再试。	

---

7. 更新与重置

- 重新登录：删除 `auth_state.json` 后再次运行 `--init-auth`。  
- 清空历史：删除 `history.json`，即可重新抓取已处理过的论文。  

---

祝使用愉快！
