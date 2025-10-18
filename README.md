```markdown
# arXiv → Copilot 自动摘要工具

一键抓取 arXiv 最新论文 → 上传至 Microsoft Copilot → 生成中文摘要并保存为 Markdown。

---

## 🚀 快速开始

1. 安装依赖  
   ```bash
   pip install arxiv playwright pyperclip
   playwright install chromium
   ```

2. 首次登录（仅一次）  
   
```bash
   python main.py --init-auth
   ```

   浏览器将自动打开 [Microsoft Copilot](https://copilot.microsoft.com)，手动完成登录后回到终端按回车。

3. 日常使用  
   
```bash
   python main.py
   ```

---

⚙️ 自定义配置

在脚本头部修改变量即可：

变量    说明    默认    
`SEARCH_QUERY`    arXiv 关键词    `"causal"`    
`MAX_RESULTS`    每次抓多少篇    `2`    
`PROMPT_TEMPLATE`    发给 Copilot 的提示    见脚本    
`PAPERS_DIR`    PDF 保存目录    `"papers"`    
`SUMMARIES_DIR`    摘要保存目录    `"summaries"`    

---

📁 目录结构

```
.
├── main.py
├── auth_state.json    # 登录会话（自动生成）
├── history.json       # 已处理记录（自动生成）
├── papers/            # PDF
└── summaries/         # Markdown 摘要
```

---

🛠️ 常见问题

问题    解决    
TimeoutError    增大脚本中的 `max_wait`    
登录失效    删除 `auth_state.json` 后重新 `--init-auth`    
想重新抓取旧论文    删除 `history.json`    

---

## 🌐 GitHub Pages 展示

1. 打开 GitHub 仓库的 **Settings → Pages** 页面。
2. 在 **Source** 下拉框中选择 `main` 分支，并将目录设置为 `docs/`。
3. 点击 **Save**，几分钟后即可通过 `https://<用户名>.github.io/<仓库名>/` 访问静态站点。

如需自定义域名或主题，可在同一页面完成进一步设置。

---

Enjoy 🎉

```
