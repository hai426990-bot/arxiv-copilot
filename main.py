import os
import json
import time
import random
import argparse
from pathlib import Path
from datetime import datetime
import arxiv
from playwright.sync_api import Download, sync_playwright, TimeoutError as PWTimeout
import pyperclip

from quant_analysis import ANALYSIS_DIR, PaperRecord, analyze_records

# ===== 配置 =====
PAPERS_DIR = "papers"
SUMMARIES_DIR = "summaries"
HISTORY_FILE = "history.json"
AUTH_STATE_FILE = "auth_state.json"

SEARCH_QUERY = "causal"
MAX_RESULTS = 2
PROMPT_TEMPLATE = "请总结这篇论文的题目，作者机构，贡献与方法，不用公式表达。只需要回答不需要其他内容。"
COPILOT_URL = "https://copilot.microsoft.com"


os.makedirs(PAPERS_DIR, exist_ok=True)
os.makedirs(SUMMARIES_DIR, exist_ok=True)

# ===== 历史记录 =====
def load_history():
    return set(json.load(open(HISTORY_FILE, encoding="utf-8"))) if os.path.exists(HISTORY_FILE) else set()

def save_history(history):
    json.dump(list(history), open(HISTORY_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# ===== 抓取新论文 =====

def fetch_new_papers(history, max_results=MAX_RESULTS,
                     base_delay: float = 1.0,
                     jitter: float = 0.5,
                     max_delay: float = 8.0):
    """
    抓取 arXiv 新论文，带指数退避 + 随机抖动防限流。
    :param history: set / History 对象，记录已处理的论文 ID
    :param max_results: 本次最多抓多少篇
    :param base_delay: 初始间隔（秒）
    :param jitter: 随机抖动范围 ±jitter
    :param max_delay: 最大间隔（秒）
    :return: list[PaperRecord] 新论文及其本地 PDF 路径
    """
    search = arxiv.Search(
        query=SEARCH_QUERY,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    client = arxiv.Client()
    new_files = []

    delay = base_delay
    for result in client.results(search):
        pid = result.get_short_id()
        if pid in history:
            continue

        pdf_path = Path(PAPERS_DIR) / f"{pid}.pdf"
        result.download_pdf(filename=str(pdf_path))
        record = PaperRecord(
            arxiv_id=pid,
            title=result.title.strip(),
            abstract=result.summary.strip(),
            authors=[author.name for author in result.authors],
            categories=list(result.categories),
            primary_category=getattr(result.primary_category, "category", result.primary_category),
            published=result.published.isoformat(),
            updated=result.updated.isoformat(),
            comment=result.comment or "",
            journal_ref=getattr(result, "journal_ref", "") or "",
            doi=getattr(result, "doi", "") or "",
            pdf_path=str(pdf_path),
        )
        new_files.append(record)
        history.add(pid)
        print(f"[⬇] 下载: {pid}")

        # ---- 防限流等待 ----
        time.sleep(delay + random.uniform(-jitter, jitter))
        delay = min(delay * 2, max_delay)  # 指数退避，封顶 max_delay

    return new_files

# ===== 登录确认 =====
def wait_for_login_confirm():
    input("\n💡 请在浏览器中完成登录并进入 Copilot 对话界面，然后回车继续...")

# ===== 初始化登录模式 =====
def init_auth_state():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=False)
        context = browser.new_context()
        page = context.new_page()
        print("[▶] 打开 Copilot，请手动登录...")
        page.goto(COPILOT_URL, wait_until="domcontentloaded", timeout=120000)
        for text in ["接受所有", "同意", "Accept all", "Agree"]:
            try:
                btn = page.get_by_text(text, exact=False)
                if btn and btn.count() > 0:
                    btn.first.click(timeout=2000)
                    print(f"[🍪] 点击了 Cookie 按钮: {text}")
                    break
            except:
                pass
        wait_for_login_confirm()
        context.storage_state(path=AUTH_STATE_FILE)
        print(f"[✅] 登录状态已保存到 {AUTH_STATE_FILE}")
        browser.close()

# ===== 打开已登录环境 =====
def open_copilot_with_auth(p):
    if not os.path.exists(AUTH_STATE_FILE):
        raise FileNotFoundError("未找到登录会话，请先运行 --init-auth 完成一次登录。")
    browser = p.chromium.launch(channel="chrome", headless=False)
    context = browser.new_context(storage_state=AUTH_STATE_FILE)
    page = context.new_page()
    page.goto(COPILOT_URL, wait_until="domcontentloaded", timeout=120000)
    return browser, context, page

# ===== 从容器提取文本 =====
def extract_answer_text(page):
    sel = 'div.space-y-3.mt-3'
    try:
        page.wait_for_selector(sel, timeout=120000)
        texts = page.locator(sel).all_inner_texts()
        return "\n".join(t.strip() for t in texts if t.strip())
    except Exception as e:
        print(f"[⚠] 提取回答失败: {e}")
        return ""

# ===== 摘要生成 =====
def summarize_one_pdf(page, pdf_path):
    print(f"[📂] 上传文件: {Path(pdf_path).name}")
    page.set_input_files('input[type="file"]', pdf_path)
    try:
        page.wait_for_selector('div[class*="file-card"], div:has-text("已上传")', timeout=30000)
        print("[✅] 文件上传完成")
    except PWTimeout:
        print("[⚠] 上传完成提示未检测到，继续")

    page.fill('textarea', PROMPT_TEMPLATE)
    page.keyboard.press("Enter")
    print("[✉] 已发送摘要请求")

    # 等复制按钮出现
    copy_btn_sel = '[data-testid="copy-message-button"]'
    try:
        page.wait_for_selector(copy_btn_sel, timeout=300000)
        print("[📋] 检测到复制按钮，开始监听容器文本...")
    except PWTimeout:
        raise RuntimeError("等待复制按钮超时")

    # 监听变化
    start_time = time.time()
    last_text = ""
    last_change = time.time()
    idle_time = 12
    min_output_time = 15
    max_wait = 480

    while True:
        current_text = extract_answer_text(page)
        if current_text and current_text != last_text:
            last_text = current_text
            last_change = time.time()
        now = time.time()
        if (now - start_time) >= min_output_time and (now - last_change) >= idle_time:
            break
        if (now - start_time) >= max_wait:
            print("[⚠] 达到最大等待时间，结束监听")
            break
        time.sleep(1)

    # 复制按钮兜底（获取包含公式的内容）
    text_final = last_text
    try:
        page.click(copy_btn_sel, timeout=3000)
        time.sleep(0.5)
        copied_text = copied_text = pyperclip.paste()
        if copied_text and copied_text.strip():
            print("[✅] 从复制按钮获取到完整内容（含公式）")
            text_final = copied_text
    except Exception as e:
        print(f"[⚠] 复制按钮获取失败: {e}")

    if not text_final.strip():
        raise RuntimeError("未获取到任何回答内容")

    
    return text_final

# ===== 主流程：每篇单独浏览器 =====
def run_pipeline(quant_enabled=True, enrich=True):
    history = load_history()
    new_files = fetch_new_papers(history)
    if not new_files:
        print("[ℹ] 没有新论文")
        save_history(history)
        return
    for record in new_files:
        fid = record.arxiv_id
        try:
            with sync_playwright() as p:
                browser, context, page = open_copilot_with_auth(p)
                summary = summarize_one_pdf(page, record.pdf_path)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(SUMMARIES_DIR, f"{fid}_{timestamp}.md")
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(summary)
                print(f"[💾] 摘要已保存 (md格式): {save_path}")
                browser.close()
        except Exception as e:
            print(f"[❌] {fid} 处理失败: {e}")
            try:
                shot_path = os.path.join(SUMMARIES_DIR, f"{fid}_error.png")
                page.screenshot(path=shot_path, full_page=True)
                print(f"[📸] 错误截图已保存: {shot_path}")
            except:
                pass
        save_history(history)

    if quant_enabled:
        try:
            print("[📊] 生成量化分析报告...")
            _, paths, warnings = analyze_records(
                new_files,
                SEARCH_QUERY,
                enrich=enrich,
                out_dir=ANALYSIS_DIR,
            )
            for warning in warnings:
                print(f"[⚠] {warning}")
            print(f"[📊] 量化报告: {paths['markdown']}")
        except Exception as e:
            print(f"[⚠] 量化分析失败: {e}")

# ===== 入口 =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="arXiv→Copilot 自动化")
    parser.add_argument("--init-auth", action="store_true", help="初始化登录并保存会话")
    parser.add_argument("--no-quant", action="store_true", help="跳过量化分析")
    parser.add_argument("--offline", action="store_true", help="量化分析不访问 Semantic Scholar")
    args = parser.parse_args()
    if args.init_auth:
        init_auth_state()
    else:
        run_pipeline(quant_enabled=not args.no_quant, enrich=not args.offline)
