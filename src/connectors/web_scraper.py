"""
Web 数据采集连接器

支持三种采集模式:
  1. static  — HTTP GET + HTML 解析 (requests + BeautifulSoup)
  2. dynamic — 浏览器渲染 (Playwright, 适用于 JS 渲染页面)
  3. api     — REST API 直接调用 (JSON 数据源)

适用平台: 拼多多/淘宝(API模式) / 头条/小红书(动态模式) / 通用网页(静态模式)
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import re
import time
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from src.connectors.base import BaseConnector, ConnectorFactory, QueryResult, TableInfo

logger = logging.getLogger(__name__)

# 常见反爬 User-Agent
DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)


class WebScraperConnector(BaseConnector):
    """
    Web 数据采集连接器

    配置:
      - mode: "static" | "dynamic" | "api"
      - urls: 目标 URL 列表
      - selector: CSS/XPath 选择器 (static 模式)
      - headers: 自定义 HTTP 头
      - extract_rules: 字段提取规则 [{name, selector, attr}]
      - pagination: 分页配置 {type: "url_pattern"|"scroll", param: "page", start, end}
    """

    connector_type = "web"
    supports_read = True
    supports_write = False

    # 平台域名 → autocli 命令映射
    AUTOCLI_PLATFORMS = {
        "toutiao.com": "toutiao",
        "今日头条": "toutiao",
        "ixigua.com": "toutiao",
        "xiaohongshu.com": "xiaohongshu",
        "小红书": "xiaohongshu",
        "bilibili.com": "bilibili",
        "zhihu.com": "zhihu",
        "weibo.com": "weibo",
        "douyin.com": "douyin",
        "xueqiu.com": "xueqiu",
        "douban.com": "douban",
    }

    @staticmethod
    def _detect_platform(url: str) -> Optional[str]:
        """检测 URL 对应的 autocli 平台"""

        domain = urlparse(url).netloc.lower()
        for key, platform in WebScraperConnector.AUTOCLI_PLATFORMS.items():
            if key in domain:
                return platform
        return None

    def __init__(self, name: str, config: Optional[dict[str, Any]] = None):
        super().__init__(name, config)
        cfg = config or {}
        self.mode = cfg.get("mode", "static")
        self.urls = cfg.get("urls", [])
        self.selector = cfg.get("selector", "body")
        self.headers = cfg.get("headers", {"User-Agent": DEFAULT_UA})
        self.extract_rules = cfg.get("extract_rules", [])
        self.pagination = cfg.get("pagination", {})
        self.timeout = cfg.get("timeout", 30)
        self._session = None

    async def connect(self) -> None:
        if self.mode == "static":

            self._session = httpx.AsyncClient(
                headers=self.headers,
                timeout=self.timeout,
                follow_redirects=True,
            )
        elif self.mode == "dynamic":
            # Playwright 延迟导入
            pass
        self._connected = True
        logger.info(f"[{self.name}] Web scraper connected (mode={self.mode})")

    async def close(self) -> None:
        if self._session:
            await self._session.aclose()
            self._session = None
        self._connected = False

    # ── 核心采集方法 ──

    async def scrape(self, urls: Optional[list[str]] = None) -> dict[str, Any]:
        """
        执行采集

        Returns:
            {
                "total_pages": N,
                "total_rows": M,
                "columns": [...],
                "rows": [[...]],
                "metadata": {...}
            }
        """
        targets = urls or self.urls
        if not targets:
            return {"error": "No URLs configured", "total_rows": 0}

        if self.mode == "static":
            return await self._scrape_static(targets)
        elif self.mode == "dynamic":
            return await self._scrape_dynamic(targets)
        elif self.mode == "api":
            return await self._scrape_api(targets)
        elif self.mode == "autocli":
            return await self._scrape_autocli(targets)
        else:
            return {"error": f"Unknown mode: {self.mode}"}

    # ── 静态采集 ──

    async def _scrape_static(self, urls: list[str]) -> dict[str, Any]:
        """HTTP GET + BeautifulSoup 解析"""

        all_rows = []
        columns = []
        pages_ok = 0
        pages_fail = 0

        # 自动生成分页 URL
        all_urls = self._generate_paginated_urls(urls)

        for url in all_urls:
            try:
                resp = await self._session.get(url)  # type: ignore[union-attr]
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")

                if self.extract_rules:
                    # 结构化提取
                    rows = self._extract_structured(soup, self.extract_rules)
                    if rows and not columns:
                        columns = [r["name"] for r in self.extract_rules]
                    all_rows.extend(rows)
                else:
                    # 纯文本提取
                    text = soup.get_text(separator="\n", strip=True)
                    lines = [l.strip() for l in text.split("\n") if l.strip()]
                    all_rows.extend([[l] for l in lines])
                    columns = columns or ["content"]

                pages_ok += 1
            except Exception as e:
                logger.warning(f"[WebScraper] Failed {url}: {e}")
                pages_fail += 1

        return {
            "total_pages": pages_ok + pages_fail,
            "pages_ok": pages_ok,
            "pages_fail": pages_fail,
            "total_rows": len(all_rows),
            "columns": columns,
            "rows": all_rows,
            "metadata": {"mode": "static", "extract_rules": len(self.extract_rules)},
        }

    # ── 动态采集 ──

    async def _scrape_dynamic(self, urls: list[str]) -> dict[str, Any]:
        """Playwright 浏览器渲染采集"""
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return {
                "error": "playwright not installed. Run: pip install playwright && playwright install chromium",
                "total_rows": 0,
            }

        all_rows = []
        pages_ok = 0

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            for url in urls:
                try:
                    await page.goto(url, wait_until="networkidle", timeout=self.timeout * 1000)

                    # 等待内容加载
                    await page.wait_for_timeout(2000)

                    # 滚动加载更多（小红书/头条等懒加载页面）
                    if self.pagination.get("type") == "scroll":
                        await self._auto_scroll(page)

                    content = await page.content()

                    soup = BeautifulSoup(content, "html.parser")

                    if self.extract_rules:
                        rows = self._extract_structured(soup, self.extract_rules)
                        all_rows.extend(rows)
                    else:
                        text = soup.get_text(separator="\n", strip=True)
                        all_rows.extend([[l.strip()] for l in text.split("\n") if l.strip()])

                    pages_ok += 1
                except Exception as e:
                    logger.warning(f"[WebScraper:dynamic] Failed {url}: {e}")

            await browser.close()

        columns = [r["name"] for r in self.extract_rules] if self.extract_rules else ["content"]
        return {
            "total_pages": len(urls),
            "pages_ok": pages_ok,
            "total_rows": len(all_rows),
            "columns": columns,
            "rows": all_rows,
            "metadata": {"mode": "dynamic"},
        }

    async def _auto_scroll(self, page, max_scrolls: int = 20):
        """自动滚动加载更多内容"""
        for _ in range(max_scrolls):
            prev_height = await page.evaluate("document.body.scrollHeight")
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(1500)
            new_height = await page.evaluate("document.body.scrollHeight")
            if new_height == prev_height:
                break

    # ── API 采集 ──

    async def _scrape_api(self, urls: list[str]) -> dict[str, Any]:
        """直接调用 REST API"""
        all_rows = []
        pages_ok = 0

        for url in urls:
            try:
                resp = await self._session.get(url)  # type: ignore[union-attr]
                resp.raise_for_status()
                data = resp.json()

                # 扁平化 JSON 为行
                if isinstance(data, list):
                    for item in data:
                        all_rows.append(list(item.values()) if isinstance(item, dict) else [item])
                elif isinstance(data, dict):
                    # 尝试找到数据数组
                    for key in ("data", "items", "results", "list", "records"):
                        if key in data and isinstance(data[key], list):
                            for item in data[key]:
                                all_rows.append(
                                    list(item.values()) if isinstance(item, dict) else [item]
                                )
                            break
                    else:
                        all_rows.append(list(data.values()))

                pages_ok += 1
            except Exception as e:
                logger.warning(f"[WebScraper:api] Failed {url}: {e}")

        # 推断列名
        columns = []
        if all_rows and isinstance(all_rows[0], list):
            columns = [f"field_{i}" for i in range(len(all_rows[0]))]

        return {
            "total_pages": len(urls),
            "pages_ok": pages_ok,
            "total_rows": len(all_rows),
            "columns": columns,
            "rows": all_rows,
            "metadata": {"mode": "api"},
        }

    # ── 辅助方法 ──

    def _extract_structured(self, soup, rules: list[dict]) -> list[list]:
        """按 CSS 选择器规则提取结构化数据"""
        # 策略1: 容器模式 — 第一个规则有 container 字段
        first_rule = rules[0]
        container_sel = first_rule.get("container", "")

        if container_sel:
            containers = soup.select(container_sel)
            rows = []
            for container in containers:
                row = []
                for rule in rules:
                    sel = rule.get("selector", "")
                    attr = rule.get("attr", "text")
                    elements = container.select(sel) if sel else [container]
                    row.append(self._extract_value(elements, attr))
                if any(row):
                    rows.append(row)
            return rows

        # 策略2: 列对齐模式 — 每个规则独立提取，按列对齐
        columns_data = []
        max_len = 0
        for rule in rules:
            sel = rule.get("selector", "body")
            attr = rule.get("attr", "text")
            elements = soup.select(sel)
            values = [self._extract_value([el], attr) for el in elements]
            columns_data.append(values)
            max_len = max(max_len, len(values))

        # 按行对齐
        rows = []
        for i in range(max_len):
            row = [col[i] if i < len(col) else "" for col in columns_data]
            if any(row):
                rows.append(row)
        return rows

    @staticmethod
    def _extract_value(elements: list, attr: str) -> str:
        if not elements:
            return ""
        el = elements[0]
        if attr == "text":
            return el.get_text(strip=True)
        if attr == "href":
            return el.get("href", "") or ""
        if attr == "src":
            return el.get("src", "") or ""
        return el.get(attr, "") or ""

    def _generate_paginated_urls(self, urls: list[str]) -> list[str]:
        """根据分页配置生成 URL 列表"""
        pag = self.pagination
        if not pag or pag.get("type") != "url_pattern":
            return urls

        param = pag.get("param", "page")
        start = pag.get("start", 1)
        end = pag.get("end", start)
        result = []
        for url in urls:
            for p in range(start, end + 1):
                if "{" + param + "}" in url:
                    result.append(url.replace("{" + param + "}", str(p)))
                else:
                    sep = "&" if "?" in url else "?"
                    result.append(f"{url}{sep}{param}={p}")
        return result or urls

    # ── 必须实现的抽象方法 ──

    async def execute(self, query: str, **params: Any) -> QueryResult:
        raise NotImplementedError("Use scrape() instead")

    async def list_tables(self) -> list[TableInfo]:
        return [TableInfo(name="web_scraped_data")]

    async def get_table_info(self, table_name: str) -> TableInfo:
        return TableInfo(name=table_name)

    # ── AutoCLI 采集（55+ 中文平台）──

    async def _scrape_autocli(self, urls: list[str]) -> dict[str, Any]:
        """
        通过 AutoCLI 采集中文平台内容
        """

        all_rows = []
        pages_ok = 0

        for url in urls:
            if '..' in url or '\x00' in url:
                logger.warning(f"[AutoCLI] Invalid URL skipped: {url}")
                continue
            platform = self._detect_platform(url)
            # 头条分享链接 → 提取纯 article ID
            clean_url = self._clean_toutiao_url(url) if platform == "toutiao" else url

            try:
                if platform:
                    # 平台专用命令: autocli <platform> search <keyword> 等
                    # 对于文章链接，统一用 autocli read
                    cmd = ["autocli", "read", clean_url, "--format", "json"]
                else:
                    cmd = ["autocli", "read", clean_url, "--format", "json"]

                proc = await asyncio.create_subprocess_exec(
                    cmd[0], *cmd[1:],
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(), timeout=25
                    )
                except asyncio.TimeoutError:
                    proc.kill()
                    logger.warning(f"[AutoCLI] Timeout: {url}")
                    continue

                if proc.returncode == 0 and stdout:
                    text = stdout.decode("utf-8", errors="replace").strip()
                    if text:
                        try:
                            data = _json.loads(text)
                            # JSON 格式: {title, content, url, ...}
                            row = [
                                data.get("title", ""),
                                data.get("content", "") or data.get("text", ""),
                                data.get("url", url),
                                data.get("author", ""),
                                data.get("date", ""),
                            ]
                            all_rows.append(row)
                            pages_ok += 1
                        except _json.JSONDecodeError:
                            # 纯文本输出
                            all_rows.append([text, "", url, "", ""])
                            pages_ok += 1
                else:
                    err = stderr.decode("utf-8", errors="replace")[:200] if stderr else ""
                    logger.warning(f"[AutoCLI] Failed ({proc.returncode}): {url} — {err}")

            except FileNotFoundError:
                return {
                    "error": "autocli not installed. Download from: https://github.com/nashsu/AutoCLI",
                    "total_rows": 0,
                }
            except Exception as e:
                logger.warning(f"[AutoCLI] Error: {url} — {e}")

        return {
            "total_pages": len(urls),
            "pages_ok": pages_ok,
            "total_rows": len(all_rows),
            "columns": ["title", "content", "url", "author", "date"],
            "rows": all_rows,
            "metadata": {"mode": "autocli", "engine": "AutoCLI (Mozilla Readability + Chrome login)"},
        }

    @staticmethod
    def _clean_toutiao_url(url: str) -> str:
        """
        清理头条分享链接，提取纯 article ID
        """
        match = re.search(r'/article/(\d+)', url)
        if match:
            return f"https://www.toutiao.com/article/{match.group(1)}/"
        return url


# ── 智能模式选择 ──

def smart_scrape_mode(url: str, prefer: str = "") -> str:
    """
    根据 URL 智能选择最佳采集模式

    优先级: autocli > static > dynamic
    """
    if prefer:
        return prefer

    # 中文平台 → autocli
    platform = WebScraperConnector._detect_platform(url)
    if platform:
        return "autocli"

    # API 特征: 路径含 /api/ 或子域名为 api.

    if "/api/" in url or urlparse(url).netloc.startswith("api.") or url.endswith(".json"):
        return "api"

    # 默认静态
    return "static"


# 注册
ConnectorFactory.register("web", WebScraperConnector)
