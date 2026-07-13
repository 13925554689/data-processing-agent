"""Test Web Scraper Connector + AutoCLI integration."""

import pytest

from src.connectors.web_scraper import (
    WebScraperConnector,
    smart_scrape_mode,
)
from src.connectors.base import ConnectorFactory


class TestWebScraperConnector:
    def test_factory_registered(self):
        conn = ConnectorFactory.create("test", "web", {"mode": "static", "urls": ["http://example.com"]})
        assert isinstance(conn, WebScraperConnector)

    def test_defaults(self):
        conn = WebScraperConnector("test")
        assert conn.mode == "static"

    def test_pagination_url_generation(self):
        conn = WebScraperConnector("test", {
            "mode": "static",
            "urls": ["https://example.com/list?page={page}"],
            "pagination": {"type": "url_pattern", "param": "page", "start": 1, "end": 3},
        })
        urls = conn._generate_paginated_urls(conn.urls)
        assert len(urls) == 3
        assert urls[0] == "https://example.com/list?page=1"

    @pytest.mark.asyncio
    async def test_connect_close(self):
        conn = WebScraperConnector("test", {"mode": "static"})
        await conn.connect()
        assert conn.is_connected
        await conn.close()
        assert not conn.is_connected

    @pytest.mark.asyncio
    async def test_scrape_static_mock(self, httpx_mock):
        httpx_mock.add_response(
            url="https://example.com",
            html="<html><body><h1>Title</h1><p>Content here</p></body></html>",
        )
        conn = WebScraperConnector("test", {"mode": "static", "urls": ["https://example.com"]})
        await conn.connect()
        result = await conn.scrape()
        assert result["total_rows"] > 0
        await conn.close()

    @pytest.mark.asyncio
    async def test_scrape_with_extract_rules(self, httpx_mock):
        httpx_mock.add_response(
            url="https://example.com/products",
            html="""<html><body>
              <h2>Product A</h2><span class="price">99</span>
              <h2>Product B</h2><span class="price">199</span>
            </body></html>""",
        )
        conn = WebScraperConnector("test", {
            "mode": "static",
            "urls": ["https://example.com/products"],
            "extract_rules": [
                {"name": "title", "selector": "h2", "attr": "text"},
                {"name": "price", "selector": ".price", "attr": "text"},
            ],
        })
        await conn.connect()
        result = await conn.scrape()
        assert result["total_rows"] >= 1
        assert "title" in result["columns"]
        await conn.close()


class TestAutoCLIIntegration:
    """AutoCLI 平台检测 + 智能模式选择"""

    def test_detect_toutiao(self):
        assert WebScraperConnector._detect_platform(
            "https://www.toutiao.com/article/123456/"
        ) == "toutiao"

    def test_detect_xiaohongshu(self):
        assert WebScraperConnector._detect_platform(
            "https://www.xiaohongshu.com/explore/abc123"
        ) == "xiaohongshu"

    def test_detect_bilibili(self):
        assert WebScraperConnector._detect_platform(
            "https://www.bilibili.com/video/BV1xx411c7mD"
        ) == "bilibili"

    def test_detect_zhihu(self):
        assert WebScraperConnector._detect_platform(
            "https://www.zhihu.com/question/123456"
        ) == "zhihu"

    def test_detect_weibo(self):
        assert WebScraperConnector._detect_platform(
            "https://weibo.com/1234567890/AbCdEfGhI"
        ) == "weibo"

    def test_detect_unknown(self):
        assert WebScraperConnector._detect_platform(
            "https://example.com/article/1"
        ) is None

    def test_smart_toutiao(self):
        assert smart_scrape_mode("https://www.toutiao.com/article/123/") == "autocli"

    def test_smart_xiaohongshu(self):
        assert smart_scrape_mode("https://www.xiaohongshu.com/explore/abc") == "autocli"

    def test_smart_api(self):
        assert smart_scrape_mode("https://api.example.com/v1/data") == "api"

    def test_smart_default(self):
        assert smart_scrape_mode("https://example.com/page") == "static"

    def test_clean_toutiao_share_url(self):
        dirty = "https://www.toutiao.com/article/7628820165253136902/?app=news_article&timestamp=123&share_token=abc"
        clean = WebScraperConnector._clean_toutiao_url(dirty)
        assert clean == "https://www.toutiao.com/article/7628820165253136902/"

    def test_clean_toutiao_no_match(self):
        url = "https://www.toutiao.com/"
        assert WebScraperConnector._clean_toutiao_url(url) == url


class TestIngestURLDetection:
    """IngestAgent URL 自动识别"""

    def test_url_to_web_type(self):
        from src.agents.ingest_agent import IngestAgent
        agent = IngestAgent()
        assert agent._infer_type("https://www.toutiao.com/article/123/") == "web"
        assert agent._infer_type("http://api.example.com/v1/items") == "web"

    def test_url_name_inference(self):
        from src.agents.ingest_agent import IngestAgent
        agent = IngestAgent()
        name = agent._infer_name("https://www.toutiao.com/article/123456/")
        assert "toutiao" in name.lower()
