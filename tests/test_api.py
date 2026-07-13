"""
API 端点测试 — 使用 FastAPI TestClient

覆盖: health, agents, ingest, clean, scrape, compliance, storage, drap
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from src.api.app import app
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "agents" in data
        assert "version" in data


class TestAgentsEndpoint:
    def test_list_agents(self, client):
        resp = client.get("/api/agents")
        assert resp.status_code == 200
        data = resp.json()
        assert "agents" in data
        assert isinstance(data["agents"], list)
        assert len(data["agents"]) >= 8


class TestIngestEndpoint:
    def test_ingest_missing_path(self, client):
        resp = client.post("/api/ingest", json={
            "source_path": "",
            "source_type": "csv",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False

    def test_ingest_path_traversal(self, client):
        resp = client.post("/api/ingest", json={
            "source_path": "../../etc/passwd",
            "source_type": "csv",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False


class TestScrapeEndpoint:
    def test_scrape_internal_url_blocked(self, client):
        resp = client.post("/api/scrape", json={
            "url": "http://169.254.169.254/latest/meta-data/",
            "mode": "static",
        })
        assert resp.status_code == 400

    def test_scrape_localhost_blocked(self, client):
        resp = client.post("/api/scrape", json={
            "url": "http://127.0.0.1:6379/",
            "mode": "static",
        })
        assert resp.status_code == 400

    def test_scrape_invalid_scheme(self, client):
        resp = client.post("/api/scrape", json={
            "url": "ftp://example.com/data",
            "mode": "static",
        })
        assert resp.status_code == 400


class TestComplianceEndpoint:
    def test_compliance_check(self, client):
        resp = client.post("/api/compliance/check", json={
            "stage": "数据采集",
            "operation": "",
            "data_desc": "test",
        })
        assert resp.status_code == 200

    def test_compliance_unknown_stage(self, client):
        resp = client.post("/api/compliance/check", json={
            "stage": "未知阶段",
            "operation": "",
            "data_desc": "",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["passed"] is True


class TestStorageEndpoint:
    def test_storage_status(self, client):
        resp = client.get("/api/storage/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "bronze" in data
        assert "silver" in data
        assert "gold" in data