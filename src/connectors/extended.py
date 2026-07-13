"""
扩展连接器 — SQLite (stdlib)、API (HTTP)、DRAP 台账 API
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from typing import Any, Optional

import httpx

from src.connectors.base import (
    BaseConnector,
    ColumnInfo,
    ConnectorFactory,
    QueryResult,
    TableInfo,
)
from src.utils.sql import safe_identifier

logger = logging.getLogger(__name__)


# ── SQLite 连接器（使用 Python stdlib sqlite3） ──────────────────────

class SQLiteConnector(BaseConnector):
    """SQLite 数据库连接器（零外部依赖）"""

    connector_type = "sqlite"
    supports_write = True

    def __init__(self, name: str, config: Optional[dict[str, Any]] = None):
        super().__init__(name, config)
        self.db_path = (config or {}).get("path", ":memory:")
        self._con: Optional[sqlite3.Connection] = None

    async def connect(self) -> None:

        if self.db_path != ":memory:" and not os.path.exists(self.db_path):
            raise FileNotFoundError(f"SQLite file not found: {self.db_path}")
        self._con = sqlite3.connect(self.db_path)
        self._con.row_factory = sqlite3.Row
        self._connected = True
        logger.info(f"[{self.name}] Connected to SQLite: {self.db_path}")

    async def close(self) -> None:
        if self._con:
            self._con.close()
            self._con = None
        self._connected = False

    async def execute(self, query: str, **params: Any) -> QueryResult:
        if not self._con:
            raise RuntimeError("Not connected")
        t0 = time.perf_counter()
        cur = self._con.execute(query)
        rows = [list(row) for row in cur.fetchall()]
        cols = [d[0] for d in cur.description] if cur.description else []
        duration = (time.perf_counter() - t0) * 1000
        return QueryResult(
            columns=cols, rows=rows, row_count=len(rows), duration_ms=duration
        )

    async def list_tables(self) -> list[TableInfo]:
        result = await self.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        return [TableInfo(name=r[0]) for r in result.rows]

    async def get_table_info(self, table_name: str) -> TableInfo:
        safe_name = safe_identifier(table_name)
        info_rows = await self.execute(f"PRAGMA table_info({safe_name})")
        columns = [
            ColumnInfo(name=r[1], dtype=r[2], nullable=not r[3])
            for r in info_rows.rows
        ]
        count = await self.execute(f"SELECT COUNT(*) FROM {safe_name}")
        return TableInfo(
            name=table_name, row_count=count.rows[0][0], columns=columns
        )

    async def read_table(
        self, table_name: str, limit: int = 1000, offset: int = 0
    ) -> QueryResult:
        safe_name = safe_identifier(table_name)
        if not isinstance(limit, int) or limit < 0:
            raise ValueError("limit must be a non-negative integer")
        if not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        return await self.execute(
            f"SELECT * FROM {safe_name} LIMIT {limit} OFFSET {offset}"
        )


# ── API 连接器（通用 HTTP） ──────────────────────────────────────────

class APIConnector(BaseConnector):
    """通用 HTTP API 连接器"""

    connector_type = "api"

    def __init__(self, name: str, config: Optional[dict[str, Any]] = None):
        super().__init__(name, config)
        cfg = config or {}
        self.base_url = cfg.get("base_url", "")
        self.headers = cfg.get("headers", {})
        self.timeout = cfg.get("timeout", 30)
        self._client = None

    async def connect(self) -> None:

        self._client = httpx.AsyncClient(
            base_url=self.base_url, headers=self.headers, timeout=self.timeout,
        )
        self._connected = True

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
        self._connected = False

    async def execute(self, query: str, **params: Any) -> QueryResult:
        raise NotImplementedError("Use get/post methods directly")

    async def get(self, path: str, **params: Any) -> dict[str, Any]:
        if not self._client:
            raise RuntimeError("Not connected")
        resp = await self._client.get(path, params=params)
        resp.raise_for_status()
        return resp.json()

    async def post(self, path: str, json_data: dict[str, Any] = None) -> dict[str, Any]:
        if not self._client:
            raise RuntimeError("Not connected")
        resp = await self._client.post(path, json=json_data or {})
        resp.raise_for_status()
        return resp.json()

    async def list_tables(self) -> list[TableInfo]:
        return [TableInfo(name=f"{self.base_url} (API)")]

    async def get_table_info(self, table_name: str) -> TableInfo:
        return TableInfo(name=table_name)


# ── DRAP 台账 API 连接器 ─────────────────────────────────────────────

class DrapConnector(APIConnector):
    """
    DRAP 数据资产估值引擎连接器
    封装 DRAP (D:/drap) 的台账管理和估值 API。
    """

    connector_type = "drap"

    async def get_ledger_list(self, page: int = 1, page_size: int = 20, **filters: Any) -> dict[str, Any]:
        return await self.get("/api/ledger/list", page=page, page_size=page_size, **filters)

    async def get_ledger_detail(self, ledger_id: str) -> dict[str, Any]:
        return await self.get(f"/api/ledger/{ledger_id}")

    async def create_ledger(self, data: dict[str, Any]) -> dict[str, Any]:
        return await self.post("/api/ledger/create", data)

    async def run_valuation(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        endpoints = {
            "bsc": "/api/valuation/bsc", "anp": "/api/valuation/anp",
            "monetization": "/api/valuation/monetization-path",
            "cultural": "/api/valuation/cultural", "industry": "/api/industry/evaluate",
        }
        path = endpoints.get(method, f"/api/valuation/{method}")
        return await self.post(path, params)

    async def generate_audit_package(self, project_id: str) -> dict[str, Any]:
        return await self.post("/api/audit/generate-package", {"project_id": project_id})

    def set_auth_token(self, token: str) -> None:
        self.headers["Authorization"] = f"Bearer {token}"


# ── 注册 ──

ConnectorFactory.register("sqlite", SQLiteConnector)
ConnectorFactory.register("api", APIConnector)
ConnectorFactory.register("drap", DrapConnector)
