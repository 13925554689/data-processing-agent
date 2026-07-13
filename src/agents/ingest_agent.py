"""
数据探查与接入 Agent (Ingest Agent)

职责:
  1. 自动探测数据源（CSV/Excel/SQLite/API）
  2. 数据探查（schema, 统计, 质量初检）
  3. 原始数据写入 Bronze 层
  4. 可选：同步到 DRAP 台账系统
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlparse

from src.agents.base import AgentBase, AgentCategory, AgentResult
from src.config import get_settings
from src.connectors.base import ConnectorFactory, FileConnector, QueryResult
from src.connectors.extended import SQLiteConnector, DrapConnector
from src.layers.bronze import BronzeLayer
from dateutil.parser import parse as dt_parse

logger = logging.getLogger(__name__)


class IngestAgent(AgentBase):
    """数据探查与接入 Agent"""

    name = "ingest"
    description = "自动探测数据源、探查数据概况、写入 Bronze 层"
    category = AgentCategory.INGEST
    version = "0.1.0"

    async def execute(self, **kwargs: Any) -> AgentResult:
        source_path = kwargs.get("source_path", "")
        source_type = kwargs.get("source_type", "")
        source_name = kwargs.get("source_name", "")
        sync_to_drap = kwargs.get("sync_to_drap", False)

        if not source_path:
            return AgentResult.fail("Missing required parameter: source_path")

        if '..' in source_path or '\x00' in source_path:
            return AgentResult.fail("Invalid source path: path traversal detected")

        # 1. 推断类型和名称
        if not source_type:
            source_type = self._infer_type(source_path)
        if not source_name:
            source_name = self._infer_name(source_path)

        logger.info(f"[Ingest] source={source_name}, type={source_type}, path={source_path}")

        # 2. 连接数据源并探查
        try:
            profile = await self._profile_source(source_path, source_type, source_name)
        except Exception as e:
            return AgentResult.fail(f"Failed to profile source: {e}")

        # 3. 写入 Bronze 层
        try:
            bronze = BronzeLayer()
            bronze_meta = bronze.ingest_records(
                source_name=source_name,
                columns=profile["columns"],
                records=profile["sample_rows"],
                metadata={
                    "source_path": source_path,
                    "source_type": source_type,
                    "profile": {
                        k: v for k, v in profile.items()
                        if k not in ("sample_rows",)
                    },
                },
            )
        except Exception as e:
            return AgentResult.fail(f"Failed to write Bronze layer: {e}")

        # 4. 可选：同步到 DRAP
        drap_result = None
        if sync_to_drap:
            drap_result = await self._sync_to_drap(source_name, profile, bronze_meta)

        return AgentResult.ok(
            data={
                "source_name": source_name,
                "source_type": source_type,
                "profile": {
                    "rows": profile["row_count"],
                    "columns": profile["columns"],
                    "dtypes": profile["dtypes"],
                    "null_counts": profile["null_counts"],
                    "sample_size": profile["sample_size"],
                },
                "bronze": bronze_meta,
                "drap_sync": drap_result,
            },
            message=f"Ingested {source_name}: {profile['row_count']} rows, {len(profile['columns'])} cols → Bronze",
        )

    # ── 内部方法 ──

    def _infer_type(self, path: str) -> str:
        """从文件扩展名或URL推断数据源类型"""
        # URL 检测
        if path.startswith("http://") or path.startswith("https://"):
            return "web"
        ext = path.lower().rsplit(".", 1)[-1] if "." in path else ""
        mapping = {
            "csv": "csv", "tsv": "csv",
            "xlsx": "excel", "xls": "excel",
            "json": "json", "jsonl": "json",
            "parquet": "parquet",
            "db": "sqlite", "sqlite": "sqlite", "sqlite3": "sqlite",
        }
        return mapping.get(ext, "csv")

    def _infer_name(self, path: str) -> str:
        """从路径或URL推断数据源名称"""
        if path.startswith("http://") or path.startswith("https://"):
            parsed = urlparse(path)
            name = parsed.netloc.replace(".", "_") + parsed.path.replace("/", "_")[:30]
            return name.strip("_").lower()[:50]
        name = os.path.splitext(os.path.basename(path))[0]
        return name.replace(" ", "_").replace("-", "_").lower()

    async def _profile_source(
        self, path: str, stype: str, name: str
    ) -> dict[str, Any]:
        """探查数据源：schema、行数、列类型、空值分布"""
        # Web 采集走专用路径
        if stype == "web":
            return await self._profile_web(path, name)


        if stype in ("csv", "excel", "json", "parquet"):
            conn = FileConnector(name, {"path": path, "type": stype})
        elif stype == "sqlite":
            conn = SQLiteConnector(name, {"path": path})
        else:
            raise ValueError(f"Unsupported source type for profiling: {stype}")

        try:
            await conn.connect()

            # 获取表结构
            info = await conn.get_table_info(path)

            # 读取样本数据（最多 10000 行用于探查）
            result = await conn.execute(
                f"SELECT * FROM _src LIMIT 10000"
                if stype != "sqlite"
                else f"SELECT * FROM (SELECT * FROM sqlite_master WHERE type='table' LIMIT 1)"
            )

            # 实际数据读取
            table_result = await conn.read_table(
                info.name if info.name else path, limit=10000
            )

            # 分析每列
            columns = table_result.columns
            rows = table_result.rows
            dtypes = {}
            null_counts = {}

            for i, col in enumerate(columns):
                values = [r[i] for r in rows if i < len(r)]
                null_counts[col] = sum(1 for v in values if v is None)
                dtypes[col] = self._guess_dtype(values)

            return {
                "source_name": name,
                "row_count": info.row_count,
                "columns": columns,
                "dtypes": dtypes,
                "null_counts": null_counts,
                "sample_size": len(rows),
                "sample_rows": rows[:100],  # 前 100 行作为元数据样本
                "profiled_at": datetime.now(timezone.utc).isoformat(),
            }
        finally:
            await conn.close()

    @staticmethod
    def _guess_dtype(values: list) -> str:
        """推断列的数据类型"""
        non_null = [v for v in values if v is not None]
        if not non_null:
            return "null"
        if all(isinstance(v, bool) for v in non_null):
            return "boolean"
        if all(isinstance(v, int) for v in non_null):
            return "integer"
        if all(isinstance(v, (int, float)) for v in non_null):
            return "float"
        if all(isinstance(v, str) for v in non_null):

            try:
                dt_parse(non_null[0])
                return "datetime"
            except Exception:
                pass
            return "string"
        return "mixed"

    async def _sync_to_drap(
        self, source_name: str, profile: dict, bronze_meta: dict
    ) -> Optional[dict[str, Any]]:
        """同步数据到 DRAP 台账系统"""
        try:

            settings = get_settings()
            conn = DrapConnector("drap_sync", {
                "base_url": settings.drap.base_url,
                "timeout": settings.drap.timeout,
            })

            # 如果配置了 auth token
            if settings.drap.auth_token.get_secret_value():
                conn.set_auth_token(settings.drap.auth_token.get_secret_value())

            await conn.connect()
            try:
                result = await conn.create_ledger({
                    "asset_name": source_name,
                    "asset_type": "dataset",
                    "data_volume": profile["row_count"],
                    "data_fields": len(profile["columns"]),
                    "storage_path": bronze_meta["path"],
                    "source_system": profile.get("source_path", ""),
                    "description": f"Auto-ingested: {profile['row_count']} rows, {len(profile['columns'])} columns",
                })
                return result
            finally:
                await conn.close()
        except Exception as e:
            logger.warning(f"[Ingest] DRAP sync failed (non-fatal): {e}")
            return {"error": str(e), "synced": False}


    async def _profile_web(self, url: str, name: str) -> dict[str, Any]:
        """Web 数据源探查：采集→统计"""
        from src.connectors.web_scraper import WebScraperConnector

        conn = WebScraperConnector(name, {
            "mode": "static", "urls": [url],
        })
        try:
            await conn.connect()
            result = await conn.scrape()
            rows = result.get("rows", [])
            columns = result.get("columns", [])
            return {
                "source_name": name,
                "row_count": len(rows),
                "columns": columns,
                "dtypes": {c: "string" for c in columns},
                "null_counts": {c: 0 for c in columns},
                "sample_size": len(rows),
                "sample_rows": rows[:100],
                "profiled_at": datetime.now(timezone.utc).isoformat(),
                "source_path": url,
            }
        finally:
            await conn.close()
