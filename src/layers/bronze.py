"""
Medallion Bronze 层 — 原始数据不可变落地（DuckDB 实现）

核心原则:
  - 原始数据 append-only，永不修改
  - 按 source/date 分区
  - Parquet 格式 + 元数据追踪
  - 零外部依赖（仅需 DuckDB）
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import duckdb

from src.config import get_settings
from src.utils.sql import safe_identifier, safe_literal, safe_path_for_duckdb

logger = logging.getLogger(__name__)


class BronzeLayer:
    """
    Medallion Bronze 层管理器

    职责：接收原始数据 → 不可变落地 → 元数据记录
    格式：Parquet (Snappy 压缩，由 DuckDB 写入)
    分区：{base_path}/bronze/{source_name}/{YYYY-MM-DD}/
    """

    def __init__(self, base_path: Optional[str] = None):
        settings = get_settings()
        self.base_path = Path(base_path or settings.medallion.bronze_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def ingest_records(
        self,
        source_name: str,
        columns: list[str],
        records: list[list[Any]],
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        写入 Bronze 层

        Args:
            source_name: 数据源标识
            columns: 列名列表
            records: 行数据（列表的列表）
            metadata: 附加元数据

        Returns:
            {"path": str, "rows": int, "columns": list, "partition": str, ...}
        """
        t0 = time.perf_counter()
        n_rows = len(records)
        n_cols = len(columns)

        if n_rows == 0:
            return {
                "source": source_name, "rows": 0, "columns": columns,
                "path": "", "partition": "", "size_bytes": 0, "duration_ms": 0,
            }

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        partition_dir = self.base_path / source_name / today
        partition_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{timestamp}_{source_name}.parquet"
        filepath = partition_dir / filename

        con = duckdb.connect(":memory:")
        try:
            placeholders = ", ".join(
                f"({', '.join(safe_literal(v) for v in row)})"
                for row in records
            )
            col_defs = ", ".join(safe_identifier(c) for c in columns)
            sql = f"CREATE TABLE _tmp({col_defs}) AS SELECT * FROM (VALUES {placeholders}) AS t({col_defs})"
            con.execute(sql)

            # 写入 Parquet
            con.execute(f"COPY _tmp TO '{filepath}' (FORMAT PARQUET, COMPRESSION SNAPPY)")
        finally:
            con.close()

        schema_info = [{"name": c, "type": self._infer_column_type(records, i)}
                       for i, c in enumerate(columns)]
        file_size = filepath.stat().st_size if filepath.exists() else 0
        duration = (time.perf_counter() - t0) * 1000

        meta = {
            "source": source_name,
            "path": str(filepath),
            "rows": n_rows,
            "columns": n_cols,
            "schema": schema_info,
            "partition": today,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "size_bytes": file_size,
            "duration_ms": duration,
        }
        if metadata:
            meta["metadata"] = metadata

        logger.info(
            f"[Bronze] {source_name}: {n_rows} rows, {n_cols} cols → {filepath.name} "
            f"({file_size} bytes, {duration:.0f}ms)"
        )
        return meta

    def ingest_csv(
        self,
        source_name: str,
        csv_path: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """从 CSV 文件直接写入 Bronze 层"""
        safe_path = safe_path_for_duckdb(csv_path)
        con = duckdb.connect(":memory:")
        try:
            con.execute(f"CREATE TABLE _tmp AS SELECT * FROM read_csv_auto('{safe_path}')")
            result = con.execute("SELECT * FROM _tmp").fetchall()
            cols = [d[0] for d in con.description]
            rows = [list(r) for r in result]
            return self.ingest_records(source_name, cols, rows, metadata)
        finally:
            con.close()

    # ── 查询方法 ──

    def list_partitions(self, source_name: str) -> list[str]:
        src_dir = self.base_path / source_name
        if not src_dir.exists():
            return []
        return sorted(
            [d.name for d in src_dir.iterdir() if d.is_dir()], reverse=True
        )

    def list_sources(self) -> list[str]:
        if not self.base_path.exists():
            return []
        return sorted([
            d.name for d in self.base_path.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])

    def read_partition(self, source_name: str, partition: str) -> list[dict[str, Any]]:
        """读取指定分区所有数据，返回 dict 列表"""
        part_dir = self.base_path / source_name / partition
        if not part_dir.exists():
            raise FileNotFoundError(f"Partition not found: {part_dir}")
        files = list(part_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files in: {part_dir}")
        con = duckdb.connect(":memory:")
        try:
            safe_path = safe_path_for_duckdb(str(part_dir))
            result = con.execute(
                f"SELECT * FROM read_parquet('{safe_path}/*.parquet')"
            ).fetchall()
            cols = [d[0] for d in con.description]
            return [dict(zip(cols, row)) for row in result]
        finally:
            con.close()

    def read_latest(self, source_name: str) -> list[dict[str, Any]]:
        partitions = self.list_partitions(source_name)
        if not partitions:
            raise FileNotFoundError(f"No data for source: {source_name}")
        return self.read_partition(source_name, partitions[0])

    def get_stats(self) -> dict[str, Any]:
        sources = self.list_sources()
        total_bytes = 0
        source_stats = {}
        for src in sources:
            src_path = self.base_path / src
            src_bytes = sum(
                f.stat().st_size for f in src_path.rglob("*.parquet") if f.is_file()
            )
            parts = self.list_partitions(src)
            source_stats[src] = {
                "partitions": len(parts),
                "latest": parts[0] if parts else None,
                "size_bytes": src_bytes,
            }
            total_bytes += src_bytes
        return {
            "sources": len(sources),
            "total_size_bytes": total_bytes,
            "sources_detail": source_stats,
        }

    @staticmethod
    def _infer_column_type(records: list[list], col_idx: int) -> str:
        for row in records:
            v = row[col_idx] if col_idx < len(row) else None
            if v is not None:
                if isinstance(v, bool):
                    return "boolean"
                if isinstance(v, int):
                    return "integer"
                if isinstance(v, float):
                    return "float"
                if isinstance(v, str):
                    return "string"
        return "null"

