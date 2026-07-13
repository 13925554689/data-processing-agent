"""
Gold 层 — 面向业务的分析就绪数据集

聚合、KPI计算、特征宽表构建。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import duckdb

from src.config import get_settings
from src.utils.sql import safe_identifier, safe_literal, safe_path_for_duckdb, validate_sql_read_only

logger = logging.getLogger(__name__)


class GoldLayer:
    """Medallion Gold 层管理器"""

    def __init__(self, base_path: Optional[str] = None):
        settings = get_settings()
        self.base_path = Path(base_path or settings.medallion.gold_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def write_aggregate(
        self,
        name: str,
        columns: list[str],
        rows: list[list[Any]],
    ) -> dict[str, Any]:
        """写入聚合表"""
        t0 = time.perf_counter()
        if not rows:
            return {"name": name, "rows": 0}

        ts = time.strftime("%Y%m%d_%H%M%S")
        filepath = self.base_path / f"{name}_{ts}.parquet"

        con = duckdb.connect(":memory:")
        try:
            safe_cols = [c.replace(" ", "_").replace("-", "_") for c in columns]
            col_def = ", ".join(safe_identifier(c) for c in safe_cols)
            placeholders = ", ".join(
                f"({', '.join(safe_literal(v) for v in row)})" for row in rows
            )
            con.execute(
                f"CREATE TABLE _t({col_def}) AS "
                f"SELECT * FROM (VALUES {placeholders}) AS t({col_def})"
            )
            con.execute(f"COPY _t TO '{filepath}' (FORMAT PARQUET, COMPRESSION SNAPPY)")
        finally:
            con.close()

        duration = (time.perf_counter() - t0) * 1000
        logger.info(f"[Gold] {name}: {len(rows)} rows ({duration:.0f}ms)")
        return {
            "name": name, "path": str(filepath), "rows": len(rows),
            "columns": len(columns), "duration_ms": duration,
        }

    def aggregate_from_sql(
        self, sql: str, output_name: str
    ) -> dict[str, Any]:
        """从 SQL 查询结果直接写入 Gold 层"""
        validate_sql_read_only(sql)
        con = duckdb.connect(":memory:")
        try:
            result = con.execute(sql).fetchall()
            cols = [d[0] for d in con.description]
            rows = [list(r) for r in result]
            return self.write_aggregate(output_name, cols, rows)
        finally:
            con.close()

    def read(self, name_pattern: str) -> list[dict[str, Any]]:
        """读取最新的 Gold 数据集"""
        if '..' in name_pattern or '\x00' in name_pattern:
            raise ValueError(f"Invalid name pattern: {name_pattern!r}")
        files = sorted(
            self.base_path.glob(f"{name_pattern}_*.parquet"), reverse=True
        )
        if not files:
            return []
        con = duckdb.connect(":memory:")
        try:
            safe_path = safe_path_for_duckdb(str(files[0]))
            result = con.execute(f"SELECT * FROM read_parquet('{safe_path}')").fetchall()
            cols = [d[0] for d in con.description]
            return [dict(zip(cols, row)) for row in result]
        finally:
            con.close()

    def list_datasets(self) -> list[str]:
        """列出所有 Gold 数据集"""
        names = set()
        for f in self.base_path.glob("*.parquet"):
            # 去掉时间戳后缀: name_20260710_120000.parquet → name
            base = f.stem.rsplit("_", 2)[0] if "_" in f.stem else f.stem
            names.add(base)
        return sorted(names)

