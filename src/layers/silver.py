"""
Medallion Silver 层 — 清洗后数据的标准化存储

核心原则:
  - 仅接受清洗后的数据（从 Clean Agent 或直接传入）
  - 支持 SCD Type 2（缓慢变化维度）
  - 按主题域分区
  - Parquet 格式
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import duckdb

from src.config import get_settings
from src.utils.sql import safe_identifier, safe_literal, safe_path_for_duckdb

logger = logging.getLogger(__name__)


class SilverLayer:
    """Medallion Silver 层管理器"""

    def __init__(self, base_path: Optional[str] = None):
        settings = get_settings()
        self.base_path = Path(base_path or settings.medallion.silver_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def write_table(
        self,
        domain: str,       # 主题域，如 'customer', 'product', 'sales'
        table_name: str,   # 表名
        columns: list[str],
        rows: list[list[Any]],
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """写入清洗后的数据到 Silver 层"""
        t0 = time.perf_counter()
        n_rows = len(rows)

        if n_rows == 0:
            return {"domain": domain, "table": table_name, "rows": 0, "path": ""}

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        domain_dir = self.base_path / domain / table_name
        domain_dir.mkdir(parents=True, exist_ok=True)

        con = duckdb.connect(":memory:")
        try:
            safe_cols = [c.replace(" ", "_").replace("-", "_") for c in columns]
            col_def = ", ".join(safe_identifier(c) for c in safe_cols)
            placeholders = ", ".join(
                f"({', '.join(safe_literal(v) for v in row)})"
                for row in rows
            )
            sql = f"CREATE TABLE _t({col_def}) AS SELECT * FROM (VALUES {placeholders}) AS t({col_def})"
            con.execute(sql)

            filepath = domain_dir / f"{ts}.parquet"
            con.execute(f"COPY _t TO '{filepath}' (FORMAT PARQUET, COMPRESSION SNAPPY)")
        finally:
            con.close()

        duration = (time.perf_counter() - t0) * 1000
        file_size = filepath.stat().st_size if filepath.exists() else 0

        logger.info(
            f"[Silver] {domain}/{table_name}: {n_rows} rows → {filepath.name} "
            f"({file_size} bytes, {duration:.0f}ms)"
        )

        return {
            "domain": domain,
            "table": table_name,
            "path": str(filepath),
            "rows": n_rows,
            "columns": len(columns),
            "partition": today,
            "size_bytes": file_size,
            "duration_ms": duration,
            "metadata": metadata or {},
        }

    def read_latest(self, domain: str, table_name: str) -> list[dict[str, Any]]:
        """读取最新版本"""
        domain_dir = self.base_path / domain / table_name
        if not domain_dir.exists():
            raise FileNotFoundError(f"No data: silver/{domain}/{table_name}")
        files = sorted(domain_dir.glob("*.parquet"), reverse=True)
        if not files:
            raise FileNotFoundError(f"No parquet files in: {domain_dir}")
        con = duckdb.connect(":memory:")
        try:
            safe_path = safe_path_for_duckdb(str(files[0]))
            result = con.execute(
                f"SELECT * FROM read_parquet('{safe_path}')"
            ).fetchall()
            cols = [d[0] for d in con.description]
            return [dict(zip(cols, row)) for row in result]
        finally:
            con.close()

    def list_domains(self) -> list[str]:
        if not self.base_path.exists():
            return []
        return sorted([d.name for d in self.base_path.iterdir() if d.is_dir()])

    def get_stats(self) -> dict[str, Any]:
        domains = self.list_domains()
        result = {}
        total_bytes = 0
        for domain in domains:
            domain_path = self.base_path / domain
            tables = [d.name for d in domain_path.iterdir() if d.is_dir()]
            domain_bytes = sum(
                f.stat().st_size for f in domain_path.rglob("*.parquet") if f.is_file()
            )
            total_bytes += domain_bytes
            result[domain] = {"tables": len(tables), "table_names": tables, "size_bytes": domain_bytes}
        return {"domains": len(domains), "total_size_bytes": total_bytes, "detail": result}

