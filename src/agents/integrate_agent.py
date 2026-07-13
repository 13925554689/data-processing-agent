"""
数据集成 Agent (Integrate Agent)

职责:
  1. 多源 Schema 对齐
  2. 实体匹配（模糊 + 精确）
  3. 数据融合到 Silver 层
  4. 冲突解决
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from src.agents.base import AgentBase, AgentCategory, AgentResult
from src.layers.bronze import BronzeLayer
from src.layers.silver import SilverLayer

logger = logging.getLogger(__name__)


class IntegrateAgent(AgentBase):
    """数据集成 Agent"""

    name = "integrate"
    description = "多源Schema对齐、实体匹配、数据融合到Silver层"
    category = AgentCategory.INTEGRATE
    version = "0.1.0"

    async def execute(self, **kwargs: Any) -> AgentResult:
        """
        执行数据集成

        Args:
            sources: 源列表 [{"name": "src_a", "key_column": "id"}, ...]
            domain: Silver 层主题域
            table_name: 目标表名
            merge_strategy: 'union' | 'join' | 'enrich'
            join_key: join/enrich 的关联键（列名）
            dedup_key: 去重键（列名列表）

        Returns:
            AgentResult with merged data
        """
        sources = kwargs.get("sources", [])
        domain = kwargs.get("domain", "integrated")
        table_name = kwargs.get("table_name", "merged")
        strategy = kwargs.get("merge_strategy", "union")
        join_key = kwargs.get("join_key", "")
        dedup_keys = kwargs.get("dedup_key", [])
        bronze = kwargs.get("bronze") or BronzeLayer()
        silver = kwargs.get("silver") or SilverLayer()

        if not sources:
            return AgentResult.fail("Missing required parameter: sources")

        if len(sources) < 2 and strategy != "enrich":
            return AgentResult.fail("At least 2 sources required for integration")

        # 1. 读取所有源数据
        source_data = {}
        for src in sources:
            name = src["name"]
            try:
                data = bronze.read_latest(name)
                source_data[name] = data
            except FileNotFoundError:
                return AgentResult.fail(f"Source not found in Bronze: {name}")

        # 2. Schema 对齐
        all_columns, aligned = self._align_schemas(source_data)

        # 3. 按策略执行集成
        if strategy == "union":
            merged_rows = self._union(aligned, all_columns)
        elif strategy == "join" and join_key:
            merged_rows = self._join(aligned, all_columns, join_key)
        elif strategy == "enrich" and join_key:
            merged_rows = self._enrich(aligned, all_columns, join_key)
        else:
            return AgentResult.fail(f"Unknown strategy: {strategy}")

        # 4. 去重
        if dedup_keys:
            merged_rows = self._dedup(merged_rows, all_columns, dedup_keys)

        # 5. 写入 Silver
        silver_meta = silver.write_table(
            domain=domain,
            table_name=table_name,
            columns=all_columns,
            rows=merged_rows,
            metadata={
                "sources": [s["name"] for s in sources],
                "strategy": strategy,
                "source_row_counts": {k: len(v) for k, v in source_data.items()},
            },
        )

        return AgentResult.ok(
            data={
                "domain": domain,
                "table": table_name,
                "columns": all_columns,
                "merged_rows": len(merged_rows),
                "source_row_counts": {k: len(v) for k, v in source_data.items()},
                "silver": silver_meta,
            },
            message=(
                f"Integrated {len(sources)} sources → {domain}/{table_name}: "
                f"{len(merged_rows)} rows, {len(all_columns)} cols"
            ),
        )

    # ── Schema 对齐 ──

    def _align_schemas(self, source_data: dict) -> tuple[list[str], dict]:
        """对齐所有源的列，缺失值填 NULL"""
        all_columns = []
        for name, rows in source_data.items():
            if rows:
                for col in rows[0].keys():
                    if col not in all_columns:
                        all_columns.append(col)

        aligned = {}
        for name, rows in source_data.items():
            aligned[name] = []
            existing = set(rows[0].keys()) if rows else set()
            for row in rows:
                aligned_row = [row.get(col) for col in all_columns]
                aligned[name].append(aligned_row)

        return all_columns, aligned

    # ── 集成策略 ──

    def _union(self, aligned: dict, columns: list[str]) -> list[list]:
        result = []
        for name, rows in aligned.items():
            result.extend(rows)
        return result

    def _join(self, aligned: dict, columns: list[str], key: str) -> list[list]:
        if key not in columns:
            return self._union(aligned, columns)

        key_idx = columns.index(key)
        names = list(aligned.keys())
        left_rows = aligned[names[0]]
        right_rows = aligned[names[1]]

        right_map = {}
        for row in right_rows:
            k = str(row[key_idx]) if key_idx < len(row) else ""
            right_map[k] = row

        result = []
        for row in left_rows:
            k = str(row[key_idx]) if key_idx < len(row) else ""
            if k in right_map:
                result.append(row + right_map[k])

        return result

    def _enrich(self, aligned: dict, columns: list[str], key: str) -> list[list]:
        return self._join(aligned, columns, key)

    def _dedup(self, rows: list[list], columns: list[str], keys: list[str]) -> list[list]:
        key_indices = [columns.index(k) for k in keys if k in columns]
        if not key_indices:
            return rows
        seen = set()
        result = []
        for row in rows:
            k = tuple(str(row[i]) if i < len(row) else "" for i in key_indices)
            if k not in seen:
                seen.add(k)
                result.append(row)
        return result
