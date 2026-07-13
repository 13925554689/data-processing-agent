"""
数据血缘追踪模块 — Column-level Lineage (2026标准)

对齐:
  - OpenLineage 标准
  - Unity Catalog lineage model
  - OpenMetadata lineage API

功能:
  - 表级血缘: 源表 → 目标表
  - 列级血缘: 源列 → 目标列 (transform追踪)
  - 血缘图导出: JSON/Mermaid
  - SQL解析: 自动提取血缘关系
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DataLineage:
    """
    数据血缘追踪器

    记录:
      - TABLE_LINEAGE: 表级血缘 (源→目标)
      - COLUMN_LINEAGE: 列级血缘 (源列→目标列+变换)
      - JOB_LINEAGE: 作业血缘 (哪个Agent/任务产生)
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path or "data/lineage")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._lineage_file = self.storage_path / "lineage.json"
        self._graph: dict[str, Any] = self._load()

    def _load(self) -> dict:
        if self._lineage_file.exists():
            try:
                return json.loads(self._lineage_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"nodes": {}, "edges": [], "column_edges": [], "updated": ""}

    def _save(self) -> None:
        self._graph["updated"] = datetime.now(timezone.utc).isoformat()
        self._lineage_file.write_text(
            json.dumps(self._graph, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ── 表级血缘 ──

    def track_table(
        self,
        source: str,        # 源表 (如 "bronze:erp_sales")
        target: str,        # 目标表 (如 "silver:sales/cleaned")
        operation: str,     # 操作类型: ingest/clean/integrate/aggregate
        agent: str = "",    # 执行Agent
        metadata: Optional[dict] = None,
    ) -> str:
        """记录表级血缘"""
        edge_id = hashlib.sha256(f"{source}→{target}".encode()).hexdigest()[:12]
        self._graph["edges"].append({
            "id": edge_id,
            "source": source,
            "target": target,
            "operation": operation,
            "agent": agent,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        })
        # 注册节点
        for node in (source, target):
            if node not in self._graph["nodes"]:
                self._graph["nodes"][node] = {"type": "table", "first_seen": datetime.now(timezone.utc).isoformat()}

        self._save()
        logger.info(f"[Lineage] {source} →[{operation}]→ {target}")
        return edge_id

    # ── 列级血缘 ──

    def track_column(
        self,
        source_table: str,
        source_column: str,
        target_table: str,
        target_column: str,
        transform: str = "direct",  # direct/rename/compute/mask/aggregate
        expression: str = "",
    ) -> str:
        """记录列级血缘"""
        edge_id = hashlib.sha256(
            f"{source_table}.{source_column}→{target_table}.{target_column}".encode()
        ).hexdigest()[:12]
        self._graph["column_edges"].append({
            "id": edge_id,
            "source_table": source_table,
            "source_column": source_column,
            "target_table": target_table,
            "target_column": target_column,
            "transform": transform,
            "expression": expression,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self._save()
        return edge_id

    def track_columns_batch(
        self,
        source_table: str,
        source_columns: list[str],
        target_table: str,
        target_columns: list[str],
        transform: str = "direct",
    ) -> list[str]:
        """批量记录列级血缘 (按位置对齐)"""
        ids = []
        for i, sc in enumerate(source_columns):
            tc = target_columns[i] if i < len(target_columns) else sc
            ids.append(self.track_column(source_table, sc, target_table, tc, transform))
        return ids

    # ── 查询 ──

    def upstream(self, table: str) -> list[dict]:
        """查询上游表"""
        return [e for e in self._graph["edges"] if e["target"] == table]

    def downstream(self, table: str) -> list[dict]:
        """查询下游表"""
        return [e for e in self._graph["edges"] if e["source"] == table]

    def full_path(self, table: str) -> list[str]:
        """追溯完整数据路径 (递归上游)"""
        path = [table]
        current = table
        while True:
            ups = self.upstream(current)
            if not ups:
                break
            current = ups[0]["source"]
            path.insert(0, current)
            if len(path) > 10:  # 防无限循环
                break
        return path

    def column_upstream(self, table: str, column: str) -> list[dict]:
        """查询列的来源"""
        return [
            e for e in self._graph["column_edges"]
            if e["target_table"] == table and e["target_column"] == column
        ]

    # ── 导出 ──

    def to_mermaid(self, table: Optional[str] = None) -> str:
        """导出 Mermaid 流程图"""
        lines = ["```mermaid", "graph LR"]
        edges = self._graph["edges"]
        if table:
            edges = [e for e in edges if e["source"] == table or e["target"] == table]
        seen = set()
        for e in edges:
            key = f"{e['source']}→{e['target']}"
            if key not in seen:
                src = e["source"].replace(":", "_").replace("/", "_")
                tgt = e["target"].replace(":", "_").replace("/", "_")
                op = e["operation"][:8]
                lines.append(f"    {src}[{e['source']}] -->|{op}| {tgt}[{e['target']}]")
                seen.add(key)
        lines.append("```")
        return "\n".join(lines)

    def stats(self) -> dict[str, Any]:
        """血缘统计"""
        edges = self._graph["edges"]
        col_edges = self._graph["column_edges"]
        ops = {}
        for e in edges:
            op = e["operation"]
            ops[op] = ops.get(op, 0) + 1
        return {
            "tables": len(self._graph["nodes"]),
            "table_edges": len(edges),
            "column_edges": len(col_edges),
            "operations": ops,
            "last_updated": self._graph.get("updated", ""),
        }
