"""
数据分析 Agent (Analyze Agent)

职责: NL→分析代码、统计摘要、可视化数据生成
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from src.agents.base import AgentBase, AgentCategory, AgentResult
from src.layers.bronze import BronzeLayer
from src.layers.silver import SilverLayer

logger = logging.getLogger(__name__)


class AnalyzeAgent(AgentBase):
    """数据分析 Agent"""

    name = "analyze"
    description = "数据统计分析、聚合计算、可视化数据生成"
    category = AgentCategory.ANALYZE
    version = "0.1.0"

    async def execute(self, **kwargs: Any) -> AgentResult:
        """
        Args:
            source: 'bronze:source_name' | 'silver:domain/table' | 'gold:dataset'
            analysis: 'summary' | 'distribution' | 'correlation' | 'top_n'
            column: 目标列（summary/all 时可选）
            top_n: top_n 分析时的 N
        """
        analysis = kwargs.get("analysis", "summary")
        column = kwargs.get("column", "")
        top_n = kwargs.get("top_n", 10)
        source = kwargs.get("source", "")

        # 解析数据源
        data = await self._load_source(source, kwargs)
        if data is None:
            return AgentResult.fail(f"Cannot load source: {source}")

        if not data:
            return AgentResult.ok(data={}, message="Empty dataset")

        columns = list(data[0].keys())
        rows = [list(r.values()) for r in data]

        if analysis == "summary":
            result = self._summary(columns, rows)
        elif analysis == "distribution" and column:
            result = self._distribution(column, columns, rows)
        elif analysis == "top_n" and column:
            result = self._top_n(column, columns, rows, top_n)
        else:
            result = self._summary(columns, rows)

        return AgentResult.ok(data=result, message=f"Analysis '{analysis}' complete")

    async def _load_source(self, source: str, kwargs: dict) -> list[dict] | None:
        if source.startswith("bronze:"):
            name = source.split(":", 1)[1]
            bronze = kwargs.get("bronze") or BronzeLayer()
            try:
                return bronze.read_latest(name)
            except Exception:
                return None
        elif source.startswith("silver:"):
            path = source.split(":", 1)[1]
            domain, table = path.split("/", 1)
            try:
                return SilverLayer().read_latest(domain, table)
            except Exception:
                return None
        return None

    def _summary(self, columns: list[str], rows: list[list]) -> dict:
        n = len(rows)
        summary = {"row_count": n, "column_count": len(columns), "columns": {}}
        for i, col in enumerate(columns):
            vals = [r[i] for r in rows if i < len(r) and r[i] is not None]
            col_info = {
                "non_null": len(vals),
                "null_pct": round((n - len(vals)) / n * 100, 1) if n else 0,
                "dtype": self._dtype(vals),
            }
            if vals and all(isinstance(v, (int, float)) for v in vals):
                s = sorted(vals)
                col_info.update({
                    "min": s[0], "max": s[-1],
                    "mean": round(sum(s) / len(s), 2),
                    "median": s[len(s) // 2],
                })
            elif vals:
                col_info["unique"] = len(set(str(v) for v in vals))
            summary["columns"][col] = col_info
        return summary

    def _distribution(self, column: str, columns: list[str], rows: list[list]) -> dict:
        if column not in columns:
            return {"error": f"Column '{column}' not found"}
        idx = columns.index(column)
        vals = [r[idx] for r in rows if idx < len(r) and r[idx] is not None]
        counter = Counter(str(v) for v in vals)
        return {
            "column": column,
            "total": len(vals),
            "unique": len(counter),
            "top_values": counter.most_common(20),
        }

    def _top_n(self, column: str, columns: list[str], rows: list[list], n: int) -> dict:
        dist = self._distribution(column, columns, rows)
        dist["top_values"] = dist["top_values"][:n]
        return dist

    @staticmethod
    def _dtype(vals: list) -> str:
        if not vals:
            return "null"
        if all(isinstance(v, bool) for v in vals):
            return "boolean"
        if all(isinstance(v, int) for v in vals):
            return "integer"
        if all(isinstance(v, (int, float)) for v in vals):
            return "number"
        return "string"
