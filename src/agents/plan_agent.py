"""
任务规划 Agent (Plan Agent)

职责: 解析用户意图 → 制定执行计划 → 分派子任务
"""

from __future__ import annotations

from typing import Any

from src.agents.base import AgentBase, AgentCategory, AgentResult


class PlanAgent(AgentBase):
    """任务规划 Agent — 用户意图 → 执行计划"""

    name = "plan"
    description = "解析用户意图，制定数据处理执行计划，分派子任务"
    category = AgentCategory.PLAN
    version = "0.1.0"

    async def execute(self, **kwargs: Any) -> AgentResult:
        """
        Args:
            intent: 用户意图描述
            context: 额外上下文（数据源路径等）

        Returns:
            AgentResult with execution plan
        """
        intent = kwargs.get("intent", "")
        context = kwargs.get("context", {})

        # 关键词匹配 → 计划生成
        plan = self._build_plan(intent, context)

        return AgentResult.ok(
            data=plan,
            message=f"Generated plan with {len(plan['steps'])} steps",
        )

    def _build_plan(self, intent: str, context: dict) -> dict:
        intent_lower = intent.lower()
        steps = []
        agents_needed = []

        # 检测意图并构建步骤
        if any(k in intent_lower for k in ("采集", "接入", "导入", "ingest", "import")):
            agents_needed.append("ingest")
            steps.append({
                "id": "ingest",
                "agent": "ingest",
                "description": "数据源探查与接入",
                "params": context.get("ingest", {}),
            })

        if any(k in intent_lower for k in ("清洗", "清理", "clean", "预处理")):
            agents_needed.append("clean")
            steps.append({
                "id": "clean",
                "agent": "clean",
                "description": "数据清洗：缺失值/异常值/去重",
                "params": context.get("clean", {}),
            })

        if any(k in intent_lower for k in ("集成", "融合", "合并", "integrate", "merge")):
            agents_needed.append("integrate")
            steps.append({
                "id": "integrate",
                "agent": "integrate",
                "description": "多源数据融合到 Silver 层",
                "params": context.get("integrate", {}),
            })

        if any(k in intent_lower for k in ("治理", "目录", "分类", "govern", "catalog")):
            agents_needed.append("govern")
            steps.append({
                "id": "govern",
                "agent": "govern",
                "description": "数据治理：元数据/分类分级",
                "params": context.get("govern", {}),
            })

        if any(k in intent_lower for k in ("分析", "统计", "报表", "analyze", "analysis")):
            agents_needed.append("analyze")
            steps.append({
                "id": "analyze",
                "agent": "analyze",
                "description": "数据分析与统计",
                "params": context.get("analyze", {}),
            })

        if any(k in intent_lower for k in ("估值", "入表", "资产", "valuate", "asset")):
            agents_needed.append("asset")
            steps.append({
                "id": "asset",
                "agent": "asset",
                "description": "数据资产化：估值→入表→审计 (DRAP)",
                "params": context.get("asset", {}),
            })

        if any(k in intent_lower for k in ("合规", "法规", "compliance")):
            steps.append({
                "id": "compliance",
                "agent": "regulation_checker",
                "description": "法规合规检查 (→法规智能体)",
                "params": context.get("compliance", {}),
            })

        # 如果没有匹配到任何关键词，默认全流程
        if not steps:
            steps = [
                {"id": "ingest", "agent": "ingest", "description": "数据接入", "params": {}},
                {"id": "clean", "agent": "clean", "description": "数据清洗", "params": {}},
                {"id": "govern", "agent": "govern", "description": "数据治理", "params": {}},
            ]

        return {
            "intent": intent,
            "total_steps": len(steps),
            "agents_required": agents_needed,
            "steps": steps,
        }
