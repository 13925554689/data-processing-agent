"""
数据资产化 Agent (Asset Agent)

编排调用 DRAP 估值引擎进行估值、入表、审计。
DRAP API: http://localhost:8000

流程: 数据准备 → 合规检查 → 质量评价 → 估值 → 入表 → 审计
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.agents.base import AgentBase, AgentCategory, AgentResult
from src.config import get_settings
from src.connectors.extended import DrapConnector

logger = logging.getLogger(__name__)


class AssetAgent(AgentBase):
    """数据资产化 Agent — DRAP 编排"""

    name = "asset"
    description = "数据确权→质量评价→估值编排(DRAP)→入表辅助→审计底稿"
    category = AgentCategory.ASSET
    version = "0.1.0"

    async def execute(self, **kwargs: Any) -> AgentResult:
        """
        Args:
            action: 'valuate' | 'accounting' | 'audit' | 'full'
            source_name: 数据源名称
            asset_data: 资产元数据 {'rows': N, 'columns': M, 'dtypes': {...}, ...}
            valuation_method: 'bsc' | 'anp' | 'industry'  (默认 'bsc')
            industry: 行业 (金融/制造/互联网)
        """
        action = kwargs.get("action", "valuate")
        source_name = kwargs.get("source_name", "unknown")
        asset_data = kwargs.get("asset_data", {})
        method = kwargs.get("valuation_method", "bsc")
        industry = kwargs.get("industry", "")

        settings = get_settings()
        connector = DrapConnector("asset_drap", {
            "base_url": settings.drap.base_url,
            "timeout": settings.drap.timeout,
        })
        if settings.drap.auth_token:
            connector.set_auth_token(settings.drap.auth_token)

        try:
            await connector.connect()

            if action == "valuate" or action == "full":
                valuation = await self._valuate(connector, source_name, asset_data, method, industry)
            else:
                valuation = None

            if action == "accounting" or action == "full":
                ledger = await self._create_ledger(connector, source_name, asset_data, valuation)
            else:
                ledger = None

            if action == "audit" or action == "full":
                audit = await self._generate_audit(connector, source_name)
            else:
                audit = None

            return AgentResult.ok(
                data={
                    "source": source_name,
                    "valuation": valuation,
                    "ledger": ledger,
                    "audit": audit,
                },
                message=f"Asset pipeline '{action}' completed for {source_name}",
            )
        except Exception as e:
            return AgentResult.fail(f"DRAP pipeline failed: {e}", data={"drap_error": str(e)})
        finally:
            await connector.close()

    async def _valuate(
        self, conn: DrapConnector, name: str, data: dict, method: str, industry: str
    ) -> dict:
        """调用 DRAP 估值 API"""
        try:
            if method == "bsc":
                return await conn.run_valuation("bsc", {
                    "asset_id": name,
                    "organization_strategy": {
                        "financial_weight": 0.35, "customer_weight": 0.25,
                        "internal_weight": 0.25, "learning_weight": 0.15,
                    },
                })
            elif method == "anp":
                return await conn.run_valuation("anp", {
                    "asset_id": name,
                    "scenario": "入表",
                })
            elif method == "industry" and industry:
                return await conn.run_valuation("industry", {
                    "asset_id": name,
                    "industry": industry,
                })
        except Exception as e:
            return {"error": str(e), "method": method, "status": "drap_unavailable"}

        return {"method": method, "status": "no_execution", "note": "Unknown method"}

    async def _create_ledger(
        self, conn: DrapConnector, name: str, data: dict, valuation: Optional[dict]
    ) -> dict:
        """创建 DRAP 台账"""
        try:
            return await conn.create_ledger({
                "asset_name": name,
                "asset_type": "dataset",
                "data_volume": data.get("rows", 0),
                "data_fields": data.get("columns", 0),
                "description": f"Auto-created from Data Processing Agent. {data.get('rows', 0)} rows.",
            })
        except Exception as e:
            return {"error": str(e), "status": "drap_unavailable"}

    async def _generate_audit(self, conn: DrapConnector, name: str) -> dict:
        """生成审计底稿"""
        try:
            return await conn.generate_audit_package(name)
        except Exception as e:
            return {"error": str(e), "status": "drap_unavailable"}
