"""
法规智能体连接器 — 合规检查网关

在数据处理操作前调用法规智能体进行合规检查。
法规智能体地址: http://localhost:8200 (建议端口)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


class RegulationChecker:
    """
    法规合规检查器

    封装法规智能体 API 调用（D:\数据资产智能体）。
    在每个数据处理操作前自动检查合规性。
    """

    def __init__(self, base_url: str = "http://localhost:8200"):
        self.base_url = base_url
        self.timeout = 15  # 合规检查超时（秒）

    # ── 10阶段合规检查 ──

    async def check_collection(self, data_type: str, source: str) -> dict[str, Any]:
        """检查数据采集合规性"""
        return await self._query(f"企业从{source}采集{data_type}数据需要遵循哪些合规要求")

    async def check_crawling(self, target: str, method: str) -> dict[str, Any]:
        """检查数据爬取合规性"""
        return await self._query(f"通过{method}方式爬取{target}数据的法律边界是什么")

    async def check_standardization(self, data_type: str) -> dict[str, Any]:
        """检查数据标准化要求"""
        return await self._query(f"{data_type}数据的分类分级和标准化有哪些国家标准要求")

    async def check_cleaning(self, operation: str) -> dict[str, Any]:
        """检查数据清洗合规性"""
        return await self._query(f"对数据进行{operation}操作时，脱敏和匿名化的法律要求")

    async def check_storage(self, data_category: str) -> dict[str, Any]:
        """检查数据存储合规性"""
        return await self._query(f"{data_category}数据的存储安全要求，包括等保和备份规定")

    async def check_ownership(self, data_desc: str) -> dict[str, Any]:
        """检查数据确权"""
        return await self._query(f"{data_desc}的数据三权分置确权流程和法规要求")

    async def check_valuation(self, method: str) -> dict[str, Any]:
        """检查估值方法合规性"""
        return await self._query(f"采用{method}进行数据资产价值评估时需遵循的国家标准")

    async def check_accounting(self, asset_type: str) -> dict[str, Any]:
        """检查入表合规性"""
        return await self._query(f"{asset_type}类数据资产入表的会计处理规定和披露要求")

    async def check_trading(self, product_type: str) -> dict[str, Any]:
        """检查交易合规性"""
        return await self._query(f"{product_type}类数据产品在数据交易所交易需满足的条件")

    async def check_cross_border(self, data_desc: str) -> dict[str, Any]:
        """检查跨境合规性"""
        return await self._query(f"{data_desc}数据出境需要经过哪些安全评估和审批流程")

    # ── 通用检查 ──

    async def check_compliance(
        self, stage: str, operation: str = "", data_desc: str = ""
    ) -> dict[str, Any]:
        """通用合规检查入口"""
        stage_map = {
            "数据采集": self.check_collection,
            "数据爬取": self.check_crawling,
            "数据规范": self.check_standardization,
            "数据清洗": self.check_cleaning,
            "数据存储": self.check_storage,
            "数据确权": self.check_ownership,
            "数据定价": self.check_valuation,
            "数据入账": self.check_accounting,
            "数据交易": self.check_trading,
            "数据跨境": self.check_cross_border,
        }

        checker = stage_map.get(stage)
        if checker:
            return await self._execute_check(checker, operation, data_desc, stage=stage)
        return {"passed": True, "note": f"Stage '{stage}' has no specific compliance check"}

    async def _execute_check(self, checker, operation: str, data_desc: str, stage: str = "") -> dict:
        try:
            result = await checker(operation or data_desc)
            return {"passed": True, "stage": stage, "result": result}
        except Exception as e:
            logger.warning(f"Compliance check failed (blocking): {e}")
            return {"passed": False, "error": f"Compliance check unavailable: {e}"}

    # ── 内部 HTTP 调用 ──

    async def _query(self, question: str) -> dict[str, Any]:
        """调用法规智能体 /query API"""

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(
                    f"{self.base_url}/query",
                    json={"question": question, "top_k": 3, "stream": False},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return {
                        "answer": data.get("answer", ""),
                        "citations": data.get("citations", []),
                        "question": question,
                    }
            except Exception as e:
                logger.debug(f"Regulation agent unavailable: {e}")

        return {"answer": "法规智能体暂不可用", "citations": [], "question": question}

    async def search_regulations(
        self, stage: str, keyword: str = "", top_k: int = 5
    ) -> dict[str, Any]:
        """检索特定阶段的法规"""

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(
                    f"{self.base_url}/search",
                    json={"question": keyword, "stage": stage, "top_k": top_k},
                )
                if resp.status_code == 200:
                    return resp.json()
            except Exception:
                pass
        return {"total_hits": 0, "results": []}
