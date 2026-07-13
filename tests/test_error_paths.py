"""
错误路径测试 — Agent异常处理 / Connector错误场景
"""

import pytest
import tempfile

from src.agents.base import AgentBase, AgentCategory, AgentResult, AgentRegistry
from src.connectors.base import FileConnector, ConnectorFactory
from src.connectors.extended import SQLiteConnector


class TestAgentErrorHandling:
    def test_agent_result_fail(self):
        result = AgentResult.fail("test error")
        assert result.success is False
        assert result.error == "test error"

    def test_agent_result_ok(self):
        result = AgentResult.ok(data={"key": "val"}, message="done")
        assert result.success is True
        assert result.data == {"key": "val"}

    @pytest.mark.asyncio
    async def test_agent_run_records_duration(self):
        class TestAgent(AgentBase):
            name = "test_duration"
            description = "test"
            category = AgentCategory.CUSTOM
            max_retries = 0

            async def execute(self, **kwargs):
                return AgentResult.ok(data={"worked": True})

        agent = TestAgent()
        result = await agent.run()
        assert result.success is True
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_agent_run_validation_failure(self):
        class FailAgent(AgentBase):
            name = "test_validation"
            description = "test"
            category = AgentCategory.CUSTOM
            max_retries = 0

            async def validate(self, **kwargs):
                return "Validation failed"

            async def execute(self, **kwargs):
                return AgentResult.ok()

        agent = FailAgent()
        result = await agent.run()
        assert result.success is False
        assert "Validation failed" in result.error


class TestConnectorErrors:
    @pytest.mark.asyncio
    async def test_file_connector_nonexistent(self):
        conn = FileConnector("test", {"path": "/nonexistent/file.csv", "type": "csv"})
        with pytest.raises(FileNotFoundError):
            await conn.connect()

    @pytest.mark.asyncio
    async def test_sqlite_connector_nonexistent(self):
        conn = SQLiteConnector("test", {"path": "/nonexistent/db.sqlite3"})
        with pytest.raises(FileNotFoundError):
            await conn.connect()

    def test_connector_factory_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown connector type"):
            ConnectorFactory.create("test", "unknown_type", {})


class TestRegulationCheckerFailSafe:
    @pytest.mark.asyncio
    async def test_compliance_check_fails_closed(self):
        from src.connectors.regulation_checker import RegulationChecker
        checker = RegulationChecker(base_url="http://localhost:99999")
        result = await checker.check_compliance("数据采集", "test", "test")
        assert result["passed"] is False