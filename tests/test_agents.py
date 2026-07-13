"""Test Agent base class and registry."""

import time

import pytest

from src.agents.base import (
    AgentBase,
    AgentCategory,
    AgentRegistry,
    AgentResult,
    AgentStatus,
    registry,
)


# ── 测试用 Agent ──

class EchoAgent(AgentBase):
    name = "echo"
    description = "Echo back the input"
    category = AgentCategory.CUSTOM

    async def execute(self, **kwargs):
        text = kwargs.get("text", "")
        return AgentResult.ok(data={"echo": text}, message=f"Echo: {text}")


class FailingAgent(AgentBase):
    name = "failing"
    description = "Always fails"
    category = AgentCategory.CUSTOM
    max_retries = 0

    async def execute(self, **kwargs):
        raise RuntimeError("I always fail")


class ValidatingAgent(AgentBase):
    name = "validating"
    description = "Validates input"
    category = AgentCategory.CLEAN

    async def validate(self, **kwargs):
        if "data" not in kwargs:
            return "Missing required field: data"
        return None

    async def execute(self, **kwargs):
        return AgentResult.ok(data=kwargs["data"] * 2)


class HookAgent(AgentBase):
    name = "hook"
    description = "Tests hooks"
    category = AgentCategory.CUSTOM

    async def pre_execute(self, **kwargs):
        self._pre_called = True

    async def execute(self, **kwargs):
        return AgentResult.ok(data={"value": 42})

    async def post_execute(self, result, **kwargs):
        result.metadata["post_hook"] = True
        return result


# ── 测试 ──

class TestAgentResult:
    def test_ok(self):
        r = AgentResult.ok(data={"x": 1}, message="done")
        assert r.success
        assert r.data == {"x": 1}
        assert r.message == "done"
        assert r.error is None

    def test_fail(self):
        r = AgentResult.fail("something broke", message="oops")
        assert not r.success
        assert r.error == "something broke"
        assert r.message == "oops"

    def test_fail_default_message(self):
        r = AgentResult.fail("err")
        assert r.message == "err"


class TestAgentBase:
    @pytest.mark.asyncio
    async def test_echo_agent(self):
        agent = EchoAgent()
        result = await agent.run(text="hello")
        assert result.success
        assert result.data == {"echo": "hello"}

    @pytest.mark.asyncio
    async def test_failing_agent(self):
        agent = FailingAgent()
        result = await agent.run()
        assert not result.success
        assert "I always fail" in (result.error or "")

    @pytest.mark.asyncio
    async def test_validation_rejects(self):
        agent = ValidatingAgent()
        result = await agent.run()
        assert not result.success
        assert "Missing required field" in (result.error or result.message)

    @pytest.mark.asyncio
    async def test_validation_passes(self):
        agent = ValidatingAgent()
        result = await agent.run(data=5)
        assert result.success
        assert result.data == 10

    @pytest.mark.asyncio
    async def test_hooks(self):
        agent = HookAgent()
        result = await agent.run()
        assert result.success
        assert result.metadata.get("post_hook") is True

    def test_info(self):
        agent = EchoAgent()
        info = agent.info
        assert info["name"] == "echo"
        assert info["category"] == "custom"
        assert info["status"] == "IDLE"

    def test_status_transition(self):
        agent = EchoAgent()
        assert agent.status == AgentStatus.IDLE


class TestAgentRegistry:
    def setup_method(self):
        AgentRegistry.reset()

    def teardown_method(self):
        AgentRegistry.reset()

    def test_singleton(self):
        r1 = registry()
        r2 = registry()
        assert r1 is r2

    def test_register_and_get(self):
        reg = registry()
        reg.register(EchoAgent)
        agent = reg.get("echo")
        assert agent is not None
        assert agent.name == "echo"

    def test_lazy_loading(self):
        reg = registry()
        reg.register(EchoAgent)
        # 注册后不应立即实例化
        assert "echo" not in reg._instances
        # 首次访问时实例化
        agent = reg.get("echo")
        assert "echo" in reg._instances
        # 再次访问返回同一实例
        agent2 = reg.get("echo")
        assert agent is agent2

    def test_get_nonexistent(self):
        reg = registry()
        assert reg.get("nonexistent") is None

    def test_get_by_category(self):
        reg = registry()
        reg.register(EchoAgent)
        reg.register(ValidatingAgent)
        clean_agents = reg.get_by_category(AgentCategory.CLEAN)
        assert len(clean_agents) == 1
        assert clean_agents[0].name == "validating"

    def test_list_names(self):
        reg = registry()
        reg.register(EchoAgent)
        reg.register(ValidatingAgent)
        names = reg.list_names()
        assert "echo" in names
        assert "validating" in names

    def test_list_info(self):
        reg = registry()
        reg.register(EchoAgent)
        info = reg.list_info()
        assert len(info) == 1
        assert info[0]["name"] == "echo"

    def test_unregister(self):
        reg = registry()
        reg.register(EchoAgent)
        assert reg.get("echo") is not None
        reg.unregister("echo")
        assert reg.get("echo") is None

    def test_duplicate_register_raises(self):
        reg = registry()
        reg.register(EchoAgent)
        with pytest.raises(ValueError, match="already registered"):
            reg.register(EchoAgent)

    def test_override_register(self):
        reg = registry()
        reg.register(EchoAgent)

        class NewEcho(EchoAgent):
            name = "echo"
            description = "new echo"

        reg.register(NewEcho, override=True)
        agent = reg.get("echo")
        assert agent is not None
        assert agent.description == "new echo"
