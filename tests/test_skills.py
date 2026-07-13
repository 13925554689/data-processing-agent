"""Test skills module."""

import pytest

from src.skills.base import (
    BaseSkill,
    SkillMeta,
    SkillPipeline,
    SkillRegistry,
    SkillResult,
    SkillStatus,
    skill_registry,
)


# ── 测试用 Skill ──

class DoubleSkill(BaseSkill):
    meta = SkillMeta(
        name="double",
        description="Double the input number",
        category="math",
        tags=["transform"],
    )

    async def execute(self, **kwargs):
        data = kwargs.get("data", 0)
        return SkillResult(
            success=True,
            data=data * 2,
            message=f"Doubled {data} -> {data * 2}",
            status=SkillStatus.SUCCESS,
            metadata={"skill": "double"},
        )


class AddTenSkill(BaseSkill):
    meta = SkillMeta(
        name="add_ten",
        description="Add 10 to input",
        category="math",
        tags=["transform"],
    )

    async def execute(self, **kwargs):
        data = kwargs.get("data", 0)
        return SkillResult(
            success=True,
            data=data + 10,
            message=f"Added 10: {data} -> {data + 10}",
            status=SkillStatus.SUCCESS,
            metadata={"skill": "add_ten"},
        )


class FailingSkill(BaseSkill):
    meta = SkillMeta(name="fail", description="Always fails", category="test")

    async def execute(self, **kwargs):
        return SkillResult(
            success=False,
            error="Intentional failure",
            status=SkillStatus.FAILED,
            metadata={"skill": "fail"},
        )


class SchemaSkill(BaseSkill):
    meta = SkillMeta(
        name="schema_test",
        description="Tests input schema",
        category="test",
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        },
    )

    async def execute(self, **kwargs):
        return SkillResult(success=True, data=kwargs, metadata={"skill": "schema_test"})


# ── 测试 ──

class TestSkillResult:
    def test_default(self):
        r = SkillResult(success=True)
        assert r.success
        assert r.status == SkillStatus.PENDING


class TestBaseSkill:
    @pytest.mark.asyncio
    async def test_double_skill(self):
        skill = DoubleSkill()
        result = await skill.execute(data=5)
        assert result.success
        assert result.data == 10

    @pytest.mark.asyncio
    async def test_info(self):
        skill = DoubleSkill()
        info = skill.info
        assert info["name"] == "double"
        assert info["category"] == "math"
        assert "transform" in info["tags"]

    @pytest.mark.asyncio
    async def test_schema_validation_pass(self):
        skill = SchemaSkill()
        error = await skill.validate_input(name="Alice", age=30)
        assert error is None

    @pytest.mark.asyncio
    async def test_schema_validation_missing_required(self):
        skill = SchemaSkill()
        error = await skill.validate_input(age=30)
        assert error is not None
        assert "name" in error


class TestSkillRegistry:
    def setup_method(self):
        SkillRegistry.reset()

    def teardown_method(self):
        SkillRegistry.reset()

    def test_register_and_get(self):
        reg = skill_registry()
        reg.register(DoubleSkill)
        skill = reg.get("double")
        assert skill is not None
        assert isinstance(skill, DoubleSkill)

    def test_list_names(self):
        reg = skill_registry()
        reg.register(DoubleSkill)
        reg.register(AddTenSkill)
        assert "double" in reg.list_names()
        assert "add_ten" in reg.list_names()

    def test_list_by_category(self):
        reg = skill_registry()
        reg.register(DoubleSkill)
        reg.register(FailingSkill)
        math_skills = reg.list_by_category("math")
        assert len(math_skills) == 1
        assert math_skills[0].meta.name == "double"


class TestSkillPipeline:
    def setup_method(self):
        SkillRegistry.reset()
        reg = skill_registry()
        reg.register(DoubleSkill)
        reg.register(AddTenSkill)
        reg.register(FailingSkill)

    def teardown_method(self):
        SkillRegistry.reset()

    @pytest.mark.asyncio
    async def test_single_skill_pipeline(self):
        pipeline = SkillPipeline("test")
        pipeline.add("double")
        result = await pipeline.run(initial_input=5)
        assert result.success
        assert result.data == 10

    @pytest.mark.asyncio
    async def test_multi_skill_pipeline(self):
        pipeline = SkillPipeline("test")
        pipeline.add("double").add("add_ten")
        result = await pipeline.run(initial_input=5)
        assert result.success
        assert result.data == 20  # 5*2 + 10

    @pytest.mark.asyncio
    async def test_pipeline_failure_stops(self):
        pipeline = SkillPipeline("test")
        pipeline.add("fail").add("double")
        result = await pipeline.run(initial_input=5)
        assert not result.success
        assert "fail" in (result.error or "")

    @pytest.mark.asyncio
    async def test_pipeline_missing_skill(self):
        pipeline = SkillPipeline("test")
        pipeline.add("nonexistent")
        result = await pipeline.run(initial_input=5)
        assert not result.success
        assert "not found" in (result.error or "")
