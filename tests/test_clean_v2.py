"""Test Clean Agent v2.0 — DCMM评分 / 语义清洗 / 可观测性 / 六维质量"""

import tempfile
import pytest

from src.agents.clean_agent import CleanAgent
from src.layers.bronze import BronzeLayer


@pytest.fixture
def bronze():
    with tempfile.TemporaryDirectory() as td:
        yield BronzeLayer(base_path=td)


@pytest.fixture
def dirty_data(bronze):
    bronze.ingest_records(
        "dirty_v2",
        columns=["id", "name", "phone", "email", "amount", "city", "date"],
        records=[
            [1, "Alice", "13800001111", "alice@test.com", 95.0, "Beijing", "2026-01-15"],
            [2, "Bob", None, "bob@test.com", None, "Shanghai", "2026-02-20"],
            [3, "Charlie", "136-0000-3333", "charlie@test.com", 85000.0, "East", "2025-06-01"],
            [4, "Diana", "137 0000 4444", "DIANA@TEST.COM", 999.0, "Beijing", "2026-03-10"],
            [1, "Alice", "13800001111", "alice@test.com", 95.0, "Beijing", "2026-01-15"],  # dup
            [5, None, None, None, 70.0, None, "2026-04-01"],
            [6, "Frank", "13900006666", "frank@test.com", -10.0, "Shenzhen", "2026-05-01"],
        ],
    )
    return ("dirty_v2", bronze)


class TestCleanAgentV2:
    @pytest.mark.asyncio
    async def test_dcmm_quality_scores(self, dirty_data):
        """六维质量评分输出"""
        name, bronze = dirty_data
        agent = CleanAgent()
        result = await agent.run(
            source_name=name, bronze=bronze,
            dedup=True, outlier_method="iqr", quality_score=True,
        )
        assert result.success, result.error
        scores = result.data["report"]["quality_scores"]
        assert "completeness" in scores
        assert "accuracy" in scores
        assert "consistency" in scores
        assert "uniqueness" in scores
        assert "validity" in scores
        assert "overall" in scores
        assert "dcmm_level" in scores
        # overall应在0-100之间
        assert 0 <= scores["overall"] <= 100

    @pytest.mark.asyncio
    async def test_dcmm_level_mapping(self):
        """DCMM等级映射"""
        agent = CleanAgent()
        assert "优化级" in agent._score_to_dcmm(96)
        assert "量化管理级" in agent._score_to_dcmm(88)
        assert "稳健级" in agent._score_to_dcmm(75)
        assert "受管理级" in agent._score_to_dcmm(55)
        assert "初始级" in agent._score_to_dcmm(30)

    @pytest.mark.asyncio
    async def test_semantic_dedup(self, dirty_data):
        """语义去重"""
        name, bronze = dirty_data
        agent = CleanAgent()
        result = await agent.run(
            source_name=name, bronze=bronze,
            dedup=True, semantic_clean=True,
        )
        assert result.success, result.error
        ops = result.data["report"]["operations"]
        op_names = [o["op"] for o in ops]
        assert "semantic_dedup" in op_names or "dedup" in op_names

    @pytest.mark.asyncio
    async def test_phone_normalization(self, dirty_data):
        """手机号标准化"""
        name, bronze = dirty_data
        agent = CleanAgent()
        result = await agent.run(
            source_name=name, bronze=bronze,
            semantic_clean=True, dedup=False,
        )
        assert result.success, result.error
        ops = result.data["report"]["operations"]
        op_names = [o["op"] for o in ops]
        # phone标准化应在语义清洗中
        assert "semantic_normalize" in op_names

    @pytest.mark.asyncio
    async def test_observability_snapshot(self, dirty_data):
        """可观测性快照"""
        name, bronze = dirty_data
        agent = CleanAgent()
        result = await agent.run(
            source_name=name, bronze=bronze,
            quality_score=True,
        )
        assert result.success
        obs = result.data["report"]["observability"]
        assert "timestamp" in obs
        assert "schema" in obs
        assert "distribution" in obs
        assert "health" in obs

    @pytest.mark.asyncio
    async def test_completeness_score(self, dirty_data):
        """完整性评分—有缺失值应低于100"""
        name, bronze = dirty_data
        agent = CleanAgent()
        result = await agent.run(
            source_name=name, bronze=bronze,
            quality_score=True,
        )
        scores = result.data["report"]["quality_scores"]
        # 数据有空值，完整性应 < 100
        assert scores["completeness"] < 100

    @pytest.mark.asyncio
    async def test_version_bump(self):
        """版本号升级到2.0.0"""
        agent = CleanAgent()
        assert agent.version == "2.0.0"
        assert "2.0.0" in agent.version

    @pytest.mark.asyncio
    async def test_column_looks_like(self):
        """列名推断"""
        assert CleanAgent._looks_like("phone_number", "phone")
        assert CleanAgent._looks_like("手机号", "phone")
        assert CleanAgent._looks_like("email_addr", "email")
        assert CleanAgent._looks_like("created_date", "date")
        assert not CleanAgent._looks_like("amount", "phone")

    @pytest.mark.asyncio
    async def test_text_normalization(self):
        """文本标准化"""
        agent = CleanAgent()
        assert "ltd" in agent._normalize_text("Limited")
        assert "北京" in agent._normalize_text("北京市")
        assert "st" in agent._normalize_text("Street")

    @pytest.mark.asyncio
    async def test_phone_clean(self):
        """手机号清洗"""
        agent = CleanAgent()
        assert agent._normalize_phone("138-0000-1111") == "13800001111"
        assert agent._normalize_phone("137 0000 4444") == "13700004444"


class TestCleanAgentRegression:
    """回归：原有功能不受影响"""

    @pytest.mark.asyncio
    async def test_basic_clean_still_works(self, dirty_data):
        name, bronze = dirty_data
        agent = CleanAgent()
        result = await agent.run(
            source_name=name, bronze=bronze,
            dedup=True, outlier_method="iqr",
            semantic_clean=False, quality_score=False,
        )
        assert result.success
        assert result.data["report"]["rows_before"] == 7
        assert "dedup" in str(result.data["report"]["operations"])

    @pytest.mark.asyncio
    async def test_empty_source(self):
        agent = CleanAgent()
        result = await agent.run(source_name="ghost")
        assert not result.success
