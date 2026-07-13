"""Test Phase 4-7: Govern/Analyze/Asset/Plan Agents + Gold Layer + API."""

import tempfile

import pytest

from src.agents.govern_agent import GovernAgent
from src.agents.analyze_agent import AnalyzeAgent
from src.agents.asset_agent import AssetAgent
from src.agents.plan_agent import PlanAgent
from src.layers.bronze import BronzeLayer
from src.layers.gold import GoldLayer


# ── Fixtures ──

@pytest.fixture
def tmpdir():
    with tempfile.TemporaryDirectory() as td:
        yield td


@pytest.fixture
def bronze(tmpdir):
    return BronzeLayer(base_path=tmpdir)


@pytest.fixture
def gold_layer(tmpdir):
    return GoldLayer(base_path=tmpdir)


@pytest.fixture
def sample_data(bronze):
    bronze.ingest_records(
        "test_customers",
        columns=["id", "name", "phone", "age", "city"],
        records=[
            [1, "Alice", "13800001111", 30, "Beijing"],
            [2, "Bob", "13900002222", 25, "Shanghai"],
            [3, "Charlie", None, 35, "Beijing"],
            [4, "Diana", "13600004444", None, "Guangzhou"],
            [5, "Eve", "13700005555", 28, "Beijing"],
        ],
    )
    raw = bronze.read_latest("test_customers")
    columns = list(raw[0].keys())
    rows = [list(r.values()) for r in raw]
    return ("test_customers", columns, rows, bronze)


# ── Govern Agent ──

class TestGovernAgent:
    @pytest.mark.asyncio
    async def test_catalog(self, sample_data):
        name, cols, rows, _ = sample_data
        agent = GovernAgent()
        result = await agent.run(action="catalog", source_name=name, columns=cols, sample_rows=rows)
        assert result.success
        assert result.data["summary"]["columns"] == 5
        assert len(result.data["fields"]) == 5

    @pytest.mark.asyncio
    async def test_classify(self, sample_data):
        _, cols, rows, _ = sample_data
        agent = GovernAgent()
        result = await agent.run(action="classify", columns=cols, sample_rows=rows)
        assert result.success
        assert "high_risk" in result.data
        # phone column should be detected
        high = result.data["high_risk"] + result.data["medium_risk"]
        assert len(high) > 0

    @pytest.mark.asyncio
    async def test_audit(self, sample_data):
        name, cols, rows, _ = sample_data
        agent = GovernAgent()
        result = await agent.run(action="audit", source_name=name, columns=cols, sample_rows=rows)
        assert result.success
        assert "passed" in result.data


# ── Gold Layer ──

class TestGoldLayer:
    def test_write_and_read(self, gold_layer):
        gold_layer.write_aggregate("kpi_summary", ["metric", "value"], [["revenue", 1000], ["cost", 600]])
        datasets = gold_layer.list_datasets()
        assert "kpi_summary" in datasets

        rows = gold_layer.read("kpi_summary")
        assert len(rows) == 2

    def test_empty(self, gold_layer):
        meta = gold_layer.write_aggregate("empty", ["a"], [])
        assert meta["rows"] == 0

    def test_list_empty(self, gold_layer):
        datasets = gold_layer.list_datasets()
        assert isinstance(datasets, list)


# ── Analyze Agent ──

class TestAnalyzeAgent:
    @pytest.mark.asyncio
    async def test_summary(self, sample_data):
        name, cols, rows, bronze = sample_data
        agent = AnalyzeAgent()
        result = await agent.run(source=f"bronze:{name}", bronze=bronze, analysis="summary")
        assert result.success
        assert result.data["row_count"] == 5

    @pytest.mark.asyncio
    async def test_distribution(self, sample_data):
        name, cols, rows, bronze = sample_data
        agent = AnalyzeAgent()
        result = await agent.run(source=f"bronze:{name}", bronze=bronze,
                                  analysis="distribution", column="city")
        assert result.success
        assert "top_values" in result.data

    @pytest.mark.asyncio
    async def test_top_n(self, sample_data):
        name, cols, rows, bronze = sample_data
        agent = AnalyzeAgent()
        result = await agent.run(source=f"bronze:{name}", bronze=bronze,
                                  analysis="top_n", column="city", top_n=3)
        assert result.success


# ── Asset Agent ──

class TestAssetAgent:
    @pytest.mark.asyncio
    async def test_valuate_offline(self):
        """DRAP 不在线时优雅降级"""
        agent = AssetAgent()
        result = await agent.run(
            action="valuate",
            source_name="test_asset",
            asset_data={"rows": 100, "columns": 10},
            valuation_method="bsc",
        )
        # 应当返回 result（可能失败但不应崩溃）
        assert result.data is not None

    @pytest.mark.asyncio
    async def test_full_pipeline_offline(self):
        agent = AssetAgent()
        result = await agent.run(
            action="full",
            source_name="test_full",
            asset_data={"rows": 500, "columns": 20},
        )
        assert result.data is not None


# ── Plan Agent ──

class TestPlanAgent:
    @pytest.mark.asyncio
    async def test_ingest_intent(self):
        agent = PlanAgent()
        result = await agent.run(intent="采集CSV数据")
        assert result.success
        steps = [s["agent"] for s in result.data["steps"]]
        assert "ingest" in steps

    @pytest.mark.asyncio
    async def test_full_pipeline_intent(self):
        agent = PlanAgent()
        result = await agent.run(intent="数据全流程处理：采集、清洗、估值入表")
        steps = [s["agent"] for s in result.data["steps"]]
        assert "ingest" in steps
        assert "clean" in steps
        assert "asset" in steps

    @pytest.mark.asyncio
    async def test_compliance_intent(self):
        agent = PlanAgent()
        result = await agent.run(intent="检查数据采集合规性")
        assert result.success

    @pytest.mark.asyncio
    async def test_default_pipeline(self):
        agent = PlanAgent()
        result = await agent.run(intent="处理一下这个数据")
        assert result.success
        assert len(result.data["steps"]) >= 2
