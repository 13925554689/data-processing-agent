"""Test Clean Agent, Silver Layer, Integrate Agent, and Regulation Checker."""

import tempfile

import pytest

from src.agents.clean_agent import CleanAgent
from src.agents.integrate_agent import IntegrateAgent
from src.layers.bronze import BronzeLayer
from src.layers.silver import SilverLayer
from src.connectors.regulation_checker import RegulationChecker


# ── Fixtures ──

@pytest.fixture
def tmpdir():
    with tempfile.TemporaryDirectory() as td:
        yield td


@pytest.fixture
def bronze(tmpdir):
    return BronzeLayer(base_path=tmpdir)


@pytest.fixture
def silver(tmpdir):
    return SilverLayer(base_path=tmpdir)


@pytest.fixture
def dirty_data(bronze):
    bronze.ingest_records(
        "dirty_test",
        columns=["id", "name", "score", "city"],
        records=[
            [1, "Alice", 95.0, "Beijing"],
            [2, "Bob", None, "Shanghai"],
            [3, "Charlie", 85.0, "  Guangzhou "],
            [4, "Diana", 999.0, "Beijing"],
            [1, "Alice", 95.0, "Beijing"],
            [5, None, 70.0, None],
            [6, "Frank", -10.0, "Shenzhen"],
        ],
    )
    return ("dirty_test", bronze)


@pytest.fixture
def multi_source_data(bronze):
    bronze.ingest_records(
        "customers_a",
        columns=["cust_id", "name", "region"],
        records=[[1, "Alice", "North"], [2, "Bob", "South"], [3, "Eve", "East"]],
    )
    bronze.ingest_records(
        "customers_b",
        columns=["cust_id", "phone", "tier"],
        records=[[1, "111-1111", "Gold"], [2, "222-2222", "Silver"], [4, "444-4444", "Bronze"]],
    )
    return (["customers_a", "customers_b"], bronze)


# ── Clean Agent ──

class TestCleanAgent:
    @pytest.mark.asyncio
    async def test_clean_dirty_data(self, dirty_data):
        name, bronze = dirty_data
        agent = CleanAgent()
        result = await agent.run(
            source_name=name,
            bronze=bronze,
            strategy="auto",
            missing_strategy="fill_median",
            outlier_method="iqr",
            dedup=True,
            normalize=True,
        )
        assert result.success, result.error
        report = result.data["report"]
        assert report["rows_before"] == 7
        assert report["rows_after"] < 7

    @pytest.mark.asyncio
    async def test_profile_in_report(self, dirty_data):
        name, bronze = dirty_data
        agent = CleanAgent()
        result = await agent.run(
            source_name=name, bronze=bronze,
            strategy="auto", dedup=False, outlier_method="none",
        )
        assert result.success, result.error
        report = result.data["report"]
        assert "profile" in report

    @pytest.mark.asyncio
    async def test_no_source(self):
        agent = CleanAgent()
        result = await agent.run(source_name="nonexistent")
        assert not result.success

    @pytest.mark.asyncio
    async def test_empty_data(self, bronze):
        """空数据应当优雅处理"""
        agent = CleanAgent()
        result = await agent.run(source_name="empty", bronze=bronze)
        assert not result.success  # "No data found" is expected for truly empty

    @pytest.mark.asyncio
    async def test_types_inferred(self, dirty_data):
        name, bronze = dirty_data
        agent = CleanAgent()
        result = await agent.run(
            source_name=name, bronze=bronze,
            dedup=False, outlier_method="none",
        )
        assert result.success, result.error
        dtypes = result.data["report"]["profile"]["dtypes"]
        assert dtypes["id"] in ("integer", "float")  # DuckDB may return int as float
        assert "score" in dtypes


# ── Silver Layer ──

class TestSilverLayer:
    def test_write_and_read(self, silver):
        silver.write_table(
            "customer", "profiles",
            columns=["id", "name", "score"],
            rows=[[1, "Alice", 95.0], [2, "Bob", 87.0]],
        )
        rows = silver.read_latest("customer", "profiles")
        assert len(rows) == 2
        assert rows[0]["name"] == "Alice"

    def test_list_domains(self, silver):
        silver.write_table("sales", "orders", ["id"], [[1]])
        silver.write_table("hr", "staff", ["id"], [[1]])
        domains = silver.list_domains()
        assert "sales" in domains
        assert "hr" in domains

    def test_get_stats(self, silver):
        silver.write_table("d1", "t1", ["a"], [[1]])
        silver.write_table("d1", "t2", ["b"], [[2]])
        stats = silver.get_stats()
        assert stats["domains"] == 1
        assert stats["detail"]["d1"]["tables"] == 2

    def test_empty_write(self, silver):
        meta = silver.write_table("test", "empty", ["x"], [])
        assert meta["rows"] == 0

    def test_special_chars(self, silver):
        """列名含空格和横线时，Silver层使用安全列名（空格→_ 横线→_）"""
        meta = silver.write_table(
            "test", "special",
            columns=["na me", "val-ue"],
            rows=[["Alice", 100]],
        )
        assert meta["rows"] == 1
        rows = silver.read_latest("test", "special")
        # Silver层将列名中的空格和横线替换为下划线
        assert rows[0]["na_me"] == "Alice"

    def test_read_nonexistent(self, silver):
        with pytest.raises(FileNotFoundError):
            silver.read_latest("no", "nope")


# ── Integrate Agent ──

class TestIntegrateAgent:
    @pytest.mark.asyncio
    async def test_union(self, multi_source_data):
        sources, bronze = multi_source_data
        agent = IntegrateAgent()
        result = await agent.run(
            sources=[{"name": s} for s in sources],
            bronze=bronze,
            merge_strategy="union",
        )
        assert result.success, result.error
        assert result.data["merged_rows"] == 6

    @pytest.mark.asyncio
    async def test_join(self, multi_source_data):
        sources, bronze = multi_source_data
        agent = IntegrateAgent()
        result = await agent.run(
            sources=[{"name": s} for s in sources],
            bronze=bronze,
            merge_strategy="join",
            join_key="cust_id",
        )
        assert result.success, result.error
        assert result.data["merged_rows"] == 2

    @pytest.mark.asyncio
    async def test_dedup(self, multi_source_data):
        sources, bronze = multi_source_data
        agent = IntegrateAgent()
        result = await agent.run(
            sources=[{"name": s} for s in sources],
            bronze=bronze,
            merge_strategy="union",
            dedup_key=["cust_id"],
        )
        assert result.success, result.error
        assert result.data["merged_rows"] == 4

    @pytest.mark.asyncio
    async def test_single_source_error(self):
        agent = IntegrateAgent()
        result = await agent.run(sources=[{"name": "only_one"}], merge_strategy="union")
        assert not result.success

    @pytest.mark.asyncio
    async def test_source_not_found(self):
        agent = IntegrateAgent()
        result = await agent.run(
            sources=[{"name": "ghost_a"}, {"name": "ghost_b"}],
            merge_strategy="union",
        )
        assert not result.success


# ── Regulation Checker ──

class TestRegulationChecker:
    def test_init(self):
        checker = RegulationChecker()
        assert checker.base_url == "http://localhost:8200"

    def test_custom_url(self):
        checker = RegulationChecker(base_url="http://custom:9000")
        assert checker.base_url == "http://custom:9000"

    @pytest.mark.asyncio
    async def test_check_compliance_unknown_stage(self):
        checker = RegulationChecker()
        result = await checker.check_compliance("未知阶段")
        assert result["passed"] is True

    @pytest.mark.asyncio
    async def test_search_regulations_offline(self):
        checker = RegulationChecker()
        result = await checker.search_regulations("数据采集", "个人信息")
        assert result["total_hits"] == 0

    @pytest.mark.asyncio
    async def test_query_offline(self):
        checker = RegulationChecker()
        result = await checker._query("测试问题")
        assert "暂不可用" in result["answer"]
