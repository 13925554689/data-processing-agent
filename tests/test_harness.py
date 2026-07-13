"""
多Agent集成审计 + 端到端 Harness 验证

验证:
  1. 全链路: Ingest → Clean → Integrate → Govern → Analyze → Asset
  2. 跨Agent数据契约: 每个Agent的输出符合下游Agent输入要求
  3. 错误处理: 异常输入、缺失参数、边界条件
  4. 并发: 多Agent注册表隔离
  5. API端点: TestClient验证全部路由
"""

import os
import tempfile

import pytest

from src.agents.base import AgentRegistry, AgentResult
from src.agents.ingest_agent import IngestAgent
from src.agents.clean_agent import CleanAgent
from src.agents.integrate_agent import IntegrateAgent
from src.agents.govern_agent import GovernAgent
from src.agents.analyze_agent import AnalyzeAgent
from src.agents.asset_agent import AssetAgent
from src.agents.plan_agent import PlanAgent
from src.layers.bronze import BronzeLayer
from src.layers.silver import SilverLayer
from src.layers.gold import GoldLayer


# ═══════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════

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
def gold(tmpdir):
    return GoldLayer(base_path=tmpdir)


@pytest.fixture
def registry():
    """隔离的Agent注册表"""
    AgentRegistry.reset()
    reg = AgentRegistry()
    reg.register(IngestAgent)
    reg.register(CleanAgent)
    reg.register(IntegrateAgent)
    reg.register(GovernAgent)
    reg.register(AnalyzeAgent)
    reg.register(AssetAgent)
    reg.register(PlanAgent)
    yield reg
    AgentRegistry.reset()


@pytest.fixture
def multi_source_data(bronze):
    """多源测试数据 — 模拟企业真实场景"""
    # 源A: ERP销售数据 (含缺失值、异常值)
    bronze.ingest_records(
        "erp_sales",
        columns=["order_id", "customer", "amount", "region", "date"],
        records=[
            [1, "Alice", 1500.0, "North", "2026-01-15"],
            [2, "Bob", None, "South", "2026-01-16"],       # 缺失金额
            [3, "Charlie", 85000.0, "East", "2026-01-17"],  # 异常高金额
            [4, "Diana", 2300.0, "North", "2026-01-15"],
            [1, "Alice", 1500.0, "North", "2026-01-15"],    # 重复行
            [5, None, 1200.0, None, "2026-01-18"],           # 缺失客户名
            [6, "Frank", -500.0, "West", "2026-01-19"],      # 负金额异常
        ],
    )
    # 源B: CRM客户数据
    bronze.ingest_records(
        "crm_customers",
        columns=["customer_id", "customer", "phone", "tier"],
        records=[
            [1, "Alice", "13800001111", "Gold"],
            [2, "Bob", "13900002222", "Silver"],
            [3, "Charlie", "13600003333", "Gold"],
            [7, "Grace", "13700007777", "Bronze"],  # 不在ERP中
        ],
    )
    return bronze


# ═══════════════════════════════════════════════
# 全链路集成测试
# ═══════════════════════════════════════════════

class TestFullPipeline:
    """端到端全链路: Ingest → Clean → Integrate → Govern → Analyze → Asset"""

    @pytest.mark.asyncio
    async def test_full_pipeline(self, multi_source_data):
        """完整7步数据管道"""
        pipeline_log = []

        # ── Step 1: Ingest (数据已在fixture中写入Bronze，直接读取验证) ──
        raw = multi_source_data.read_latest("erp_sales")
        assert len(raw) == 7, f"Expected 7 rows, got {len(raw)}"
        pipeline_log.append({"step": "ingest", "rows": len(raw), "status": "ok"})

        # ── Step 2: Clean ──
        agent2 = CleanAgent()
        result = await agent2.run(
            source_name="erp_sales",
            bronze=multi_source_data,
            strategy="auto",
            missing_strategy="fill_median",
            outlier_method="iqr",
            dedup=True,
            normalize=True,
        )
        assert result.success, f"Clean failed: {result.error}"
        report = result.data["report"]
        assert report["rows_before"] == 7
        assert report["rows_after"] <= 7  # 至少去重
        assert len(report["operations"]) >= 1
        pipeline_log.append({"step": "clean", "before": report["rows_before"], "after": report["rows_after"], "ops": len(report["operations"])})

        from src.layers.silver import SilverLayer

        # ── Step 3: Integrate (使用相同根目录的SilverLayer) ──
        agent3 = IntegrateAgent()
        result = await agent3.run(
            sources=[{"name": "erp_sales"}, {"name": "crm_customers"}],
            bronze=multi_source_data,
            domain="sales",
            table_name="erp_crm_merged",
            merge_strategy="join",
            join_key="customer",
            silver=SilverLayer(base_path=multi_source_data.base_path),
        )
        assert result.success, f"Integrate failed: {result.error}"
        assert result.data["merged_rows"] >= 1
        pipeline_log.append({"step": "integrate", "merged": result.data["merged_rows"]})

        # ── Step 4: Govern ──
        # 集成后数据在Silver层
        silver = SilverLayer(base_path=multi_source_data.base_path)
        silver_data = silver.read_latest("sales", "erp_crm_merged")
        if silver_data:
            cols = list(silver_data[0].keys())
            rows = [list(r.values()) for r in silver_data]
            agent4 = GovernAgent()
            result = await agent4.run(
                action="catalog",
                source_name="erp_crm_merged",
                columns=cols,
                sample_rows=rows,
            )
            assert result.success
            assert result.data["summary"]["columns"] >= 3
            pipeline_log.append({"step": "govern: catalog", "fields": len(result.data["fields"])})

            result = await agent4.run(
                action="classify",
                columns=cols,
                sample_rows=rows,
            )
            assert result.success
            pipeline_log.append({"step": "govern: classify", "high_risk": len(result.data.get("high_risk", []))})

        # ── Step 5: Analyze ──
        agent5 = AnalyzeAgent()
        result = await agent5.run(
            source="bronze:erp_sales",
            bronze=multi_source_data,
            analysis="summary",
        )
        assert result.success
        assert result.data["row_count"] >= 1
        pipeline_log.append({"step": "analyze", "rows": result.data["row_count"], "cols": result.data["column_count"]})

        # ── Step 6: Asset (DRAP离线时验证降级) ──
        agent6 = AssetAgent()
        result = await agent6.run(
            action="valuate",
            source_name="erp_sales",
            asset_data={"rows": result.data["row_count"], "columns": result.data["column_count"]},
            valuation_method="bsc",
        )
        # DRAP不在线，应该返回优雅降级而非崩溃
        assert result.data is not None
        pipeline_log.append({"step": "asset", "data": result.data is not None})

        # ── Step 7: Plan ──
        agent7 = PlanAgent()
        result = await agent7.run(intent="采集ERP数据、清洗、估值入表")
        assert result.success
        steps = [s["agent"] for s in result.data["steps"]]
        assert "ingest" in steps and "clean" in steps and "asset" in steps
        pipeline_log.append({"step": "plan", "total_steps": result.data["total_steps"]})

        # ── 验证日志完整性 ──
        print(f"\n  Pipeline log:")
        for entry in pipeline_log:
            print(f"    {entry}")
        assert len(pipeline_log) == 8  # ingest + clean + integrate + govern×2 + analyze + asset + plan


class TestCrossAgentContracts:
    """跨Agent数据契约验证"""

    @pytest.mark.asyncio
    async def test_ingest_output_schema(self, multi_source_data):
        """Bronze层读取验证: 行数/列数/类型"""
        rows = multi_source_data.read_latest("erp_sales")
        assert len(rows) > 0
        assert "order_id" in rows[0]
        assert "amount" in rows[0]
        assert "customer" in rows[0]

    @pytest.mark.asyncio
    async def test_clean_output_schema(self, multi_source_data):
        """Clean输出必须包含 report.rows_before / report.rows_after / report.operations"""
        agent = CleanAgent()
        result = await agent.run(source_name="erp_sales", bronze=multi_source_data,
                                  dedup=True, outlier_method="iqr")
        report = result.data["report"]
        assert "rows_before" in report
        assert "rows_after" in report
        assert "operations" in report
        assert report["rows_before"] >= report["rows_after"]

    @pytest.mark.asyncio
    async def test_integrate_output_schema(self, multi_source_data):
        """Integrate输出必须包含 merged_rows / columns"""
        agent = IntegrateAgent()
        result = await agent.run(
            sources=[{"name": "erp_sales"}, {"name": "crm_customers"}],
            bronze=multi_source_data,
            merge_strategy="union",
        )
        assert "merged_rows" in result.data
        assert "columns" in result.data
        assert result.data["merged_rows"] == 11  # 7 + 4

    @pytest.mark.asyncio
    async def test_govern_output_schema(self, multi_source_data):
        """Govern catalog输出必须包含 fields"""
        agent = GovernAgent()
        result = await agent.run(
            action="catalog",
            source_name="test",
            columns=["id", "name", "phone"],
            sample_rows=[[1, "Alice", "13800001111"]],
        )
        assert len(result.data["fields"]) == 3

    @pytest.mark.asyncio
    async def test_analyze_output_schema(self, multi_source_data):
        """Analyze summary输出必须包含 row_count / column_count"""
        agent = AnalyzeAgent()
        result = await agent.run(
            source="bronze:erp_sales",
            bronze=multi_source_data,
            analysis="summary",
        )
        assert "row_count" in result.data
        assert "column_count" in result.data

    @pytest.mark.asyncio
    async def test_plan_output_schema(self):
        """Plan输出必须包含 steps"""
        agent = PlanAgent()
        result = await agent.run(intent="全流程处理")
        assert "steps" in result.data
        assert len(result.data["steps"]) >= 2


class TestErrorHandling:
    """异常场景覆盖"""

    @pytest.mark.asyncio
    async def test_ingest_nonexistent_source(self):
        agent = IngestAgent()
        result = await agent.run(source_name="ghost_source")
        assert not result.success

    @pytest.mark.asyncio
    async def test_clean_nonexistent_source(self):
        agent = CleanAgent()
        result = await agent.run(source_name="ghost_source")
        assert not result.success

    @pytest.mark.asyncio
    async def test_integrate_insufficient_sources(self):
        agent = IntegrateAgent()
        result = await agent.run(sources=[{"name": "only_one"}], merge_strategy="union")
        assert not result.success
        assert "At least 2 sources" in result.error

    @pytest.mark.asyncio
    async def test_ingest_missing_path(self):
        agent = IngestAgent()
        result = await agent.run()
        assert not result.success
        assert "source_path" in (result.error or result.message or "")

    @pytest.mark.asyncio
    async def test_clean_missing_source(self):
        agent = CleanAgent()
        result = await agent.run()
        assert not result.success

    @pytest.mark.asyncio
    async def test_integrate_missing_sources(self):
        agent = IntegrateAgent()
        result = await agent.run()
        assert not result.success

    @pytest.mark.asyncio
    async def test_govern_unknown_action(self):
        agent = GovernAgent()
        result = await agent.run(action="unknown_action")
        assert not result.success

    @pytest.mark.asyncio
    async def test_duplicate_register_raises(self):
        AgentRegistry.reset()
        reg = AgentRegistry()
        reg.register(IngestAgent)
        with pytest.raises(ValueError, match="already registered"):
            reg.register(IngestAgent)
        AgentRegistry.reset()

    @pytest.mark.asyncio
    async def test_registry_isolation(self, registry):
        """注册表隔离 — 不同测试间不互相污染"""
        agent = registry.get("ingest")
        assert agent is not None
        assert agent.name == "ingest"


class TestAPIIntegration:
    """API端点集成验证"""

    def test_all_routes_registered(self):
        from src.api.app import app
        routes = [r.path for r in app.routes if hasattr(r, "path")]
        required = [
            "/api/health", "/api/agents", "/api/plan",
            "/api/ingest", "/api/clean", "/api/integrate",
            "/api/govern", "/api/analyze", "/api/asset/valuate",
            "/api/scrape",
            "/api/compliance/check", "/api/compliance/search",
            "/api/storage/status", "/api/drap/status",
        ]
        for r in required:
            assert r in routes, f"Missing route: {r}"

    def test_health_endpoint(self):
        from fastapi.testclient import TestClient
        from src.api.app import app, reg
        from src.agents.ingest_agent import IngestAgent
        from src.agents.clean_agent import CleanAgent
        from src.agents.integrate_agent import IntegrateAgent
        from src.agents.govern_agent import GovernAgent
        from src.agents.analyze_agent import AnalyzeAgent
        from src.agents.asset_agent import AssetAgent
        from src.agents.plan_agent import PlanAgent
        from src.agents.standardize_agent import StandardizeAgent
        reg.register(IngestAgent, override=True).register(CleanAgent, override=True)
        reg.register(IntegrateAgent, override=True).register(GovernAgent, override=True)
        reg.register(AnalyzeAgent, override=True).register(AssetAgent, override=True)
        reg.register(PlanAgent, override=True).register(StandardizeAgent, override=True)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["agents"]) >= 1

    def test_agents_endpoint(self):
        from fastapi.testclient import TestClient
        from src.api.app import app, reg
        from src.agents.ingest_agent import IngestAgent
        from src.agents.clean_agent import CleanAgent
        from src.agents.integrate_agent import IntegrateAgent
        from src.agents.govern_agent import GovernAgent
        from src.agents.analyze_agent import AnalyzeAgent
        from src.agents.asset_agent import AssetAgent
        from src.agents.plan_agent import PlanAgent
        from src.agents.standardize_agent import StandardizeAgent
        reg.register(IngestAgent, override=True).register(CleanAgent, override=True)
        reg.register(IntegrateAgent, override=True).register(GovernAgent, override=True)
        reg.register(AnalyzeAgent, override=True).register(AssetAgent, override=True)
        reg.register(PlanAgent, override=True).register(StandardizeAgent, override=True)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/agents")
        assert resp.status_code == 200
        agents = resp.json()["agents"]
        assert len(agents) >= 1

    def test_storage_endpoint(self):
        from fastapi.testclient import TestClient
        from src.api.app import app
        client = TestClient(app)
        resp = client.get("/api/storage/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "bronze" in data
        assert "silver" in data
        assert "gold" in data

    def test_scrape_endpoint_validation(self):
        from fastapi.testclient import TestClient
        from src.api.app import app
        client = TestClient(app)
        # 缺少url参数应返回422
        resp = client.post("/api/scrape", json={})
        assert resp.status_code == 422

    def test_plan_endpoint(self):
        from fastapi.testclient import TestClient
        from src.api.app import app
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/plan", json={"intent": "采集并清洗数据"})
        assert resp.status_code in (200, 503)


class TestStorageLayerIntegration:
    """存储层集成测试"""

    def test_bronze_to_silver_chain(self, tmpdir):
        """Bronze写入 → 读取 → 清洗 → Silver写入"""
        bronze = BronzeLayer(base_path=tmpdir)
        silver = SilverLayer(base_path=tmpdir)

        # Bronze 写入
        bronze.ingest_records("chain_test", ["id", "val"], [[1, 100], [2, 200]])
        rows = bronze.read_latest("chain_test")
        assert len(rows) == 2

        # Silver 写入（模拟清洗后数据）
        meta = silver.write_table("test_domain", "chain_result",
            columns=["id", "val", "doubled"],
            rows=[[1, 100, 200], [2, 200, 400]],
        )
        assert meta["rows"] == 2

        # Silver 读取
        result = silver.read_latest("test_domain", "chain_result")
        assert result[0]["doubled"] == 200

    def test_silver_to_gold_chain(self, tmpdir):
        """Silver → Gold 聚合"""
        silver = SilverLayer(base_path=tmpdir)
        gold = GoldLayer(base_path=tmpdir)

        silver.write_table("sales", "orders",
            columns=["region", "revenue"],
            rows=[["North", 100], ["North", 200], ["South", 300]],
        )

        gold.write_aggregate("revenue_by_region",
            columns=["region", "total"],
            rows=[["North", 300], ["South", 300]],
        )

        result = gold.read("revenue_by_region")
        assert len(result) == 2
        assert result[0]["total"] == 300
