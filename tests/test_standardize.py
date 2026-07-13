"""Test Standardize Agent — 数据分类分级+格式标准化+国标对齐"""

import tempfile
import pytest

from src.agents.standardize_agent import StandardizeAgent
from src.layers.bronze import BronzeLayer


@pytest.fixture
def bronze():
    with tempfile.TemporaryDirectory() as td:
        yield BronzeLayer(base_path=td)


@pytest.fixture
def enterprise_data(bronze):
    """企业ERP数据"""
    bronze.ingest_records(
        "erp_standard",
        columns=["order_id", "customer", "phone", "amount", "create_date", "status"],
        records=[
            ["ORD001", "甲公司", "13800001111", 15000, "2026-01-15", "已完成"],
            ["ORD002", "乙公司", "139-0000-2222", 28000, "2026/02/20", "processing"],
            ["ORD003", "丙公司", "13600003333", 9500, "2025-12-01", "已完成"],
        ],
    )
    return ("erp_standard", bronze)


class TestStandardizeAgent:
    @pytest.mark.asyncio
    async def test_full_standardize(self, enterprise_data):
        """全流程: 分类+分级+格式+DCMM"""
        name, bronze = enterprise_data
        agent = StandardizeAgent()
        result = await agent.run(source_name=name, bronze=bronze, action="full")
        assert result.success, result.error
        assert "classification" in result.data
        assert "grading" in result.data
        assert "dcmm_score" in result.data

    @pytest.mark.asyncio
    async def test_classify_industry_inference(self, enterprise_data):
        """行业推断"""
        name, bronze = enterprise_data
        agent = StandardizeAgent()
        result = await agent.run(source_name=name, bronze=bronze, action="classify")
        c = result.data["classification"]
        assert "industry" in c
        assert "field_classes" in c
        assert len(c["field_classes"]) == 6

    @pytest.mark.asyncio
    async def test_classify_with_hint(self, enterprise_data):
        """指定行业"""
        name, bronze = enterprise_data
        agent = StandardizeAgent()
        result = await agent.run(source_name=name, bronze=bronze,
                                  action="classify", industry_hint="金融")
        assert result.data["classification"]["industry"] == "金融"

    @pytest.mark.asyncio
    async def test_grading(self, enterprise_data):
        """三级分级"""
        name, bronze = enterprise_data
        agent = StandardizeAgent()
        result = await agent.run(source_name=name, bronze=bronze, action="grade")
        g = result.data["grading"]
        assert "overall_grade" in g
        assert "field_grades" in g
        assert g["reference"] == "GB/T 43697-2024"
        # phone列应被检测为"重要数据"
        assert g["field_grades"].get("phone", "") == "重要数据"

    @pytest.mark.asyncio
    async def test_format_check(self, enterprise_data):
        """格式检查"""
        name, bronze = enterprise_data
        agent = StandardizeAgent()
        result = await agent.run(source_name=name, bronze=bronze, action="format")
        fmt = result.data.get("format_issues", [])
        assert isinstance(fmt, list)

    @pytest.mark.asyncio
    async def test_dcmm_level_output(self, enterprise_data):
        """DCMM等级"""
        name, bronze = enterprise_data
        agent = StandardizeAgent()
        result = await agent.run(source_name=name, bronze=bronze, action="full")
        score = result.data["dcmm_score"]
        assert 0 <= score["score"] <= 100
        assert "级" in score["level"]

    @pytest.mark.asyncio
    async def test_no_source(self):
        agent = StandardizeAgent()
        result = await agent.run(source_name="ghost")
        assert not result.success

    @pytest.mark.asyncio
    async def test_field_classification(self, enterprise_data):
        """字段业务分类"""
        name, bronze = enterprise_data
        agent = StandardizeAgent()
        result = await agent.run(source_name=name, bronze=bronze, action="classify")
        fc = result.data["classification"]["field_classes"]
        assert fc["order_id"] == "标识类"
        assert fc["amount"] == "金额类"
        assert fc["create_date"] == "时间类"

    @pytest.mark.asyncio
    async def test_industry_standard_reference(self, enterprise_data):
        """行业标准引用"""
        name, bronze = enterprise_data
        agent = StandardizeAgent()
        result = await agent.run(source_name=name, bronze=bronze,
                                  action="classify", industry_hint="金融")
        ref = result.data["classification"]["industry_standard"]
        assert "JR/T" in ref or "GB/T" in ref

    @pytest.mark.asyncio
    async def test_version(self):
        agent = StandardizeAgent()
        assert agent.version == "1.0.0"
        assert "GB/T 43697" in agent.description
