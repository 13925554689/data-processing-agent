"""Test Clean Agent v2.1 — 脱敏/脱密处理"""

import tempfile
import pytest

from src.agents.clean_agent import CleanAgent
from src.layers.bronze import BronzeLayer


@pytest.fixture
def bronze():
    with tempfile.TemporaryDirectory() as td:
        yield BronzeLayer(base_path=td)


@pytest.fixture
def pii_data(bronze):
    """含身份证/手机/邮箱/银行卡/姓名的测试数据"""
    bronze.ingest_records(
        "pii_test",
        columns=["id", "name", "phone", "email", "id_card", "bank_card", "address"],
        records=[
            [1, "张三", "13800001111", "zhangsan@test.com",
             "110101199001011234", "6222021234567890123", "北京市朝阳区某某路100号"],
            [2, "李四", "13900002222", "lisi@test.com",
             "310101199202022345", "6228480012345678901", "上海市浦东新区某某大厦"],
            [3, "王五", "13600003333", "wangwu@test.com",
             "440101199303033456", "6217001234567890123", "广东省广州市天河区某街"],
        ],
    )
    return ("pii_test", bronze)


class TestDesensitize:
    @pytest.mark.asyncio
    async def test_mask_phone(self, pii_data):
        """手机号部分遮盖"""
        name, bronze = pii_data
        agent = CleanAgent()
        result = await agent.run(
            source_name=name, bronze=bronze,
            dedup=False, outlier_method="none",
            desensitize=True, desensitize_strategy="mask",
        )
        assert result.success, result.error
        data = result.data["cleaned_data"]
        # 手机号应被遮盖
        assert "****" in str(data[0][2])  # phone列

    @pytest.mark.asyncio
    async def test_mask_idcard(self, pii_data):
        """身份证号部分遮盖"""
        name, bronze = pii_data
        agent = CleanAgent()
        result = await agent.run(
            source_name=name, bronze=bronze,
            dedup=False, outlier_method="none",
            desensitize=True, desensitize_strategy="mask",
        )
        data = result.data["cleaned_data"]
        # 身份证应保留前4后4
        phone_val = str(data[0][4])
        assert "**********" in phone_val or "****" in phone_val

    @pytest.mark.asyncio
    async def test_mask_email(self, pii_data):
        """邮箱遮盖"""
        name, bronze = pii_data
        agent = CleanAgent()
        result = await agent.run(
            source_name=name, bronze=bronze,
            desensitize=True, desensitize_strategy="mask",
        )
        data = result.data["cleaned_data"]
        email = str(data[0][3])
        assert "***" in email or "@" not in email or "*" in email

    @pytest.mark.asyncio
    async def test_mask_name(self, pii_data):
        """姓名遮盖"""
        name, bronze = pii_data
        agent = CleanAgent()
        result = await agent.run(
            source_name=name, bronze=bronze,
            desensitize=True, desensitize_strategy="mask",
        )
        data = result.data["cleaned_data"]
        # 张三 → 张*
        assert "*" in str(data[0][1])

    @pytest.mark.asyncio
    async def test_mask_address(self, pii_data):
        """地址遮盖—保留省市"""
        name, bronze = pii_data
        agent = CleanAgent()
        result = await agent.run(
            source_name=name, bronze=bronze,
            desensitize=True, desensitize_strategy="mask",
        )
        data = result.data["cleaned_data"]
        addr = str(data[0][6])
        assert "北京市" in addr or "上海市" in addr or "广东省" in addr

    @pytest.mark.asyncio
    async def test_full_mask(self, pii_data):
        """全遮盖策略"""
        name, bronze = pii_data
        agent = CleanAgent()
        result = await agent.run(
            source_name=name, bronze=bronze,
            desensitize=True, desensitize_strategy="full",
        )
        data = result.data["cleaned_data"]
        assert str(data[0][2]) == "***"

    @pytest.mark.asyncio
    async def test_no_desensitize(self, pii_data):
        """关闭脱敏—原始数据保留"""
        name, bronze = pii_data
        agent = CleanAgent()
        result = await agent.run(
            source_name=name, bronze=bronze,
            desensitize=False,
        )
        data = result.data["cleaned_data"]
        # 手机号应保持不变
        assert "13800001111" == str(data[0][2])

    @pytest.mark.asyncio
    async def test_desensitize_report(self, pii_data):
        """脱敏操作日志"""
        name, bronze = pii_data
        agent = CleanAgent()
        result = await agent.run(
            source_name=name, bronze=bronze,
            desensitize=True,
        )
        ops = result.data["report"]["operations"]
        mask_ops = [o for o in ops if o.get("op") == "desensitize"]
        assert len(mask_ops) >= 1
        # 第一条是汇总
        assert "detected" in mask_ops[0]
        assert mask_ops[0]["detected"] >= 5  # name/phone/email/id_card/bank_card/address

    @pytest.mark.asyncio
    async def test_no_pii_data(self, bronze):
        """无PII数据—不报错"""
        bronze.ingest_records("clean_data", ["product", "price"], [["Widget", 99], ["Gadget", 199]])
        agent = CleanAgent()
        result = await agent.run(
            source_name="clean_data", bronze=bronze,
            desensitize=True,
        )
        assert result.success
        ops = result.data["report"]["operations"]
        # 应有desensitize操作但detected=0
        mask_ops = [o for o in ops if o.get("op") == "desensitize"]
        assert mask_ops[0]["detected"] == 0

    @pytest.mark.asyncio
    async def test_mixed_strategies(self, pii_data):
        """不同策略不互相干扰"""
        name, bronze = pii_data
        for strategy in ["mask", "full", "hash", "token"]:
            result = await CleanAgent().run(
                source_name=name, bronze=bronze,
                desensitize=True, desensitize_strategy=strategy,
            )
            assert result.success, f"Strategy {strategy} failed: {result.error}"
