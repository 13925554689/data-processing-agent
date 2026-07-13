"""
存储层测试 — Bronze / Silver / Gold Layer

覆盖: 写入/读取/统计 + SQL注入防护 + 路径遍历防护
"""

import os
import tempfile
import pytest

from src.layers.bronze import BronzeLayer
from src.layers.silver import SilverLayer
from src.layers.gold import GoldLayer


@pytest.fixture
def tmp_base(tmp_path):
    return str(tmp_path / "medallion")


class TestBronzeLayer:
    def test_ingest_records(self, tmp_base):
        bronze = BronzeLayer(base_path=tmp_base)
        result = bronze.ingest_records(
            source_name="test_source",
            columns=["id", "name", "value"],
            records=[[1, "Alice", 100.0], [2, "Bob", 200.0]],
        )
        assert result["rows"] == 2
        assert result["columns"] == 3
        assert result["source"] == "test_source"
        assert os.path.exists(result["path"])

    def test_ingest_empty_records(self, tmp_base):
        bronze = BronzeLayer(base_path=tmp_base)
        result = bronze.ingest_records("empty", ["col"], [])
        assert result["rows"] == 0

    def test_list_sources(self, tmp_base):
        bronze = BronzeLayer(base_path=tmp_base)
        bronze.ingest_records("src1", ["x"], [[1]])
        bronze.ingest_records("src2", ["y"], [[2]])
        sources = bronze.list_sources()
        assert "src1" in sources
        assert "src2" in sources

    def test_read_latest(self, tmp_base):
        bronze = BronzeLayer(base_path=tmp_base)
        bronze.ingest_records("read_test", ["id", "val"], [[10, "hello"]])
        data = bronze.read_latest("read_test")
        assert len(data) == 1
        assert data[0]["id"] == 10

    def test_read_nonexistent_source(self, tmp_base):
        bronze = BronzeLayer(base_path=tmp_base)
        with pytest.raises(FileNotFoundError):
            bronze.read_latest("nonexistent")

    def test_get_stats(self, tmp_base):
        bronze = BronzeLayer(base_path=tmp_base)
        bronze.ingest_records("stat_src", ["a"], [[1], [2]])
        stats = bronze.get_stats()
        assert stats["sources"] >= 1


class TestSilverLayer:
    def test_write_table(self, tmp_base):
        silver = SilverLayer(base_path=tmp_base)
        result = silver.write_table(
            domain="customer",
            table_name="orders",
            columns=["order_id", "amount"],
            rows=[[1, 99.9], [2, 199.9]],
        )
        assert result["rows"] == 2
        assert result["domain"] == "customer"

    def test_write_empty_table(self, tmp_base):
        silver = SilverLayer(base_path=tmp_base)
        result = silver.write_table("test", "empty", ["col"], [])
        assert result["rows"] == 0

    def test_read_latest(self, tmp_base):
        silver = SilverLayer(base_path=tmp_base)
        silver.write_table("finance", "transactions", ["tx_id", "val"], [[100, 50.0]])
        data = silver.read_latest("finance", "transactions")
        assert len(data) == 1

    def test_list_domains(self, tmp_base):
        silver = SilverLayer(base_path=tmp_base)
        silver.write_table("domain_a", "t1", ["x"], [[1]])
        silver.write_table("domain_b", "t2", ["y"], [[2]])
        domains = silver.list_domains()
        assert "domain_a" in domains
        assert "domain_b" in domains


class TestGoldLayer:
    def test_write_aggregate(self, tmp_base):
        gold = GoldLayer(base_path=tmp_base)
        result = gold.write_aggregate(
            name="sales_summary",
            columns=["region", "total"],
            rows=[["East", 1000], ["West", 2000]],
        )
        assert result["rows"] == 2

    def test_write_empty_aggregate(self, tmp_base):
        gold = GoldLayer(base_path=tmp_base)
        result = gold.write_aggregate("empty", ["col"], [])
        assert result["rows"] == 0

    def test_list_datasets(self, tmp_base):
        gold = GoldLayer(base_path=tmp_base)
        gold.write_aggregate("test_ds", ["a"], [[1]])
        datasets = gold.list_datasets()
        assert "test_ds" in datasets

    def test_aggregate_from_sql_rejects_non_select(self, tmp_base):
        gold = GoldLayer(base_path=tmp_base)
        with pytest.raises(ValueError):
            gold.aggregate_from_sql("DROP TABLE _t", "bad")

    def test_aggregate_from_sql_rejects_dangerous(self, tmp_base):
        gold = GoldLayer(base_path=tmp_base)
        with pytest.raises(ValueError, match="dangerous"):
            gold.aggregate_from_sql("SELECT * FROM _t; DROP TABLE _t", "bad")

    def test_read_rejects_path_traversal(self, tmp_base):
        gold = GoldLayer(base_path=tmp_base)
        with pytest.raises(ValueError, match="Invalid name pattern"):
            gold.read("../../etc")