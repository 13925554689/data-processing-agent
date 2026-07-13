"""
数据血缘追踪测试
"""

import pytest
import tempfile

from src.lineage.tracker import DataLineage


@pytest.fixture
def lineage(tmp_path):
    return DataLineage(storage_path=str(tmp_path / "lineage"))


class TestDataLineage:
    def test_track_table(self, lineage):
        edge_id = lineage.track_table("bronze:src", "silver:cleaned", "clean", agent="clean")
        assert edge_id
        assert len(lineage._graph["edges"]) == 1

    def test_track_column(self, lineage):
        edge_id = lineage.track_column("bronze:src", "name", "silver:cleaned", "name", "direct")
        assert edge_id
        assert len(lineage._graph["column_edges"]) == 1

    def test_upstream(self, lineage):
        lineage.track_table("bronze:src", "silver:cleaned", "clean")
        ups = lineage.upstream("silver:cleaned")
        assert len(ups) == 1
        assert ups[0]["source"] == "bronze:src"

    def test_downstream(self, lineage):
        lineage.track_table("bronze:src", "silver:cleaned", "clean")
        downs = lineage.downstream("bronze:src")
        assert len(downs) == 1
        assert downs[0]["target"] == "silver:cleaned"

    def test_full_path(self, lineage):
        lineage.track_table("bronze:a", "silver:b", "clean")
        lineage.track_table("silver:b", "gold:c", "aggregate")
        path = lineage.full_path("gold:c")
        assert path == ["bronze:a", "silver:b", "gold:c"]

    def test_to_mermaid(self, lineage):
        lineage.track_table("bronze:src", "silver:dst", "clean")
        mermaid = lineage.to_mermaid()
        assert "graph LR" in mermaid
        assert "bronze_src" in mermaid

    def test_stats(self, lineage):
        lineage.track_table("a", "b", "ingest")
        stats = lineage.stats()
        assert stats["tables"] == 2
        assert stats["table_edges"] == 1

    def test_track_columns_batch(self, lineage):
        ids = lineage.track_columns_batch(
            "bronze:src", ["col1", "col2"],
            "silver:dst", ["col1", "col2"],
            "direct",
        )
        assert len(ids) == 2

    def test_column_upstream(self, lineage):
        lineage.track_column("bronze:src", "name", "silver:dst", "full_name", "rename")
        ups = lineage.column_upstream("silver:dst", "full_name")
        assert len(ups) == 1
        assert ups[0]["transform"] == "rename"