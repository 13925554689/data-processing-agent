"""Test connectors module."""

import os
import tempfile

import pytest

from src.connectors.base import (
    BaseConnector,
    ColumnInfo,
    ConnectorFactory,
    FileConnector,
    QueryResult,
    TableInfo,
)


class TestColumnInfo:
    def test_defaults(self):
        c = ColumnInfo(name="id", dtype="INTEGER")
        assert c.name == "id"
        assert c.dtype == "INTEGER"
        assert c.nullable is True


class TestQueryResult:
    def test_basic(self):
        r = QueryResult(columns=["a", "b"], rows=[[1, 2], [3, 4]], row_count=2)
        assert r.columns == ["a", "b"]
        assert r.row_count == 2


class TestConnectorFactory:
    def test_available_types(self):
        types = ConnectorFactory.available_types()
        assert "csv" in types
        assert "parquet" in types

    def test_create_file_csv(self):
        conn = ConnectorFactory.create("test", "csv", {"path": "/tmp/test.csv"})
        assert isinstance(conn, FileConnector)
        assert conn.file_type == "csv"

    def test_create_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown connector type"):
            ConnectorFactory.create("test", "unknown_db", {"path": "/tmp"})


class TestFileConnector:
    @pytest.fixture
    def csv_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write("id,name,value\n1,Alice,100\n2,Bob,200\n3,Charlie,300\n")
            path = f.name
        yield path
        os.unlink(path)

    @pytest.mark.asyncio
    async def test_connect_csv(self, csv_file):
        conn = FileConnector("test", {"path": csv_file, "type": "csv"})
        await conn.connect()
        assert conn.is_connected
        await conn.close()

    @pytest.mark.asyncio
    async def test_connect_nonexistent(self):
        conn = FileConnector("test", {"path": "/nonexistent/file.csv", "type": "csv"})
        with pytest.raises(FileNotFoundError):
            await conn.connect()

    @pytest.mark.asyncio
    async def test_execute_query_csv(self, csv_file):
        conn = FileConnector("test", {"path": csv_file, "type": "csv"})
        await conn.connect()
        result = await conn.execute("SELECT * FROM _src")
        assert result.row_count == 3
        assert result.columns == ["id", "name", "value"]
        await conn.close()

    @pytest.mark.asyncio
    async def test_execute_filtered_csv(self, csv_file):
        conn = FileConnector("test", {"path": csv_file, "type": "csv"})
        await conn.connect()
        result = await conn.execute("SELECT * FROM _src WHERE value > 150")
        assert result.row_count == 2
        await conn.close()

    @pytest.mark.asyncio
    async def test_get_table_info_csv(self, csv_file):
        conn = FileConnector("test", {"path": csv_file, "type": "csv"})
        await conn.connect()
        info = await conn.get_table_info(csv_file)
        assert info.row_count == 3
        assert len(info.columns) == 3
        await conn.close()

    @pytest.mark.asyncio
    async def test_async_context_manager(self, csv_file):
        async with FileConnector("test", {"path": csv_file, "type": "csv"}) as conn:
            result = await conn.execute("SELECT COUNT(*) AS cnt FROM _src")
            assert result.rows[0][0] == 3
