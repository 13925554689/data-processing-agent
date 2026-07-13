"""Test Bronze layer, extended connectors, and Ingest Agent — DuckDB-based."""

import os
import tempfile

import pytest

from src.layers.bronze import BronzeLayer
from src.connectors.extended import SQLiteConnector, DrapConnector
from src.connectors.base import ConnectorFactory


# ── Fixtures ──

@pytest.fixture
def bronze():
    with tempfile.TemporaryDirectory() as td:
        yield BronzeLayer(base_path=td)


@pytest.fixture
def sqlite_db():
    import sqlite3
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)")
    con.execute("INSERT INTO users VALUES (1, 'Alice', 30)")
    con.execute("INSERT INTO users VALUES (2, 'Bob', 25)")
    con.execute("INSERT INTO users VALUES (3, 'Charlie', 35)")
    con.commit()
    con.close()
    yield path
    os.unlink(path)


# ── Bronze Layer ──

class TestBronzeLayer:
    def test_ingest_records(self, bronze):
        meta = bronze.ingest_records(
            "records_test",
            columns=["a", "b"],
            records=[[1, "x"], [2, "y"], [3, "z"]],
        )
        assert meta["rows"] == 3
        assert meta["columns"] == 2
        assert meta["source"] == "records_test"
        assert os.path.exists(meta["path"])

    def test_ingest_with_nulls(self, bronze):
        meta = bronze.ingest_records(
            "nulls_test",
            columns=["id", "name", "score"],
            records=[[1, "A", 95.5], [2, None, None], [3, "C", 88.0]],
        )
        assert meta["rows"] == 3
        assert meta["columns"] == 3

    def test_ingest_empty(self, bronze):
        meta = bronze.ingest_records("empty_test", ["a"], [])
        assert meta["rows"] == 0
        assert meta["columns"] == ["a"]

    def test_list_sources(self, bronze):
        bronze.ingest_records("src_a", ["x"], [[1]])
        bronze.ingest_records("src_b", ["y"], [[2]])
        sources = bronze.list_sources()
        assert "src_a" in sources
        assert "src_b" in sources

    def test_list_partitions(self, bronze):
        bronze.ingest_records("src_x", ["v"], [[1]])
        parts = bronze.list_partitions("src_x")
        assert len(parts) == 1

    def test_read_latest(self, bronze):
        bronze.ingest_records("src_r", ["id", "name"], [[1, "Alice"], [2, "Bob"]])
        rows = bronze.read_latest("src_r")
        assert len(rows) == 2
        assert rows[0]["name"] == "Alice"

    def test_read_latest_not_found(self, bronze):
        with pytest.raises(FileNotFoundError):
            bronze.read_latest("nonexistent")

    def test_get_stats(self, bronze):
        bronze.ingest_records("s1", ["a"], [[1]])
        bronze.ingest_records("s2", ["b"], [[2]])
        stats = bronze.get_stats()
        assert stats["sources"] == 2

    def test_partition_isolation(self, bronze):
        bronze.ingest_records("iso", ["v"], [[1]])
        assert len(bronze.list_partitions("iso")) == 1

    def test_read_partition_exact(self, bronze):
        bronze.ingest_records("exact", ["k"], [["hello"]])
        parts = bronze.list_partitions("exact")
        rows = bronze.read_partition("exact", parts[0])
        assert rows[0]["k"] == "hello"

    def test_bool_and_float_types(self, bronze):
        meta = bronze.ingest_records(
            "types", ["flag", "ratio"],
            [[True, 3.14], [False, 2.71]]
        )
        schema = meta["schema"]
        assert schema[0]["type"] == "boolean"
        assert schema[1]["type"] == "float"


# ── SQLite Connector ──

class TestSQLiteConnector:
    @pytest.mark.asyncio
    async def test_connect_and_query(self, sqlite_db):
        conn = SQLiteConnector("test", {"path": sqlite_db})
        await conn.connect()
        assert conn.is_connected
        result = await conn.execute("SELECT * FROM users ORDER BY id")
        assert result.row_count == 3
        assert result.columns == ["id", "name", "age"]
        await conn.close()

    @pytest.mark.asyncio
    async def test_list_tables(self, sqlite_db):
        conn = SQLiteConnector("test", {"path": sqlite_db})
        await conn.connect()
        tables = await conn.list_tables()
        assert any(t.name == "users" for t in tables)
        await conn.close()

    @pytest.mark.asyncio
    async def test_get_table_info(self, sqlite_db):
        conn = SQLiteConnector("test", {"path": sqlite_db})
        await conn.connect()
        info = await conn.get_table_info("users")
        assert info.row_count == 3
        assert len(info.columns) == 3
        await conn.close()

    @pytest.mark.asyncio
    async def test_read_table(self, sqlite_db):
        conn = SQLiteConnector("test", {"path": sqlite_db})
        await conn.connect()
        result = await conn.read_table("users", limit=2)
        assert result.row_count == 2
        await conn.close()

    @pytest.mark.asyncio
    async def test_async_context_manager(self, sqlite_db):
        async with SQLiteConnector("test", {"path": sqlite_db}) as conn:
            result = await conn.execute("SELECT COUNT(*) FROM users")
            assert result.rows[0][0] == 3


# ── Connector Factory ──

class TestConnectorFactoryExtended:
    def test_create_sqlite(self):
        conn = ConnectorFactory.create("test", "sqlite", {"path": ":memory:"})
        assert isinstance(conn, SQLiteConnector)

    def test_create_drap(self):
        conn = ConnectorFactory.create("test", "drap", {"base_url": "http://localhost:8000"})
        assert isinstance(conn, DrapConnector)
        assert conn.base_url == "http://localhost:8000"

    def test_drap_set_auth(self):
        conn = DrapConnector("test")
        conn.set_auth_token("test-jwt")
        assert conn.headers["Authorization"] == "Bearer test-jwt"
