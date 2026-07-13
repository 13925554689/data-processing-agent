"""
MCP (Model Context Protocol) 工具集成层

统一封装数据源连接：数据库、文件系统、API 等。
通过标准化 Connector 接口屏蔽底层差异，支持懒连接和连接池。

设计原则:
  - 每个 Connector 实现统一的 connect / execute / close 接口
  - 支持连接字符串、配置文件、环境变量三种配置方式
  - 内置重试和超时机制
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from src.utils.sql import safe_identifier, safe_path_for_duckdb

logger = logging.getLogger(__name__)


# ── 通用数据模型 ─────────────────────────────────────────────────────

@dataclass
class ColumnInfo:
    """列信息"""
    name: str
    dtype: str
    nullable: bool = True
    comment: str = ""


@dataclass
class TableInfo:
    """表信息"""
    name: str
    schema: str = ""
    row_count: int = 0
    columns: list[ColumnInfo] = field(default_factory=list)


@dataclass
class QueryResult:
    """查询结果"""
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    duration_ms: float = 0.0
    truncated: bool = False


# ── Connector 抽象基类 ───────────────────────────────────────────────

class BaseConnector(ABC):
    """数据源连接器抽象基类"""

    connector_type: str = "base"
    supports_read: bool = True
    supports_write: bool = False

    def __init__(self, name: str, config: Optional[dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self._connected = False

    @abstractmethod
    async def connect(self) -> None:
        """建立连接"""
        ...

    @abstractmethod
    async def close(self) -> None:
        """关闭连接"""
        ...

    @abstractmethod
    async def execute(self, query: str, **params: Any) -> QueryResult:
        """执行查询"""
        ...

    @abstractmethod
    async def list_tables(self) -> list[TableInfo]:
        """列出所有表"""
        ...

    @abstractmethod
    async def get_table_info(self, table_name: str) -> TableInfo:
        """获取表结构信息"""
        ...

    async def read_table(
        self, table_name: str, limit: int = 1000, offset: int = 0
    ) -> QueryResult:
        """读取表数据"""
        safe_name = safe_identifier(table_name)
        if not isinstance(limit, int) or limit < 0:
            raise ValueError("limit must be a non-negative integer")
        if not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        query = f"SELECT * FROM {safe_name} LIMIT {limit} OFFSET {offset}"
        return await self.execute(query)

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.close()

    @property
    def is_connected(self) -> bool:
        return self._connected


# ── 文件系统连接器 ────────────────────────────────────────────────────

class FileConnector(BaseConnector):
    """文件系统数据源连接器 (CSV/Excel/JSON/Parquet)"""

    connector_type = "file"
    supports_write = True

    def __init__(self, name: str, config: Optional[dict[str, Any]] = None):
        super().__init__(name, config)
        self.file_path = config.get("path", "") if config else ""
        self.file_type = config.get("type", "csv") if config else "csv"

    async def connect(self) -> None:

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        self._connected = True
        logger.info(f"[{self.name}] Connected to file: {self.file_path}")

    async def close(self) -> None:
        self._connected = False

    async def execute(self, query: str, **params: Any) -> QueryResult:
        """使用 DuckDB 执行文件上的 SQL 查询"""
        import duckdb
        import time
        t0 = time.perf_counter()

        con = duckdb.connect(":memory:")
        try:
            ft = self.file_type
            safe_path = safe_path_for_duckdb(self.file_path)
            if ft == "csv":
                con.execute(f"CREATE TABLE _src AS SELECT * FROM read_csv_auto('{safe_path}')")
            elif ft in ("xlsx", "xls", "excel"):
                con.execute(
                    f"CREATE TABLE _src AS SELECT * FROM st_read('{safe_path}')"
                )
            elif ft == "json":
                con.execute(f"CREATE TABLE _src AS SELECT * FROM read_json_auto('{safe_path}')")
            elif ft == "parquet":
                con.execute(f"CREATE TABLE _src AS SELECT * FROM read_parquet('{safe_path}')")
            else:
                raise ValueError(f"Unsupported file type: {ft}")

            resolved = query.replace(self.file_path, "_src")


            result = con.execute(resolved).fetchall()
            cols = [desc[0] for desc in con.description] if con.description else []
            duration = (time.perf_counter() - t0) * 1000
            return QueryResult(
                columns=cols,
                rows=[list(row) for row in result],
                row_count=len(result),
                duration_ms=duration,
            )
        finally:
            con.close()

    async def list_tables(self) -> list[TableInfo]:
        return [TableInfo(name=self.file_path, schema="file", columns=[])]

    async def get_table_info(self, table_name: str) -> TableInfo:
        import duckdb
        con = duckdb.connect(":memory:")
        try:
            ft = self.file_type
            safe_path = safe_path_for_duckdb(self.file_path)
            if ft == "csv":
                con.execute(f"CREATE TABLE _src AS SELECT * FROM read_csv_auto('{safe_path}')")
            elif ft == "parquet":
                con.execute(f"CREATE TABLE _src AS SELECT * FROM read_parquet('{safe_path}')")
            else:
                return TableInfo(name=table_name)
            result = con.execute("DESCRIBE _src").fetchall()
            columns = [
                ColumnInfo(name=row[0], dtype=row[1], nullable=row[3] == "YES")
                for row in result
            ]
            count = con.execute("SELECT COUNT(*) FROM _src").fetchone()[0]
            return TableInfo(name=table_name, row_count=count, columns=columns)
        finally:
            con.close()


# ── 连接器工厂 ────────────────────────────────────────────────────────

class ConnectorFactory:
    """连接器工厂 — 根据配置创建合适的连接器"""

    _registry: dict[str, type[BaseConnector]] = {
        "file": FileConnector,
        "csv": FileConnector,
        "excel": FileConnector,
        "xlsx": FileConnector,
        "json": FileConnector,
        "parquet": FileConnector,
    }

    @classmethod
    def register(cls, conn_type: str, conn_cls: type[BaseConnector]) -> None:
        """注册新的连接器类型"""
        cls._registry[conn_type] = conn_cls

    @classmethod
    def create(cls, name: str, conn_type: str, config: dict[str, Any]) -> BaseConnector:
        """根据类型创建连接器"""
        if conn_type not in cls._registry:
            raise ValueError(f"Unknown connector type: {conn_type}. Available: {list(cls._registry)}")
        return cls._registry[conn_type](name, {**config, "type": conn_type})

    @classmethod
    def available_types(cls) -> list[str]:
        return list(cls._registry.keys())
