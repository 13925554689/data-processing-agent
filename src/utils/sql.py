"""
SQL 安全工具模块

提供标识符校验、SQL只读验证、参数化值构建等安全函数，
消除全项目中的 SQL 注入风险。
"""

from __future__ import annotations

import re
from typing import Any

_IDENTIFIER_RE = re.compile(r'^[A-Za-z_\u4e00-\u9fff][A-Za-z0-9_\u4e00-\u9fff]*$')

_DANGEROUS_PATTERNS = re.compile(
    r';|--|/\*|\*/|xp_|sp_|exec\s|execute\s|drop\s|alter\s|create\s|insert\s|update\s|delete\s|truncate\s|grant\s|revoke\s',
    re.IGNORECASE,
)

_ALLOWED_SELECT_RE = re.compile(
    r'^\s*SELECT\s', re.IGNORECASE,
)


def safe_identifier(name: str) -> str:
    if not name or not _IDENTIFIER_RE.match(name):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return f'"{name}"'


def validate_sql_read_only(sql: str) -> None:
    if _DANGEROUS_PATTERNS.search(sql):
        raise ValueError("SQL contains potentially dangerous statements")
    if not _ALLOWED_SELECT_RE.match(sql.strip()):
        raise ValueError("Only SELECT queries are allowed in this context")


def safe_literal(v: Any) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return str(v)
    if isinstance(v, str):
        if '\x00' in v:
            raise ValueError("String contains NULL byte")
        escaped = v.replace("'", "''")
        return f"'{escaped}'"
    return f"'{str(v).replace(chr(39), chr(39)+chr(39))}'"


def safe_path_for_duckdb(path: str) -> str:
    if '..' in path or '\x00' in path:
        raise ValueError(f"Invalid file path: {path!r}")
    return path.replace("'", "''")