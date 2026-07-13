"""
安全测试 — SQL注入防护 / 标识符校验 / 路径校验 / SecretStr
"""

import pytest

from src.utils.sql import safe_identifier, safe_literal, safe_path_for_duckdb, validate_sql_read_only
from src.config import DrapConfig, LLMConfig


class TestSafeIdentifier:
    def test_valid_identifier(self):
        assert safe_identifier("my_table") == '"my_table"'

    def test_chinese_identifier(self):
        assert safe_identifier("数据表") == '"数据表"'

    def test_rejects_semicolon(self):
        with pytest.raises(ValueError):
            safe_identifier("t; DROP TABLE _t")

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            safe_identifier("")

    def test_rejects_special_chars(self):
        with pytest.raises(ValueError):
            safe_identifier("table'; DROP TABLE--")


class TestSafeLiteral:
    def test_none(self):
        assert safe_literal(None) == "NULL"

    def test_bool(self):
        assert safe_literal(True) == "TRUE"
        assert safe_literal(False) == "FALSE"

    def test_int(self):
        assert safe_literal(42) == "42"

    def test_float(self):
        assert safe_literal(3.14) == "3.14"

    def test_string_escapes_quotes(self):
        result = safe_literal("it's")
        assert result == "'it''s'"

    def test_rejects_null_byte(self):
        with pytest.raises(ValueError):
            safe_literal("hello\x00world")


class TestSafePathForDuckDB:
    def test_normal_path(self):
        assert safe_path_for_duckdb("/data/file.csv") == "/data/file.csv"

    def test_rejects_path_traversal(self):
        with pytest.raises(ValueError):
            safe_path_for_duckdb("../../etc/passwd")

    def test_rejects_null_byte(self):
        with pytest.raises(ValueError):
            safe_path_for_duckdb("/data/file\x00.csv")


class TestValidateSQLReadOnly:
    def test_allows_select(self):
        validate_sql_read_only("SELECT * FROM _t WHERE id = 1")

    def test_rejects_drop(self):
        with pytest.raises(ValueError):
            validate_sql_read_only("DROP TABLE _t")

    def test_rejects_insert(self):
        with pytest.raises(ValueError):
            validate_sql_read_only("INSERT INTO _t VALUES (1)")

    def test_rejects_semicolon_injection(self):
        with pytest.raises(ValueError):
            validate_sql_read_only("SELECT 1; DROP TABLE _t")


class TestSecretStr:
    def test_drap_auth_token_is_secret(self):
        cfg = DrapConfig()
        assert hasattr(cfg.auth_token, "get_secret_value")
        assert cfg.auth_token.get_secret_value() == ""

    def test_llm_api_key_is_secret(self):
        cfg = LLMConfig()
        assert hasattr(cfg.api_key, "get_secret_value")
        assert cfg.api_key.get_secret_value() == ""