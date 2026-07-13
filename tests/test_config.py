"""Test config module."""

import os
from pathlib import Path

import pytest

from src.config import (
    LLMConfig,
    MedallionConfig,
    DrapConfig,
    QualityConfig,
    Settings,
    get_settings,
    reload_settings,
)


class TestLLMConfig:
    def test_defaults(self):
        cfg = LLMConfig()
        assert cfg.provider == "deepseek"
        assert cfg.model == "deepseek-v4-pro"
        assert cfg.temperature == 0.1

    def test_env_override(self, monkeypatch):
        """LLMConfig 直接读取环境变量 (DPA_LLM_ prefix)"""
        monkeypatch.setenv("DPA_LLM_MODEL", "gpt-4")
        monkeypatch.setenv("DPA_LLM_TEMPERATURE", "0.5")
        cfg = LLMConfig()
        assert cfg.model == "gpt-4"
        assert float(cfg.temperature) == 0.5

    def test_custom_values(self):
        cfg = LLMConfig(provider="openai", model="gpt-4o", temperature=0.7)
        assert cfg.provider == "openai"
        assert cfg.model == "gpt-4o"


class TestMedallionConfig:
    def test_default_paths(self):
        cfg = MedallionConfig(base_path="/tmp/data")
        assert "bronze" in cfg.bronze_path
        assert "silver" in cfg.silver_path
        assert "gold" in cfg.gold_path

    def test_custom_paths(self):
        cfg = MedallionConfig(
            base_path="/tmp/data",
            bronze_path="/custom/bronze",
            silver_path="/custom/silver",
            gold_path="/custom/gold",
        )
        assert cfg.bronze_path == "/custom/bronze"


class TestDrapConfig:
    def test_defaults(self):
        cfg = DrapConfig()
        assert cfg.base_url == "http://localhost:8000"
        assert cfg.timeout == 30


class TestQualityConfig:
    def test_defaults(self):
        cfg = QualityConfig()
        assert cfg.completeness_threshold == 0.95
        assert cfg.accuracy_threshold == 0.90


class TestSettings:
    def test_singleton(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_reload(self):
        s1 = get_settings()
        s2 = reload_settings()
        assert s1 is not s2  # new instance
        assert s1.app_name == s2.app_name

    def test_default_values(self):
        s = get_settings()
        assert s.app_name == "DataProcessingAgent"
        assert s.dcmm_target_level == 2
        assert s.api_port == 8100
        assert isinstance(s.llm, LLMConfig)
        assert isinstance(s.medallion, MedallionConfig)
        assert isinstance(s.drap, DrapConfig)
        assert isinstance(s.quality, QualityConfig)

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("DPA_APP_NAME", "TestApp")
        monkeypatch.setenv("DPA_API_PORT", "9999")
        s = reload_settings()
        assert s.app_name == "TestApp"
        assert s.api_port == 9999

    def test_medallion_paths_exist(self):
        s = get_settings()
        for p in [s.medallion.bronze_path, s.medallion.silver_path, s.medallion.gold_path]:
            assert Path(p).exists(), f"Path should exist: {p}"

    def test_drap_connected_to_config(self):
        s = get_settings()
        assert s.drap.base_url == "http://localhost:8000"
