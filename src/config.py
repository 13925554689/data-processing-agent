"""
核心配置模块 — 统一管理所有配置，支持 YAML + 环境变量覆盖

配置优先级 (低 → 高):
  1. 默认值 (代码内置)
  2. config.yaml (项目根目录)
  3. 环境变量 (DPA_ 前缀)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Project root discovery ──────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _find_config(paths: list[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


_YAML_PATH = _find_config([
    _PROJECT_ROOT / "config.yaml",
    _PROJECT_ROOT / "config.yml",
])


def _load_yaml() -> dict[str, Any]:
    if _YAML_PATH and _YAML_PATH.exists():
        with open(_YAML_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


# ── Pydantic settings ────────────────────────────────────────────────

class DataSourceConfig(BaseSettings):
    """单个数据源配置"""
    name: str = ""
    type: str = "csv"  # csv | excel | json | postgres | mysql | api
    path: str = ""  # 文件路径或连接字符串
    options: dict[str, Any] = Field(default_factory=dict)

    model_config = SettingsConfigDict(extra="allow")


class DrapConfig(BaseSettings):
    """DRAP 估值引擎连接配置"""
    base_url: str = "http://localhost:8000"
    api_version: str = "v1"
    timeout: int = 30
    auth_token: SecretStr = SecretStr("")

    model_config = SettingsConfigDict(env_prefix="DPA_DRAP_")


class MedallionConfig(BaseSettings):
    """Medallion 分层存储路径"""
    base_path: str = str(_PROJECT_ROOT / "data")
    bronze_path: str = ""
    silver_path: str = ""
    gold_path: str = ""

    def model_post_init(self, __context: Any) -> None:
        base = Path(self.base_path)
        if not self.bronze_path:
            self.bronze_path = str(base / "bronze")
        if not self.silver_path:
            self.silver_path = str(base / "silver")
        if not self.gold_path:
            self.gold_path = str(base / "gold")

    model_config = SettingsConfigDict(env_prefix="DPA_MEDALLION_")


class LLMConfig(BaseSettings):
    """LLM 配置"""
    provider: str = "deepseek"
    model: str = "deepseek-v4-pro"
    api_key: SecretStr = SecretStr("")
    base_url: str = ""
    temperature: float = 0.1
    max_tokens: int = 4096

    model_config = SettingsConfigDict(env_prefix="DPA_LLM_")


class QualityConfig(BaseSettings):
    """数据质量配置"""
    completeness_threshold: float = 0.95
    accuracy_threshold: float = 0.90
    consistency_threshold: float = 0.95
    timeliness_hours: int = 24
    uniqueness_threshold: float = 0.98

    model_config = SettingsConfigDict(env_prefix="DPA_QUALITY_")


class Settings(BaseSettings):
    """应用总配置"""

    # ── 应用信息 ──
    app_name: str = "DataProcessingAgent"
    app_version: str = "0.1.0"
    debug: bool = False

    # ── 项目路径 ──
    project_root: str = str(_PROJECT_ROOT)
    data_dir: str = str(_PROJECT_ROOT / "data")
    logs_dir: str = str(_PROJECT_ROOT / "logs")

    # ── DCMM 目标等级 (1-5) ──
    dcmm_target_level: int = 2  # 默认受管理级

    # ── 子配置 ──
    llm: LLMConfig = Field(default_factory=LLMConfig)
    medallion: MedallionConfig = Field(default_factory=MedallionConfig)
    drap: DrapConfig = Field(default_factory=DrapConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)

    # ── 数据源列表 ──
    data_sources: list[DataSourceConfig] = Field(default_factory=list)

    # ── API 服务 ──
    api_host: str = "0.0.0.0"
    api_port: int = 8100

    # ── 日志 ──
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    model_config = SettingsConfigDict(
        env_prefix="DPA_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="allow",
    )

    @classmethod
    def from_yaml(cls, yaml_path: Optional[Path] = None) -> "Settings":
        """从 YAML 文件加载配置，环境变量覆盖"""
        path = yaml_path or _YAML_PATH
        yaml_data = {}
        if path and path.exists():
            with open(path, encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}

        # 展平嵌套 YAML 为 env-like keys
        flat: dict[str, Any] = {}
        _flatten(yaml_data, "", flat)

        # 用环境变量覆盖 YAML 值
        for key, value in flat.items():
            env_val = os.environ.get(f"DPA_{key.upper()}")
            if env_val is not None:
                flat[key] = env_val

        # 重建嵌套结构
        nested = _unflatten(flat)

        # 处理子配置
        llm_raw = nested.pop("llm", {})
        medallion_raw = nested.pop("medallion", {})
        drap_raw = nested.pop("drap", {})
        quality_raw = nested.pop("quality", {})

        return cls(
            **nested,
            llm=LLMConfig(**llm_raw) if llm_raw else LLMConfig(),
            medallion=MedallionConfig(**medallion_raw) if medallion_raw else MedallionConfig(),
            drap=DrapConfig(**drap_raw) if drap_raw else DrapConfig(),
            quality=QualityConfig(**quality_raw) if quality_raw else QualityConfig(),
        )


def _flatten(d: dict, prefix: str, out: dict) -> None:
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}__{k}"
        if isinstance(v, dict):
            _flatten(v, key, out)
        else:
            out[key] = v


def _unflatten(flat: dict) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in flat.items():
        parts = key.split("__")
        d = result
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return result


# ── 全局单例 ──
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """获取配置单例"""
    global _settings
    if _settings is None:
        _settings = Settings.from_yaml()
        for d in [_settings.data_dir, _settings.logs_dir,
                   _settings.medallion.bronze_path,
                   _settings.medallion.silver_path,
                   _settings.medallion.gold_path]:
            Path(d).mkdir(parents=True, exist_ok=True)
        import logging
        logging.basicConfig(
            level=getattr(logging, _settings.log_level, logging.INFO),
            format=_settings.log_format,
        )
        if _settings.debug:
            logging.getLogger().setLevel(logging.DEBUG)
    return _settings


def reload_settings(yaml_path: Optional[Path] = None) -> Settings:
    """重新加载配置"""
    global _settings
    _settings = Settings.from_yaml(yaml_path)
    return _settings
