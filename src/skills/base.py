"""
Skill 技能体系框架

Skill 是 Agent 的可复用能力单元。每个 Skill 定义:
  - 输入参数模式
  - 执行逻辑
  - 输出结果格式
  - 依赖关系

设计原则:
  - 每个 Skill 是独立的、可组合的函数
  - 支持链式调用（管道）
  - 支持 lazy loading
  - 内置输入/输出校验
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


# ── 枚举 ────────────────────────────────────────────────────────────

class SkillStatus(Enum):
    """Skill 执行状态"""
    PENDING = auto()
    RUNNING = auto()
    SUCCESS = auto()
    FAILED = auto()
    SKIPPED = auto()


# ── 数据模型 ─────────────────────────────────────────────────────────

class SkillResult(BaseModel):
    """Skill 执行结果"""
    success: bool
    data: Any = None
    message: str = ""
    error: Optional[str] = None
    status: SkillStatus = SkillStatus.PENDING
    duration_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass
class SkillMeta:
    """Skill 元信息"""
    name: str
    description: str
    version: str = "0.1.0"
    category: str = "general"
    tags: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)  # 依赖的其他 skill
    input_schema: Optional[dict] = None  # JSON Schema
    output_schema: Optional[dict] = None


# ── Skill 抽象基类 ───────────────────────────────────────────────────

class BaseSkill(ABC):
    """Skill 抽象基类"""

    # 子类必须定义
    meta: SkillMeta

    def __init__(self):
        self._last_result: Optional[SkillResult] = None

    @abstractmethod
    async def execute(self, **kwargs: Any) -> SkillResult:
        """执行 Skill 逻辑"""
        ...

    async def validate_input(self, **kwargs: Any) -> Optional[str]:
        """输入校验（可选覆盖）"""
        if self.meta.input_schema:
            try:
                # 使用 pydantic 动态校验
                from pydantic import create_model
                fields = {}
                for key, spec in self.meta.input_schema.get("properties", {}).items():
                    py_type = self._json_type_to_python(spec.get("type", "string"))
                    required = key in self.meta.input_schema.get("required", [])
                    if required:
                        fields[key] = (py_type, ...)
                    else:
                        fields[key] = (Optional[py_type], None)
                if fields:
                    model = create_model(f"{self.meta.name}Input", **fields)  # type: ignore[call-overload]
                    model(**kwargs)
            except ValidationError as e:
                return str(e)
            except Exception as e:
                logger.warning(f"Schema validation skipped: {e}")
        return None

    @staticmethod
    def _json_type_to_python(json_type: str) -> type:
        mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        return mapping.get(json_type, str)

    @property
    def info(self) -> dict[str, Any]:
        return {
            "name": self.meta.name,
            "description": self.meta.description,
            "version": self.meta.version,
            "category": self.meta.category,
            "tags": self.meta.tags,
        }

    def __repr__(self) -> str:
        return f"<Skill({self.meta.name} v{self.meta.version})>"


# ── Skill 注册表 ─────────────────────────────────────────────────────

class SkillRegistry:
    """Skill 注册表（单例）"""

    _instance: Optional["SkillRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "SkillRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._skills: dict[str, type[BaseSkill]] = {}
                cls._instance._instances: dict[str, BaseSkill] = {}
            return cls._instance

    def register(self, skill_cls: type[BaseSkill], override: bool = False) -> "SkillRegistry":
        name = skill_cls.meta.name
        if name in self._skills and not override:
            raise ValueError(f"Skill '{name}' already registered")
        self._skills[name] = skill_cls
        self._instances.pop(name, None)
        logger.info(f"Registered skill: {name}")
        return self

    def get(self, name: str) -> Optional[BaseSkill]:
        if name in self._instances:
            return self._instances[name]
        cls = self._skills.get(name)
        if cls is None:
            return None
        instance = cls()
        self._instances[name] = instance
        return instance

    def list_names(self) -> list[str]:
        return list(self._skills.keys())

    def list_by_category(self, category: str) -> list[BaseSkill]:
        result = []
        for name in self._skills:
            skill = self.get(name)
            if skill and skill.meta.category == category:
                result.append(skill)
        return result

    def clear(self) -> None:
        self._skills.clear()
        self._instances.clear()

    @classmethod
    def reset(cls) -> None:
        if cls._instance:
            cls._instance.clear()
        cls._instance = None


def skill_registry() -> SkillRegistry:
    return SkillRegistry()


# ── Skill 管道 ──────────────────────────────────────────────────────

class SkillPipeline:
    """
    Skill 管道 — 按顺序执行多个 Skill

    前一个 Skill 的输出作为下一个 Skill 的输入 (data 字段)。
    任何一个 Skill 失败则管道中断。
    """

    def __init__(self, name: str = "pipeline"):
        self.name = name
        self._skills: list[str] = []
        self._registry = SkillRegistry()

    def add(self, skill_name: str) -> "SkillPipeline":
        self._skills.append(skill_name)
        return self

    def add_many(self, *skill_names: str) -> "SkillPipeline":
        self._skills.extend(skill_names)
        return self

    async def run(self, initial_input: Any = None) -> SkillResult:
        """执行管道"""
        current = initial_input
        results: list[SkillResult] = []

        for name in self._skills:
            skill = self._registry.get(name)
            if skill is None:
                return SkillResult(
                    success=False,
                    error=f"Skill not found: {name}",
                    message=f"Pipeline '{self.name}' aborted at '{name}'",
                )

            result = await skill.execute(data=current)
            results.append(result)

            if not result.success:
                return SkillResult(
                    success=False,
                    data={"completed": [r.metadata.get("skill") for r in results]},
                    error=f"Skill '{name}' failed: {result.error}",
                    message=f"Pipeline '{self.name}' aborted at '{name}'",
                )

            current = result.data

        return SkillResult(
            success=True,
            data=current,
            message=f"Pipeline '{self.name}' completed ({len(results)} skills)",
            metadata={"skills_executed": [r.metadata.get("skill") for r in results]},
        )
