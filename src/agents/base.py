"""
Agent 基类模块

定义 Agent 抽象基类和注册机制，所有专业 Agent 均继承自 AgentBase。
支持懒加载、状态追踪、事件钩子。
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── 枚举 ────────────────────────────────────────────────────────────

class AgentStatus(Enum):
    """Agent 运行状态"""
    IDLE = auto()        # 空闲，等待任务
    RUNNING = auto()     # 执行中
    COMPLETED = auto()   # 已完成
    FAILED = auto()      # 执行失败
    PAUSED = auto()      # 暂停（等待人工输入）


class AgentCategory(Enum):
    """Agent 分类"""
    INGEST = "ingest"        # 数据采集/接入
    CLEAN = "clean"          # 数据清洗
    INTEGRATE = "integrate"  # 数据集成/融合
    GOVERN = "govern"        # 数据治理/编目
    ANALYZE = "analyze"      # 数据分析
    ASSET = "asset"          # 数据资产化
    SERVE = "serve"          # 数据服务
    SAFETY = "safety"        # 安全审计
    PLAN = "plan"            # 任务规划
    CUSTOM = "custom"        # 自定义


# ── 结果类型 ─────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    """Agent 执行结果"""
    success: bool
    data: Any = None
    message: str = ""
    error: Optional[str] = None
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, data: Any = None, message: str = "", **meta) -> "AgentResult":
        return cls(success=True, data=data, message=message, metadata=meta)

    @classmethod
    def fail(cls, error: str, message: str = "", **meta) -> "AgentResult":
        return cls(success=False, error=error, message=message or error, metadata=meta)


# ── Agent 基类 ───────────────────────────────────────────────────────

class AgentBase(ABC):
    """
    Agent 抽象基类

    所有专业 Agent 的基类。子类必须实现:
      - execute(): 核心执行逻辑
    子类可选覆盖:
      - validate(): 输入验证
      - pre_execute(): 执行前钩子
      - post_execute(): 执行后钩子
    """

    # ── 子类必须定义的类属性 ──
    name: str = "base_agent"
    description: str = "Base agent class"
    category: AgentCategory = AgentCategory.CUSTOM
    version: str = "0.1.0"

    # ── 可选类属性 ──
    requires_human_approval: bool = False  # 是否需要人工审批
    max_retries: int = 2                    # 最大重试次数
    timeout_seconds: int = 300              # 超时时间

    def __init__(self, config: Optional[dict[str, Any]] = None):
        self.config = config or {}
        self.status = AgentStatus.IDLE
        self._last_result: Optional[AgentResult] = None
        self._execution_count: int = 0
        self._total_duration_ms: float = 0.0

    # ── 公共接口 ──

    async def run(self, **kwargs: Any) -> AgentResult:
        """
        执行 Agent 任务（带重试和状态管理）

        Args:
            **kwargs: 任务参数，由具体 Agent 定义

        Returns:
            AgentResult: 执行结果
        """
        self.status = AgentStatus.RUNNING
        self._execution_count += 1
        t0 = time.perf_counter()

        try:
            validation_error = await self.validate(**kwargs)
            if validation_error:
                return self._finish(AgentResult.fail(validation_error), t0)

            await self.pre_execute(**kwargs)

            last_error = None
            for attempt in range(self.max_retries + 1):
                try:
                    result = await self.execute(**kwargs)
                    break
                except Exception as e:
                    last_error = str(e)
                    logger.warning(
                        f"[{self.name}] Attempt {attempt + 1}/{self.max_retries + 1} failed: {e}"
                    )
                    if attempt == self.max_retries:
                        raise

            result = await self.post_execute(result, **kwargs)
            return self._finish(result, t0)

        except Exception as e:
            logger.error(f"[{self.name}] Execution failed: {e}", exc_info=True)
            return self._finish(AgentResult.fail(str(e)), t0)

    # ── 子类接口 ──

    @abstractmethod
    async def execute(self, **kwargs: Any) -> AgentResult:
        """
        核心执行逻辑（子类必须实现）

        Returns:
            AgentResult: 执行结果
        """
        ...

    async def validate(self, **kwargs: Any) -> Optional[str]:
        """
        输入验证（可选覆盖）

        Returns:
            None 表示验证通过，否则返回错误描述字符串
        """
        return None

    async def pre_execute(self, **kwargs: Any) -> None:
        """执行前钩子（可选覆盖）"""
        pass

    async def post_execute(self, result: AgentResult, **kwargs: Any) -> AgentResult:
        """执行后钩子（可选覆盖），可修改结果"""
        return result

    # ── 工具方法 ──

    def _finish(self, result: AgentResult, t0: float) -> AgentResult:
        """完成执行，记录状态和耗时"""
        result.duration_ms = (time.perf_counter() - t0) * 1000
        self._last_result = result
        self.status = AgentStatus.COMPLETED if result.success else AgentStatus.FAILED
        return result

    @property
    def info(self) -> dict[str, Any]:
        """Agent 元信息"""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "version": self.version,
            "status": self.status.name,
            "execution_count": self._execution_count,
            "requires_human_approval": self.requires_human_approval,
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name}, status={self.status.name})>"


# ── Agent 注册表 ──────────────────────────────────────────────────────

class AgentRegistry:
    """
    Agent 注册表（单例）

    管理所有 Agent 的注册、发现和实例化。
    支持懒加载：首次访问时才实例化 Agent。
    """

    _instance: Optional["AgentRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "AgentRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._agents: dict[str, type[AgentBase]] = {}
                cls._instance._instances: dict[str, AgentBase] = {}
                cls._instance._initialized = False
            return cls._instance

    def register(
        self, agent_cls: type[AgentBase], override: bool = False
    ) -> "AgentRegistry":
        """
        注册 Agent 类

        Args:
            agent_cls: Agent 子类
            override: 是否允许覆盖已注册的同名 Agent

        Returns:
            self (链式调用)
        """
        name = agent_cls.name
        if name in self._agents and not override:
            raise ValueError(
                f"Agent '{name}' already registered. Use override=True to replace."
            )
        self._agents[name] = agent_cls
        # 清除已缓存的实例
        self._instances.pop(name, None)
        logger.info(f"Registered agent: {name} ({agent_cls.category.value})")
        return self

    def get(self, name: str) -> Optional[AgentBase]:
        """
        获取 Agent 实例（懒加载）

        Args:
            name: Agent 名称

        Returns:
            Agent 实例，未注册则返回 None
        """
        # 已缓存
        if name in self._instances:
            return self._instances[name]

        # 首次访问，实例化
        agent_cls = self._agents.get(name)
        if agent_cls is None:
            return None

        instance = agent_cls()
        self._instances[name] = instance
        return instance

    def get_by_category(self, category: AgentCategory) -> list[AgentBase]:
        """获取指定分类的所有 Agent 实例"""
        result = []
        for name, cls in self._agents.items():
            if cls.category == category:
                agent = self.get(name)
                if agent:
                    result.append(agent)
        return result

    def list_names(self) -> list[str]:
        """列出所有已注册 Agent 名称"""
        return list(self._agents.keys())

    def list_info(self) -> list[dict[str, Any]]:
        """列出所有 Agent 信息"""
        return [
            {
                "name": cls.name,
                "description": cls.description,
                "category": cls.category.value,
                "version": cls.version,
                "registered": cls.name in self._instances,
            }
            for cls in self._agents.values()
        ]

    def unregister(self, name: str) -> bool:
        """注销 Agent"""
        self._agents.pop(name, None)
        self._instances.pop(name, None)
        return True

    def clear(self) -> None:
        """清空所有注册"""
        self._agents.clear()
        self._instances.clear()

    @classmethod
    def reset(cls) -> None:
        """重置单例（仅用于测试）"""
        if cls._instance:
            cls._instance.clear()
        cls._instance = None


# ── 便捷函数 ──

def registry() -> AgentRegistry:
    """获取全局注册表单例"""
    return AgentRegistry()
