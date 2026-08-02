# Repository Guidelines — 数据处理智能体 (data-processing-agent)

## Project Overview
通用数据处理智能体，覆盖数据全生命周期：采集→清洗→集成→治理→分析→资产化→服务。
多 Agent 协作：plan / ingest / clean / standardize / integrate / govern / analyze / asset。
分层存储：bronze（原始）→ silver（清洗后）→ gold（资产化）。FastAPI 对外服务。

## Tech Stack
- 语言：Python ≥3.11
- 框架：FastAPI + uvicorn / pydantic v2 / pydantic-settings
- 数据：pandas / pyarrow / duckdb / openpyxl / python-docx
- 存储：bronze→silver→gold 分层 + lineage 血缘追踪

## Build, Test, and Development Commands
- 安装：`pip install -e .`（pyproject.toml，Python ≥3.11，依赖 pandas/duckdb/fastapi/pydantic）
- 测试：`python -m pytest tests/` — test_agents / test_clean_v2 / test_desensitize / test_api 等
- 启动 API：`uvicorn src.api.app:app --port 8000`（fastapi + uvicorn）
- 关键 Agent 入口：`src/agents/<agent>_agent.py`，统一继承 `AgentBase`，`execute(**kwargs)` 返回 `AgentResult`
- 治理评分对齐 GB/T 36073-2025 DCMM 六维：完整性/准确性/一致性/时效性/唯一性/规范性

## Coding Style & Naming Conventions
Python 3.11+，snake_case 模块/函数，PascalCase 类，pydantic 模型做配置校验。
Agent 内部用 `_method` 私有方法组织 handlers，`execute()` 只做 action 路由。
新数据处理能力（清洗/脱敏/标准化规则）放对应 agent 私有方法，不新增文件。

## Data Security Notes（红线）
- 脱敏实现：`src/agents/clean_agent.py::_desensitize` — 策略 mask/full/hash/token，PII 列自动检测（COLUMN_HINTS）
- **hash 策略必须加盐**（禁止无盐 SHA256，防彩虹表）——盐放 config，环境变量注入
- 新增 PII 类型先加 `COLUMN_HINTS` + `_apply_mask` 分支，再补 `tests/test_desensitize.py` 用例
- 生产数据禁止明文进测试/开发环境；导出交付前走静态脱敏

## Testing Guidelines
每个 Agent 行为改动至少一个确定性断言。脱敏相关改动必须跑 `tests/test_desensitize.py`。
分层改动跑 test_layers / test_lineage。API 改动跑 test_api。

## Configuration & Security Notes
配置在 `src/config.py`（pydantic-settings）。密钥从环境变量读，不进 git。
新增外部连接器放 `src/connectors/`，强制超时+重试+输入校验（参考 regulation_checker.py）。
