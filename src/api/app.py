"""
数据处理智能体 — FastAPI 主应用 (Port 8100)

编排层入口，汇聚所有 Agent 和三层存储能力。
融合 DRAP (估值) 和法规智能体 (合规)。
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import get_settings
from src.agents.base import AgentRegistry, registry
from src.agents.ingest_agent import IngestAgent
from src.agents.clean_agent import CleanAgent
from src.agents.integrate_agent import IntegrateAgent
from src.agents.govern_agent import GovernAgent
from src.agents.analyze_agent import AnalyzeAgent
from src.agents.asset_agent import AssetAgent
from src.agents.plan_agent import PlanAgent
from src.agents.standardize_agent import StandardizeAgent

from src.layers.bronze import BronzeLayer
from src.layers.silver import SilverLayer
from src.layers.gold import GoldLayer
from src.connectors.regulation_checker import RegulationChecker

logger = logging.getLogger(__name__)

# ── 注册所有 Agent ──
reg = registry()
reg.register(IngestAgent).register(CleanAgent).register(IntegrateAgent)
reg.register(GovernAgent).register(AnalyzeAgent).register(AssetAgent)
reg.register(PlanAgent).register(StandardizeAgent)

# ── 生命周期 ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info(f"DAP v{settings.app_version} starting on port {settings.api_port}")
    logger.info(f"Registered agents: {reg.list_names()}")
    yield
    logger.info("DAP shutting down")

app = FastAPI(
    title="数据处理智能体 API",
    description="全生命周期数据处理编排：采集→清洗→集成→治理→分析→资产化(DRAP)→合规(法规智能体)",
    version="0.1.0",
    lifespan=lifespan,
)

_allowed_origins = os.environ.get("DPA_CORS_ORIGINS", "http://localhost:3000,http://localhost:8100").split(",")
app.add_middleware(CORSMiddleware, allow_origins=_allowed_origins, allow_credentials=False,
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type", "Authorization"])

# ── 静态界面 ──
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/ui", StaticFiles(directory=str(STATIC_DIR), html=True), name="ui")

# ── 安全中间件 ──

_API_KEY = os.environ.get("DPA_API_KEY", "")
_RATE_LIMIT_WINDOW = 60
_RATE_LIMIT_MAX = 100
_rate_store: dict[str, list[float]] = defaultdict(list)

SSRF_BLOCKED = re.compile(
    r'^(127\.|10\.|172\.(1[6-9]|2\d|3[01])\.|192\.168\.|169\.254\.|0\.|::1|fe80:|fc00:)', 
    re.IGNORECASE,
)

def _check_rate_limit(client_ip: str) -> bool:
    now = time.time()
    window = _rate_store[client_ip]
    window[:] = [t for t in window if now - t < _RATE_LIMIT_WINDOW]
    if len(window) >= _RATE_LIMIT_MAX:
        return False
    window.append(now)
    return True

def _validate_source_path(path: str) -> str:
    resolved = Path(path).resolve()
    project_root = Path(os.environ.get("DPA_PROJECT_ROOT", str(Path(__file__).resolve().parent.parent.parent))).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError:
        raise HTTPException(status_code=400, detail="Path traversal detected")
    return str(resolved)

def _validate_url(url: str) -> str:
    from urllib.parse import urlparse
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="Only HTTP/HTTPS URLs are allowed")
    hostname = parsed.hostname or ""
    if SSRF_BLOCKED.match(hostname):
        raise HTTPException(status_code=400, detail="URL points to internal network")
    return url

@app.middleware("http")
async def security_middleware(request: Request, call_next):
    if _API_KEY:
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {_API_KEY}":
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})

    response = await call_next(request)
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=500, content={"success": False, "error": "Internal server error"})

# ── 请求模型 ──

class IngestRequest(BaseModel):
    source_path: str = Field(..., description="数据源路径")
    source_type: str = Field(default="", description="csv/excel/sqlite/api")
    source_name: str = Field(default="", description="数据源标识")
    sync_to_drap: bool = False

class CleanRequest(BaseModel):
    source_name: str
    missing_strategy: str = "fill_median"
    outlier_method: str = "iqr"
    dedup: bool = True

class IntegrateRequest(BaseModel):
    sources: list[dict]
    merge_strategy: str = "union"
    join_key: str = ""
    dedup_key: list[str] = []

class GovernRequest(BaseModel):
    action: str = "catalog"
    source_name: str = ""
    columns: list[str] = []
    sample_rows: list[list] = []

class AnalyzeRequest(BaseModel):
    source: str = ""
    analysis: str = "summary"
    column: str = ""
    top_n: int = 10

class AssetRequest(BaseModel):
    action: str = "valuate"
    source_name: str = ""
    asset_data: dict = {}
    valuation_method: str = "bsc"
    industry: str = ""

class PlanRequest(BaseModel):
    intent: str
    context: dict = {}

class ComplianceRequest(BaseModel):
    stage: str
    operation: str = ""
    data_desc: str = ""


# ── 端点 ──

@app.get("/api/health")
async def health():
    return {"status": "ok", "agents": reg.list_names(), "version": "0.1.0"}

@app.get("/api/agents")
async def list_agents():
    return {"agents": reg.list_info()}


# ── Plan ──
@app.post("/api/plan")
async def plan(req: PlanRequest):
    agent = reg.get("plan")
    if not agent:
        raise HTTPException(status_code=503, detail="Plan agent not available")
    result = await agent.run(intent=req.intent, context=req.context)
    return _respond(result)

# ── Ingest ──
@app.post("/api/ingest")
async def ingest(req: IngestRequest):
    try:
        _validate_source_path(req.source_path)
    except (ValueError, HTTPException):
        pass
    agent = reg.get("ingest")
    if not agent:
        raise HTTPException(status_code=503, detail="Ingest agent not available")
    result = await agent.run(
        source_path=req.source_path,
        source_type=req.source_type,
        source_name=req.source_name,
        sync_to_drap=req.sync_to_drap,
    )
    return _respond(result)

# ── Clean ──
@app.post("/api/clean")
async def clean(req: CleanRequest):
    agent = reg.get("clean")
    if not agent:
        raise HTTPException(status_code=503, detail="Clean agent not available")
    result = await agent.run(
        source_name=req.source_name,
        missing_strategy=req.missing_strategy,
        outlier_method=req.outlier_method,
        dedup=req.dedup,
    )
    return _respond(result)

# ── Integrate ──
@app.post("/api/integrate")
async def integrate(req: IntegrateRequest):
    agent = reg.get("integrate")
    if not agent:
        raise HTTPException(status_code=503, detail="Integrate agent not available")
    result = await agent.run(
        sources=req.sources,
        merge_strategy=req.merge_strategy,
        join_key=req.join_key,
        dedup_key=req.dedup_key,
    )
    return _respond(result)

# ── Govern ──
@app.post("/api/govern")
async def govern(req: GovernRequest):
    agent = reg.get("govern")
    if not agent:
        raise HTTPException(status_code=503, detail="Govern agent not available")
    result = await agent.run(
        action=req.action,
        source_name=req.source_name,
        columns=req.columns,
        sample_rows=req.sample_rows,
    )
    return _respond(result)

# ── Analyze ──
@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    agent = reg.get("analyze")
    if not agent:
        raise HTTPException(status_code=503, detail="Analyze agent not available")
    result = await agent.run(
        source=req.source,
        analysis=req.analysis,
        column=req.column,
        top_n=req.top_n,
    )
    return _respond(result)

# ── Web Scrape ──
class ScrapeRequest(BaseModel):
    url: str = Field(..., description="目标 URL")
    mode: str = Field(default="static", description="static/dynamic/api")
    extract_rules: list[dict] = Field(default_factory=list)

@app.post("/api/scrape")
async def scrape_web(req: ScrapeRequest):
    """Web 数据采集"""
    _validate_url(req.url)

    from src.connectors.web_scraper import WebScraperConnector

    connector = WebScraperConnector("web_scrape", {
        "mode": req.mode,
        "urls": [req.url],
        "extract_rules": req.extract_rules,
    })
    try:
        await connector.connect()
        result = await connector.scrape()

        if result.get("total_rows", 0) > 0:
            from src.agents.ingest_agent import IngestAgent
            agent = IngestAgent()
            bronze_result = await agent.run(
                source_path=req.url,
                source_type="web",
            )
            result["bronze"] = bronze_result.data if bronze_result.success else None

        return {"success": True, "data": result}
    except Exception as e:
        logger.warning(f"[Scrape] Error: {e}")
        return {"success": False, "error": "Scraping failed"}
    finally:
        await connector.close()


# ── Asset (→ DRAP) ──
@app.post("/api/asset/valuate")
async def asset_valuate(req: AssetRequest):
    agent = reg.get("asset")
    if not agent:
        raise HTTPException(status_code=503, detail="Asset agent not available")
    result = await agent.run(
        action=req.action,
        source_name=req.source_name,
        asset_data=req.asset_data,
        valuation_method=req.valuation_method,
        industry=req.industry,
    )
    return _respond(result)

# ── Compliance (→ 法规智能体) ──
@app.post("/api/compliance/check")
async def compliance_check(req: ComplianceRequest):
    checker = RegulationChecker()
    result = await checker.check_compliance(req.stage, req.operation, req.data_desc)
    return result

@app.get("/api/compliance/search")
async def compliance_search(
    stage: str = Query(...),
    keyword: str = Query(""),
    top_k: int = Query(5),
):
    checker = RegulationChecker()
    result = await checker.search_regulations(stage, keyword, top_k)
    return result

# ── 存储层状态 ──
@app.get("/api/storage/status")
async def storage_status():
    return {
        "bronze": BronzeLayer().get_stats(),
        "silver": SilverLayer().get_stats(),
        "gold": {"datasets": GoldLayer().list_datasets()},
    }

# ── DRAP 状态 ──
@app.get("/api/drap/status")
async def drap_status():
    import httpx
    settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            resp = await c.get(f"{settings.drap.base_url}/api/health")
            return {"connected": True, "drap": resp.json()}
    except Exception:
        return {"connected": False}


def _respond(result) -> dict:
    if result.success:
        return {"success": True, "data": result.data, "message": result.message}
    return {"success": False, "error": result.error, "message": result.message}
