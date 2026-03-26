"""Charge-SKRL Training Dashboard — 訓練配置一覽介面

啟動方式：
    cd /home/aa/IsaacLab
    python scripts/reinforcement_learning/skrl/dashboard/app.py
    # 或
    python -m uvicorn scripts.reinforcement_learning.skrl.dashboard.app:app --host 0.0.0.0 --port 8050

瀏覽器開啟 http://localhost:8050
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DASHBOARD_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = DASHBOARD_DIR / "templates"
STATIC_DIR = DASHBOARD_DIR / "static"

# ---------------------------------------------------------------------------
# Data — loaded once at startup (no GPU, no torch)
# ---------------------------------------------------------------------------
_CFG: dict = {}


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load all config data on startup."""
    global _CFG
    from config_parser import load_all_configs  # noqa: E402
    from nn_diagram import generate_nn_svg  # noqa: E402

    _CFG = load_all_configs()
    _CFG["nn_svg"] = generate_nn_svg()
    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Charge-SKRL Training Dashboard", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "cfg": _CFG, "cfg_json": json.dumps(_CFG, ensure_ascii=False, default=str)},
    )


@app.get("/api/config", response_class=JSONResponse)
async def api_config():
    return JSONResponse(content=_CFG)


# ---------------------------------------------------------------------------
# __main__ entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    # Add dashboard dir to sys.path so `from config_parser import ...` works
    sys.path.insert(0, str(DASHBOARD_DIR))

    import uvicorn

    uvicorn.run(
        f"{Path(__file__).stem}:app",
        host="0.0.0.0",
        port=8050,
        reload=False,
        log_level="info",
        app_dir=str(DASHBOARD_DIR),
    )
