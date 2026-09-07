"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import __version__
from .api.routes import router
from .config import PACKAGE_ROOT, settings
from .core.loader import manager

log = logging.getLogger("meld_gui")

WEB_ROOT = PACKAGE_ROOT / "web"
STATIC_ROOT = WEB_ROOT / "static"
TEMPLATE_ROOT = WEB_ROOT / "templates"

templates = Jinja2Templates(directory=str(TEMPLATE_ROOT))

DESCRIPTION = """
A local interface for the **MELD** AI-generated text detector.

Every screen in the GUI is backed by an endpoint documented here, so the same
functionality is available from scripts and CI. Load the checkpoint with
`POST /api/model/load`, then score documents with `POST /api/analyze`.

MELD returns a calibrated *raw* score. Compare it to the threshold for your
document type and chosen false-positive rate rather than to 0.5.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    if settings.eager_load:
        log.info("Eager load requested; fetching checkpoint in the background.")
        manager.ensure_async()
    yield
    manager.unload()


def create_app() -> FastAPI:
    app = FastAPI(
        title="MELD Detector",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )

    app.include_router(router)
    app.mount("/static", StaticFiles(directory=str(STATIC_ROOT)), name="static")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request, "index.html", {"version": __version__, "repo_id": settings.repo_id}
        )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": f"{type(exc).__name__}: {exc}", "code": "internal"},
        )

    return app


app = create_app()
