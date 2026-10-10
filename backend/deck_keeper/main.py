import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__, migrate
from .config import get_env
from .routes import admin_settings, auth, cards, decks, oidc, scans, users
from .services import scanner, scryfall

STATIC = Path(__file__).parent / "static"
log = logging.getLogger("deck_keeper")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=get_env().log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    migrate.upgrade()
    tasks = []
    if app.state.background:
        tasks = [asyncio.create_task(scanner.worker_loop()), asyncio.create_task(scryfall.sync_loop())]
    log.info("MTG Deck Keeper %s ready", __version__)
    yield
    for t in tasks:
        t.cancel()


def create_app(background: bool = True) -> FastAPI:
    app = FastAPI(title="MTG Deck Keeper", version=__version__, lifespan=lifespan,
                  docs_url="/api/docs", openapi_url="/api/openapi.json", redoc_url=None)
    app.state.background = background
    for r in (auth.router, oidc.router, users.router, admin_settings.router, cards.router, scans.router,
              decks.router):
        app.include_router(r)

    @app.get("/api/health")
    def health():
        return {"ok": True, "version": __version__}

    if (STATIC / "assets").exists():
        app.mount("/assets", StaticFiles(directory=STATIC / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str, request: Request):
        if path.startswith("api/"):
            return JSONResponse({"detail": "Not found"}, status_code=404)
        file = (STATIC / path).resolve()
        if path and file.is_file() and STATIC.resolve() in file.parents:
            return FileResponse(file)
        index = STATIC / "index.html"
        if index.exists():
            return FileResponse(index, headers={"Cache-Control": "no-cache"})
        return JSONResponse({"detail": "Frontend not built. Run npm run build in frontend/."}, status_code=404)

    return app


app = create_app()
