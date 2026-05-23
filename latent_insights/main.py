"""
FastAPI application — entry point.

Run with: uv run uvicorn latent_insights.main:app --reload
Or after pip install: latent-insights
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from latent_insights.config import AppConfig
from latent_insights.core.llm import LLMClient
from latent_insights.core.queue import Queue
from latent_insights.core.store import InvestigationStore
from latent_insights.db.connection import Database
from latent_insights.api import routes, sse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = AppConfig.from_env()
    logger.info(f"Starting {config.app_name}")
    logger.info(f"LLM provider: {config.llm_provider}, base: {config.llm_base_url}")
    logger.info(f"Models: {config.models}")

    llm = LLMClient(
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
        app_name=config.app_name,
        app_url=config.app_url,
        think=config.llm_think,
    )

    db = Database(data_dir=config.data_dir)
    queue_instance = Queue()
    store = InvestigationStore(data_dir=config.data_dir, queue=queue_instance)

    # Store on app.state for access in route handlers
    app.state.config = config
    app.state.llm = llm
    app.state.db = db
    app.state.queue = queue_instance
    app.state.store = store

    logger.info("Ready")
    yield

    # Cleanup — save all state to disk
    store.save_all()
    queue_instance.cancel_session("*")
    queue_instance.shutdown(wait=False)
    db.close()
    logger.info("Shutdown complete")


app = FastAPI(title="Latent Insights", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes.router, prefix="/api")
app.include_router(sse.router, prefix="/api")


@app.get("/health")
async def health():
    return {"status": "ok", "app": "latent-insights"}


def cli():
    """Console entry point — run the Latent Insights server."""
    import uvicorn
    uvicorn.run("latent_insights.main:app", host="0.0.0.0", port=8000)
