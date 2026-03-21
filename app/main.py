"""
FastAPI application — entry point.

Run with: uv run uvicorn app.main:app --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import AppConfig
from app.core.llm import LLMClient
from app.core.queue import Queue
from app.db.connection import Database
from app.api import routes, sse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Global app state — set during lifespan
config: AppConfig | None = None
llm: LLMClient | None = None
db: Database | None = None
queue_instance: Queue | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global config, llm, db, queue_instance

    config = AppConfig.from_env()
    logger.info(f"Starting {config.app_name}")
    logger.info(f"Models: {config.models}")

    # Init LLM client
    llm = LLMClient(
        api_key=config.openrouter_api_key,
        base_url=config.openrouter_base_url,
        app_name=config.app_name,
        app_url=config.app_url,
    )

    # Init database
    db = Database(data_dir=config.data_dir)
    main_db = db.get_main_db()

    # Attach cache to LLM client
    llm.set_cache_db(main_db)

    # Init queue
    queue_instance = Queue()
    sse.queue = queue_instance

    logger.info("Ready")
    yield

    # Cleanup
    await queue_instance.cancel_session("*")  # cancel all
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
