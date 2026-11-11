import os
import sys
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool
from starlette.background import BackgroundTask

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.schemas import (
    HealthResponse, APIInfoResponse, UploadResponse,
    GenerateResponse, ResultResponse, ErrorResponse
)
from backend.config import settings
from backend.services import privsynth_service as svc

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("privsynth.api")


# ---------------------------------------------------------------------------
# Background Cleanup & Lifespan Management
# ---------------------------------------------------------------------------
CLEANUP_INTERVAL_SECONDS = 3600  # Run every hour


async def periodic_cleanup_task(interval_seconds: int = CLEANUP_INTERVAL_SECONDS):
    """Background task running periodic cleanup of expired synthetic results."""
    logger.info("Periodic cleanup worker started (interval: %ds, TTL: %dh)", interval_seconds, settings.RESULT_TTL_HOURS)
    # Run once at startup
    try:
        summary = await run_in_threadpool(svc.clean_expired_results)
        logger.info("Startup cleanup finished: %d file(s) deleted, %d error(s).", summary["deleted_files"], summary["errors"])
    except Exception as e:
        logger.warning("Startup cleanup encountered an error: %s", e)

    while True:
        try:
            await asyncio.sleep(interval_seconds)
            summary = await run_in_threadpool(svc.clean_expired_results)
            if summary["deleted_files"] > 0 or summary["errors"] > 0:
                logger.info("Periodic cleanup finished: %d file(s) deleted, %d error(s).", summary["deleted_files"], summary["errors"])
        except asyncio.CancelledError:
            logger.info("Periodic cleanup worker received cancellation, shutting down.")
            break
        except Exception as e:
            logger.warning("Periodic cleanup task encountered an error: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan managing startup validation and background workers."""
    logger.info("PrivSynth API starting up on %s:%d", settings.HOST, settings.PORT)
    logger.info("Storage directory: %s", settings.PRIVSYNTH_STORAGE_DIR)
    logger.info("Limits: max file size %d bytes, max rows %d", settings.MAX_FILE_BYTES, settings.MAX_UPLOAD_ROWS)

    try:
        storage_path = svc.ensure_storage_dir()
        logger.info("Storage directory verified and writable: %s", storage_path)
    except Exception as e:
        logger.critical("Storage initialization failure: %s", e)
        if not settings.IS_VERCEL:
            raise
        logger.warning("Running on Vercel: continuing startup despite storage warning (%s)", e)

    cleanup_task = None
    if not settings.IS_VERCEL:
        cleanup_task = asyncio.create_task(periodic_cleanup_task())
    else:
        logger.info("Running in Vercel Serverless environment: background periodic cleanup loop disabled.")
    try:
        yield
    finally:
        if cleanup_task is not None:
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("PrivSynth API shutdown complete.")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PrivSynth API",
    description=(
        "FastAPI backend for PrivSynth: An Adaptive Framework for "
        "Privacy-Preserving Synthetic Tabular Data Generation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS Configuration
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _error(msg: str, status: int = 400):
    return JSONResponse(status_code=status, content={"success": False, "error": msg})


def _require_csv(file: UploadFile):
    if not file or not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse, tags=["System"])
def health():
    return HealthResponse(status="ok", service="PrivSynth API")


@app.get("/api/", response_model=APIInfoResponse, tags=["System"])
def api_info():
    return APIInfoResponse(
        name="PrivSynth API",
        version="1.0.0",
        description="Synthetic tabular data generation with privacy-utility trade-off analysis.",
        endpoints=[
            "GET  /api/health",
            "GET  /api/",
            "POST /api/upload",
            "POST /api/generate",
            "GET  /api/results/{result_id}",
            "GET  /api/results/{result_id}/download",
        ]
    )


@app.post("/api/upload", response_model=UploadResponse, tags=["Data"])
async def upload(file: UploadFile = File(...)):
    _require_csv(file)
    content = await file.read()
    try:
        result = await run_in_threadpool(svc.upload_dataset, content, file.filename)
    except svc.RowLimitExceededError as e:
        logger.warning("Upload rejected (row limit): %s", e)
        return _error(str(e), status=413)
    except ValueError as e:
        logger.warning("Upload rejected (validation): %s", e)
        return _error(str(e), status=400)
    except RuntimeError as e:
        logger.error("Storage error during upload: %s", e)
        return _error("Storage service is temporarily unavailable.", status=503)
    except Exception:
        logger.exception("Unexpected error while processing uploaded file.")
        return _error("An unexpected error occurred while processing the file.", status=500)
    return result


@app.post("/api/generate", response_model=GenerateResponse, tags=["Generation"])
async def generate(file: UploadFile = File(...), generator: str = "ctgan"):
    _require_csv(file)
    if generator not in ("ctgan", "gaussian_copula"):
        return _error("Invalid generator. Choose 'ctgan' or 'gaussian_copula'.")
    content = await file.read()
    try:
        result = await run_in_threadpool(svc.generate_synthetic, content, generator)
    except svc.RowLimitExceededError as e:
        logger.warning("Generation rejected (row limit): %s", e)
        return _error(str(e), status=413)
    except ValueError as e:
        logger.warning("Generation rejected (validation): %s", e)
        return _error(str(e), status=400)
    except RuntimeError as e:
        logger.error("Storage error during generation: %s", e)
        return _error("Storage service is temporarily unavailable.", status=503)
    except Exception:
        logger.exception("Unexpected failure during generation pipeline.")
        return _error("An unexpected error occurred during synthetic data generation.", status=500)
    return result


@app.get("/api/results/{result_id}", response_model=ResultResponse, tags=["Results"])
def get_result(result_id: str):
    meta = svc.get_result(result_id)
    if meta is None:
        return _error(f"Result '{result_id}' not found.", status=404)
    return {**meta, "success": True}


@app.get("/api/results/{result_id}/download", tags=["Results"])
def download_result(result_id: str):
    dl_info = svc.get_csv_download(result_id)
    if dl_info is None:
        return _error(f"Result '{result_id}' not found.", status=404)

    if dl_info.get("type") == "redirect":
        return RedirectResponse(url=dl_info["url"], status_code=307)

    csv_path = dl_info["path"]
    # Shield file from background cleanup while client downloads
    svc.acquire_download(result_id)
    return FileResponse(
        path=csv_path,
        media_type="text/csv",
        filename=dl_info.get("filename", f"privsynth_synthetic_{result_id}.csv"),
        background=BackgroundTask(svc.release_download, result_id)
    )
