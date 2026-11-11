"""
backend/services/storage_backend.py - Pluggable Storage Backend for PrivSynth.

Provides:
  - LocalStorageBackend: Filesystem-based storage for local development & Docker VPS.
  - VercelBlobStorageBackend: Serverless storage via Vercel Blob REST API.
  - get_storage_backend(): Factory returning appropriate backend based on configuration.
"""
import os
import json
import time
import logging
import threading
from contextlib import contextmanager
from typing import Optional, Dict, Any, Tuple

from backend.config import settings

logger = logging.getLogger("privsynth.storage")

RESULT_META_SUFFIX = "_meta.json"

# In-process active file registry (for local filesystem synchronization)
_active_lock = threading.Lock()
_active_generations: set[str] = set()
_active_downloads: dict[str, int] = {}


def acquire_download_lock(result_id: str) -> None:
    with _active_lock:
        _active_downloads[result_id] = _active_downloads.get(result_id, 0) + 1


def release_download_lock(result_id: str) -> None:
    with _active_lock:
        if result_id in _active_downloads:
            _active_downloads[result_id] -= 1
            if _active_downloads[result_id] <= 0:
                del _active_downloads[result_id]


@contextmanager
def track_active_generation_lock(result_id: str):
    with _active_lock:
        _active_generations.add(result_id)
    try:
        yield
    finally:
        with _active_lock:
            _active_generations.discard(result_id)


def is_result_active_lock(result_id: str) -> bool:
    with _active_lock:
        return (result_id in _active_generations) or (_active_downloads.get(result_id, 0) > 0)


# Aliases for compatibility
acquire_download = acquire_download_lock
release_download = release_download_lock
track_active_generation = track_active_generation_lock
is_result_active = is_result_active_lock


# =============================================================================
# Local Storage Backend
# =============================================================================
class LocalStorageBackend:
    """Stores results on the local filesystem. Used for local development and Docker VPS."""

    def __init__(self, storage_dir: Optional[str] = None):
        self._custom_dir = storage_dir

    def get_storage_dir(self) -> str:
        target_dir = os.path.abspath(self._custom_dir or settings.PRIVSYNTH_STORAGE_DIR)
        try:
            os.makedirs(target_dir, exist_ok=True)
        except OSError as e:
            logger.error("Failed to access storage directory '%s': %s", target_dir, e)
            if settings.IS_VERCEL:
                fallback_dir = "/tmp/privsynth_storage"
                os.makedirs(fallback_dir, exist_ok=True)
                return fallback_dir
            raise RuntimeError(f"Storage directory '{target_dir}' is inaccessible.") from e
        return target_dir

    def ensure_writable(self) -> str:
        s_dir = self.get_storage_dir()
        test_file = os.path.join(s_dir, f".write_test_{os.getpid()}_{int(time.time())}.tmp")
        try:
            with open(test_file, "w") as f:
                f.write("ok")
            os.remove(test_file)
        except OSError as e:
            logger.critical("Storage directory '%s' is not writable: %s", s_dir, e)
            if settings.IS_VERCEL:
                fallback_dir = "/tmp/privsynth_storage"
                try:
                    os.makedirs(fallback_dir, exist_ok=True)
                    fb_test = os.path.join(fallback_dir, f".write_test_{os.getpid()}.tmp")
                    with open(fb_test, "w") as f:
                        f.write("ok")
                    os.remove(fb_test)
                    return fallback_dir
                except Exception:
                    pass
            raise RuntimeError(f"Storage directory '{s_dir}' is not writable.") from e
        return s_dir

    def save_result(self, result_id: str, csv_bytes: bytes, meta: dict) -> None:
        s_dir = self.get_storage_dir()
        csv_final = os.path.join(s_dir, f"{result_id}.csv")
        meta_final = os.path.join(s_dir, f"{result_id}{RESULT_META_SUFFIX}")
        csv_tmp = csv_final + ".tmp"
        meta_tmp = meta_final + ".tmp"

        with track_active_generation_lock(result_id):
            try:
                with open(csv_tmp, "wb") as f:
                    f.write(csv_bytes)
                with open(meta_tmp, "w", encoding="utf-8") as f:
                    json.dump(meta, f, default=str)
                os.replace(csv_tmp, csv_final)
                os.replace(meta_tmp, meta_final)
            finally:
                for tmp in (csv_tmp, meta_tmp):
                    if os.path.exists(tmp):
                        try:
                            os.remove(tmp)
                        except OSError:
                            pass

    def get_meta(self, result_id: str) -> Optional[dict]:
        s_dir = self.get_storage_dir()
        meta_path = os.path.join(s_dir, f"{result_id}{RESULT_META_SUFFIX}")
        if not os.path.exists(meta_path):
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to read metadata for result '%s': %s", result_id, e)
            return None

    def get_csv_download(self, result_id: str) -> Optional[dict]:
        s_dir = self.get_storage_dir()
        csv_path = os.path.join(s_dir, f"{result_id}.csv")
        if not os.path.exists(csv_path):
            return None
        return {
            "type": "file",
            "path": csv_path,
            "filename": f"privsynth_synthetic_{result_id}.csv"
        }

    def clean_expired(self, ttl_hours: Optional[int] = None) -> dict:
        ttl = ttl_hours if ttl_hours is not None else settings.RESULT_TTL_HOURS
        cutoff = time.time() - (ttl * 3600)
        s_dir = self.get_storage_dir()

        if not os.path.isdir(s_dir):
            return {"deleted_files": 0, "errors": 0}

        deleted = 0
        errors = 0
        try:
            entries = os.listdir(s_dir)
        except OSError as e:
            logger.error("Failed to list storage directory '%s': %s", s_dir, e)
            return {"deleted_files": 0, "errors": 1}

        for filename in entries:
            if filename == ".gitkeep":
                continue
            filepath = os.path.join(s_dir, filename)
            try:
                stat = os.stat(filepath)
            except OSError:
                continue

            res_id = None
            if filename.endswith(".csv"):
                res_id = filename[:-4]
            elif filename.endswith(RESULT_META_SUFFIX):
                res_id = filename[:-len(RESULT_META_SUFFIX)]
            elif filename.endswith(".tmp"):
                res_id = filename.split(".")[0]

            if res_id and is_result_active_lock(res_id):
                continue

            if stat.st_mtime < cutoff:
                try:
                    os.remove(filepath)
                    deleted += 1
                except OSError as e:
                    errors += 1
                    logger.warning("Failed to delete expired file '%s': %s", filename, e)

        return {"deleted_files": deleted, "errors": errors}


# =============================================================================
# Vercel Blob Storage Backend
# =============================================================================
class VercelBlobStorageBackend:
    """Stores generated CSVs and metadata in Vercel Blob for serverless deployment."""

    def __init__(self, token: Optional[str] = None):
        self.token = token or settings.BLOB_READ_WRITE_TOKEN
        if not self.token:
            raise ValueError("Vercel Blob token is missing. Configure BLOB_READ_WRITE_TOKEN.")
        self.api_url = "https://blob.vercel-storage.com"

    def ensure_writable(self) -> str:
        return "Vercel Blob (Cloud Storage)"

    def _headers(self, content_type: str) -> dict:
        return {
            "Authorization": f"Bearer {self.token}",
            "x-api-version": "7",
            "content-type": content_type,
        }

    def save_result(self, result_id: str, csv_bytes: bytes, meta: dict) -> None:
        import urllib.request
        # 1. Upload CSV to Vercel Blob
        csv_url = f"{self.api_url}/{result_id}.csv?addRandomSuffix=false"
        req_csv = urllib.request.Request(
            csv_url,
            data=csv_bytes,
            headers=self._headers("text/csv"),
            method="PUT"
        )
        try:
            with urllib.request.urlopen(req_csv) as resp:
                csv_resp = json.loads(resp.read().decode())
                download_url = csv_resp.get("downloadUrl", csv_resp.get("url"))
        except Exception as e:
            logger.error("Vercel Blob CSV upload failed for result %s: %s", result_id, e)
            raise RuntimeError(f"Failed to persist synthetic dataset to cloud storage: {e}") from e

        # Store download url directly into metadata envelope
        meta["blob_download_url"] = download_url

        # 2. Upload metadata to Vercel Blob
        meta_bytes = json.dumps(meta, default=str).encode("utf-8")
        meta_url = f"{self.api_url}/{result_id}{RESULT_META_SUFFIX}?addRandomSuffix=false"
        req_meta = urllib.request.Request(
            meta_url,
            data=meta_bytes,
            headers=self._headers("application/json"),
            method="PUT"
        )
        try:
            with urllib.request.urlopen(req_meta) as resp:
                meta_resp = json.loads(resp.read().decode())
                meta["blob_meta_url"] = meta_resp.get("url")
        except Exception as e:
            logger.error("Vercel Blob metadata upload failed for result %s: %s", result_id, e)
            raise RuntimeError(f"Failed to persist result metadata to cloud storage: {e}") from e

    def get_meta(self, result_id: str) -> Optional[dict]:
        import urllib.request
        # Fetch blob directly using token or list API
        url = f"{self.api_url}/{result_id}{RESULT_META_SUFFIX}"
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {self.token}", "x-api-version": "7"}
        )
        try:
            with urllib.request.urlopen(req) as resp:
                data = resp.read().decode()
                return json.loads(data)
        except Exception as e:
            logger.warning("Vercel Blob metadata lookup for '%s' returned: %s", result_id, e)
            return None

    def get_csv_download(self, result_id: str) -> Optional[dict]:
        meta = self.get_meta(result_id)
        if not meta or "blob_download_url" not in meta:
            return None
        # Return redirect to edge CDN URL to avoid function egress bandwidth overhead
        return {
            "type": "redirect",
            "url": meta["blob_download_url"],
            "filename": f"privsynth_synthetic_{result_id}.csv"
        }

    def clean_expired(self, ttl_hours: Optional[int] = None) -> dict:
        # In Vercel Blob, objects can be deleted via the delete API or managed via lifecycle
        return {"deleted_files": 0, "errors": 0}


# =============================================================================
# Backend Factory
# =============================================================================
def get_storage_backend():
    """Return VercelBlobStorageBackend if token is configured, else LocalStorageBackend."""
    if settings.BLOB_READ_WRITE_TOKEN:
        logger.info("Using Vercel Blob cloud storage backend.")
        return VercelBlobStorageBackend(token=settings.BLOB_READ_WRITE_TOKEN)
    return LocalStorageBackend()
