import os
import uuid
import json
import time
import sys
import threading
import logging
import pandas as pd
from contextlib import contextmanager
from typing import Optional

# Ensure the privsynth package is importable from project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from privsynth.pipeline.pipeline import run_pipeline
from privsynth.profiling.profiler import profile_dataset
from privsynth.evaluation.fidelity import evaluate_fidelity
from privsynth.evaluation.privacy import evaluate_privacy_proximity
from privsynth.config import CTGAN_DEFAULT_CONFIG
from backend.config import settings
from backend.services.storage_backend import (
    get_storage_backend,
    acquire_download_lock,
    release_download_lock,
    track_active_generation_lock,
    is_result_active_lock,
    RESULT_META_SUFFIX,
)

logger = logging.getLogger("privsynth.service")

MIN_ROWS = 20

# Aliases preserving thread-safe active file registry compatibility
acquire_download = acquire_download_lock
release_download = release_download_lock
track_active_generation = track_active_generation_lock
is_result_active = is_result_active_lock


# ---------------------------------------------------------------------------
# Storage Management
# ---------------------------------------------------------------------------
def get_storage_dir() -> str:
    """Return the configured storage directory path, creating it if necessary."""
    backend = get_storage_backend()
    if hasattr(backend, "get_storage_dir"):
        return backend.get_storage_dir()
    return os.path.abspath(settings.PRIVSYNTH_STORAGE_DIR)


def ensure_storage_dir() -> str:
    """Startup check validating storage directory availability and writability."""
    backend = get_storage_backend()
    return backend.ensure_writable()


# ---------------------------------------------------------------------------
# Custom Exceptions & Validation
# ---------------------------------------------------------------------------
class RowLimitExceededError(ValueError):
    """Raised when an uploaded or generation dataset exceeds MAX_UPLOAD_ROWS."""
    def __init__(self, row_count: int, max_rows: int):
        self.row_count = row_count
        self.max_rows = max_rows
        super().__init__(
            f"Dataset exceeds the maximum allowed limit of {max_rows:,} rows (received {row_count:,} rows)."
        )


def validate_dataframe(df: pd.DataFrame, max_rows: Optional[int] = None) -> None:
    """Validate DataFrame dimensions and constraints. Raises ValueError or RowLimitExceededError."""
    if df.empty:
        raise ValueError("The uploaded dataset is empty.")
    if len(df) < MIN_ROWS:
        raise ValueError(f"The uploaded dataset contains too few rows ({len(df)}). Minimum required: {MIN_ROWS}.")
    if len(df.columns) < 2:
        raise ValueError("The uploaded dataset must have at least 2 columns.")
    limit = max_rows if max_rows is not None else settings.MAX_UPLOAD_ROWS
    if len(df) > limit:
        raise RowLimitExceededError(len(df), limit)


def is_valid_uuid(val: str) -> bool:
    """Check if the provided string is a valid UUID to prevent path traversal."""
    try:
        u = uuid.UUID(str(val).strip())
        return str(u) == str(val).strip().lower()
    except (ValueError, AttributeError, TypeError):
        return False


def read_csv_bytes(content: bytes) -> pd.DataFrame:
    """Parse raw CSV bytes into a DataFrame, with encoding fallback and NA handling."""
    import io
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(content), encoding=enc, na_values="?", skipinitialspace=True)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return pd.read_csv(io.BytesIO(content), na_values="?", skipinitialspace=True)


# ---------------------------------------------------------------------------
# Core Service Functions
# ---------------------------------------------------------------------------
def upload_dataset(content: bytes, filename: str) -> dict:
    """Validate and profile an uploaded CSV, returning summary info."""
    if len(content) == 0:
        raise ValueError("Uploaded file is empty.")
    if len(content) > settings.MAX_FILE_BYTES:
        raise ValueError(f"Uploaded file exceeds the size limit ({len(content)} > {settings.MAX_FILE_BYTES} bytes).")
    try:
        df = read_csv_bytes(content)
    except Exception:
        raise ValueError("The uploaded file could not be parsed as a valid CSV.")

    validate_dataframe(df)

    profile = profile_dataset(df)
    preview = df.head(5).fillna("").to_dict(orient="records")

    return {
        "success": True,
        "filename": os.path.basename(filename),
        "rows": profile["num_rows"],
        "columns": profile["num_cols"],
        "column_names": profile["columns"],
        "missing_values": profile["missing_values_total"],
        "preview": preview
    }


def generate_synthetic(content: bytes, generator: str = "ctgan") -> dict:
    """Run the PrivSynth pipeline and atomically persist results; return result_id."""
    if len(content) == 0:
        raise ValueError("Uploaded file is empty.")
    if len(content) > settings.MAX_FILE_BYTES:
        raise ValueError(f"Uploaded file exceeds the size limit ({len(content)} > {settings.MAX_FILE_BYTES} bytes).")
    try:
        df = read_csv_bytes(content)
    except Exception:
        raise ValueError("The uploaded file could not be parsed as a valid CSV.")

    validate_dataframe(df)

    # Serverless runtime protection: prevent CTGAN from timing out on Vercel Hobby
    if generator == "ctgan" and settings.IS_VERCEL and len(df) > settings.VERCEL_MAX_CTGAN_ROWS:
        raise ValueError(
            f"CTGAN generation on Vercel Serverless is limited to {settings.VERCEL_MAX_CTGAN_ROWS:,} rows "
            f"to prevent serverless execution timeout (received {len(df):,} rows). "
            f"Please use 'Gaussian Copula' for larger datasets on Vercel, or run PrivSynth locally/Docker."
        )

    # Run the existing pipeline
    result = run_pipeline(df, generator=generator)
    synth_df: pd.DataFrame = result["synthetic_data"]

    # Light fidelity & privacy metrics (no ML utility — avoids min-row issues)
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in df.select_dtypes(exclude=["number"]).columns.tolist()]
    cat_cols_no_target = cat_cols

    try:
        fidelity = evaluate_fidelity(df, synth_df, num_cols, cat_cols_no_target)
    except Exception as e:
        fidelity = {"error": str(e)}

    try:
        privacy = evaluate_privacy_proximity(df, synth_df, num_cols, cat_cols_no_target)
    except Exception as e:
        privacy = {"error": str(e)}

    result_id = str(uuid.uuid4())

    meta = {
        "result_id": result_id,
        "generator": generator,
        "configuration": CTGAN_DEFAULT_CONFIG,
        "rows": len(synth_df),
        "columns": len(synth_df.columns),
        "validation": result["validation"],
        "metrics": {
            "fidelity": fidelity,
            "utility": "Not evaluated in API pipeline (requires labeled target column)",
            "privacy": privacy
        },
        "preview": synth_df.head(5).fillna("").to_dict(orient="records")
    }

    # Delegate persistence to configured storage backend (local filesystem or Vercel Blob)
    backend = get_storage_backend()
    csv_bytes = synth_df.to_csv(index=False).encode("utf-8")
    backend.save_result(result_id, csv_bytes, meta)

    return {
        "success": True,
        "result_id": result_id,
        "generator": generator,
        "message": "Synthetic dataset generated successfully."
    }


def get_result(result_id: str) -> Optional[dict]:
    """Load and return persisted result metadata by ID."""
    if not is_valid_uuid(result_id):
        return None
    backend = get_storage_backend()
    return backend.get_meta(result_id)


def get_csv_download(result_id: str) -> Optional[dict]:
    """Retrieve download routing dictionary ({type: 'file', path: ...} or {type: 'redirect', url: ...})."""
    if not is_valid_uuid(result_id):
        return None
    backend = get_storage_backend()
    return backend.get_csv_download(result_id)


def get_csv_path(result_id: str) -> Optional[str]:
    """Return local path to generated CSV if it exists on local filesystem."""
    if not is_valid_uuid(result_id):
        return None
    info = get_csv_download(result_id)
    if info and info.get("type") == "file":
        return info.get("path")
    return None


# ---------------------------------------------------------------------------
# Result Cleanup Mechanism
# ---------------------------------------------------------------------------
def clean_expired_results(ttl_hours: Optional[int] = None) -> dict:
    """Scan storage and prune expired results via the configured storage backend."""
    backend = get_storage_backend()
    return backend.clean_expired(ttl_hours=ttl_hours)
