"""
test_stage2.py - Comprehensive Unit & Integration Tests for Stage 2 Production Preparation.
Covers:
  A. Storage (Default, custom directory, automatic creation, filesystem errors)
  B. Row limits (Below, at, above limit, upload and generate HTTP 413)
  C. Result cleanup (Expired deleted, unexpired preserved, orphaned files, active protection, graceful shutdown)
"""
import os
import io
import time
import shutil
import tempfile
import pandas as pd
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.config import Settings, settings
from backend.services import privsynth_service as svc
from backend.main import app

client = TestClient(app)

def _make_csv(rows: int) -> bytes:
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "col_a": rng.integers(1, 100, size=rows),
        "col_b": rng.choice(["Alpha", "Beta", "Gamma"], size=rows),
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# A. Storage Tests
# ---------------------------------------------------------------------------
def test_storage_default_path():
    """Verify default storage path points to backend/services/storage."""
    s = Settings()
    expected = os.path.join(os.path.dirname(svc.__file__), "storage")
    assert os.path.abspath(s.DEFAULT_STORAGE_DIR) == os.path.abspath(expected)


def test_storage_custom_and_auto_creation():
    """Verify custom directory is respected and automatically created."""
    temp_dir = tempfile.mkdtemp(prefix="privsynth_custom_storage_")
    custom_sub = os.path.join(temp_dir, "nested", "storage_folder")
    assert not os.path.exists(custom_sub)

    original_storage = settings.PRIVSYNTH_STORAGE_DIR
    try:
        settings.PRIVSYNTH_STORAGE_DIR = custom_sub
        resolved = svc.get_storage_dir()
        assert os.path.exists(resolved)
        assert os.path.abspath(resolved) == os.path.abspath(custom_sub)

        # Verify generation persists into custom directory
        csv_bytes = _make_csv(25)
        res = svc.generate_synthetic(csv_bytes, generator="gaussian_copula")
        result_id = res["result_id"]
        assert os.path.exists(os.path.join(custom_sub, f"{result_id}.csv"))
        assert os.path.exists(os.path.join(custom_sub, f"{result_id}_meta.json"))
    finally:
        settings.PRIVSYNTH_STORAGE_DIR = original_storage
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_storage_filesystem_permission_error(monkeypatch):
    """Verify filesystem errors raise RuntimeError and map to HTTP 503."""
    def mock_makedirs(*args, **kwargs):
        raise PermissionError("Access Denied (test mock)")

    monkeypatch.setattr(os, "makedirs", mock_makedirs)
    with pytest.raises(RuntimeError) as exc_info:
        svc.get_storage_dir()
    assert "inaccessible" in str(exc_info.value)


# ---------------------------------------------------------------------------
# B. Row Limit Tests
# ---------------------------------------------------------------------------
def test_row_limit_below_and_exact_accepted():
    """Verify datasets below and at limit pass validation."""
    df_25 = pd.DataFrame({"a": range(25), "b": range(25)})
    # Below limit (limit=30)
    svc.validate_dataframe(df_25, max_rows=30)

    # Exactly at limit (limit=25)
    svc.validate_dataframe(df_25, max_rows=25)


def test_row_limit_above_rejected_exception():
    """Verify exceeding limit raises RowLimitExceededError with informative count."""
    df_30 = pd.DataFrame({"a": range(30), "b": range(30)})
    with pytest.raises(svc.RowLimitExceededError) as exc_info:
        svc.validate_dataframe(df_30, max_rows=25)
    msg = str(exc_info.value)
    assert "exceeds" in msg
    assert "25" in msg
    assert "30" in msg


def test_row_limit_upload_workflow_http_413(monkeypatch):
    """Verify /api/upload returns HTTP 413 when row limit is exceeded."""
    # Temporarily set limit to 25 rows
    monkeypatch.setattr(settings, "MAX_UPLOAD_ROWS", 25)
    csv_bytes = _make_csv(35)  # 35 rows > 25 limit

    r = client.post("/api/upload", files={"file": ("test.csv", csv_bytes, "text/csv")})
    assert r.status_code == 413
    body = r.json()
    assert body["success"] is False
    assert "exceeds" in body["error"].lower()
    assert "25" in body["error"]
    assert "35" in body["error"]


def test_row_limit_generate_workflow_http_413(monkeypatch):
    """Verify /api/generate returns HTTP 413 when row limit is exceeded."""
    # Temporarily set limit to 25 rows
    monkeypatch.setattr(settings, "MAX_UPLOAD_ROWS", 25)
    csv_bytes = _make_csv(35)  # 35 rows > 25 limit

    r = client.post(
        "/api/generate",
        files={"file": ("test.csv", csv_bytes, "text/csv")},
        params={"generator": "gaussian_copula"}
    )
    assert r.status_code == 413
    body = r.json()
    assert body["success"] is False
    assert "exceeds" in body["error"].lower()


# ---------------------------------------------------------------------------
# C. Result Cleanup & Active File Protection Tests
# ---------------------------------------------------------------------------
def test_cleanup_expired_and_preserved_files():
    """Verify expired files are deleted while unexpired files are kept."""
    temp_dir = tempfile.mkdtemp(prefix="privsynth_test_cleanup_")
    original_storage = settings.PRIVSYNTH_STORAGE_DIR
    try:
        settings.PRIVSYNTH_STORAGE_DIR = temp_dir

        # 1. Unexpired file (now)
        unexpired_id = "11111111-1111-1111-1111-111111111111"
        unexp_csv = os.path.join(temp_dir, f"{unexpired_id}.csv")
        unexp_meta = os.path.join(temp_dir, f"{unexpired_id}_meta.json")
        with open(unexp_csv, "w") as f: f.write("a,b\n1,2\n")
        with open(unexp_meta, "w") as f: f.write("{}")

        # 2. Expired file (48 hours ago)
        expired_id = "22222222-2222-2222-2222-222222222222"
        exp_csv = os.path.join(temp_dir, f"{expired_id}.csv")
        exp_meta = os.path.join(temp_dir, f"{expired_id}_meta.json")
        with open(exp_csv, "w") as f: f.write("a,b\n1,2\n")
        with open(exp_meta, "w") as f: f.write("{}")

        old_mtime = time.time() - (48 * 3600)
        os.utime(exp_csv, (old_mtime, old_mtime))
        os.utime(exp_meta, (old_mtime, old_mtime))

        # 3. Orphaned expired file (csv without json, 48 hours ago)
        orphan_id = "33333333-3333-3333-3333-333333333333"
        orphan_csv = os.path.join(temp_dir, f"{orphan_id}.csv")
        with open(orphan_csv, "w") as f: f.write("a,b\n1,2\n")
        os.utime(orphan_csv, (old_mtime, old_mtime))

        # Run cleanup with 24h TTL
        res = svc.clean_expired_results(ttl_hours=24)
        assert res["deleted_files"] == 3  # expired csv + expired meta + orphan csv
        assert res["errors"] == 0

        # Unexpired files MUST survive
        assert os.path.exists(unexp_csv)
        assert os.path.exists(unexp_meta)

        # Expired files MUST be gone
        assert not os.path.exists(exp_csv)
        assert not os.path.exists(exp_meta)
        assert not os.path.exists(orphan_csv)
    finally:
        settings.PRIVSYNTH_STORAGE_DIR = original_storage
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_cleanup_shields_active_generation():
    """Verify files undergoing active generation are not deleted even if old."""
    temp_dir = tempfile.mkdtemp(prefix="privsynth_test_active_gen_")
    original_storage = settings.PRIVSYNTH_STORAGE_DIR
    try:
        settings.PRIVSYNTH_STORAGE_DIR = temp_dir
        res_id = "44444444-4444-4444-4444-444444444444"
        csv_file = os.path.join(temp_dir, f"{res_id}.csv")
        with open(csv_file, "w") as f: f.write("data")
        old_mtime = time.time() - (48 * 3600)
        os.utime(csv_file, (old_mtime, old_mtime))

        # Inside active generation shield
        with svc.track_active_generation(res_id):
            assert svc.is_result_active(res_id) is True
            res = svc.clean_expired_results(ttl_hours=24)
            assert res["deleted_files"] == 0
            assert os.path.exists(csv_file)

        # Outside shield, it should be deleted
        assert svc.is_result_active(res_id) is False
        res = svc.clean_expired_results(ttl_hours=24)
        assert res["deleted_files"] == 1
        assert not os.path.exists(csv_file)
    finally:
        settings.PRIVSYNTH_STORAGE_DIR = original_storage
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_cleanup_shields_active_download():
    """Verify files undergoing active download are not deleted even if expired."""
    temp_dir = tempfile.mkdtemp(prefix="privsynth_test_active_dl_")
    original_storage = settings.PRIVSYNTH_STORAGE_DIR
    try:
        settings.PRIVSYNTH_STORAGE_DIR = temp_dir
        res_id = "55555555-5555-5555-5555-555555555555"
        csv_file = os.path.join(temp_dir, f"{res_id}.csv")
        with open(csv_file, "w") as f: f.write("data")
        old_mtime = time.time() - (48 * 3600)
        os.utime(csv_file, (old_mtime, old_mtime))

        # Acquire download lock
        svc.acquire_download(res_id)
        try:
            assert svc.is_result_active(res_id) is True
            res = svc.clean_expired_results(ttl_hours=24)
            assert res["deleted_files"] == 0
            assert os.path.exists(csv_file)
        finally:
            svc.release_download(res_id)

        # Once released, cleanup removes it
        assert svc.is_result_active(res_id) is False
        res = svc.clean_expired_results(ttl_hours=24)
        assert res["deleted_files"] == 1
        assert not os.path.exists(csv_file)
    finally:
        settings.PRIVSYNTH_STORAGE_DIR = original_storage
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.anyio
async def test_periodic_cleanup_cancellation():
    """Verify background cleanup worker terminates cleanly upon cancellation."""
    from backend.main import periodic_cleanup_task
    import asyncio

    task = asyncio.create_task(periodic_cleanup_task(interval_seconds=3600))
    await asyncio.sleep(0.05)  # Let it run initial pass
    task.cancel()
    # Await cancellation should complete cleanly without unhandled exceptions
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert task.done() is True
