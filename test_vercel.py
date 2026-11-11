"""
test_vercel.py - Vercel Serverless Architecture & Compatibility Test Suite.

Tests:
1. api/index.py entrypoint export and ASGI application loading.
2. Vercel environment detection and CTGAN row limit enforcement.
3. Pluggable Storage factory (LocalStorageBackend vs VercelBlobStorageBackend).
4. VercelBlobStorageBackend PUT/GET operations and error handling (mocked).
5. Download endpoint behavior: HTTP 307 Redirect for Blob vs FileResponse for Local.
6. Static verification of vercel.json and .vercelignore.
"""
import io
import json
import os
import pytest
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.config import settings, Settings
from backend.services.storage_backend import (
    LocalStorageBackend,
    VercelBlobStorageBackend,
    get_storage_backend,
)
from backend.services import privsynth_service as svc
from backend.main import app

client = TestClient(app)


# -----------------------------------------------------------------------------
# 1. api/index.py entrypoint
# -----------------------------------------------------------------------------
def test_api_index_export():
    """Verify api/index.py successfully imports and exposes FastAPI app."""
    import ast
    import sys
    import api.index as api_entry
    assert hasattr(api_entry, "app")
    assert isinstance(api_entry.app, FastAPI)
    assert api_entry.app.title == "PrivSynth API"

    # Verify all expected API endpoints are registered on the app
    routes = [r.path for r in api_entry.app.routes]
    expected_endpoints = [
        "/api/health",
        "/api/",
        "/api/upload",
        "/api/generate",
        "/api/results/{result_id}",
        "/api/results/{result_id}/download",
    ]
    for ep in expected_endpoints:
        assert ep in routes, f"Expected endpoint {ep} not found in {routes}"

    # Verify app is defined as a top-level module statement for Vercel AST scanners
    repo_root = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(repo_root, "api", "index.py")
    with open(index_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    top_level_targets = []
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Name):
                    top_level_targets.append(t.id)
    assert "app" in top_level_targets, "app must be defined as an explicit top-level assignment for Vercel"


def test_pyproject_toml_entrypoint():
    """Verify pyproject.toml exists and configures the explicit Vercel tool entrypoint."""
    repo_root = os.path.dirname(os.path.abspath(__file__))
    pyproj_path = os.path.join(repo_root, "pyproject.toml")
    assert os.path.exists(pyproj_path), "pyproject.toml must exist"
    with open(pyproj_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "[tool.vercel]" in content
    assert 'entrypoint = "api.index:app"' in content


def test_api_index_does_not_import_torch(monkeypatch):
    """Verify api/index.py startup does not trigger heavy PyTorch/CTGAN imports."""
    import sys
    import subprocess
    cmd = [
        sys.executable,
        "-c",
        "import sys; from api.index import app; assert 'torch' not in sys.modules, 'torch was loaded during startup!'; print('OK')"
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"Startup imported torch: {proc.stderr}"
    assert "OK" in proc.stdout


# -----------------------------------------------------------------------------
# 2. Vercel Environment & CTGAN Limit Enforcement
# -----------------------------------------------------------------------------
def test_vercel_environment_flag(monkeypatch):
    """Verify Settings detects VERCEL and VERCEL_ENV environment variables."""
    monkeypatch.setenv("VERCEL", "1")
    s = Settings()
    assert s.IS_VERCEL is True

    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.setenv("VERCEL_ENV", "production")
    s2 = Settings()
    assert s2.IS_VERCEL is True

    monkeypatch.delenv("VERCEL_ENV", raising=False)
    s3 = Settings()
    assert s3.IS_VERCEL is False


def test_ctgan_row_limit_on_vercel(monkeypatch):
    """Verify CTGAN enforces VERCEL_MAX_CTGAN_ROWS when IS_VERCEL is True."""
    monkeypatch.setattr(settings, "IS_VERCEL", True)
    monkeypatch.setattr(settings, "VERCEL_MAX_CTGAN_ROWS", 50)

    # Generate CSV content with 60 rows (> limit of 50)
    lines = ["age,income,category"]
    for i in range(60):
        lines.append(f"{20 + i % 40},{30000 + i * 500},Cat_{(i % 3) + 1}")
    csv_bytes = "\n".join(lines).encode("utf-8")

    # CTGAN should be rejected with a clear, honest explanation
    with pytest.raises(ValueError) as excinfo:
        svc.generate_synthetic(csv_bytes, generator="ctgan")
    assert "CTGAN generation on Vercel Serverless is limited to 50 rows" in str(excinfo.value)
    assert "Gaussian Copula" in str(excinfo.value)

    # When IS_VERCEL is False, it should NOT trigger the Vercel limit
    monkeypatch.setattr(settings, "IS_VERCEL", False)
    # Validate dataframe alone accepts 60 rows
    import pandas as pd
    df = pd.read_csv(io.BytesIO(csv_bytes))
    svc.validate_dataframe(df)  # Should not raise


# -----------------------------------------------------------------------------
# 3. Storage Backend Factory
# -----------------------------------------------------------------------------
def test_storage_backend_factory(monkeypatch):
    """Verify get_storage_backend() switches based on BLOB_READ_WRITE_TOKEN."""
    monkeypatch.setattr(settings, "BLOB_READ_WRITE_TOKEN", None)
    backend_local = get_storage_backend()
    assert isinstance(backend_local, LocalStorageBackend)

    monkeypatch.setattr(settings, "BLOB_READ_WRITE_TOKEN", "vercel_blob_rw_test_token_123")
    backend_blob = get_storage_backend()
    assert isinstance(backend_blob, VercelBlobStorageBackend)
    assert backend_blob.token == "vercel_blob_rw_test_token_123"


# -----------------------------------------------------------------------------
# 4. VercelBlobStorageBackend REST Operations (Mocked)
# -----------------------------------------------------------------------------
def test_vercel_blob_save_and_retrieve():
    """Verify VercelBlobStorageBackend performs correct REST calls to Vercel Blob."""
    backend = VercelBlobStorageBackend(token="test_token_xyz")
    res_id = "11111111-2222-3333-4444-555555555555"
    sample_csv = b"col1,col2\n1,2\n3,4"
    sample_meta = {"result_id": res_id, "rows": 2}

    csv_blob_resp = json.dumps({
        "url": f"https://mock-blob.vercel-storage.com/{res_id}.csv",
        "downloadUrl": f"https://mock-blob.vercel-storage.com/{res_id}.csv?download=1"
    }).encode("utf-8")

    meta_blob_resp = json.dumps({
        "url": f"https://mock-blob.vercel-storage.com/{res_id}_meta.json"
    }).encode("utf-8")

    calls = []

    def mock_urlopen(request):
        calls.append(request)
        resp = MagicMock()
        url = request.full_url
        if ".csv" in url:
            resp.read.return_value = csv_blob_resp
        else:
            resp.read.return_value = meta_blob_resp
        resp.__enter__.return_value = resp
        return resp

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        backend.save_result(res_id, sample_csv, sample_meta)

    assert len(calls) == 2
    # Verify CSV call headers and URL
    assert calls[0].get_header("Authorization") == "Bearer test_token_xyz"
    assert calls[0].get_header("X-api-version") == "7"
    assert calls[0].get_method() == "PUT"

    # Verify download URL was injected into metadata envelope
    assert sample_meta["blob_download_url"] == f"https://mock-blob.vercel-storage.com/{res_id}.csv?download=1"

    # Test get_csv_download returns redirect descriptor
    with patch.object(backend, "get_meta", return_value=sample_meta):
        dl_info = backend.get_csv_download(res_id)
        assert dl_info["type"] == "redirect"
        assert dl_info["url"] == sample_meta["blob_download_url"]


# -----------------------------------------------------------------------------
# 5. Download Endpoint HTTP Routing (Redirect vs File)
# -----------------------------------------------------------------------------
def test_download_endpoint_vercel_blob_redirect(monkeypatch):
    """Verify GET /api/results/{id}/download issues HTTP 307 when storage is Vercel Blob."""
    test_id = "abcdef12-3456-7890-abcd-ef1234567890"
    redirect_target = f"https://mock-blob.vercel-storage.com/{test_id}.csv?download=1"

    def mock_get_download(rid):
        if rid == test_id:
            return {
                "type": "redirect",
                "url": redirect_target,
                "filename": f"privsynth_synthetic_{test_id}.csv"
            }
        return None

    monkeypatch.setattr(svc, "get_csv_download", mock_get_download)

    # Disable automatic redirect following to inspect 307 status
    response = client.get(f"/api/results/{test_id}/download", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers.get("location") == redirect_target


def test_download_endpoint_local_file(monkeypatch, tmp_path):
    """Verify GET /api/results/{id}/download serves FileResponse when storage is Local."""
    test_id = "12345678-1234-1234-1234-123456789abc"
    dummy_csv = tmp_path / f"{test_id}.csv"
    dummy_csv.write_text("a,b\n1,2\n")

    def mock_get_download(rid):
        if rid == test_id:
            return {
                "type": "file",
                "path": str(dummy_csv),
                "filename": f"privsynth_synthetic_{test_id}.csv"
            }
        return None

    monkeypatch.setattr(svc, "get_csv_download", mock_get_download)

    response = client.get(f"/api/results/{test_id}/download")
    assert response.status_code == 200
    assert "text/csv" in response.headers.get("content-type", "")
    assert f"privsynth_synthetic_{test_id}.csv" in response.headers.get("content-disposition", "")
    assert response.text.replace("\r\n", "\n") == "a,b\n1,2\n"


# -----------------------------------------------------------------------------
# 6. Static Verification of Vercel Configuration Files
# -----------------------------------------------------------------------------
def test_vercel_json_structure():
    """Verify vercel.json is present, valid JSON, and correctly configured."""
    repo_root = os.path.dirname(os.path.abspath(__file__))
    vj_path = os.path.join(repo_root, "vercel.json")
    assert os.path.exists(vj_path), "vercel.json must exist in root."

    with open(vj_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check functions configuration
    assert "functions" in data
    assert "api/index.py" in data["functions"]
    assert data["functions"]["api/index.py"]["memory"] == 1024
    assert data["functions"]["api/index.py"]["maxDuration"] == 60

    # Check rewrites
    assert "rewrites" in data
    sources = [r["source"] for r in data["rewrites"]]
    assert "/api/(.*)" in sources
    assert any("frontend/index.html" in r["destination"] for r in data["rewrites"])

    api_rewrite = next(r for r in data["rewrites"] if r["source"] == "/api/(.*)")
    assert api_rewrite["destination"] == "/api/index.py"


def test_vercelignore_exclusions():
    """Verify .vercelignore excludes non-production artifacts."""
    repo_root = os.path.dirname(os.path.abspath(__file__))
    vi_path = os.path.join(repo_root, ".vercelignore")
    assert os.path.exists(vi_path), ".vercelignore must exist in root."

    with open(vi_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    assert "notebooks/" in lines
    assert "data/" in lines
    assert ".venv/" in lines
    assert "Dockerfile" in lines
    assert "docker-compose.yml" in lines
    assert "test_*.py" in lines


def test_requirements_vercel_lightweight():
    """Verify requirements.txt, requirements-vercel.txt, and api/requirements.txt omit torch/ctgan/sdv."""
    repo_root = os.path.dirname(os.path.abspath(__file__))
    for fname in ["requirements.txt", "requirements-vercel.txt", os.path.join("api", "requirements.txt")]:
        fpath = os.path.join(repo_root, fname)
        assert os.path.exists(fpath), f"{fname} must exist."
        with open(fpath, "r", encoding="utf-8") as f:
            lines = [line.strip().lower() for line in f if line.strip() and not line.strip().startswith("#")]

        # Heavy deep learning packages must NOT be present as requirements
        for req in ["torch", "ctgan", "sdv"]:
            assert not any(line.startswith(req) for line in lines), f"{fname} must not require {req}."

        # Lightweight packages must be present
        for req in ["copulas", "rdt", "fastapi", "pandas", "scipy"]:
            assert any(line.startswith(req) for line in lines), f"{fname} must require {req}."


def test_vercel_json_install_command():
    """Verify vercel.json specifies custom installCommand and large function support."""
    repo_root = os.path.dirname(os.path.abspath(__file__))
    vj_path = os.path.join(repo_root, "vercel.json")
    with open(vj_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("installCommand") in ["pip install -r requirements.txt", "pip install -r requirements-vercel.txt"]
    assert data.get("env", {}).get("VERCEL_SUPPORT_LARGE_FUNCTIONS") == "1"


def test_vercel_read_only_storage_fallback(monkeypatch, tmp_path):
    """Verify storage backend handles read-only filesystem on Vercel gracefully."""
    monkeypatch.setattr(settings, "IS_VERCEL", True)
    # Point storage dir to a non-writable path simulation
    read_only_dir = tmp_path / "read_only_dir"
    read_only_dir.mkdir()
    # Make directory non-writable
    os.chmod(str(read_only_dir), 0o444)

    backend = LocalStorageBackend(storage_dir=str(read_only_dir))
    try:
        # Should not raise error on Vercel; should fallback to /tmp
        writable_path = backend.ensure_writable()
        assert writable_path is not None
    finally:
        os.chmod(str(read_only_dir), 0o777)


def test_gaussian_copula_api_generation_workflow():
    """Verify full upload -> Gaussian Copula generation -> results -> download workflow."""
    sample_csv = (
        "age,income,education\n"
        + "\n".join([f"{25 + i},{50000 + i * 1000},Bachelors" for i in range(25)])
    )
    files = {"file": ("test_gc.csv", sample_csv.encode("utf-8"), "text/csv")}

    # 1. Health check
    h_resp = client.get("/api/health")
    assert h_resp.status_code == 200
    assert h_resp.json()["status"] == "ok"

    # 2. Upload check
    up_resp = client.post("/api/upload", files=files)
    assert up_resp.status_code == 200
    assert up_resp.json()["rows"] == 25

    # 3. Generate check with Gaussian Copula
    files = {"file": ("test_gc.csv", sample_csv.encode("utf-8"), "text/csv")}
    gen_resp = client.post("/api/generate?generator=gaussian_copula", files=files)
    assert gen_resp.status_code == 200
    res_id = gen_resp.json()["result_id"]
    assert res_id is not None

    # 4. Results check
    res_resp = client.get(f"/api/results/{res_id}")
    assert res_resp.status_code == 200
    meta = res_resp.json()
    assert meta["generator"] == "gaussian_copula"
    assert meta["rows"] == 25
    assert "fidelity" in meta["metrics"]
    assert "privacy" in meta["metrics"]

    # 5. Download check
    dl_resp = client.get(f"/api/results/{res_id}/download")
    assert dl_resp.status_code == 200
    assert "text/csv" in dl_resp.headers.get("content-type", "")
    lines = dl_resp.text.strip().split("\n")
    assert len(lines) == 26  # header + 25 rows
