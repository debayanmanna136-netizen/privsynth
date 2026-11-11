"""
test_backend.py — Integration tests for the PrivSynth FastAPI backend.

Uses a tiny in-memory CSV fixture to avoid expensive full training runs.
The CTGAN generation test is REAL (calls the actual pipeline), but uses
only 50 rows so training completes in under 30 seconds.
"""

import io
import os
import sys
import pytest

# Ensure project root importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Small CSV fixture (50 rows, no adult.data dependency)
# ---------------------------------------------------------------------------

def _make_tiny_csv(rows: int = 50) -> bytes:
    import pandas as pd
    import numpy as np
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "age": rng.integers(18, 70, size=rows),
        "hours_per_week": rng.integers(1, 80, size=rows),
        "capital_gain": rng.integers(0, 10000, size=rows),
        "education": rng.choice(["Bachelors", "HS-grad", "Masters", "PhD"], size=rows),
        "marital_status": rng.choice(["Married", "Single", "Divorced"], size=rows),
        "income": rng.choice(["<=50K", ">50K"], size=rows),
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


TINY_CSV = _make_tiny_csv(50)


# ---------------------------------------------------------------------------
# System endpoints
# ---------------------------------------------------------------------------

def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "PrivSynth" in body["service"]


def test_api_info():
    r = client.get("/api/")
    assert r.status_code == 200
    body = r.json()
    assert "name" in body
    assert "endpoints" in body


# ---------------------------------------------------------------------------
# Upload endpoint
# ---------------------------------------------------------------------------

def test_upload_valid():
    r = client.post("/api/upload", files={"file": ("test.csv", TINY_CSV, "text/csv")})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["rows"] == 50
    assert body["columns"] == 6
    assert len(body["preview"]) <= 5


def test_upload_non_csv():
    r = client.post("/api/upload", files={"file": ("data.txt", b"a,b\n1,2", "text/plain")})
    assert r.status_code in (400, 422)


def test_upload_empty_file():
    r = client.post("/api/upload", files={"file": ("empty.csv", b"", "text/csv")})
    body = r.json()
    assert body.get("success") is False or r.status_code == 400


def test_upload_invalid_csv():
    r = client.post("/api/upload", files={"file": ("bad.csv", b"\x00\x01\x02garbage", "text/csv")})
    body = r.json()
    # Either a parse error or success=False
    assert r.status_code in (200, 400)


# ---------------------------------------------------------------------------
# Generation endpoint (real pipeline, 50 rows)
# ---------------------------------------------------------------------------

def test_generate_real_ctgan():
    """Calls the actual CTGAN pipeline on 50 rows. ~20-30 seconds on CPU."""
    r = client.post(
        "/api/generate",
        files={"file": ("test.csv", TINY_CSV, "text/csv")},
        params={"generator": "ctgan"},
        timeout=300,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    result_id = body["result_id"]
    assert result_id

    # Result retrieval
    r2 = client.get(f"/api/results/{result_id}")
    assert r2.status_code == 200
    meta = r2.json()
    assert meta["success"] is True
    assert meta["rows"] == 50
    assert "fidelity" in meta["metrics"]
    assert len(meta["preview"]) <= 5

    # Download
    r3 = client.get(f"/api/results/{result_id}/download")
    assert r3.status_code == 200
    assert "text/csv" in r3.headers.get("content-type", "")
    # Verify downloaded content is non-empty and parseable
    import pandas as pd
    df = pd.read_csv(io.StringIO(r3.text))
    assert len(df) == 50


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_unknown_result_id():
    r = client.get("/api/results/nonexistent-id-12345")
    body = r.json()
    assert body.get("success") is False or r.status_code == 404


def test_invalid_generator():
    r = client.post(
        "/api/generate",
        files={"file": ("test.csv", TINY_CSV, "text/csv")},
        params={"generator": "unknown_gen"},
    )
    body = r.json()
    assert body.get("success") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
