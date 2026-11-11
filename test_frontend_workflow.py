import io
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def make_test_csv(rows=50):
    rng = np.random.default_rng(123)
    df = pd.DataFrame({
        "age": rng.integers(18, 65, rows),
        "hours": rng.integers(20, 60, rows),
        "capital": rng.integers(0, 5000, rows),
        "education": rng.choice(["BSc", "MSc", "PhD"], rows),
        "income": rng.choice(["<=50K", ">50K"], rows),
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()

TEST_CSV = make_test_csv(50)
PASS = []
FAIL = []

def check(name, cond, details=""):
    if cond:
        PASS.append(name)
        print(f"  [PASS] {name}")
    else:
        FAIL.append(name)
        print(f"  [FAIL] {name}: {details}")

print("=== Running Frontend-Backend Simulation Tests ===")

# 1. Health check
r = client.get("/api/health")
check("Health 200 OK", r.status_code == 200)
check("Health status ok", r.json().get("status") == "ok")

# 2. Upload valid CSV
r = client.post("/api/upload", files={"file": ("test.csv", TEST_CSV, "text/csv")})
check("Upload 200 OK", r.status_code == 200)
d = r.json()
check("Upload success is True", d.get("success") is True)
check("Upload row count is 50", d.get("rows") == 50)
check("Upload column count is 5", d.get("columns") == 5)
check("Upload column names present", d.get("column_names") == ["age", "hours", "capital", "education", "income"])
check("Upload preview has 5 rows", len(d.get("preview", [])) == 5)

# 3. Upload non-CSV
r = client.post("/api/upload", files={"file": ("test.txt", b"a,b\n1,2", "text/plain")})
check("Upload non-CSV rejected (400)", r.status_code == 400)

# 4. Upload empty file
r = client.post("/api/upload", files={"file": ("test.csv", b"", "text/csv")})
check("Upload empty CSV rejected (400)", r.status_code == 400)

# 5. Invalid generator parameter
r = client.post("/api/generate?generator=invalid", files={"file": ("test.csv", TEST_CSV, "text/csv")})
check("Invalid generator rejected (400)", r.status_code == 400)
check("Invalid generator error message", "Invalid generator" in r.json().get("error", ""))

# 6. Real CTGAN generation with query param
print("Running generation (CTGAN 50 rows)...")
r = client.post("/api/generate?generator=ctgan", files={"file": ("test.csv", TEST_CSV, "text/csv")})
check("Generate 200 OK", r.status_code == 200)
gen_resp = r.json()
check("Generate success is True", gen_resp.get("success") is True)
result_id = gen_resp.get("result_id", "")
check("Result ID returned", bool(result_id))
check("Generator is ctgan", gen_resp.get("generator") == "ctgan")

# 7. Result metadata retrieval
r = client.get(f"/api/results/{result_id}")
check("Results 200 OK", r.status_code == 200)
meta = r.json()
check("Results rows count 50", meta.get("rows") == 50)
check("Results columns count 5", meta.get("columns") == 5)
check("Results preview <= 5", len(meta.get("preview", [])) <= 5)
metrics = meta.get("metrics", {})
check("Metrics object present", bool(metrics))
check("Fidelity metrics present", "fidelity" in metrics)
check("Privacy metrics present", "privacy" in metrics)
check("Utility message present", "utility" in metrics)

# 8. Download generated CSV
r = client.get(f"/api/results/{result_id}/download")
check("Download 200 OK", r.status_code == 200)
check("Download Content-Type text/csv", "text/csv" in r.headers.get("content-type", ""))
df_dl = pd.read_csv(io.StringIO(r.text))
check("Downloaded rows count 50", len(df_dl) == 50)
check("Downloaded columns match original", list(df_dl.columns) == ["age", "hours", "capital", "education", "income"])

# 9. Non-existent result ID
r = client.get("/api/results/non-existent-uuid-1234")
check("Unknown ID 404 Not Found", r.status_code == 404)

print(f"\n==========================================")
print(f"TOTAL: {len(PASS) + len(FAIL)} | PASSED: {len(PASS)} | FAILED: {len(FAIL)}")
if FAIL:
    print(f"FAILED TESTS: {FAIL}")
    sys.exit(1)
else:
    print("ALL FRONTEND-BACKEND SIMULATION TESTS PASSED!")

