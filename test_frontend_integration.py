# -*- coding: utf-8 -*-
"""
Integration smoke test for the frontend<->backend workflow.
Simulates exactly what the browser JS does:
  1. GET /api/health
  2. POST /api/upload  (tiny CSV)
  3. POST /api/generate (tiny CSV, ctgan)
  4. GET  /api/results/{id}
  5. GET  /api/results/{id}/download → verify CSV
"""
import io
import time
import requests
import pandas as pd
import numpy as np

BASE = "http://127.0.0.1:8000"

def tiny_csv(rows=50):
    rng = np.random.default_rng(99)
    df = pd.DataFrame({
        "age": rng.integers(18, 70, rows),
        "hours": rng.integers(1, 80, rows),
        "capital": rng.integers(0, 5000, rows),
        "education": rng.choice(["HS", "BSc", "MSc"], rows),
        "income": rng.choice(["<=50K", ">50K"], rows),
    })
    buf = io.BytesIO(); df.to_csv(buf, index=False); return buf.getvalue()

CSV = tiny_csv()
PASS = []
FAIL = []

def check(name, cond, info=""):
    if cond: PASS.append(name); print(f"  PASS  {name}")
    else: FAIL.append(name); print(f"  FAIL  {name}  {info}")

# 1. Health
print("\n[1] Health")
r = requests.get(f"{BASE}/api/health", timeout=5)
check("status 200", r.status_code == 200)
check("status=ok", r.json().get("status") == "ok")

# 2. Upload
print("\n[2] Upload")
r = requests.post(f"{BASE}/api/upload", files={"file": ("t.csv", CSV, "text/csv")}, timeout=15)
check("upload 200", r.status_code == 200)
d = r.json()
check("success=true", d.get("success"))
check("rows=50", d.get("rows") == 50)
check("columns=5", d.get("columns") == 5)
check("preview<=5 rows", len(d.get("preview", [])) <= 5)
check("column_names present", "column_names" in d)

# 2b. Upload non-CSV
print("\n[2b] Non-CSV rejection")
r2 = requests.post(f"{BASE}/api/upload", files={"file": ("t.txt", b"a,b\n1,2", "text/plain")}, timeout=5)
check("non-csv rejected (4xx)", r2.status_code >= 400)

# 3. Generate
print("\n[3] Generate (real CTGAN, 50 rows — may take ~30s)")
r = requests.post(
    f"{BASE}/api/generate",
    files={"file": ("t.csv", CSV, "text/csv")},
    params={"generator": "ctgan"},
    timeout=300
)
check("generate 200", r.status_code == 200)
d = r.json()
check("success=true", d.get("success"))
result_id = d.get("result_id", "")
check("result_id non-empty", bool(result_id))
check("generator=ctgan", d.get("generator") == "ctgan")

# 4. Result metadata
print("\n[4] Result metadata")
r = requests.get(f"{BASE}/api/results/{result_id}", timeout=10)
check("result 200", r.status_code == 200)
m = r.json()
check("rows=50", m.get("rows") == 50)
check("metrics present", "metrics" in m)
check("fidelity key in metrics", "fidelity" in (m.get("metrics") or {}))
check("preview in result", len(m.get("preview", [])) <= 5)

# 5. Download
print("\n[5] Download")
r = requests.get(f"{BASE}/api/results/{result_id}/download", timeout=10)
check("download 200", r.status_code == 200)
check("content-type csv", "text/csv" in r.headers.get("content-type",""))
df_dl = pd.read_csv(io.StringIO(r.text))
check("downloaded rows=50", len(df_dl) == 50)
check("downloaded cols=5", len(df_dl.columns) == 5)

# 6. Unknown ID
print("\n[6] Unknown result ID")
r = requests.get(f"{BASE}/api/results/definitely-not-real-id", timeout=5)
b = r.json()
check("error returned", b.get("success") is False or r.status_code == 404)

# Report
print(f"\n{'='*40}")
print(f"PASS: {len(PASS)}   FAIL: {len(FAIL)}")
if FAIL: print(f"Failed: {FAIL}")
else: print("All checks passed!")
