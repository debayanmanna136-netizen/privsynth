# PrivSynth

**PrivSynth: An Adaptive Framework for Privacy-Preserving Synthetic Tabular Data Generation**

A research and engineering project evaluating the fidelity-utility-privacy trade-off of synthetic tabular data generators.

---

## Project Structure

```
privsynth/          # Reusable Python package (generators, evaluation, pipeline)
notebooks/          # Research experiment notebooks (01–14)
backend/            # FastAPI HTTP backend
frontend/           # Vanilla HTML/CSS/JS frontend application
│   ├── index.html  # Landing page and interactive workspace
│   ├── css/
│   │   └── styles.css  # Editorial design system and responsive styles
│   ├── js/
│   │   └── app.js      # Vanilla JS SPA and Fetch API integration
│   └── assets/
│       └── .gitkeep
data/               # Raw datasets, synthetic outputs, results
```

---

## Installation & Environment Setup

### 1. Environment Configuration

PrivSynth supports centralized configuration via environment variables with a local `.env` file fallback:

```bash
# Copy the template to create your local environment file
cp .env.example .env
```

**How Environment Variables Are Loaded:**
- Environment variables are managed centrally via [`backend/config.py`](file:///c:/.vscode/privsynth/backend/config.py).
- Platform-provided environment variables (e.g., in Docker, Cloud Run, or CI/CD) always take precedence over values defined in `.env` (`override=False`).
- If no `.env` file or environment variables are provided, sensible local development defaults are used automatically.

### Supported Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `HOST` | `127.0.0.1` | Network interface to bind the API server. |
| `PORT` | `8000` | Port for the API server (validated positive integer). |
| `ALLOWED_ORIGINS` | Dev ports (`3000, 5500, 8080`) | Comma-separated allowed CORS origins or `*`. |
| `PRIVSYNTH_STORAGE_DIR` | `backend/services/storage/` | Path to persistent directory for synthetic CSVs and metadata. Automatically created if missing. |
| `MAX_FILE_BYTES` | `52428800` (50 MB) | File size ceiling for CSV uploads. Exceeding returns HTTP 400. |
| `MAX_UPLOAD_ROWS` | `50000` | Dataset row count ceiling. Exceeding returns HTTP 413. |
| `RESULT_TTL_HOURS` | `24` | Retention period in hours before synthetic files are automatically purged. |

### 2. Dependency Installation

PrivSynth maintains separated, strictly pinned dependency manifests:

```bash
# Full Production Runtime (Docker VPS + CTGAN Deep Learning + PyTorch)
pip install -r requirements.txt

# Lightweight Serverless Runtime (Vercel Serverless + Gaussian Copula, Zero PyTorch)
pip install -r requirements-vercel.txt

# Development, Testing & Research (Pytest, HTTPX, Jupyter, Matplotlib, Seaborn)
pip install -r requirements-dev.txt
```

---

## Quick Start (Running Backend & Frontend)

### 1. Start the FastAPI Backend

From the project root (`c:\.vscode\privsynth`):

```bash
# Using project virtual environment
.venv\Scripts\uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at:
- **API Base**: `http://127.0.0.1:8000`
- **Interactive Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc UI**: `http://127.0.0.1:8000/redoc`

### 2. Start the Frontend Server

In a new terminal, serve the `frontend/` directory on port 5500 (or use the VS Code Live Server extension on port 5500):

```bash
# Serve frontend on port 5500
python -m http.server 5500 --directory frontend
```

### 3. Open the Application in Your Browser

Navigate to:
👉 **`http://127.0.0.1:5500`** (or `http://localhost:5500`)

---

## Frontend Architecture & API Communication

The PrivSynth frontend is built strictly with **HTML5, CSS3, and vanilla JavaScript** without heavy external frameworks.

```
Browser (Vanilla JS / Fetch API)
    │
    ├─► GET  /api/health                     (Pings backend status dot)
    ├─► POST /api/upload                     (Multipart CSV upload & profiling)
    ├─► POST /api/generate?generator=...     (Dispatches CTGAN or Gaussian Copula)
    ├─► GET  /api/results/{result_id}        (Fetches fidelity, privacy metrics & preview)
    └─► GET  /api/results/{result_id}/download (Streams generated synthetic CSV file)
```

- **Configurable Base URL**: Defined once in `frontend/js/app.js` (`const API_BASE_URL = "http://127.0.0.1:8000"`).
- **Safe DOM Rendering**: All user-supplied tabular content, column names, and cell values are escaped and rendered strictly through `textContent` and DOM nodes to eliminate XSS risks.
- **Indeterminate Progress Indicator**: Synchronous backend generation displays an honest indeterminate loading state and clear guidance without fabricated percentage bars.
- **Honest Metrics**: Metrics are extracted directly from backend results. Skipped or unavailable metrics are explicitly noted ("Not available for this dataset"). No synthetic scores or fake percentages are fabricated.
- **Empirical Privacy Caveat**: Every privacy evaluation card displays a clear reminder that nearest-neighbour distance is an empirical proximity heuristic and does not constitute a formal Differential Privacy guarantee.

---

## FastAPI Backend Details

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/` | API information & route list |
| POST | `/api/upload` | Upload & profile a CSV dataset |
| POST | `/api/generate` | Generate synthetic data from uploaded CSV |
| GET | `/api/results/{result_id}` | Retrieve result metrics & preview |
| GET | `/api/results/{result_id}/download` | Download generated synthetic CSV |

### Default Generator Configuration

| Parameter | Value |
|-----------|-------|
| Generator | CTGAN |
| epochs | 25 |
| batch_size | 500 |
| embedding_dim | 256 |
| seed | 42 |

> **Important:** The nearest-neighbour distance metric used by the privacy module is an empirical proximity heuristic. It does NOT constitute a formal Differential Privacy guarantee.

### Storage Architecture & Lifecycle Cleanup

- **Configurable Persistent Storage**: All synthetic CSV outputs (`{result_id}.csv`) and metadata envelopes (`{result_id}_meta.json`) are written to `PRIVSYNTH_STORAGE_DIR`. The directory is created automatically with permission validation upon startup.
- **Atomic File Writing**: Datasets and metadata are written to temporary `.tmp` files and committed via atomic `os.replace` to prevent external requests from reading partial files.
- **Automated Hourly Cleanup**: An asynchronous background worker managed by FastAPI lifespan periodically scans `PRIVSYNTH_STORAGE_DIR` and removes files older than `RESULT_TTL_HOURS` (including orphaned CSV or metadata files).
- **Active File Shielding**: An in-memory, thread-safe lock registry protects files actively being generated or downloaded (`is_result_active`). Active downloads hold locks via Starlette `BackgroundTask` until streaming completes, ensuring no file is deleted mid-download.
- **Row-Limit Guard**: Both `/api/upload` and `/api/generate` enforce `MAX_UPLOAD_ROWS` (default: 50,000 rows), rejecting oversized datasets with HTTP 413 Payload Too Large.

---

## Running Verification & Tests

### Backend Test Suite (Pytest)
Runs 9 comprehensive backend tests including a real CTGAN generation run:

```bash
.venv\Scripts\pytest test_backend.py -v
```

### Production Readiness & Stage 2 Test Suite (Pytest)
Tests custom storage directories, permission errors, row limits (HTTP 413), atomic operations, TTL expiration, active generation/download shielding, and graceful worker shutdown:

```bash
.venv\Scripts\pytest test_stage2.py -v
```

### Vercel Serverless Architecture & Compatibility Test Suite (Pytest)
Tests `api/index.py` export, Vercel environment detection, CTGAN row limits on serverless, Vercel Blob storage adapter, HTTP 307 download redirects, and deployment bundle exclusions:

```bash
.venv\Scripts\pytest test_vercel.py -v
```

### Frontend Integration & Simulation Suite
Simulates the full browser workflow across all API endpoints:

```bash
.venv\Scripts\python.exe test_frontend_workflow.py
```

### Frontend Code Quality & Security Checker
Verifies DOM IDs, script/style links, and ensures zero unsafe `innerHTML` usage:

```bash
.venv\Scripts\python.exe verify_frontend.py
```

---

## Vercel Serverless Deployment (Hobby Plan & Cloud Storage)

PrivSynth supports deployment to Vercel's free Hobby plan. In this architecture:
- Static frontend assets (`frontend/`) are served at the root (`/`) via Vercel's global Edge CDN.
- API endpoints (`/api/*`) are routed to a Python ASGI serverless function (`api/index.py`).
- Persistent storage is backed by **Vercel Blob** (`BLOB_READ_WRITE_TOKEN`), replacing local disk with cloud storage.
- Download requests issue **HTTP 307 Temporary Redirects** straight to Vercel Blob CDN URLs, bypassing serverless memory and egress limits.

```
                      Internet / User Browser
                                 │
                 ┌───────────────┴───────────────┐
                 ▼ (Static files: /)             ▼ (API calls: /api/*)
          Vercel Edge Network            Python Serverless Function
          (HTML, CSS, JS)                (FastAPI / api/index.py)
                                                 │
                                                 ▼
                                         Vercel Blob Storage
                                         (CSVs & Metadata JSON)
```

### 1. Configure Vercel Blob Storage

1. Open your project dashboard at [vercel.com](https://vercel.com).
2. Navigate to **Storage** -> **Create Database** -> select **Blob**.
3. Create the store (e.g. `privsynth-storage`).
4. Note your read-write access token (`BLOB_READ_WRITE_TOKEN`).

### 2. Configure Environment Variables in Vercel

In your Vercel Project Settings -> **Environment Variables**, add:

| Variable | Recommended Value | Purpose |
| :--- | :--- | :--- |
| `BLOB_READ_WRITE_TOKEN` | `vercel_blob_rw_...` | Read-write authentication token for Vercel Blob cloud storage. |
| `VERCEL_MAX_CTGAN_ROWS` | `1000` | Safety limit for CTGAN on serverless to prevent Lambda timeouts. |
| `ALLOWED_ORIGINS` | `*` or `https://<your-project>.vercel.app` | Allowed CORS origins. |

### 3. Deploy via Vercel CLI or GitHub

**Via Vercel CLI:**
```bash
# Install Vercel CLI if needed
npm install -g vercel

# Deploy to preview
vercel

# Deploy to production
vercel --prod
```

**Via GitHub Integration:**
Push the repository to GitHub and import it in Vercel. Vercel automatically detects `vercel.json` and routes requests.

### 4. Serverless Constraints & Technical Feasibility

- **Function Duration (Hobby: 60s max)**: Deep learning CTGAN training on thousands of rows requires significant CPU compute (3–5 minutes on 30,000 rows). PrivSynth enforces `VERCEL_MAX_CTGAN_ROWS` (default: 1,000 rows) on Vercel. Requests exceeding this threshold receive a clear, explanatory message recommending Gaussian Copula or running on VPS/Docker.
- **Gaussian Copula Generator**: Fast statistical synthesizer (<2 seconds on serverless), fully functional on Vercel Hobby for datasets up to 50,000 rows.
- **Bundle Size Limit (250 MB uncompressed)**: AWS Lambda / Vercel Serverless limits deployment packages to 250 MB uncompressed. Standard PyTorch alone occupies >500 MB. In serverless environments where bundle size limits prevent native PyTorch wheels, running Gaussian Copula or deploying to Docker VPS remains the recommended production deployment path.

---

## Production Docker Deployment

PrivSynth is configured for production deployment using Docker, Docker Compose, and Nginx. This architecture runs a single backend worker process alongside an Nginx reverse proxy and a persistent volume for generated datasets.

```
                    Internet / User Browser
                              │
                              ▼ (Public Port 80)
                 ┌─────────────────────────┐
                 │       Nginx Proxy       │
                 │   - Serves / (frontend) │
                 │   - Proxies /api/       │
                 └────────────┬────────────┘
                              │
                              ▼ (Internal Port 8000)
                 ┌─────────────────────────┐
                 │     FastAPI Backend     │
                 │   - 1 Uvicorn Worker    │
                 │   - Non-root user       │
                 └────────────┬────────────┘
                              │
                              ▼ (Persistent Mount)
                 ┌─────────────────────────┐
                 │  privsynth_storage Vol  │
                 │  - Persists across restarts
                 └─────────────────────────┘
```

### 1. Prerequisites

- **Docker Engine** (version 24.0 or later)
- **Docker Compose** (v2.0 or later)

### 2. Configure Environment

Create your production `.env` file from `.env.example`:

**Linux / macOS:**
```bash
cp .env.example .env
```

**Windows PowerShell:**
```powershell
Copy-Item .env.example .env
```

Configure your target settings in `.env`:
```env
PUBLIC_PORT=80
ALLOWED_ORIGINS=*
MAX_FILE_BYTES=52428800
MAX_UPLOAD_ROWS=50000
RESULT_TTL_HOURS=24
```

### 3. Build & Start the Application

Build the backend image and launch the containers in the background:

**Linux / PowerShell:**
```bash
docker compose up -d --build
```

Nginx will start listening on port 80 (or your configured `PUBLIC_PORT`), serving the static frontend and routing `/api/` calls to the FastAPI backend.

### 4. Check Container Status & Health

Verify both services are healthy:

```bash
docker compose ps
```

You should see:
- `privsynth-backend` in state `Up (healthy)`
- `privsynth-nginx` in state `Up`

### 5. View Logs

Stream real-time structured logs from all services or a specific container:

```bash
# All logs
docker compose logs -f

# Backend service logs only
docker compose logs -f backend

# Nginx access & error logs only
docker compose logs -f nginx
```

### 6. Stop the Application

To gracefully stop the running containers without losing generated datasets:

```bash
docker compose down
```

### 7. Persistent Storage & Restart Resilience

- Generated CSVs and metadata are stored in the Docker named volume `privsynth_storage` mounted at `/app/storage`.
- Restarting or rebuilding the backend container **does not lose** previously generated synthetic datasets or results:

```bash
# Restart backend without losing data
docker compose restart backend
```

- To inspect the volume:
```bash
docker volume inspect privsynth_storage
```

### 8. Update & Rebuild

To deploy updates to the application:

```bash
git pull
docker compose up -d --build
```

---

## Known Limitations

1. **Synchronous Execution**: The backend generation endpoint executes synchronously on the CPU. Training CTGAN on larger datasets may take 1 to 5 minutes.
2. **Generic Upload ML Utility**: In the generic upload workflow, downstream ML utility (TSTR classification) is marked as not evaluated because arbitrary uploaded CSVs do not define a default target label column.
3. **CORS Origins**: CORS defaults to standard local development origins (`http://localhost:5500`, `http://127.0.0.1:5500`, `http://localhost:3000`, `http://127.0.0.1:3000`, `http://localhost:8080`, `http://127.0.0.1:8080`), and can be customized or set to `*` via the `ALLOWED_ORIGINS` environment variable in `.env`.
