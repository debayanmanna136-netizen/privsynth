/* ============================================================
   PrivSynth: Core Application
   Vanilla ES6+ Frontend Client
   ============================================================ */

const API_BASE_URL = window.PRIVSYNTH_API_URL || (
  (window.location.port === "5500" || window.location.port === "3000")
    ? `${window.location.protocol}//${window.location.hostname || "127.0.0.1"}:8000`
    : ""
);

// --------------- State Management ---------------
const state = {
  file: null,
  uploadData: null,
  resultId: null,
  generating: false,
  timerInterval: null,
  secondsElapsed: 0,
};

// --------------- Formatting Utilities ---------------
function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / 1048576).toFixed(1) + " MB";
}

function formatMetricLabel(key) {
  return key
    .replace(/^syn_to_real_/, "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// --------------- Toast System ---------------
function showToast(element, message, type = "info") {
  if (!element) return;
  element.className = `toast toast--${type}`;
  if (type === "loading") {
    element.innerHTML = `<span class="spinner" aria-hidden="true"></span>${escapeHtml(message)}`;
  } else {
    element.textContent = message;
  }
  element.classList.remove("hidden");
}

function hideToast(element) {
  if (!element) return;
  element.classList.add("hidden");
}

// --------------- View Switching System ---------------
function showView(name) {
  const views = document.querySelectorAll(".view");
  views.forEach((v) => v.classList.remove("view--active"));

  const target = document.getElementById(`view-${name}`);
  if (target) {
    target.classList.add("view--active");
  }

  // Update nav buttons contextual state
  const navBtn = document.getElementById("nav-get-started");
  if (navBtn) {
    if (name === "workspace") {
      navBtn.textContent = "Back to Overview";
      navBtn.setAttribute("data-goto", "landing");
    } else {
      navBtn.textContent = "Open Workspace";
      navBtn.setAttribute("data-goto", "workspace");
    }
  }

  window.scrollTo({ top: 0, behavior: "instant" });
}

// --------------- Mobile Navigation ---------------
function toggleMobileNav() {
  const nav = document.getElementById("main-nav");
  const toggle = document.getElementById("nav-toggle");
  const isOpen = nav.classList.toggle("nav--open");
  toggle.setAttribute("aria-expanded", String(isOpen));
}

function closeMobileNav() {
  const nav = document.getElementById("main-nav");
  const toggle = document.getElementById("nav-toggle");
  if (nav) nav.classList.remove("nav--open");
  if (toggle) toggle.setAttribute("aria-expanded", "false");
}

// --------------- Backend Health Check ---------------
async function checkBackendHealth() {
  const container = document.getElementById("backend-status");
  if (!container) return;

  container.textContent = "";

  const dot = document.createElement("span");
  dot.className = "status-dot";

  const label = document.createElement("span");
  label.className = "status-label";

  try {
    const res = await fetch(`${API_BASE_URL}/api/health`);
    if (res.ok) {
      const data = await res.json();
      if (data.status === "ok") {
        dot.className = "status-dot status-dot--connected";
        label.textContent = "Connected";
      } else {
        dot.className = "status-dot status-dot--offline";
        label.textContent = "Offline";
      }
    } else {
      dot.className = "status-dot status-dot--offline";
      label.textContent = "Offline";
    }
  } catch {
    dot.className = "status-dot status-dot--offline";
    label.textContent = "Offline";
  }

  container.appendChild(dot);
  container.appendChild(label);
}

// --------------- Built-in Demo Dataset ---------------
function createDemoDatasetFile() {
  const header = "age,workclass,education,hours_per_week,income\n";
  const rows = [
    "39,State-gov,Bachelors,40,<=50K",
    "50,Self-emp-not-inc,Bachelors,13,<=50K",
    "38,Private,HS-grad,40,<=50K",
    "53,Private,11th,40,<=50K",
    "28,Private,Bachelors,40,<=50K",
    "37,Private,Masters,40,<=50K",
    "49,Private,9th,16,<=50K",
    "52,Self-emp-not-inc,HS-grad,45,>50K",
    "31,Private,Masters,50,>50K",
    "42,Private,Bachelors,40,>50K",
    "37,Private,Some-college,80,>50K",
    "30,State-gov,Bachelors,40,>50K",
    "23,Private,Bachelors,30,<=50K",
    "32,Private,Assoc-acdm,50,<=50K",
    "40,Private,Assoc-voc,40,>50K",
    "34,Private,7th-8th,45,<=50K",
    "25,Self-emp-not-inc,HS-grad,35,<=50K",
    "32,Private,HS-grad,40,<=50K",
    "38,Private,11th,50,<=50K",
    "43,Self-emp-not-inc,Masters,45,>50K",
    "40,Private,Doctorate,60,>50K",
    "54,Private,HS-grad,20,<=50K",
    "35,Private,9th,40,<=50K",
    "43,Private,11th,40,<=50K",
    "59,Private,HS-grad,40,<=50K",
    "56,Local-gov,Bachelors,40,>50K",
    "19,Private,HS-grad,20,<=50K",
    "54,Private,Some-college,60,>50K",
    "39,Private,HS-grad,80,<=50K",
    "49,Private,HS-grad,40,<=50K",
    "23,Local-gov,Some-college,52,<=50K",
    "20,Private,Some-college,44,<=50K",
    "45,Private,Bachelors,40,<=50K",
    "30,Federal-gov,Some-college,40,<=50K",
    "22,State-gov,Some-college,15,<=50K",
    "48,Private,11th,40,<=50K",
    "21,Private,Some-college,40,<=50K",
    "19,Private,HS-grad,25,<=50K",
    "31,Private,Some-college,38,>50K",
    "48,Self-emp-not-inc,Assoc-acdm,40,<=50K",
    "31,Private,9th,43,<=50K",
    "53,Private,Bachelors,40,<=50K",
    "24,Private,Bachelors,40,<=50K",
    "49,Private,HS-grad,40,<=50K",
    "25,Private,HS-grad,35,<=50K",
    "57,Private,Bachelors,40,>50K",
    "53,Private,HS-grad,38,<=50K",
    "44,Private,Masters,50,<=50K",
    "41,Private,9th,40,<=50K",
    "29,Private,Assoc-voc,42,<=50K"
  ];
  const blob = new Blob([header + rows.join("\n")], { type: "text/csv" });
  return new File([blob], "adult_census_demo.csv", { type: "text/csv" });
}

// --------------- File Selection & Validation ---------------
function handleFileSelection(file) {
  if (!file) return;

  const ext = file.name.split(".").pop().toLowerCase();
  if (ext !== "csv") {
    const status = document.getElementById("upload-status");
    showToast(status, "Only CSV files are supported. Please select a valid .csv file.", "error");
    return;
  }

  state.file = file;

  const pill = document.getElementById("file-pill");
  const pillName = document.getElementById("file-pill-name");
  const pillSize = document.getElementById("file-pill-size");
  const dropZone = document.getElementById("drop-zone");
  const sampleBtnWrapper = document.querySelector(".demo-dataset-action");

  pillName.textContent = file.name;
  pillSize.textContent = formatFileSize(file.size);

  pill.classList.remove("hidden");
  dropZone.classList.add("hidden");
  if (sampleBtnWrapper) sampleBtnWrapper.classList.add("hidden");

  uploadDataset(file);
}

function clearFile() {
  state.file = null;
  state.uploadData = null;
  state.resultId = null;
  state.generating = false;

  if (state.timerInterval) {
    clearInterval(state.timerInterval);
    state.timerInterval = null;
  }

  const pill = document.getElementById("file-pill");
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const sampleBtnWrapper = document.querySelector(".demo-dataset-action");
  const genBtn = document.getElementById("gen-btn");
  const dlBtn = document.getElementById("dl-btn");
  const overviewPanel = document.getElementById("overview-panel");
  const resultsPanel = document.getElementById("results-panel");
  const placeholder = document.getElementById("ws-placeholder");
  const uploadStatus = document.getElementById("upload-status");
  const genStatus = document.getElementById("gen-status");
  const dlStatus = document.getElementById("dl-status");
  const progressBar = document.getElementById("gen-progress-bar");
  const timerLabel = document.getElementById("gen-timer");

  pill.classList.add("hidden");
  dropZone.classList.remove("hidden");
  if (sampleBtnWrapper) sampleBtnWrapper.classList.remove("hidden");
  fileInput.value = "";
  genBtn.disabled = true;
  dlBtn.disabled = true;

  overviewPanel.classList.add("hidden");
  resultsPanel.classList.add("hidden");
  if (progressBar) progressBar.classList.add("hidden");
  if (timerLabel) timerLabel.classList.add("hidden");
  if (placeholder) placeholder.classList.remove("hidden");

  hideToast(uploadStatus);
  hideToast(genStatus);
  hideToast(dlStatus);
}

// --------------- API Upload Handler ---------------
async function uploadDataset(file) {
  const status = document.getElementById("upload-status");
  const genBtn = document.getElementById("gen-btn");
  genBtn.disabled = true;

  showToast(status, "Analyzing dataset schema and distributions…", "loading");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(`${API_BASE_URL}/api/upload`, {
      method: "POST",
      body: formData,
    });

    const data = await res.json();

    if (!res.ok || !data.success) {
      showToast(status, handleApiError(res, data), "error");
      return;
    }

    state.uploadData = data;
    renderOverview(data);
    genBtn.disabled = false;
  } catch {
    showToast(status, "Could not reach the backend API. Is the server running?", "error");
  }
}

// --------------- Dataset Overview Rendering ---------------
function renderOverview(data) {
  const status = document.getElementById("upload-status");

  document.getElementById("ov-rows").textContent = Number(data.rows).toLocaleString();
  document.getElementById("ov-cols").textContent = Number(data.columns).toLocaleString();
  document.getElementById("ov-missing").textContent = Number(data.missing_values).toLocaleString();

  const colContainer = document.getElementById("ov-columns");
  colContainer.textContent = "";
  data.column_names.forEach((name) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = name;
    colContainer.appendChild(chip);
  });

  const tableContainer = document.getElementById("ov-table");
  renderTable(tableContainer, data.preview);

  const placeholder = document.getElementById("ws-placeholder");
  if (placeholder) placeholder.classList.add("hidden");

  document.getElementById("overview-panel").classList.remove("hidden");
  showToast(status, `Dataset loaded successfully (${data.rows} rows, ${data.columns} features).`, "success");
}

// --------------- Synthetic Data Generation ---------------
async function generateSyntheticData() {
  if (!state.file || state.generating) return;

  state.generating = true;
  state.secondsElapsed = 0;

  const genBtn = document.getElementById("gen-btn");
  const genStatus = document.getElementById("gen-status");
  const progressBar = document.getElementById("gen-progress-bar");
  const timerLabel = document.getElementById("gen-timer");

  genBtn.disabled = true;
  if (progressBar) progressBar.classList.remove("hidden");
  if (timerLabel) {
    timerLabel.textContent = "Training generator on CPU (0s elapsed)...";
    timerLabel.classList.remove("hidden");
  }

  showToast(genStatus, "Training generator model on CPU. This may take a moment…", "loading");

  state.timerInterval = setInterval(() => {
    state.secondsElapsed += 1;
    if (timerLabel) {
      timerLabel.textContent = `Training generator on CPU (${state.secondsElapsed}s elapsed)...`;
    }
  }, 1000);

  const formData = new FormData();
  formData.append("file", state.file);

  const generator = document.getElementById("gen-select").value;

  try {
    const res = await fetch(
      `${API_BASE_URL}/api/generate?generator=${encodeURIComponent(generator)}`,
      { method: "POST", body: formData }
    );

    const data = await res.json();

    if (!res.ok || !data.success) {
      showToast(genStatus, handleApiError(res, data), "error");
      genBtn.disabled = false;
      state.generating = false;
      return;
    }

    state.resultId = data.result_id;
    showToast(genStatus, `Synthetic records generated in ${state.secondsElapsed}s.`, "success");
    await fetchResultMetadata(data.result_id);
  } catch {
    showToast(genStatus, "Synthetic generation failed. Is the backend server running?", "error");
    genBtn.disabled = false;
  } finally {
    state.generating = false;
    if (state.timerInterval) {
      clearInterval(state.timerInterval);
      state.timerInterval = null;
    }
    if (progressBar) progressBar.classList.add("hidden");
    if (timerLabel) timerLabel.classList.add("hidden");
  }
}

// --------------- Fetch Result Details ---------------
async function fetchResultMetadata(resultId) {
  const genStatus = document.getElementById("gen-status");

  try {
    const res = await fetch(`${API_BASE_URL}/api/results/${resultId}`);
    const data = await res.json();

    if (!data.success) {
      showToast(genStatus, handleApiError(res, data), "error");
      return;
    }

    renderResults(data);
  } catch {
    showToast(genStatus, "Failed to retrieve evaluation results metadata.", "error");
  }
}

// --------------- Render Results Dashboard ---------------
function renderResults(data) {
  document.getElementById("res-rows").textContent = Number(data.rows).toLocaleString();
  document.getElementById("res-cols").textContent = Number(data.columns).toLocaleString();
  document.getElementById("res-gen").textContent = (data.generator || "").toUpperCase();
  document.getElementById("res-id").textContent = data.result_id.substring(0, 8);

  // --- Statistical Fidelity ---
  const fidelityEl = document.getElementById("m-fidelity");
  fidelityEl.textContent = "";
  const fidelity = data.metrics && data.metrics.fidelity;

  if (fidelity && typeof fidelity === "object") {
    if (fidelity.correlation_mae != null) {
      fidelityEl.appendChild(createMetricRow("Correlation MAE", Number(fidelity.correlation_mae).toFixed(4)));
    }
    if (fidelity.ks_statistic && typeof fidelity.ks_statistic === "object") {
      Object.entries(fidelity.ks_statistic).forEach(([col, val]) => {
        fidelityEl.appendChild(createMetricRow(`KS Distance (${col})`, Number(val).toFixed(4)));
      });
    }
    if (fidelity.js_divergence && typeof fidelity.js_divergence === "object") {
      Object.entries(fidelity.js_divergence).forEach(([col, val]) => {
        fidelityEl.appendChild(createMetricRow(`JS Divergence (${col})`, Number(val).toFixed(4)));
      });
    }
    if (fidelityEl.childElementCount === 0) {
      const note = document.createElement("p");
      note.className = "metric-empty";
      note.textContent = "Not available for this dataset.";
      fidelityEl.appendChild(note);
    }
  } else {
    const note = document.createElement("p");
    note.className = "metric-empty";
    note.textContent = "Not available for this dataset.";
    fidelityEl.appendChild(note);
  }

  // --- Downstream ML Utility ---
  const utilityEl = document.getElementById("m-utility");
  utilityEl.textContent = "";
  const utility = data.metrics && data.metrics.utility;

  if (utility != null) {
    if (typeof utility === "string") {
      const note = document.createElement("p");
      note.className = "metric-note";
      note.textContent = utility;
      utilityEl.appendChild(note);
    } else if (typeof utility === "object") {
      Object.entries(utility).forEach(([key, val]) => {
        utilityEl.appendChild(createMetricRow(formatMetricLabel(key), val));
      });
    }
  } else {
    const note = document.createElement("p");
    note.className = "metric-empty";
    note.textContent = "Not available for this dataset.";
    utilityEl.appendChild(note);
  }

  // --- Empirical Privacy Proximity ---
  const privacyEl = document.getElementById("m-privacy");
  privacyEl.textContent = "";
  const privacy = data.metrics && data.metrics.privacy;

  const privacyLabels = {
    syn_to_real_mean_distance: "Mean Distance",
    syn_to_real_median_distance: "Median Distance",
    syn_to_real_min_distance: "Min Distance",
    syn_to_real_p5_distance: "5th Percentile Distance",
  };

  if (privacy && typeof privacy === "object") {
    Object.entries(privacy).forEach(([key, val]) => {
      const label = privacyLabels[key] || formatMetricLabel(key);
      privacyEl.appendChild(createMetricRow(label, Number(val).toFixed(4)));
    });
    const disclaimer = document.createElement("p");
    disclaimer.className = "metric-note";
    disclaimer.textContent = "Nearest-neighbour distance is an empirical proximity heuristic, not a formal Differential Privacy guarantee.";
    privacyEl.appendChild(disclaimer);
  } else {
    const note = document.createElement("p");
    note.className = "metric-empty";
    note.textContent = "Not available for this dataset.";
    privacyEl.appendChild(note);
  }

  // --- Synthetic Data Preview ---
  const synthTable = document.getElementById("synth-table");
  renderTable(synthTable, data.preview);

  // --- Enable Download ---
  const dlBtn = document.getElementById("dl-btn");
  dlBtn.disabled = false;

  const resultsPanel = document.getElementById("results-panel");
  resultsPanel.classList.remove("hidden");

  // Smooth scroll into results
  resultsPanel.scrollIntoView({ behavior: "smooth", block: "start" });
}

function createMetricRow(label, value) {
  const row = document.createElement("div");
  row.className = "metric-row";

  const keyEl = document.createElement("span");
  keyEl.className = "metric-key";
  keyEl.textContent = label;

  const valEl = document.createElement("span");
  valEl.className = "metric-val";
  valEl.textContent = String(value);

  row.appendChild(keyEl);
  row.appendChild(valEl);
  return row;
}

// --------------- Safe DOM Table Rendering ---------------
function renderTable(container, rows) {
  container.textContent = "";

  if (!Array.isArray(rows) || rows.length === 0) {
    const msg = document.createElement("p");
    msg.className = "metric-empty";
    msg.textContent = "No preview available.";
    container.appendChild(msg);
    return;
  }

  const wrap = document.createElement("div");
  wrap.className = "tbl-wrap";

  const table = document.createElement("table");
  table.className = "data-table";

  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  const keys = Object.keys(rows[0]);

  keys.forEach((key) => {
    const th = document.createElement("th");
    th.textContent = key;
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    keys.forEach((key) => {
      const td = document.createElement("td");
      const val = row[key];
      td.textContent = val == null ? "" : String(val);
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);

  wrap.appendChild(table);
  container.appendChild(wrap);
}

// --------------- CSV Download Handler ---------------
async function downloadSyntheticDataset() {
  if (!state.resultId) return;

  const dlBtn = document.getElementById("dl-btn");
  const dlStatus = document.getElementById("dl-status");
  dlBtn.disabled = true;

  try {
    const res = await fetch(`${API_BASE_URL}/api/results/${state.resultId}/download`);

    if (!res.ok) {
      showToast(dlStatus, "Download failed. Please try again.", "error");
      dlBtn.disabled = false;
      return;
    }

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `privsynth_synthetic_${state.resultId.substring(0, 8)}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    showToast(dlStatus, "Synthetic dataset CSV downloaded successfully.", "success");
  } catch {
    showToast(dlStatus, "Download request failed. Is the backend server running?", "error");
  } finally {
    dlBtn.disabled = false;
  }
}

// --------------- Error Message Formatter ---------------
function handleApiError(response, data) {
  if (data && data.error) return data.error;
  return `Server request failed (HTTP ${response.status}).`;
}

// --------------- Smooth Navigation Scrolling ---------------
function setupSmoothScroll() {
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", (e) => {
      const href = anchor.getAttribute("href");
      if (!href || href === "#") return;

      const id = href.substring(1);
      e.preventDefault();

      // If we are currently in workspace view, return to landing view first!
      const landing = document.getElementById("view-landing");
      if (!landing.classList.contains("view--active")) {
        showView("landing");
      }

      const target = document.getElementById(id);
      if (target) {
        target.scrollIntoView({ behavior: "smooth" });
        closeMobileNav();
      }
    });
  });
}

// --------------- App Initialization ---------------
function initializeApp() {
  // Navigation View Toggles
  document.querySelectorAll('[data-goto="workspace"]').forEach((el) => {
    el.addEventListener("click", (e) => {
      e.preventDefault();
      showView("workspace");
    });
  });

  const wsBack = document.getElementById("ws-back");
  if (wsBack) wsBack.addEventListener("click", () => showView("landing"));

  const logoHome = document.getElementById("logo-home");
  if (logoHome) {
    logoHome.addEventListener("click", (e) => {
      e.preventDefault();
      showView("landing");
    });
  }

  // Dynamic Nav CTA Button
  const navGetStarted = document.getElementById("nav-get-started");
  if (navGetStarted) {
    navGetStarted.addEventListener("click", (e) => {
      e.preventDefault();
      const targetView = navGetStarted.getAttribute("data-goto") || "workspace";
      showView(targetView);
    });
  }

  // Mobile Nav Burger
  const navToggle = document.getElementById("nav-toggle");
  if (navToggle) navToggle.addEventListener("click", toggleMobileNav);

  const navLinks = document.getElementById("nav-links");
  if (navLinks) {
    navLinks.querySelectorAll("a").forEach((link) => {
      link.addEventListener("click", closeMobileNav);
    });
  }

  // File Drop Zone Setup (Clean & Safe)
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");

  if (dropZone && fileInput) {
    // Click triggers hidden input
    dropZone.addEventListener("click", () => {
      fileInput.click();
    });

    dropZone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        fileInput.click();
      }
    });

    // Drag-and-drop
    dropZone.addEventListener("dragenter", (e) => {
      e.preventDefault();
      dropZone.classList.add("dropzone--hover");
    });

    dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropZone.classList.add("dropzone--hover");
    });

    dropZone.addEventListener("dragleave", (e) => {
      e.preventDefault();
      dropZone.classList.remove("dropzone--hover");
    });

    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.classList.remove("dropzone--hover");
      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        handleFileSelection(files[0]);
      }
    });

    fileInput.addEventListener("change", () => {
      if (fileInput.files && fileInput.files.length > 0) {
        handleFileSelection(fileInput.files[0]);
      }
    });
  }

  // Instant Demo Dataset Button
  const btnSampleData = document.getElementById("btn-sample-data");
  if (btnSampleData) {
    btnSampleData.addEventListener("click", () => {
      const demoFile = createDemoDatasetFile();
      handleFileSelection(demoFile);
    });
  }

  // File Pill Remove Button
  const filePillX = document.getElementById("file-pill-x");
  if (filePillX) filePillX.addEventListener("click", clearFile);

  // Generator Action
  const genBtn = document.getElementById("gen-btn");
  if (genBtn) genBtn.addEventListener("click", generateSyntheticData);

  // Download Action
  const dlBtn = document.getElementById("dl-btn");
  if (dlBtn) dlBtn.addEventListener("click", downloadSyntheticDataset);

  // Smooth Scroll
  setupSmoothScroll();

  // Backend Health Status Check
  checkBackendHealth();
  setInterval(checkBackendHealth, 30000);
}

document.addEventListener("DOMContentLoaded", initializeApp);
