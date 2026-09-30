/* ============================================================
   TKIE — app.js
   Handles: file selection/drag-drop, API calls, JSON rendering,
            syntax highlighting, bad-sample polling
   ============================================================ */

const API_BASE = (window.TKIE_API_BASE || "http://localhost:8000").replace(/\/$/, "");

// ── DOM refs ──────────────────────────────────────────────────────────────
const dropZone     = document.getElementById("drop-zone");
const fileInput    = document.getElementById("file-input");
const previewWrap  = document.getElementById("preview-wrap");
const previewImg   = document.getElementById("preview-img");
const previewName  = document.getElementById("preview-name");
const clearBtn     = document.getElementById("clear-btn");
const extractBtn   = document.getElementById("extract-btn");
const spinner      = document.getElementById("spinner");
const errorBanner  = document.getElementById("error-banner");
const jsonOutput   = document.getElementById("json-output");
const placeholder  = document.getElementById("results-placeholder");
const fieldPills   = document.getElementById("field-pills");
const copyBtn      = document.getElementById("copy-btn");
const downloadBtn  = document.getElementById("download-btn");
const statusDot    = document.getElementById("status-dot");
const badCount     = document.getElementById("bad-count");
const badList      = document.getElementById("bad-samples-list");
const refreshBad   = document.getElementById("refresh-bad-btn");

let selectedFile = null;
let lastResult   = null;

// ── Health check ──────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const r = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(4000) });
    statusDot.className = r.ok ? "status-dot online" : "status-dot offline";
    statusDot.title     = r.ok ? "API online" : "API error";
  } catch {
    statusDot.className = "status-dot offline";
    statusDot.title     = "API unreachable";
  }
}
checkHealth();
setInterval(checkHealth, 30_000);

// ── Bad samples ───────────────────────────────────────────────────────────
async function loadBadSamples() {
  try {
    const r = await fetch(`${API_BASE}/bad-samples`);
    if (!r.ok) return;
    const data = await r.json();
    badCount.textContent = data.count;

    if (data.count === 0) {
      badList.innerHTML = `<p class="muted">No quarantined documents.</p>`;
      return;
    }
    badList.innerHTML = data.samples.map(s => `
      <div class="bad-sample-item">
        <span class="icon">☠</span>
        <span>${escapeHTML(s.name)}</span>
        ${s.sidecar ? `<span class="muted" style="margin-left:auto;">has sidecar log</span>` : ""}
      </div>
    `).join("");
  } catch {
    badCount.textContent = "?";
  }
}
loadBadSamples();
refreshBad.addEventListener("click", loadBadSamples);

// ── File selection ────────────────────────────────────────────────────────
function setFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    showError("Please select a valid image file (JPEG, PNG, TIFF, WebP, BMP).");
    return;
  }
  selectedFile = file;
  hideError();

  // Show preview
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewName.textContent = `${file.name}  (${formatBytes(file.size)})`;
  dropZone.classList.add("hidden");
  previewWrap.classList.remove("hidden");
  extractBtn.disabled = false;
}

function clearFile() {
  selectedFile = null;
  fileInput.value = "";
  previewImg.src = "";
  previewWrap.classList.add("hidden");
  dropZone.classList.remove("hidden");
  extractBtn.disabled = true;
  hideError();
}

dropZone.addEventListener("click",   () => fileInput.click());
dropZone.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") fileInput.click(); });
fileInput.addEventListener("change", e => { if (e.target.files[0]) setFile(e.target.files[0]); });
clearBtn.addEventListener("click",   clearFile);

// Drag & drop
dropZone.addEventListener("dragover",  e => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", ()  => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const f = e.dataTransfer.files[0];
  if (f) setFile(f);
});

// ── Extraction ────────────────────────────────────────────────────────────
extractBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  // UI: loading state
  setLoading(true);
  hideError();
  resetResults();

  const fd = new FormData();
  fd.append("file", selectedFile);

  try {
    const res = await fetch(`${API_BASE}/extract`, { method: "POST", body: fd });

    if (res.status === 422) {
      const body = await res.json();
      showError(`⚠ Document quarantined: ${body.detail}`);
      loadBadSamples();
      return;
    }
    if (!res.ok) {
      const body = await res.json().catch(() => ({ detail: res.statusText }));
      showError(`Error ${res.status}: ${body.detail || res.statusText}`);
      return;
    }

    const data = await res.json();
    lastResult = data;
    renderResult(data);
    loadBadSamples();

  } catch (err) {
    showError(`Network error: ${err.message}`);
  } finally {
    setLoading(false);
  }
});

// ── Render result ─────────────────────────────────────────────────────────
function renderResult(data) {
  placeholder.classList.add("hidden");
  jsonOutput.classList.remove("hidden");
  jsonOutput.innerHTML = syntaxHighlight(JSON.stringify(data, null, 2));
  copyBtn.classList.remove("hidden");
  downloadBtn.classList.remove("hidden");

  // Field pills
  fieldPills.innerHTML = Object.entries(data).map(([k, v]) => {
    const filled = v !== null && v !== undefined && !(Array.isArray(v) && v.length === 0);
    return `<span class="pill ${filled ? "filled" : "empty"}">
      ${filled ? "✓" : "○"} ${escapeHTML(k)}
    </span>`;
  }).join("");
  fieldPills.classList.remove("hidden");
}

function resetResults() {
  placeholder.classList.remove("hidden");
  jsonOutput.classList.add("hidden");
  jsonOutput.innerHTML = "";
  fieldPills.classList.add("hidden");
  fieldPills.innerHTML = "";
  copyBtn.classList.add("hidden");
  downloadBtn.classList.add("hidden");
  lastResult = null;
}

// ── Copy / Download ───────────────────────────────────────────────────────
copyBtn.addEventListener("click", async () => {
  if (!lastResult) return;
  try {
    await navigator.clipboard.writeText(JSON.stringify(lastResult, null, 2));
    const orig = copyBtn.textContent;
    copyBtn.textContent = "✓ Copied!";
    setTimeout(() => (copyBtn.textContent = orig), 1800);
  } catch { /* clipboard blocked */ }
});

downloadBtn.addEventListener("click", () => {
  if (!lastResult) return;
  const blob = new Blob([JSON.stringify(lastResult, null, 2)], { type: "application/json" });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href     = url;
  a.download = `tkie_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
});

// ── Syntax highlighting ───────────────────────────────────────────────────
function syntaxHighlight(json) {
  return escapeHTML(json).replace(
    /("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g,
    match => {
      let cls = "json-num";
      if (/^"/.test(match)) {
        cls = /:$/.test(match) ? "json-key" : "json-str";
      } else if (/true|false/.test(match)) {
        cls = "json-bool";
      } else if (/null/.test(match)) {
        cls = "json-null";
      }
      return `<span class="${cls}">${match}</span>`;
    }
  );
}

// ── Helpers ───────────────────────────────────────────────────────────────
function setLoading(on) {
  extractBtn.disabled = on;
  spinner.classList.toggle("hidden", !on);
}

function showError(msg) {
  errorBanner.textContent = msg;
  errorBanner.classList.remove("hidden");
}

function hideError() {
  errorBanner.textContent = "";
  errorBanner.classList.add("hidden");
}

function escapeHTML(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function formatBytes(bytes) {
  if (bytes < 1024)       return `${bytes} B`;
  if (bytes < 1024 ** 2)  return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}
