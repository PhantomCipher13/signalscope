/**
 * SignalScope — static/js/app.js
 * Frontend logic: upload → analyze → render results → feedback
 */
"use strict";

// API_BASE: reads from window.SIGNALSCOPE_API_BASE (injected via /static/js/config.js
// at build time for Vercel/production). Falls back to "" (same-origin) for local dev
// where FastAPI serves both the frontend and API.
const API_BASE = (typeof window !== "undefined" && window.SIGNALSCOPE_API_BASE) || "";

// ── DOM refs ──────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const dropZone        = $("drop-zone");
const fileInput       = $("file-input");
const previewSection  = $("preview-section");
const previewImg      = $("preview-img");
const previewName     = $("preview-name");
const previewSize     = $("preview-size");
const changeBtn       = $("change-btn");
const analyzeBtn      = $("analyze-btn");
const optStress       = $("opt-stress");
const optGradcam      = $("opt-gradcam");
const optProvenance   = $("opt-provenance");

const uploadSection   = $("upload-section");
const loadingSection  = $("loading-section");
const resultSection   = $("result-section");
const errorSection    = $("error-section");

let _currentFile      = null;
let _currentResult    = null;
let _selectedVerdict  = null;

// ── Upload & preview ──────────────────────────────────────────────────────────
dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") fileInput.click(); });
dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave",  () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", e => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const file = e.dataTransfer.files?.[0];
    if (file) setFile(file);
});
fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    if (file) setFile(file);
});
changeBtn.addEventListener("click", () => {
    fileInput.value = "";
    _currentFile = null;
    previewSection.classList.add("hidden");
    dropZone.classList.remove("hidden");
    analyzeBtn.disabled = true;
});

function setFile(file) {
    _currentFile = file;
    const mb = (file.size / 1024 / 1024).toFixed(2);
    previewName.textContent = file.name;
    previewSize.textContent = `${mb} MB`;
    const reader = new FileReader();
    reader.onload = e => { previewImg.src = e.target.result; };
    reader.readAsDataURL(file);
    dropZone.classList.add("hidden");
    previewSection.classList.remove("hidden");
    analyzeBtn.disabled = false;
}

// ── Analyze ───────────────────────────────────────────────────────────────────
analyzeBtn.addEventListener("click", runAnalysis);
$("retry-btn").addEventListener("click", () => {
    errorSection.classList.add("hidden");
    uploadSection.classList.remove("hidden");
});
$("new-analysis-btn").addEventListener("click", () => {
    resultSection.classList.add("hidden");
    uploadSection.classList.remove("hidden");
    fileInput.value = "";
    _currentFile = null;
    previewSection.classList.add("hidden");
    dropZone.classList.remove("hidden");
    analyzeBtn.disabled = true;
    _currentResult = null;
    _selectedVerdict = null;
});

async function runAnalysis() {
    if (!_currentFile) return;

    // Show loading
    uploadSection.classList.add("hidden");
    resultSection.classList.add("hidden");
    errorSection.classList.add("hidden");
    loadingSection.classList.remove("hidden");

    // Animate steps
    const steps = ["step-detect","step-cal","step-probe","step-explain","step-prov"];
    let si = 0;
    const stepInterval = setInterval(() => {
        if (si > 0) document.getElementById(steps[si-1])?.classList.replace("active","done");
        if (si < steps.length) {
            document.getElementById(steps[si])?.classList.add("active");
            $("loading-step").textContent = [
                "Running detector…","Applying calibration…",
                "Running 5 probes…","Generating attention map…","Reading metadata…"
            ][si];
        }
        si++;
        if (si > steps.length) clearInterval(stepInterval);
    }, 900);

    try {
        const formData = new FormData();
        formData.append("file", _currentFile);
        formData.append("run_stress_test", optStress.checked ? "true" : "false");
        formData.append("run_gradcam",     optGradcam.checked ? "true" : "false");
        formData.append("run_provenance",  optProvenance.checked ? "true" : "false");

        const resp = await fetch(`${API_BASE}/analyze`, {
            method: "POST",
            body: formData,
        });

        clearInterval(stepInterval);
        steps.forEach(s => {
            document.getElementById(s)?.classList.remove("active");
            document.getElementById(s)?.classList.add("done");
        });

        if (!resp.ok) {
            let detail = "";
            try { const j = await resp.json(); detail = j.detail?.message || j.detail || resp.statusText; }
            catch { detail = resp.statusText; }
            showError("Analysis Failed", detail || `Server returned ${resp.status}`);
            return;
        }

        const result = await resp.json();
        _currentResult = result;
        renderResult(result);

    } catch (err) {
        clearInterval(stepInterval);
        showError("Network Error", err.message || "Could not reach the SignalScope server.");
    }
}

// ── Render result ─────────────────────────────────────────────────────────────
function renderResult(r) {
    loadingSection.classList.add("hidden");
    resultSection.classList.remove("hidden");

    const pred   = r.prediction || "unknown";
    const calProb = r.calibrated_probability;
    const rawProb = r.raw_probability;
    const dispProb = calProb ?? rawProb ?? 0;

    // Verdict banner
    const banner = $("verdict-banner");
    banner.className = "verdict-banner";
    if      (pred === "real")      { banner.classList.add("real");      $("verdict-icon").textContent = "✓"; }
    else if (pred === "synthetic") { banner.classList.add("synthetic"); $("verdict-icon").textContent = "⚠"; }
    else                           { banner.classList.add("uncertain"); $("verdict-icon").textContent = "?"; }

    $("verdict-label").textContent   = r.label || (pred === "synthetic" ? "Likely AI-Generated" : "Likely Real");
    $("verdict-sublabel").textContent = calProb != null
        ? `Temperature-calibrated result · model: ${r.model_version || "efficientnet_b0_v1"}`
        : `Raw model output · ${r.calibration_status}`;
    $("prob-display").textContent = (dispProb * 100).toFixed(1) + "%";

    // Probability bar
    const pct = Math.round(dispProb * 100);
    $("prob-bar-fill").style.width   = `${pct}%`;
    $("prob-bar-marker").style.left  = `${pct}%`;

    // Metrics
    $("m-raw").textContent     = rawProb != null ? (rawProb * 100).toFixed(1) + "%" : "—";
    $("m-cal").textContent     = calProb != null ? (calProb * 100).toFixed(1) + "%" : "—";

    const st = r.stress_test || {};
    const stability = st.stability;
    $("m-stability").textContent = stability != null ? (stability * 100).toFixed(1) + "%" : "—";
    $("m-probes").textContent    = st.probes_successful != null
        ? `${st.probes_successful}/${st.probes_configured || 5}`
        : "—";

    // Probes table
    const probeResults = st.probe_results || [];
    const tbody = $("probe-tbody");
    tbody.innerHTML = "";
    if (probeResults.length > 0) {
        probeResults.forEach(p => {
            const tr = document.createElement("tr");
            const calCell = p.calibrated_probability != null
                ? (p.calibrated_probability * 100).toFixed(1) + "%"
                : "—";
            tr.innerHTML = `
                <td>${p.probe_name}</td>
                <td class="${p.success ? "probe-status-ok" : "probe-status-fail"}">${p.success ? "✓ OK" : "✗ FAIL"}</td>
                <td>${p.raw_probability != null ? (p.raw_probability*100).toFixed(1)+"%" : "—"}</td>
                <td>${calCell}</td>
                <td>${p.processing_time_s != null ? p.processing_time_s.toFixed(2)+"s" : "—"}</td>
            `;
            tbody.appendChild(tr);
        });
        $("probes-section").classList.remove("hidden");
    }

    // Threshold note
    const rel = r.reliability || {};
    $("threshold-note").textContent = rel.threshold_configured
        ? `Stability threshold configured. Stop reason: ${rel.stop_reason}.`
        : "Reliability thresholds not yet configured. All probes run deterministically.";

    // Reliability
    const relStatus = rel.status || "insufficient_evidence";
    const relBadge  = {
        "threshold_not_configured": ["not-configured", "Threshold not configured"],
        "stable":                   ["stable",         "Stable"],
        "unstable":                 ["unstable",       "Unstable"],
        "insufficient_evidence":    ["insufficient",   "Insufficient evidence"],
    }[relStatus] || ["not-configured", relStatus];

    $("reliability-content").innerHTML = `
        <span class="rel-badge ${relBadge[0]}">${relBadge[1]}</span>
        <p>Observations: ${rel.observation_count || 1} (baseline + ${st.probes_successful || 0} successful probes)</p>
        ${rel.mean_probability != null ? `<p>Mean probability: ${(rel.mean_probability*100).toFixed(1)}%</p>` : ""}
        ${rel.std_probability  != null ? `<p>Std deviation:    ${(rel.std_probability*100).toFixed(1)}%</p>` : ""}
        <p style="margin-top:8px;font-size:.82rem;color:var(--text-dim)">
            Stop reason: ${rel.stop_reason || "N/A"}.
            ${rel.threshold_configured ? "" : "Reliability thresholds must be validated on held-out data before early stopping is enabled."}
        </p>
    `;

    // Grad-CAM
    const gc = r.gradcam || {};
    if (gc.overlay_b64 && gc.status === "ok") {
        $("gradcam-section").classList.remove("hidden");
        $("gradcam-overlay").src = "data:image/png;base64," + gc.overlay_b64;
        if (_currentFile) {
            const reader = new FileReader();
            reader.onload = e => { $("original-img-display").src = e.target.result; };
            reader.readAsDataURL(_currentFile);
        }
        $("gradcam-layer-note").textContent =
            `Target layer: ${gc.target_layer_name || "auto-detected"} · class ${gc.target_class === 1 ? "synthetic" : "real"}`;
    } else {
        $("gradcam-section").classList.add("hidden");
    }

    // Provenance
    renderProvenance(r.provenance || {});

    // Analysis ID
    if (r.analysis_id) {
        $("analysis-id-note").textContent = `Analysis ID: ${r.analysis_id}`;
    }

    // Reset feedback
    _selectedVerdict = null;
    document.querySelectorAll(".btn-feedback").forEach(b => b.classList.remove("selected"));
    $("submit-feedback-btn").classList.add("hidden");
    $("feedback-status").textContent = "";
    $("feedback-comment").value = "";
}

function renderProvenance(prov) {
    const exif = prov.exif || {};
    const c2pa = prov.c2pa || {};
    const container = $("provenance-content");

    if (!prov.exif && !prov.c2pa) {
        container.innerHTML = `<p class="note">Provenance analysis not run.</p>`;
        return;
    }

    let html = `<div class="prov-grid">`;

    if (exif.status === "metadata_available") {
        const fields = [
            ["Camera", [exif.camera_make, exif.camera_model].filter(Boolean).join(" ") || "—"],
            ["Software",  exif.software || "—"],
            ["Taken",     exif.datetime_original || "—"],
            ["GPS",       exif.gps_available ? "Available" : "Not present"],
            ["ISO",       exif.iso_speed != null ? String(exif.iso_speed) : "—"],
            ["Exposure",  exif.exposure_time || "—"],
        ];
        fields.forEach(([k, v]) => {
            html += `<div class="prov-item"><span class="prov-key">${k}</span><span class="prov-value">${v}</span></div>`;
        });
    } else {
        html += `<div class="prov-item"><span class="prov-key">EXIF Status</span><span class="prov-value">${exif.status || "unknown"}</span></div>`;
    }

    html += `</div>`;

    if (c2pa.status) {
        html += `<p class="note" style="margin-top:8px">C2PA: ${c2pa.status}${c2pa.status === "c2pa_library_unavailable" ? " (library not installed)" : ""}</p>`;
    }

    if (prov.summary) {
        html += `<p class="note" style="margin-top:4px">${prov.summary}</p>`;
    }

    container.innerHTML = html;
}

// ── Feedback ──────────────────────────────────────────────────────────────────
document.querySelectorAll(".btn-feedback").forEach(btn => {
    btn.addEventListener("click", () => {
        _selectedVerdict = btn.dataset.verdict;
        document.querySelectorAll(".btn-feedback").forEach(b => b.classList.remove("selected"));
        btn.classList.add("selected");
        $("submit-feedback-btn").classList.remove("hidden");
    });
});

$("submit-feedback-btn").addEventListener("click", async () => {
    if (!_selectedVerdict || !_currentResult) return;
    const analysisId = _currentResult.analysis_id;
    if (!analysisId) {
        $("feedback-status").textContent = "Cannot submit feedback — no analysis ID.";
        $("feedback-status").className   = "feedback-status error";
        return;
    }

    $("submit-feedback-btn").disabled = true;
    $("feedback-status").textContent  = "Submitting…";
    $("feedback-status").className    = "feedback-status";

    try {
        const resp = await fetch(`${API_BASE}/feedback`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                analysis_id: analysisId,
                verdict:     _selectedVerdict,
                comment:     $("feedback-comment").value.trim() || null,
                tester_id:   null,
            }),
        });
        const r = await resp.json();
        if (r.status === "stored") {
            $("feedback-status").textContent = "✓ Feedback recorded. Thank you.";
            $("feedback-status").className   = "feedback-status success";
        } else {
            $("feedback-status").textContent = "Could not store feedback: " + (r.message || "unknown error");
            $("feedback-status").className   = "feedback-status error";
        }
    } catch (err) {
        $("feedback-status").textContent = "Network error: " + err.message;
        $("feedback-status").className   = "feedback-status error";
    } finally {
        $("submit-feedback-btn").disabled = false;
    }
});

// ── Helpers ───────────────────────────────────────────────────────────────────
function showError(title, msg) {
    loadingSection.classList.add("hidden");
    resultSection.classList.add("hidden");
    uploadSection.classList.add("hidden");
    errorSection.classList.remove("hidden");
    $("error-title").textContent   = title;
    $("error-message").textContent = msg;
}
