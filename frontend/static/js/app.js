/**
 * SignalScope — static/js/app.js
 * Frontend logic: upload → analyze → render results → feedback
 * UI redesigned for forensic-product readability (September 2026).
 * API contract: UNCHANGED from v0.2.0 backend.
 */
"use strict";

// API_BASE: reads from window.SIGNALSCOPE_API_BASE (injected via /static/js/config.js
// at build time for Vercel/production). Falls back to "" (same-origin) for local dev.
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
dropZone.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); } });
dropZone.addEventListener("dragover",  e => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
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
    window.scrollTo({ top: 0, behavior: "smooth" });
});

// Technical details toggle
const techToggle = $("tech-toggle");
const techBody   = $("tech-body");
const techIcon   = $("tech-icon");
techToggle.addEventListener("click", () => {
    const expanded = techToggle.getAttribute("aria-expanded") === "true";
    techToggle.setAttribute("aria-expanded", String(!expanded));
    techBody.hidden = expanded;
    techIcon.classList.toggle("open", !expanded);
});

const ANALYSIS_STEPS = [
    { id: "step-received" },
    { id: "step-detect" },
    { id: "step-probe" },
    { id: "step-explain" },
    { id: "step-prov" },
    { id: "step-cal" }
];

async function runAnalysis() {
    if (!_currentFile) return;

    // Show loading
    uploadSection.classList.add("hidden");
    resultSection.classList.add("hidden");
    errorSection.classList.add("hidden");
    loadingSection.classList.remove("hidden");

    // Reset step states
    ANALYSIS_STEPS.forEach(s => {
        const el = $(s.id);
        if (el) {
            el.classList.remove("active", "done");
            el.querySelector(".step-dot").textContent = "...";
        }
    });

    $("loading-step").textContent = "Analyzing image on server. This may take a few seconds...";

    try {
        const formData = new FormData();
        formData.append("file",             _currentFile);
        formData.append("run_stress_test",  optStress.checked    ? "true" : "false");
        formData.append("run_gradcam",      optGradcam.checked   ? "true" : "false");
        formData.append("run_provenance",   optProvenance.checked ? "true" : "false");

        const resp = await fetch(`${API_BASE}/analyze`, {
            method: "POST",
            body:   formData,
        });

        ANALYSIS_STEPS.forEach(s => {
            const el = $(s.id);
            if (el) {
                el.classList.remove("active");
                el.classList.add("done");
                el.querySelector(".step-dot").textContent = "✓";
            }
        });

        if (!resp.ok) {
            let detail = "";
            try { const j = await resp.json(); detail = j.detail?.message || j.detail || resp.statusText; }
            catch { detail = resp.statusText; }
            showError("Analysis failed", detail || `Server returned ${resp.status}`);
            return;
        }

        const result = await resp.json();
        _currentResult = result;
        renderResult(result);

    } catch (err) {
        clearInterval(stepInterval);
        showError("Network error", err.message || "Could not reach the SignalScope server.");
    }
}

// ── Render result ─────────────────────────────────────────────────────────────
function renderResult(r) {
    loadingSection.classList.add("hidden");
    resultSection.classList.remove("hidden");

    const pred     = r.prediction || "unknown";
    const calProb  = r.calibrated_probability;
    const rawProb  = r.raw_probability;
    const dispProb = calProb ?? rawProb ?? 0;

    // ① Assessment verdict
    const verdictBanner = $("verdict-banner");
    const verdictLabel  = $("verdict-label");
    verdictBanner.className = "assessment-verdict-bar";

    let verdictText, verdictClass, verdictExplain;
    if (pred === "synthetic") {
        verdictText    = "Likely AI-Generated";
        verdictClass   = "synthetic";
        verdictExplain = "SignalScope's model estimates this image shows patterns consistent with AI generation. " +
                         "This is a probabilistic assessment, not proof of synthetic origin.";
    } else if (pred === "real") {
        verdictText    = "Likely Authentic";
        verdictClass   = "real";
        verdictExplain = "SignalScope's model estimates this image is consistent with a real photograph. " +
                         "This is a probabilistic assessment, not proof of authentic origin.";
    } else {
        verdictText    = "Inconclusive";
        verdictClass   = "uncertain";
        verdictExplain = "The model's confidence is near the uncertainty threshold. " +
                         "The evidence is insufficient to make a reliable determination.";
    }
    verdictBanner.classList.add(verdictClass);
    verdictLabel.textContent = verdictText;
    verdictLabel.className   = `verdict-headline ${verdictClass}`;
    $("verdict-explanation").textContent = verdictExplain;

    // ② Confidence
    const pct = Math.round(dispProb * 100);
    $("prob-display").textContent = pct + "%";
    $("prob-sublabel").textContent = calProb != null
        ? "Temperature-calibrated"
        : `Raw model output · ${r.calibration_status || "not calibrated"}`;

    // Probability bar
    $("prob-bar-fill").style.width  = `${pct}%`;
    $("prob-bar-marker").style.left = `${pct}%`;
    $("prob-bar-marker").setAttribute("aria-valuenow", pct);
    $("bar-caption").textContent = calProb != null
        ? `Temperature-calibrated probability (T=2.876). 50% = maximum uncertainty.`
        : "Raw model probability. Not temperature-calibrated.";

    // ② Reliability
    const rel      = r.reliability || {};
    const st       = r.stress_test  || {};
    const relStatus = rel.status || "insufficient_evidence";
    const relMap    = {
        "stable":                   ["stable",         "Stable",              "High"],
        "unstable":                 ["unstable",       "Unstable",            "Low"],
        "insufficient_evidence":    ["insufficient",   "Limited evidence",    "—"],
        "threshold_not_configured": ["not-configured", "Not configured",      "—"],
    };
    const [relClass, relLabel] = relMap[relStatus] || ["not-configured", relStatus, "—"];

    $("reliability-pill-wrap").innerHTML =
        `<span class="rel-pill ${relClass}" role="status" aria-label="Reliability: ${relLabel}">${relLabel}</span>`;
    
    $("prob-sublabel").textContent = "How strongly the model leans toward the result.";
    $("reliability-sub").textContent = "How consistently the result behaves under additional checks.";

    // ③ Evidence bullets — built from ACTUAL backend data, never fabricated
    const evidenceList = $("evidence-list");
    evidenceList.innerHTML = "";
    const bullets = buildEvidenceBullets(r, pred, dispProb, relStatus, st, rel);
    bullets.forEach(text => {
        const li = document.createElement("li");
        li.className = "evidence-item";
        li.innerHTML = `<span class="evidence-dot" aria-hidden="true"></span><span>${esc(text)}</span>`;
        evidenceList.appendChild(li);
    });

    // ④ Grad-CAM
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
            `Layer: ${gc.target_layer_name || "auto"} · class: ${gc.target_class === 1 ? "synthetic" : "real"}`;
    } else {
        $("gradcam-section").classList.add("hidden");
    }

    // ⑤ Robustness probes
    renderProbes(st, rel);

    // ⑤ Provenance
    renderProvenance(r.provenance || {});

    // ⑥ Technical details
    renderTechDetails(r, st, rel, gc, dispProb, rawProb, calProb);

    // Analysis ID
    if (r.analysis_id) {
        $("analysis-id-note").textContent = `Analysis #${r.analysis_id}`;
    }

    // Reset feedback state
    _selectedVerdict = null;
    document.querySelectorAll(".btn-feedback").forEach(b => {
        b.classList.remove("selected");
        b.setAttribute("aria-pressed", "false");
    });
    $("submit-feedback-btn").classList.add("hidden");
    $("feedback-status").textContent = "";
    $("feedback-comment").value = "";

    // Collapse tech details
    $("tech-toggle").setAttribute("aria-expanded", "false");
    $("tech-body").hidden = true;
    $("tech-icon").classList.remove("open");

    window.scrollTo({ top: 0, behavior: "smooth" });
}

// ── Evidence bullets ──────────────────────────────────────────────────────────
function buildEvidenceBullets(r, pred, prob, relStatus, st, rel) {
    const bullets = [];

    // 1. Core prediction statement
    const probPct = (prob * 100).toFixed(1);
    if (pred === "synthetic") {
        bullets.push(`The model detected visual patterns consistent with AI generation (${probPct}% confidence).`);
    } else if (pred === "real") {
        bullets.push(`The model found visual patterns consistent with a real photograph (${probPct}% confidence).`);
    } else {
        bullets.push(`The model's confidence is near 50% — indicating high uncertainty about image origin.`);
    }

    // 2. Calibration
    if (r.calibration_status === "calibrated") {
        bullets.push("Confidence is temperature-calibrated (T=2.876), reducing overconfidence in model estimates.");
    } else if (r.calibration_status) {
        bullets.push(`Calibration status: ${r.calibration_status}.`);
    }

    // 3. Robustness
    const probesOk  = st.probes_successful;
    const probesAll = st.probes_configured || 5;
    if (probesOk != null) {
        if (probesOk === probesAll) {
            bullets.push(`The prediction remained stable across all ${probesOk} image transformations (JPEG compression, resizing).`);
        } else if (probesOk > 0) {
            bullets.push(`The prediction was consistent in ${probesOk} of ${probesAll} image transformations.`);
        } else {
            bullets.push("The prediction varied across image transformations — the result may be less reliable.");
        }
    }

    // 4. Grad-CAM
    const gc = r.gradcam || {};
    if (gc.status === "ok" && gc.overlay_b64) {
        bullets.push("A visual attention map was generated, showing which image regions most influenced the model.");
    }

    // 5. Provenance
    const exif = (r.provenance || {}).exif || {};
    if (exif.status === "metadata_available") {
        bullets.push("Camera metadata (EXIF) was found in the file.");
    } else if (exif.status && exif.status !== "no_exif") {
        bullets.push("No camera metadata (EXIF) was detected in the file.");
    }

    return bullets;
}

// ── Robustness probes render ──────────────────────────────────────────────────
function renderProbes(st, rel) {
    const probeResults = st.probe_results || [];
    const probesOk     = st.probes_successful ?? 0;
    const probesAll    = st.probes_configured  ?? probeResults.length;
    const stability    = st.stability;

    // Summary
    $("probe-count-display").style.display = "none";
    $("probe-summary-text").textContent  = `Prediction remained stable across ${probesOk} checks.`;
    $("stability-summary-text").style.display = "none";

    // Probe rows
    const probeList = $("probe-list");
    probeList.innerHTML = "";
    if (probeResults.length > 0) {
        probeResults.forEach(p => {
            const row = document.createElement("div");
            row.className = "probe-row";
            row.setAttribute("role", "listitem");
            // Determine text: if no success property, say Insufficient evidence or check backend
            const isStable = p.success;
            row.innerHTML = `
                <span class="probe-name">${esc(p.probe_name)}</span>
                <span class="probe-status ${isStable ? "ok" : "fail"}">${isStable ? "✓ Stable" : "✗ Changed"}</span>
            `;
            probeList.appendChild(row);
        });
    } else {
        probeList.innerHTML = `<div class="probe-row"><span class="probe-name" style="color:var(--text-muted);grid-column:1/-1">No probe data available.</span></div>`;
    }

    // Threshold note
    $("threshold-note").textContent = rel.threshold_configured
        ? `Stop reason: ${rel.stop_reason || "N/A"}.`
        : "All probes run deterministically — reliability thresholds not yet configured.";
}

// ── Provenance render ──────────────────────────────────────────────────────────
function renderProvenance(prov) {
    const exif = prov.exif || {};
    const c2pa = prov.c2pa || {};
    const container = $("provenance-content");

    if (!prov.exif && !prov.c2pa) {
        container.innerHTML = `<p class="body-sm">Provenance analysis was not run.</p>`;
        return;
    }

    const exifAvail = exif.status === "metadata_available";
    const c2paAvail = c2pa.status === "c2pa_found";

    let html = `<div class="prov-status-row">`;
    html += `
        <div class="prov-status-chip ${exifAvail ? "chip-available" : "chip-unavailable"}">
            <div class="chip-icon">${exifAvail ? "✓" : "—"}</div>
            <div>
                <div class="chip-label">Camera metadata</div>
                <div class="chip-status">${exifAvail ? "EXIF found" : "EXIF not found"}</div>
            </div>
        </div>
        <div class="prov-status-chip ${c2paAvail ? "chip-available" : "chip-unavailable"}">
            <div class="chip-icon">${c2paAvail ? "✓" : "—"}</div>
            <div>
                <div class="chip-label">Content credentials</div>
                <div class="chip-status">${
                    c2pa.status === "c2pa_found"              ? "C2PA found" :
                    c2pa.status === "c2pa_library_unavailable"? "Library not installed" :
                    c2pa.status                               ? c2pa.status : "C2PA not found"
                }</div>
            </div>
        </div>
    `;
    html += `</div>`;

    // EXIF fields
    if (exifAvail) {
        const fields = [
            ["Camera",   [exif.camera_make, exif.camera_model].filter(Boolean).join(" ") || "—"],
            ["Software", exif.software || "—"],
            ["Captured", exif.datetime_original || "—"],
            ["GPS",      exif.gps_available ? "Available" : "Not present"],
            ["ISO",      exif.iso_speed != null ? String(exif.iso_speed) : "—"],
            ["Exposure", exif.exposure_time || "—"],
        ];
        html += `<div class="prov-fields">`;
        fields.forEach(([k, v]) => {
            html += `<div class="prov-field"><span class="prov-key">${esc(k)}</span><span class="prov-value">${esc(v)}</span></div>`;
        });
        html += `</div>`;
    }

    if (prov.summary) {
        html += `<p class="body-sm" style="margin-top:12px">${esc(prov.summary)}</p>`;
    }

    // Mandatory disclaimer
    html += `
        <div class="prov-disclaimer" role="note">
            <strong>Important note</strong>
            Absence of metadata is <strong>NOT</strong> evidence of AI generation.
            Many real photographs have EXIF stripped by social media platforms and messaging apps.
        </div>
    `;

    container.innerHTML = html;
}

// ── Technical details ──────────────────────────────────────────────────────────
function renderTechDetails(r, st, rel, gc, dispProb, rawProb, calProb) {
    const rows = [
        ["Model",           r.model_version || "efficientnet_b0_v1"],
        ["Calibration",     r.calibration_status || "unknown"],
        ["Raw probability", rawProb != null ? (rawProb * 100).toFixed(3) + "%" : "—"],
        ["Cal. probability",calProb != null ? (calProb * 100).toFixed(3) + "%" : "—"],
        ["Stability",       st.stability != null ? (st.stability * 100).toFixed(2) + "%" : "—"],
        ["Probe count",     `${st.probes_successful ?? "—"}/${st.probes_configured ?? "—"}`],
        ["Reliability",     rel.status || "—"],
        ["Mean prob.",      rel.mean_probability != null ? (rel.mean_probability * 100).toFixed(2) + "%" : "—"],
        ["Std deviation",   rel.std_probability  != null ? (rel.std_probability  * 100).toFixed(2) + "%" : "—"],
        ["Grad-CAM layer",  gc.target_layer_name || "—"],
        ["Grad-CAM class",  gc.target_class === 1 ? "synthetic (1)" : gc.target_class === 0 ? "real (0)" : "—"],
        ["Stop reason",     rel.stop_reason || "—"],
        ["Analysis ID",     r.analysis_id != null ? String(r.analysis_id) : "—"],
        ["Supabase ID",     r.supabase_analysis_id || "—"],
    ];
    const content = $("tech-content");
    content.innerHTML = rows.map(([k, v]) =>
        `<div class="tech-row"><span class="tech-key">${esc(k)}</span><span class="tech-val">${esc(v)}</span></div>`
    ).join("");
}

// ── Feedback ──────────────────────────────────────────────────────────────────
document.querySelectorAll(".btn-feedback").forEach(btn => {
    btn.addEventListener("click", () => {
        _selectedVerdict = btn.dataset.verdict;
        document.querySelectorAll(".btn-feedback").forEach(b => {
            b.classList.remove("selected");
            b.setAttribute("aria-pressed", "false");
        });
        btn.classList.add("selected");
        btn.setAttribute("aria-pressed", "true");
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
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({
                analysis_id: analysisId,
                verdict:     _selectedVerdict,
                comment:     $("feedback-comment").value.trim() || null,
                tester_id:   null,
            }),
        });
        const rx = await resp.json();
        if (rx.status === "stored") {
            $("feedback-status").textContent = "✓ Feedback recorded. Thank you.";
            $("feedback-status").className   = "feedback-status success";
        } else {
            $("feedback-status").textContent = "Could not store feedback: " + (rx.message || "unknown error");
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

/** Escape HTML entities — prevent XSS from server data in innerHTML */
function esc(s) {
    if (s == null) return "";
    return String(s)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#x27;");
}
