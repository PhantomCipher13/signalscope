-- SignalScope — Supabase Schema Migration
-- Run this in the Supabase SQL editor before deploying.
-- Version: 1.0  (2026-09-13)
--
-- Tables:
--   analyses      — per-image analysis results
--   probe_results — per-transformation probe outcomes
--   provenance    — EXIF / C2PA metadata summary
--   feedback      — tester feedback (not used for auto-retraining)
--
-- Scientific constraints:
--   - Image binary data is NEVER stored.
--   - Only SHA-256 hashes are stored.
--   - evaluation_type = 'in_distribution' for CIFAKE-based analyses.
--   - Unseen-generator results MUST use evaluation_type = 'unseen_generator'.
--   - Feedback is for UX improvement only — NOT auto-retraining signal.

-- Enable Row Level Security (RLS) — configure policies in Supabase dashboard
-- For hackathon use, RLS can be disabled temporarily:
-- ALTER TABLE analyses DISABLE ROW LEVEL SECURITY;

CREATE TABLE IF NOT EXISTS analyses (
    id                    UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    created_at            TIMESTAMPTZ DEFAULT now() NOT NULL,
    image_hash            TEXT NOT NULL,           -- SHA-256 hex, 64 chars. No binary stored.
    model_version         TEXT,
    prediction            TEXT CHECK (prediction IN ('real', 'synthetic', NULL)),
    raw_probability       REAL CHECK (raw_probability BETWEEN 0 AND 1),
    calibrated_probability REAL CHECK (calibrated_probability BETWEEN 0 AND 1),
    calibration_status    TEXT DEFAULT 'not_calibrated',
    reliability_status    TEXT,
    stability             REAL CHECK (stability BETWEEN 0 AND 1),
    probes_run            INTEGER DEFAULT 0,
    processing_time_ms    REAL,
    evaluation_type       TEXT DEFAULT 'in_distribution'
                          CHECK (evaluation_type IN ('in_distribution', 'unseen_generator'))
);

CREATE INDEX IF NOT EXISTS idx_analyses_created_at  ON analyses (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analyses_image_hash   ON analyses (image_hash);
CREATE INDEX IF NOT EXISTS idx_analyses_prediction   ON analyses (prediction);

CREATE TABLE IF NOT EXISTS probe_results (
    id                    UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    created_at            TIMESTAMPTZ DEFAULT now() NOT NULL,
    analysis_id           UUID REFERENCES analyses(id) ON DELETE CASCADE,
    transformation        TEXT NOT NULL,
    parameters            JSONB,
    probability           REAL,
    calibrated_probability REAL,
    status                TEXT CHECK (status IN ('ok', 'failed')),
    error_message         TEXT
);

CREATE INDEX IF NOT EXISTS idx_probe_analysis ON probe_results (analysis_id);

CREATE TABLE IF NOT EXISTS provenance (
    id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    created_at      TIMESTAMPTZ DEFAULT now() NOT NULL,
    analysis_id     UUID REFERENCES analyses(id) ON DELETE CASCADE,
    exif_status     TEXT,
    c2pa_status     TEXT,
    summary         TEXT
    -- IMPORTANT: Absence of EXIF is NOT evidence of AI generation.
    -- This table stores contextual information only.
);

CREATE INDEX IF NOT EXISTS idx_prov_analysis ON provenance (analysis_id);

CREATE TABLE IF NOT EXISTS feedback (
    id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    created_at      TIMESTAMPTZ DEFAULT now() NOT NULL,
    analysis_id     UUID REFERENCES analyses(id),
    verdict         TEXT NOT NULL CHECK (verdict IN ('correct', 'incorrect', 'unsure')),
    comment         TEXT,
    tester_id       TEXT    -- anonymised (no real names, no auth required)
    -- NOTE: Feedback is NOT used for automatic retraining.
    -- It is collected for UX improvement and failure-case identification only.
);

CREATE INDEX IF NOT EXISTS idx_feedback_analysis ON feedback (analysis_id);
CREATE INDEX IF NOT EXISTS idx_feedback_verdict  ON feedback (verdict);

-- Useful view for admin feedback review
CREATE OR REPLACE VIEW feedback_with_analysis AS
SELECT
    f.id AS feedback_id,
    f.created_at,
    f.verdict,
    f.comment,
    f.tester_id,
    a.id AS analysis_id,
    a.prediction,
    a.calibrated_probability,
    a.reliability_status,
    a.stability,
    a.probes_run,
    a.model_version,
    a.image_hash
FROM feedback f
LEFT JOIN analyses a ON f.analysis_id = a.id
ORDER BY f.created_at DESC;
