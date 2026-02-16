-- init_db.sql
-- Unified Schema for ECG Analysis Dashboard

-- 1. Legacy Table (Support for existing features)
CREATE TABLE IF NOT EXISTS ecg_features_annotatable (
    segment_id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    segment_index INT NOT NULL,
    segment_start_s FLOAT NOT NULL,
    segment_duration_s FLOAT NOT NULL,
    arrhythmia_label VARCHAR(50) DEFAULT 'Unlabeled',
    arrhythmia_text_notes TEXT DEFAULT '',
    r_peaks_in_segment TEXT,
    features_json JSONB,
    model_pred_label TEXT,
    model_pred_probs JSONB,
    cardiologist_notes TEXT,
    corrected_by TEXT,
    corrected_at TIMESTAMP,
    training_round INT,
    raw_signal JSONB, -- Added for consistency
    pr_interval FLOAT,
    segment_fs INT DEFAULT 250,
    dataset_source TEXT,
    is_verified BOOLEAN DEFAULT FALSE,
    mistake_target TEXT,
    events_json JSONB DEFAULT '[]'::jsonb -- Unified event storage
);

-- 2. Optimized Segment Table (Phase 3 Standard)
CREATE TABLE IF NOT EXISTS ecg_segments (
    segment_id SERIAL PRIMARY KEY,
    patient_id TEXT,
    filename VARCHAR(255) NOT NULL,
    segment_index INT NOT NULL,
    signal JSONB,
    features JSONB,
    segment_state TEXT DEFAULT 'PENDING',
    background_rhythm TEXT DEFAULT 'Sinus Rhythm',
    events_json JSONB DEFAULT '{"events": [], "final_display_events": []}'::jsonb,
    segment_fs INT DEFAULT 250,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Performance Indexes
CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_legacy_segment ON ecg_features_annotatable (filename, segment_index);
CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_optimized_segment ON ecg_segments (filename, segment_index);

-- GIN Indexes for high-speed JSONB searches (Crucial for finding PACs/PVCs across thousands of segments)
CREATE INDEX IF NOT EXISTS idx_gin_events_legacy ON ecg_features_annotatable USING GIN (events_json);
CREATE INDEX IF NOT EXISTS idx_gin_events_optimized ON ecg_segments USING GIN (events_json);

-- 4. Utility Views (Optional - for Analytics)
CREATE OR REPLACE VIEW v_segment_summary AS
SELECT 
    s.segment_id,
    s.filename,
    s.segment_index,
    s.background_rhythm,
    s.segment_state,
    COALESCE(jsonb_array_length(s.events_json->'events'), 0) as event_count
FROM ecg_segments s;
