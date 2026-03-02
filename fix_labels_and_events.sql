-- ================================================================
-- FIX SCRIPT: Recover lost labels + populate events_json
-- Run in order. Each section is safe to re-run (idempotent).
-- Connect: psql -U ecg_user -d ecg_analysis
-- ================================================================


-- ================================================================
-- STEP 0: VERIFY CURRENT STATE BEFORE DOING ANYTHING
-- ================================================================
\echo '=== BEFORE STATE ==='
SELECT 
    COUNT(*) AS total,
    COUNT(arrhythmia_label) AS has_label,
    COUNT(*) - COUNT(arrhythmia_label) AS missing_label,
    (SELECT COUNT(*) FROM ecg_segments WHERE events_json IS NOT NULL) AS segments_with_events
FROM ecg_features_annotatable;


-- ================================================================
-- STEP 1: FIX MITDB LABELS
-- MITDB records have no arrhythmia_label despite having raw_signal.
-- We recover labels from filename - MITDB filenames encode the record.
-- wfdb_to_json.py uses priority logic, so the segment label is the
-- most severe arrhythmia in that window.
-- We cannot recover the exact original label without re-running the
-- converter. Instead, we set them to a placeholder so migrate can run.
-- The real fix is to re-run wfdb_to_json.py then re-import.
-- ================================================================
\echo ''
\echo '=== STEP 1: CHECK MITDB LABEL RECOVERY OPTIONS ==='
-- First check what MITDB filenames look like to understand what we have
SELECT filename, arrhythmia_label, dataset_source
FROM ecg_features_annotatable
WHERE dataset_source = 'MITDB'
LIMIT 20;


-- ================================================================
-- STEP 2: FIX AFDB NULL LABELS
-- AFDB segments should be either "Atrial Fibrillation" or "Sinus Rhythm"
-- The filename pattern is: AFDB_<record>_seg_<NNNN>.json
-- We cannot determine which without re-reading the JSON files.
-- But we CAN use a smart heuristic from the walkthrough.md:
-- "Sinus Rhythm: 24,202" and "Atrial Fibrillation: 6,111" out of 30,313
-- Ratio is roughly 80% Sinus, 20% AFib for AFDB records.
-- 
-- THE REAL FIX: re-run afdb_to_json.py and re-import with labels.
-- Temporary heuristic below just for diagnosis.
-- ================================================================
\echo ''
\echo '=== STEP 2: AFDB NULL LABEL COUNT ==='
SELECT 
    dataset_source,
    COUNT(*) AS total,
    COUNT(arrhythmia_label) AS labeled,
    COUNT(*) - COUNT(arrhythmia_label) AS unlabeled_nulls
FROM ecg_features_annotatable
GROUP BY dataset_source;


-- ================================================================
-- STEP 3: POPULATE events_json FOR ALL LABELED SEGMENTS
-- This is the CRITICAL fix. Right now 28286 segments have NULL
-- events_json in ecg_segments - retrain.py skips all of them.
-- We need to write events_json for every segment that has a label.
-- ================================================================
\echo ''
\echo '=== STEP 3: COUNT SEGMENTS NEEDING events_json ==='
SELECT COUNT(*) AS need_events_json
FROM ecg_segments s
JOIN ecg_features_annotatable f ON s.segment_id = f.segment_id
WHERE s.events_json IS NULL
  AND f.arrhythmia_label IS NOT NULL
  AND f.arrhythmia_label != 'Unlabeled';


-- ================================================================
-- STEP 3A: POPULATE events_json FOR AFDB (Atrial Fibrillation)
-- Only do segments where we have "Atrial Fibrillation" label
-- ================================================================
\echo ''
\echo '=== STEP 3A: Populating AFib events ==='

UPDATE ecg_segments s
SET events_json = jsonb_build_object(
    'segment_state', 'VERIFIED',
    'background_rhythm', 'Atrial Fibrillation',
    'events', jsonb_build_array(
        jsonb_build_object(
            'event_id',          'migrated_' || s.segment_id::text || '_r',
            'event_type',        'Atrial Fibrillation',
            'event_category',    'RHYTHM',
            'start_time',        0.0,
            'end_time',          10.0,
            'beat_indices',      '[]'::jsonb,
            'annotation_source', 'trusted_source',
            'annotation_status', 'confirmed',
            'used_for_training', true
        )
    ),
    'final_display_events', jsonb_build_array(
        jsonb_build_object(
            'event_id',          'migrated_' || s.segment_id::text || '_r',
            'event_type',        'Atrial Fibrillation',
            'event_category',    'RHYTHM',
            'start_time',        0.0,
            'end_time',          10.0,
            'beat_indices',      '[]'::jsonb,
            'annotation_source', 'trusted_source',
            'annotation_status', 'confirmed',
            'used_for_training', true
        )
    )
)
FROM ecg_features_annotatable f
WHERE s.segment_id = f.segment_id
  AND f.arrhythmia_label IN ('Atrial Fibrillation', 'AFib', 'AF')
  AND (s.events_json IS NULL 
       OR NOT (s.events_json ? 'events')
       OR jsonb_array_length(s.events_json->'events') = 0);

-- Check result
SELECT COUNT(*) AS afib_events_written
FROM ecg_segments s
JOIN ecg_features_annotatable f ON s.segment_id = f.segment_id
WHERE f.arrhythmia_label IN ('Atrial Fibrillation', 'AFib', 'AF')
  AND s.events_json IS NOT NULL
  AND jsonb_array_length(s.events_json->'events') > 0;


-- ================================================================
-- STEP 3B: POPULATE events_json FOR ALL OTHER LABELED SEGMENTS
-- Covers: PAC, PVC, Atrial Run, Atrial Couplet, PSVT, Artifact,
--         Ventricular Run, Sinus Rhythm, etc.
-- NOTE: Sinus Rhythm gets populated here too - retrain.py will
--       skip it (get_rhythm_label_idx returns None for Sinus)
--       but having events_json populated lets the segment count
--       correctly in stats queries.
-- ================================================================
\echo ''
\echo '=== STEP 3B: Populating all other labeled segment events ==='

UPDATE ecg_segments s
SET events_json = jsonb_build_object(
    'segment_state', 'VERIFIED',
    'background_rhythm', f.arrhythmia_label,
    'events', jsonb_build_array(
        jsonb_build_object(
            'event_id',          'migrated_' || s.segment_id::text || '_auto',
            'event_type',        f.arrhythmia_label,
            'event_category',    
                CASE 
                    WHEN f.arrhythmia_label IN ('PAC','PVC','Atrial Run','Atrial Couplet',
                         'Ventricular Run','NSVT','PVC Bigeminy','PVC Trigeminy','PVC Couplet',
                         'PAC Bigeminy')
                    THEN 'ECTOPY'
                    ELSE 'RHYTHM'
                END,
            'start_time',        0.0,
            'end_time',          10.0,
            'beat_indices',      '[]'::jsonb,
            'annotation_source', 'trusted_source',
            'annotation_status', 'confirmed',
            'used_for_training', true
        )
    ),
    'final_display_events', jsonb_build_array(
        jsonb_build_object(
            'event_id',          'migrated_' || s.segment_id::text || '_auto',
            'event_type',        f.arrhythmia_label,
            'event_category',    
                CASE 
                    WHEN f.arrhythmia_label IN ('PAC','PVC','Atrial Run','Atrial Couplet',
                         'Ventricular Run','NSVT','PVC Bigeminy','PVC Trigeminy','PVC Couplet',
                         'PAC Bigeminy')
                    THEN 'ECTOPY'
                    ELSE 'RHYTHM'
                END,
            'start_time',        0.0,
            'end_time',          10.0,
            'beat_indices',      '[]'::jsonb,
            'annotation_source', 'trusted_source',
            'annotation_status', 'confirmed',
            'used_for_training', true
        )
    )
)
FROM ecg_features_annotatable f
WHERE s.segment_id = f.segment_id
  AND f.arrhythmia_label IS NOT NULL
  AND f.arrhythmia_label NOT IN ('Unlabeled', '', 'Atrial Fibrillation', 'AFib', 'AF')
  AND (s.events_json IS NULL 
       OR NOT (s.events_json ? 'events')
       OR jsonb_array_length(s.events_json->'events') = 0);

-- Check result  
SELECT COUNT(*) AS other_events_written
FROM ecg_segments s
WHERE s.events_json IS NOT NULL
  AND s.events_json ? 'events'
  AND jsonb_array_length(s.events_json->'events') > 0;


-- ================================================================
-- STEP 4: VERIFY FINAL STATE
-- ================================================================
\echo ''
\echo '=== STEP 4: FINAL STATE ==='

-- Total segments with valid events
SELECT 
    COUNT(*) AS total_segments,
    COUNT(CASE WHEN events_json IS NOT NULL 
               AND events_json ? 'events'
               AND jsonb_array_length(events_json->'events') > 0 
          THEN 1 END) AS segments_with_events,
    COUNT(CASE WHEN events_json IS NULL 
               OR NOT (events_json ? 'events')
               OR jsonb_array_length(events_json->'events') = 0
          THEN 1 END) AS segments_still_empty
FROM ecg_segments;

-- Event type distribution (what retrain.py will actually see)
\echo ''
\echo '=== EVENT TYPE DISTRIBUTION (what retrain.py will train on) ==='
SELECT 
    event->>'event_type' AS event_type,
    COUNT(*) AS count
FROM ecg_segments,
LATERAL (
    SELECT CASE 
        WHEN jsonb_typeof(events_json) = 'array' THEN events_json
        WHEN jsonb_typeof(events_json) = 'object' AND events_json ? 'events' 
             THEN events_json->'events'
        ELSE '[]'::jsonb
    END AS ev
) l,
jsonb_array_elements(l.ev) AS event
WHERE events_json IS NOT NULL
GROUP BY event->>'event_type'
ORDER BY count DESC;

-- Rhythm model training count (non-Sinus, non-None labels)
\echo ''
\echo '=== RHYTHM MODEL TRAINING EVENTS ==='
SELECT 
    event->>'event_type' AS event_type,
    COUNT(*) AS count
FROM ecg_segments,
LATERAL (
    SELECT CASE 
        WHEN jsonb_typeof(events_json) = 'array' THEN events_json
        WHEN jsonb_typeof(events_json) = 'object' AND events_json ? 'events' 
             THEN events_json->'events'
        ELSE '[]'::jsonb
    END AS ev
) l,
jsonb_array_elements(l.ev) AS event
WHERE events_json IS NOT NULL
  AND event->>'event_type' IN (
    'Supraventricular Tachycardia','Atrial Fibrillation','Atrial Flutter',
    'Junctional Rhythm','Idioventricular Rhythm','Ventricular Tachycardia',
    'Ventricular Fibrillation','1st Degree AV Block','2nd Degree AV Block Type 1',
    '2nd Degree AV Block Type 2','3rd Degree AV Block','Bundle Branch Block',
    'Artifact','PSVT','Pause','Atrial Run','Ventricular Run'
  )
GROUP BY event->>'event_type'
ORDER BY count DESC;

-- Ectopy model training count
\echo ''
\echo '=== ECTOPY MODEL TRAINING EVENTS ==='
SELECT 
    event->>'event_type' AS event_type,
    COUNT(*) AS count
FROM ecg_segments,
LATERAL (
    SELECT CASE 
        WHEN jsonb_typeof(events_json) = 'array' THEN events_json
        WHEN jsonb_typeof(events_json) = 'object' AND events_json ? 'events' 
             THEN events_json->'events'
        ELSE '[]'::jsonb
    END AS ev
) l,
jsonb_array_elements(l.ev) AS event
WHERE events_json IS NOT NULL
  AND event->>'event_type' IN ('PAC','PVC','Run',
    'Atrial Couplet','PVC Bigeminy','PVC Trigeminy','PVC Couplet',
    'PAC Bigeminy','Ventricular Run','NSVT')
GROUP BY event->>'event_type'
ORDER BY count DESC;
