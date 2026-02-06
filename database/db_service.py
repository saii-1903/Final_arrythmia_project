import psycopg2
import json
from typing import List, Dict, Any

# ---------------------------------------
# PostgreSQL Connection Settings
# ---------------------------------------
PSQL_CONN_PARAMS = {
    "dbname": "ecg_analysis",
    "user": "ecg_user",
    "password": "sais",         # <-- your password
    "host": "127.0.0.1",
    "port": "5432"
}

def _connect():
    """Create a new PostgreSQL connection."""
    return psycopg2.connect(**PSQL_CONN_PARAMS)

# =====================================================================
# FETCH LIST OF FILES
# =====================================================================
def get_segment_list() -> List[Dict[str, Any]]:
    conn = None
    try:
        conn = _connect()
        cur = conn.cursor()
        cur.execute("""
            SELECT filename,
                   COUNT(*) AS segment_count,
                   SUM(CASE WHEN arrhythmia_label IS NULL
                               OR arrhythmia_label='Unlabeled'
                            THEN 1 ELSE 0 END) AS unlabeled_count
            FROM ecg_features_annotatable
            GROUP BY filename
            ORDER BY filename;
        """)

        rows = cur.fetchall()
        return [
            {
                "filename": r[0],
                "segment_count": r[1],
                "unlabeled_count": r[2]
            }
            for r in rows
        ]
    except:
        return []
    finally:
        if conn:
            conn.close()

# =====================================================================
# FETCH A SINGLE SEGMENT
# =====================================================================
def get_segment_data(segment_id: int):
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    segment_id,
                    filename,
                    segment_index,
                    segment_start_s,
                    segment_duration_s,
                    arrhythmia_label,
                    arrhythmia_text_notes,
                    r_peaks_in_segment,
                    features_json,
                    cardiologist_notes,
                    corrected_by,
                    corrected_at,
                    training_round,
                    raw_signal,
                    pr_interval,
                    segment_fs,
                    dataset_source,
                    is_verified,
                    mistake_target
                FROM ecg_features_annotatable
                WHERE segment_id = %s
                """,
                (segment_id,),
            )
            row = cur.fetchone()
            if not row:
                return None

            cols = [
                "segment_id",
                "filename",
                "segment_index",
                "segment_start_s",
                "segment_duration_s",
                "arrhythmia_label",
                "arrhythmia_text_notes",
                "r_peaks_in_segment",
                "features_json",
                "cardiologist_notes",
                "corrected_by",
                "corrected_at",
                "training_round",
                "raw_signal",
                "pr_interval",
                "segment_fs",
                "dataset_source",
                "is_verified",
                "mistake_target"
            ]

            data = {cols[i]: row[i] for i in range(len(cols))}
            return data

    except Exception as e:
        print("DB ERROR get_segment_data:", e)
        return None
    finally:
        conn.close()
# =====================================================================
# NEW: FETCH FROM ecg_segments (Phase 3 Standard)
# =====================================================================
def get_segment_new(segment_id: int) -> Dict[str, Any]:
    """Fetches a segment from the new optimized ecg_segments table."""
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT segment_id, filename, segment_index, signal, features, 
                       segment_state, background_rhythm, events_json
                FROM ecg_segments
                WHERE segment_id = %s
            """, (segment_id,))
            row = cur.fetchone()
            if not row: return None
            
            return {
                "segment_id": row[0],
                "filename": row[1],
                "segment_index": row[2],
                "raw_signal": row[3] if isinstance(row[3], list) else json.loads(row[3]),
                "features_json": row[4] if isinstance(row[4], dict) else json.loads(row[4] or "{}"),
                "segment_state": row[5],
                "background_rhythm": row[6],
                "events_json": row[7] if isinstance(row[7], dict) else json.loads(row[7] or "{}")
            }
    except Exception as e:
        print("DB ERROR get_segment_new:", e)
        return None
    finally:
        conn.close()

def save_event_to_db(segment_id: int, event: Dict[str, Any]) -> bool:
    """Appends a cardiologist event to the events_json list for a segment."""
    conn = _connect()
    try:
        with conn.cursor() as cur:
            # Fetch existing events_json from ecg_segments
            cur.execute("SELECT events_json FROM ecg_segments WHERE segment_id = %s", (segment_id,))
            row = cur.fetchone()
            if not row:
                return False
            
            raw_data = row[0]
            # Handle if it's already a dict (full decision) or a list (events only)
            if isinstance(raw_data, str):
                data = json.loads(raw_data)
            else:
                data = raw_data or []

            if isinstance(data, list):
                # Legacy or simple list mode
                data.append(event)
            elif isinstance(data, dict) and "events" in data:
                # Full decision mode
                data["events"].append(event)
                # Ensure it appears in final_display_events too
                if "final_display_events" in data:
                    data["final_display_events"].append(event)
            else:
                # Fallback
                data = [event]
                
            cur.execute(
                "UPDATE ecg_segments SET events_json = %s WHERE segment_id = %s",
                (json.dumps(data), segment_id)
            )
            conn.commit()
            return True
    except Exception as e:
        print("DB ERROR save_event_to_db:", e)
        return False
    finally:
        conn.close()

def count_confirmed_cardiologist_events() -> int:
    """Counts how many events marked by a cardiologist exist in the ecg_segments table."""
    conn = _connect()
    try:
        with conn.cursor() as cur:
            # Check if events_json is a list of events or a decision dict
            # For robustness, we check both structures using JSONB operators
            cur.execute("""
                SELECT COUNT(*) 
                FROM ecg_segments, 
                LATERAL (
                    SELECT CASE 
                        WHEN jsonb_typeof(events_json) = 'array' THEN events_json
                        WHEN jsonb_typeof(events_json) = 'object' AND events_json ? 'events' THEN events_json->'events'
                        ELSE '[]'::jsonb
                    END as event_list
                ) AS l,
                jsonb_array_elements(l.event_list) AS event
                WHERE event->>'annotation_source' = 'cardiologist'
                  AND event->>'annotation_status' = 'confirmed';
            """)
            row = cur.fetchone()
            return row[0] if row else 0
    except Exception as e:
        print("DB ERROR count_confirmed_cardiologist_events:", e)
        return 0
    finally:
        conn.close()

# LEGACY update_annotation removed.

# =====================================================================
# SAVE MODEL PREDICTION (for XAI UI)
# =====================================================================
def save_model_prediction(segment_id: int, pred_label: str, probs_list):
    conn = None
    try:
        conn = _connect()
        cur = conn.cursor()

        cur.execute("""
            UPDATE ecg_features_annotatable
            SET model_pred_label = %s,
                model_pred_probs = %s
            WHERE segment_id = %s;
        """, (pred_label, json.dumps(probs_list), segment_id))

        conn.commit()

    except Exception as e:
        print("DB ERROR save_model_prediction:", e)
    finally:
        if conn:
            conn.close()
    return True
# =====================================================================
# FIND FIRST SEGMENT WITH raw_signal
# =====================================================================
def get_min_segment_id_with_signal() -> int:
    conn = None
    try:
        conn = _connect()
        cur = conn.cursor()

        cur.execute("""
            SELECT MIN(segment_id)
            FROM ecg_features_annotatable
            WHERE raw_signal IS NOT NULL;
        """)

        row = cur.fetchone()
        return int(row[0]) if row and row[0] else 0

    except Exception as e:
        print("DB ERROR get_min_segment_id_with_signal:", e)
        return 0
    finally:
        if conn:
            conn.close()

# =====================================================================
# GENERIC fetch_one() used by your app.py
# =====================================================================
def fetch_one(sql: str, params=None):
    conn = None
    try:
        conn = _connect()
        cur = conn.cursor()
        cur.execute(sql, params)
        return cur.fetchone()
    except Exception as e:
        print("DB fetch_one error:", e)
        return None
    finally:
        if conn:
            conn.close()

# =====================================================================
# Find first segment for a newly uploaded JSON
# =====================================================================
def get_first_segment_id_by_filename(filename_key: str) -> int:
    conn = None
    try:
        conn = _connect()
        cur = conn.cursor()

        cur.execute("""
            SELECT segment_id
            FROM ecg_features_annotatable
            WHERE filename = %s
            ORDER BY segment_index ASC
            LIMIT 1;
        """, (filename_key,))

        row = cur.fetchone()
        return row[0] if row else 0

    except Exception as e:
        print("DB ERROR get_first_segment_id_by_filename:", e)
        return 0

    finally:
        if conn:
            conn.close()
# LEGACY get_all_corrected removed.
