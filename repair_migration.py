"""
repair_migration.py
-------------------
Fixes segments in ecg_segments whose labels were incorrectly migrated 
from ecg_features_annotatable. Specifically addresses:
- Incorrect background_rhythm mappings.
- Incorrect event_type mappings (e.g., Atrial Couplet -> PVC).
- Restores cardiologist_notes into the xai_notes/events structure.
"""
import psycopg2
import json
import sys
from pathlib import Path

# Add current dir to path for imports
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from models_training.data_loader import normalize_label, get_rhythm_label_idx, get_ectopy_label_idx, RHYTHM_CLASS_NAMES, ECTOPY_CLASS_NAMES

conn = psycopg2.connect(dbname='ecg_analysis', user='ecg_user', password='sais', host='127.0.0.1')
cur = conn.cursor()

def repair():
    print(" >> Starting MIGRATION REPAIR...")
    
    # 1. Fetch segments where old table had a label
    cur.execute("""
        SELECT f.segment_id, f.arrhythmia_label, f.cardiologist_notes, s.events_json, s.background_rhythm
        FROM ecg_features_annotatable f
        JOIN ecg_segments s ON f.segment_id = s.segment_id
        WHERE f.arrhythmia_label != 'Unlabeled' AND f.arrhythmia_label IS NOT NULL
    """)
    
    rows = cur.fetchall()
    print(f" Found {len(rows)} segments to verify.")
    
    repair_count = 0
    
    for row in rows:
        seg_id, old_label, old_notes, current_events_raw, current_bg = row
        
        # Parse current events
        if isinstance(current_events_raw, str):
            current_data = json.loads(current_events_raw or "{}")
        else:
            current_data = current_events_raw or {}
            
        if isinstance(current_data, list):
            current_data = {"events": current_data}
            
        # Re-compute what it SHOULD be using fixed logic
        label_norm = normalize_label(old_label)
        
        # 1. Background Rhythm Logic (Better mapping)
        new_bg = "Sinus Rhythm"
        if "Atrial Fibrillation" in label_norm: new_bg = "Atrial Fibrillation"
        elif "Atrial Flutter" in label_norm: new_bg = "Atrial Flutter"
        elif "Sinus Bradycardia" in label_norm: new_bg = "Sinus Bradycardia"
        elif "Sinus Tachycardia" in label_norm: new_bg = "Sinus Tachycardia"
        elif "Junctional" in label_norm: new_bg = "Junctional Rhythm"
        elif "Idioventricular" in label_norm: new_bg = "Idioventricular Rhythm"
        elif "Ventricular Tachycardia" in label_norm: new_bg = "Ventricular Tachycardia"
        elif "Ventricular Fibrillation" in label_norm: new_bg = "Ventricular Fibrillation"
        elif "AV Block" in label_norm: new_bg = label_norm # Preserve blocks as background
        
        # 2. Events List
        new_events = []
        
        # Rhythm Event (if it's a specific pathology we want to highlight)
        r_idx = get_rhythm_label_idx(label_norm)
        if r_idx is not None:
            new_events.append({
                "event_id": f"mig_{seg_id}_r",
                "event_type": label_norm,
                "event_category": "RHYTHM",
                "start_time": 0.0,
                "end_time": 10.0,
                "annotation_source": "cardiologist",
                "annotation_status": "confirmed",
                "used_for_training": True
            })
            
        # Ectopy Event
        e_idx = get_ectopy_label_idx(label_norm)
        if e_idx is not None and ECTOPY_CLASS_NAMES[e_idx] != "None":
            new_events.append({
                "event_id": f"mig_{seg_id}_e",
                "event_type": ECTOPY_CLASS_NAMES[e_idx],
                "event_category": "ECTOPY",
                "start_time": 0.0,
                "end_time": 10.0,
                "annotation_source": "cardiologist",
                "annotation_status": "confirmed",
                "used_for_training": True
            })

        # Preserve any cardiologists notes in the dict
        new_data = {
            "events": new_events,
            "final_display_events": new_events,
            "cardiologist_notes": old_notes,
            "migrated_from_label": old_label
        }
        
        # Check if we actually changed anything important
        # We skip if the current_data already has manual clinician work (not prefixed with mig_)
        has_manual_cur = any(not e.get("event_id", "").startswith("mig_") for e in current_data.get("events", []))
        
        if not has_manual_cur:
            # Safe to overwrite with corrected migration
            cur.execute("""
                UPDATE ecg_segments 
                SET background_rhythm = %s,
                    events_json = %s
                WHERE segment_id = %s
            """, (new_bg, json.dumps(new_data), seg_id))
            repair_count += 1

        if (repair_count % 500 == 0) and repair_count > 0:
            conn.commit()
            print(f"  Processed {repair_count} repairs...")

    conn.commit()
    print(f"\nRepair complete! Total segments fixed: {repair_count}")
    cur.close()
    conn.close()

if __name__ == "__main__":
    repair()
