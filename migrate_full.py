
import psycopg2
import json
import numpy as np
import sys
from pathlib import Path

# Fix path to imports
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

try:
    from database.db_service import PSQL_CONN_PARAMS
    from decision_engine.rhythm_orchestrator import RhythmOrchestrator
    from xai.xai import explain_segment, explain_decision
    from models_training.data_loader import normalize_label, get_rhythm_label_idx, get_ectopy_label_idx, RHYTHM_CLASS_NAMES, ECTOPY_CLASS_NAMES
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def migrate_full():
    print(f" >> Starting FULL Migration (Trusted Data Mode)...")
    
    orchestrator = RhythmOrchestrator()
    
    try:
        conn = psycopg2.connect(**PSQL_CONN_PARAMS)
        cur = conn.cursor()
        
        # 1. Fetch ALL data (No LIMIT)
        cur.execute("""
            SELECT 
                segment_id, 
                raw_signal, 
                segment_fs, 
                features_json, 
                filename, 
                segment_index,
                dataset_source,
                arrhythmia_label
            FROM ecg_features_annotatable
            WHERE raw_signal IS NOT NULL
        """)
        
        rows = cur.fetchall()
        print(f" Found {len(rows)} segments for migration.")
        
        migrated_count = 0
        
        for row in rows:
            seg_id, signal_data, fs, features_raw, filename, seg_idx, source, label_txt = row
            
            # Convert signal to numpy
            if isinstance(signal_data, str):
                signal = np.array(list(map(float, signal_data.split(","))))
            elif isinstance(signal_data, list):
                 signal = np.array(signal_data)
            else:
                signal = np.array(signal_data)
                
            features = features_raw if isinstance(features_raw, dict) else json.loads(features_raw or "{}")
            
            # --- TRUSTED SOURCE LOGIC ---
            is_trusted_source = source in ['MITDB', 'AFDB']
            
            # Default state
            segment_state = "ANALYZED"
            background_rhythm = "Unknown"
            events_list = []
            
            if is_trusted_source and label_txt and label_txt != 'Unlabeled':
                # "GOLDEN DATA" MODE
                # We do NOT run the inference engine. We trust the label.
                
                label_norm = normalize_label(label_txt)
                
                # 1. Background Rhythm (Simplified map from label)
                if "Sinus" in label_norm:
                    background_rhythm = label_norm
                elif "Atrial Fibrillation" in label_norm:
                    background_rhythm = "Atrial Fibrillation"
                elif "Flutter" in label_norm:
                    background_rhythm = "Atrial Flutter"
                else:
                    background_rhythm = "Sinus Rhythm" # Default assumption for other rhythms unless specific
                
                # 2. Create Confirmed Event
                # Determine type (Rhythm vs Ectopy)
                # We add BOTH if it's a composite, or just one.
                
                # Rhythm Event
                if get_rhythm_label_idx(label_norm) is not None:
                     events_list.append({
                        "event_id": f"mig_{seg_id}_r",
                        "event_type": label_norm,
                        "event_category": "RHYTHM",
                        "start_time": 0.0,
                        "end_time": 10.0,
                        "annotation_source": "cardiologist", # Trusted
                        "annotation_status": "confirmed",    # Trusted
                        "used_for_training": True            # TRAIN ON THIS
                     })
                
                # Ectopy Event
                ect_idx = get_ectopy_label_idx(label_norm)
                if ect_idx is not None and ECTOPY_CLASS_NAMES[ect_idx] != "None":
                     events_list.append({
                        "event_id": f"mig_{seg_id}_e",
                        "event_type": ECTOPY_CLASS_NAMES[ect_idx], # Use the canonical class name
                        "event_category": "ECTOPY",
                        "start_time": 0.0,
                        "end_time": 10.0,
                        "annotation_source": "cardiologist",
                        "annotation_status": "confirmed",
                        "used_for_training": True
                     })
                     
                segment_state = "VERIFIED"
                
            else:
                # INFERENCE MODE (For uploaded files)
                ml_results = explain_segment(signal, features)
                sqi = {"is_acceptable": True}
                decision = orchestrator.decide(
                    ml_prediction=ml_results.get("rhythm", {}),
                    clinical_features=features,
                    sqi_result=sqi,
                    segment_index=seg_idx
                )
                segment_state = decision.segment_state.value
                background_rhythm = decision.background_rhythm
                events_list = [e.to_dict() for e in decision.events]

            
            # X. Save to NEW table ecg_segments
            # AND Update the old table to mark it as verified/trained
            if is_trusted_source:
                # Mark as Golden Data for Cumulative Training
                cur.execute("""
                    UPDATE ecg_features_annotatable
                    SET is_verified = TRUE,
                        mistake_target = FALSE,
                        annotation_type = 'CONFIRMED',
                        used_for_training = TRUE
                    WHERE segment_id = %s;
                """, (seg_id,))

            cur.execute("""
                INSERT INTO ecg_segments (
                    segment_id,
                    signal,
                    features,
                    segment_state,
                    background_rhythm,
                    events_json,
                    filename,
                    segment_index,
                    segment_fs,
                    patient_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (segment_id) DO UPDATE SET
                    segment_state = EXCLUDED.segment_state,
                    background_rhythm = EXCLUDED.background_rhythm,
                    events_json = EXCLUDED.events_json,
                    filename = EXCLUDED.filename,
                    segment_index = EXCLUDED.segment_index,
                    segment_fs = EXCLUDED.segment_fs;
            """, (
                seg_id,
                json.dumps(signal.tolist()),
                json.dumps(features),
                segment_state,
                background_rhythm,
                json.dumps(events_list),
                filename,
                seg_idx,
                fs,
                None # patient_id not available in row currently
            ))
            
            migrated_count += 1
            if migrated_count % 100 == 0:
                print(f"  ... migrated {migrated_count} segments")
                conn.commit()
            
        conn.commit()
        print(f"[SUCCESS] Full Migration Complete! Total segments: {migrated_count}")
        
    except Exception as e:
        print(f"[ERROR] Error during migration: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    migrate_full()
