
import os
import json
import psycopg2
from pathlib import Path
from tqdm import tqdm

# Config
DB_PARAMS = {
    "host": "localhost",
    "database": "ecg_analysis",
    "user": "ecg_user",
    "password": "sais",
    "port": 5432
}
JSON_FOLDER = Path("data/converted_ecg")

def import_mitdb():
    conn = psycopg2.connect(**DB_PARAMS)
    conn.autocommit = True
    cur = conn.cursor()
    
    files = sorted(list(JSON_FOLDER.glob("MITDB*")))
    print(f"Prioritizing {len(files)} MITDB files...")
    
    for f in tqdm(files):
        try:
            data = json.loads(f.read_text())
            cur.execute("""
                INSERT INTO ecg_features_annotatable 
                (filename, segment_index, raw_signal, arrhythmia_label, dataset_source) 
                VALUES (%s, %s, %s, %s, %s) 
                ON CONFLICT (filename) DO NOTHING;
            """, (
                f.name, 
                data.get("segment_index", 0), 
                data["ECG_CH_A"], 
                data.get("label", "Unlabeled"), 
                "MITDB"
            ))
        except Exception as e:
            print(f"Error in {f.name}: {e}")
            
    conn.close()
    print("[SUCCESS] MITDB prioritization complete.")

if __name__ == "__main__":
    import_mitdb()
