#!/usr/bin/env python3
"""
relabel_from_json.py

Fixes the 28275 NULL arrhythmia_label rows in ecg_features_annotatable
by re-reading the original JSON files in data/converted_ecg/.

Run from project root:
    python relabel_from_json.py

Then run fix_labels_and_events.sql to populate events_json.
Then run migrate_full.py to sync to ecg_segments.
"""

import json
import psycopg2
from pathlib import Path
from tqdm import tqdm
import sys

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "models_training"))
from data_loader import normalize_label

JSON_FOLDER = BASE_DIR / "data" / "converted_ecg"

CONN_PARAMS = {
    "dbname":   "ecg_analysis",
    "user":     "ecg_user",
    "password": "sais",
    "host":     "127.0.0.1",
    "port":     "5432"
}

def main():
    if not JSON_FOLDER.exists():
        print(f"ERROR: JSON folder not found at {JSON_FOLDER}")
        print("Make sure wfdb_to_json.py and afdb_to_json.py have been run first.")
        sys.exit(1)

    files = sorted(JSON_FOLDER.glob("*.json"))
    print(f"Found {len(files)} JSON files in {JSON_FOLDER}")

    conn = psycopg2.connect(**CONN_PARAMS)
    cur  = conn.cursor()

    # ── Only process rows that are currently NULL label ────────────────
    cur.execute("""
        SELECT filename, segment_index, dataset_source
        FROM ecg_features_annotatable
        WHERE arrhythmia_label IS NULL
           OR arrhythmia_label = 'Unlabeled'
    """)
    rows_needing_label = {(r[0], r[1]): r[2] for r in cur.fetchall()}
    print(f"Rows needing label recovery: {len(rows_needing_label)}")

    if len(rows_needing_label) == 0:
        print("Nothing to fix — all rows already have labels.")
        conn.close()
        return

    # ── Build a filename->label map from JSON files ────────────────────
    print("Reading JSON files to recover labels...")
    label_map = {}   # {filename: label}
    skipped   = 0

    for f in tqdm(files, desc="Scanning JSONs"):
        try:
            data  = json.loads(f.read_text())
            label = data.get("label") or data.get("arrhythmia_label")
            if label:
                label_map[f.name] = normalize_label(label)
            else:
                skipped += 1
        except Exception as e:
            skipped += 1

    print(f"Labels recovered from JSONs: {len(label_map)}, skipped: {skipped}")

    # ── Update the DB ──────────────────────────────────────────────────
    updated = 0
    still_null = 0

    for (filename, seg_idx), dataset_source in tqdm(rows_needing_label.items(),
                                                     desc="Updating DB"):
        label = label_map.get(filename)

        if label is None:
            still_null += 1
            continue

        cur.execute("""
            UPDATE ecg_features_annotatable
            SET arrhythmia_label = %s,
                dataset_source   = COALESCE(dataset_source, %s)
            WHERE filename       = %s
              AND segment_index  = %s
        """, (label, dataset_source, filename, seg_idx))
        updated += 1

    conn.commit()

    print(f"\n{'='*50}")
    print(f"Updated:    {updated:6d} rows with recovered labels")
    print(f"Still NULL: {still_null:6d} rows (JSON files missing or no label field)")
    print(f"{'='*50}")

    # ── Summary of what we now have ────────────────────────────────────
    cur.execute("""
        SELECT arrhythmia_label, COUNT(*) AS cnt
        FROM ecg_features_annotatable
        WHERE raw_signal IS NOT NULL
        GROUP BY arrhythmia_label
        ORDER BY cnt DESC
        LIMIT 20
    """)
    print("\nLabel distribution after fix:")
    print(f"  {'Label':<40} {'Count':>8}")
    print(f"  {'-'*50}")
    for row in cur.fetchall():
        label = row[0] if row[0] else "NULL"
        print(f"  {label:<40} {row[1]:>8}")

    conn.close()

    print("\n✅ Done. Next steps:")
    print("  1. Run:  psql -U ecg_user -d ecg_analysis -f fix_labels_and_events.sql")
    print("  2. Run:  python migrate_full.py")
    print("  3. Run:  python models_training/retrain.py --task rhythm")
    print("  4. Run:  python models_training/retrain.py --task ectopy")


if __name__ == "__main__":
    main()
