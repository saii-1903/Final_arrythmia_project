"""
Quick Database Health Check: Is everything ready for annotation?
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import db_service

passed = 0
failed = 0
total = 0

def check(name, condition, detail=""):
    global passed, failed, total
    total += 1
    if condition:
        passed += 1
        print(f"  [PASS] {name}" + (f" -- {detail}" if detail else ""))
    else:
        failed += 1
        print(f"  [FAIL] {name}" + (f" -- {detail}" if detail else ""))

print("=" * 60)
print("  DATABASE HEALTH CHECK")
print("=" * 60)

# 1. Connection Test
print("\n[1] CONNECTION")
conn = None
try:
    conn = db_service._connect()
    check("PostgreSQL connection", conn is not None)
except Exception as e:
    check("PostgreSQL connection", False, str(e))

if not conn:
    print("\nCannot proceed without database connection.")
    sys.exit(1)

cur = conn.cursor()

# 2. Table Existence
print("\n[2] TABLES")
for table in ["ecg_features_annotatable", "ecg_segments"]:
    cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)", (table,))
    exists = cur.fetchone()[0]
    check(f"Table '{table}' exists", exists)

# 3. Column checks for ecg_segments
print("\n[3] SCHEMA (ecg_segments)")
critical_cols = ["segment_id", "events_json", "segment_state", "background_rhythm", "features_json"]
cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'ecg_segments'")
seg_cols = [r[0] for r in cur.fetchall()]
for col in critical_cols:
    check(f"Column 'ecg_segments.{col}'", col in seg_cols)

# 4. Column checks for ecg_features_annotatable 
print("\n[4] SCHEMA (ecg_features_annotatable)")
critical_cols2 = ["segment_id", "filename", "segment_index", "arrhythmia_label", "r_peaks_in_segment"]
cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'ecg_features_annotatable'")
feat_cols = [r[0] for r in cur.fetchall()]
for col in critical_cols2:
    check(f"Column 'ecg_features_annotatable.{col}'", col in feat_cols)

# 5. Data counts
print("\n[5] DATA VOLUME")
cur.execute("SELECT COUNT(*) FROM ecg_segments")
seg_count = cur.fetchone()[0]
check(f"ecg_segments has data", seg_count > 0, f"{seg_count} segments")

cur.execute("SELECT COUNT(*) FROM ecg_features_annotatable")
feat_count = cur.fetchone()[0]
check(f"ecg_features_annotatable has data", feat_count > 0, f"{feat_count} records")

# 6. Annotation readiness: events_json check
print("\n[6] ANNOTATION READINESS")
cur.execute("SELECT COUNT(*) FROM ecg_segments WHERE events_json IS NOT NULL")
annotated = cur.fetchone()[0]
check(f"Segments with annotations", True, f"{annotated}/{seg_count} annotated ({100*annotated//seg_count if seg_count else 0}%)")

cur.execute("SELECT COUNT(*) FROM ecg_segments WHERE segment_state = 'VERIFIED'")
verified = cur.fetchone()[0]
check(f"Verified segments", True, f"{verified}/{seg_count} verified ({100*verified//seg_count if seg_count else 0}%)")

cur.execute("SELECT COUNT(*) FROM ecg_segments WHERE segment_state = 'PENDING'")
pending = cur.fetchone()[0]
check(f"Pending segments (ready to annotate)", True, f"{pending} segments awaiting annotation")

# 7. Save/Load round-trip test
print("\n[7] WRITE/READ TEST")
import uuid
test_id_str = f"HEALTH_CHECK_{uuid.uuid4().hex[:8]}"
# Find a segment to test on
cur.execute("SELECT segment_id FROM ecg_segments ORDER BY segment_id LIMIT 1")
test_seg = cur.fetchone()
if test_seg:
    test_segment_id = test_seg[0]
    test_event = {
        "event_id": test_id_str,
        "event_type": "PVC",
        "event_category": "ECTOPY",
        "start_time": 0.5,
        "end_time": 1.1,
        "annotation_source": "health_check",
        "used_for_training": False
    }
    save_ok = db_service.save_event_to_db(test_segment_id, test_event)
    check("Write event to DB", save_ok)

    # Read it back
    seg_data = db_service.get_segment_new(test_segment_id)
    events_json = seg_data.get("events_json") if seg_data else None
    found = False
    if events_json:
        raw = events_json.get("events", []) if isinstance(events_json, dict) else events_json
        found = any(e.get("event_id") == test_id_str for e in raw)
    check("Read event back from DB", found)

    # Cleanup
    conn2 = db_service._connect()
    cur2 = conn2.cursor()
    cur2.execute("SELECT events_json FROM ecg_segments WHERE segment_id = %s", (test_segment_id,))
    row = cur2.fetchone()
    if row and row[0]:
        data = row[0] if isinstance(row[0], dict) else json.loads(row[0])
        if isinstance(data, dict):
            for key in ["events", "final_display_events"]:
                if key in data:
                    data[key] = [e for e in data[key] if e.get("event_id") != test_id_str]
        cur2.execute("UPDATE ecg_segments SET events_json = %s WHERE segment_id = %s",
                     (json.dumps(data), test_segment_id))
        conn2.commit()
    cur2.close()
    conn2.close()
    check("Cleanup test data", True)
else:
    check("Write/Read test", False, "No segments found")

# 8. GIN index check
print("\n[8] INDEXES")
cur.execute("""
    SELECT indexname FROM pg_indexes 
    WHERE tablename IN ('ecg_segments', 'ecg_features_annotatable') 
    AND indexdef LIKE '%gin%'
""")
gin_indexes = [r[0] for r in cur.fetchall()]
check("GIN indexes for JSONB queries", len(gin_indexes) > 0, f"{len(gin_indexes)} GIN indexes found")

conn.close()

# SUMMARY
print("\n" + "=" * 60)
print(f"  RESULTS: {passed}/{total} PASSED, {failed} FAILED")
if failed == 0:
    print("  DATABASE IS HEALTHY AND READY FOR ANNOTATION!")
else:
    print("  WARNING: Some checks failed. Review above.")
print("=" * 60)
