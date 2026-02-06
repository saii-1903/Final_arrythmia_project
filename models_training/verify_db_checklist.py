
import psycopg2
from psycopg2 import sql
import pandas as pd

# Connection params
DB_PARAMS = {
    "dbname": "ecg_analysis",
    "user": "ecg_user",
    "password": "sais",
    "host": "127.0.0.1",
    "port": "5432"
}

def run_check(name, check_func, conn):
    print(f"\nExample Check: {name}")
    print("=" * 50)
    try:
        check_func(conn)
        print("✅ Check Finished")
    except Exception as e:
        print(f"❌ Check Failed: {e}")

def get_cursor(conn):
    return conn.cursor()

def check_0_connections(conn):
    print("0️⃣ CONNECTION SANITY")
    cur = get_cursor(conn)
    query = "SELECT pid, datname, application_name, state FROM pg_stat_activity WHERE datname = 'ecg_analysis';"
    cur.execute(query)
    rows = cur.fetchall()
    print(f"Active connections to ecg_analysis: {len(rows)}")
    for row in rows:
        print(row)
    cur.close()

def check_1_db_table(conn):
    print("1️⃣ TABLE EXISTS & DB CHECK")
    cur = get_cursor(conn)
    cur.execute("SELECT current_database();")
    print(f"Current Database: {cur.fetchone()[0]}")
    
    cur.execute("SELECT to_regclass('public.ecg_features_annotatable');")
    res = cur.fetchone()[0]
    print(f"Table 'ecg_features_annotatable' matches: {res}")
    cur.close()

def check_2_columns(conn):
    print("2️⃣ REQUIRED COLUMNS EXIST")
    cur = get_cursor(conn)
    query = """
    SELECT column_name, data_type, is_nullable
    FROM information_schema.columns
    WHERE table_name = 'ecg_features_annotatable'
    ORDER BY ordinal_position;
    """
    cur.execute(query)
    rows = cur.fetchall()
    cols = [r[0] for r in rows]
    required = [
        "segment_id", "filename", "segment_index", "segment_start_s", "raw_signal",
        "arrhythmia_label", "ectopy_label", "model_pred_label", "model_ectopy_label",
        "annotation_type", "mistake_target", "used_for_training", "corrected_by", "corrected_at"
    ]
    missing = [c for c in required if c not in cols]
    if missing:
        print(f"❌ MISSING COLUMNS: {missing}")
    else:
        print("✅ All required columns exist.")
    
    # Optional
    optional = ["model_confidence", "low_confidence_flag", "cardiologist_notes"]
    present_opt = [c for c in optional if c in cols]
    print(f"Optional columns present: {present_opt}")
    cur.close()

def check_3_domain(conn):
    print("3️⃣ DOMAIN CHECKS")
    cur = get_cursor(conn)
    
    print("Checking annotation_type...")
    cur.execute("SELECT DISTINCT annotation_type FROM ecg_features_annotatable;")
    print(f"Values: {[r[0] for r in cur.fetchall()]}")
    
    print("Checking mistake_target...")
    cur.execute("SELECT DISTINCT mistake_target FROM ecg_features_annotatable;")
    print(f"Values: {[r[0] for r in cur.fetchall()]}")
    cur.close()

def check_4_integrity(conn):
    print("4️⃣ ANNOTATION INTEGRITY")
    cur = get_cursor(conn)
    
    # Annotated but not classified
    cur.execute("""
        SELECT COUNT(*) FROM ecg_features_annotatable 
        WHERE corrected_by IS NOT NULL AND annotation_type IS NULL;
    """)
    n1 = cur.fetchone()[0]
    print(f"Annotated but NULL annotation_type (Should be 0): {n1}")
    
    # FN/FP without target
    cur.execute("""
        SELECT COUNT(*) FROM ecg_features_annotatable 
        WHERE annotation_type IN ('FALSE_NEGATIVE','FALSE_POSITIVE') AND mistake_target IS NULL;
    """)
    n2 = cur.fetchone()[0]
    print(f"FN/FP without mistake_target (Should be 0): {n2}")
    cur.close()

def check_5_used_flag(conn):
    print("5️⃣ USED_FOR_TRAINING FLAG BEHAVIOR")
    cur = get_cursor(conn)
    
    # New corrections reset flag
    cur.execute("""
        SELECT COUNT(*) FROM ecg_features_annotatable 
        WHERE annotation_type IS NOT NULL AND (used_for_training IS NOT FALSE);
    """)
    n1 = cur.fetchone()[0]
    print(f"Annotated rows where used_for_training is NOT FALSE (Should be 0 for new): {n1}")
    
    cur.execute("SELECT COUNT(*) FROM ecg_features_annotatable WHERE used_for_training = TRUE;")
    n2 = cur.fetchone()[0]
    print(f"Rows marked used_for_training=TRUE (Historical): {n2}")
    cur.close()

def check_6_preview(conn):
    print("6️⃣ RETRAINING DATASET PREVIEW")
    cur = get_cursor(conn)
    
    print("--- Rhythm Retraining Preview ---")
    cur.execute("""
        SELECT segment_id, arrhythmia_label, annotation_type 
        FROM ecg_features_annotatable 
        WHERE annotation_type IN ('FALSE_NEGATIVE','FALSE_POSITIVE','BORDERLINE') 
          AND mistake_target = 'RHYTHM' 
          AND used_for_training = FALSE;
    """)
    r_rows = cur.fetchall()
    print(f"Count: {len(r_rows)}")
    if len(r_rows) < 10:
        for r in r_rows: print(r)
        
    print("--- Ectopy Retraining Preview ---")
    cur.execute("""
        SELECT segment_id, ectopy_label, annotation_type 
        FROM ecg_features_annotatable 
        WHERE annotation_type IN ('FALSE_NEGATIVE','FALSE_POSITIVE','BORDERLINE') 
          AND mistake_target = 'ECTOPY' 
          AND used_for_training = FALSE;
    """)
    e_rows = cur.fetchall()
    print(f"Count: {len(e_rows)}")
    if len(e_rows) < 10:
        for r in e_rows: print(r)
    cur.close()

def check_7_leaking(conn):
    print("7️⃣ CONFIRMED CASES NOT LEAKING")
    cur = get_cursor(conn)
    cur.execute("""
        SELECT COUNT(*) FROM ecg_features_annotatable 
        WHERE annotation_type = 'CONFIRMED_CORRECT' AND used_for_training = FALSE;
    """)
    n = cur.fetchone()[0]
    print(f"Confirmed correct cases (ignored in training): {n}")
    cur.close()

def check_8_time(conn):
    print("8️⃣ TIME-BASED SPLIT SANITY")
    cur = get_cursor(conn)
    cur.execute("""
        SELECT COUNT(*) AS total, COUNT(segment_start_s) AS with_time, 
               COUNT(*) - COUNT(segment_start_s) AS missing_time 
        FROM ecg_features_annotatable;
    """)
    row = cur.fetchone()
    print(f"Total: {row[0]}, With Time: {row[1]}, Missing Time: {row[2]}")
    cur.close()

def check_10_summary(conn):
    print("🔟 ONE-SHOT HEALTH SUMMARY")
    cur = get_cursor(conn)
    cur.execute("""
        SELECT annotation_type, mistake_target, used_for_training, COUNT(*) 
        FROM ecg_features_annotatable 
        GROUP BY annotation_type, mistake_target, used_for_training 
        ORDER BY annotation_type;
    """)
    rows = cur.fetchall()
    print(f"{'Type':<20} | {'Target':<10} | {'Used':<5} | {'Count'}")
    print("-" * 50)
    for r in rows:
        print(f"{str(r[0]):<20} | {str(r[1]):<10} | {str(r[2]):<5} | {r[3]}")
    cur.close()


if __name__ == "__main__":
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        print("Connected to DB.")
        
        run_check("0. Connections", check_0_connections, conn)
        run_check("1. Table & DB", check_1_db_table, conn)
        run_check("2. Columns", check_2_columns, conn)
        run_check("3. Domain", check_3_domain, conn)
        run_check("4. Integrity", check_4_integrity, conn)
        run_check("5. Used Flag", check_5_used_flag, conn)
        run_check("6. Preview", check_6_preview, conn)
        run_check("7. Leaking", check_7_leaking, conn)
        run_check("8. Time Sanity", check_8_time, conn)
        run_check("10. Connectivity Summary", check_10_summary, conn)
        
        conn.close()
    except Exception as e:
        print(f"CRITICAL FAILURE: {e}")
