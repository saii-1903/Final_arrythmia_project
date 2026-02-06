
import psycopg2
import sys

def run_migration():
    print("Starting DB Migration for Retraining Validations...")
    
    conn_params = {
        "host": "localhost",
        "database": "ecg_analysis",
        "user": "ecg_user",
        "password": "sais"
    }
    
    try:
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                # 1. Add used_for_training column
                print("Checking 'used_for_training' column...")
                cur.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name='ecg_features_annotatable' AND column_name='used_for_training';
                """)
                if not cur.fetchone():
                    print("  -> Adding 'used_for_training' column...")
                    cur.execute("""
                        ALTER TABLE ecg_features_annotatable
                        ADD COLUMN used_for_training BOOLEAN DEFAULT FALSE;
                    """)
                else:
                    print("  -> 'used_for_training' already exists.")

                # 2. Add mistake_target column
                print("Checking 'mistake_target' column...")
                cur.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name='ecg_features_annotatable' AND column_name='mistake_target';
                """)
                if not cur.fetchone():
                    print("  -> Adding 'mistake_target' column...")
                    # We use Text instead of ENUM to avoid migration complexity with types, 
                    # but we can enforce check constraint if needed. For now, simple text is flexible.
                    cur.execute("""
                        ALTER TABLE ecg_features_annotatable
                        ADD COLUMN mistake_target VARCHAR(20);
                    """)
                    
                    # OPTIONAL: Backfill simple heuristics for existing data so we don't lose everything
                    print("  -> Backfilling 'mistake_target' based on labels (Heuristic)...")
                    # Ectopy: PVC, PAC, Runs
                    cur.execute("""
                        UPDATE ecg_features_annotatable
                        SET mistake_target = 'ECTOPY'
                        WHERE mistake_target IS NULL 
                          AND (
                              arrhythmia_label ILIKE '%PVC%' OR 
                              arrhythmia_label ILIKE '%PAC%' OR 
                              arrhythmia_label ILIKE '%Run%' OR
                              arrhythmia_label ILIKE '%Bigeminy%' OR
                              arrhythmia_label ILIKE '%Trigeminy%' OR
                              arrhythmia_label ILIKE '%Couplet%'
                          );
                    """)
                    # Rhythm: Everything else (AF, Flutter, VT, Blocks, etc)
                    cur.execute("""
                        UPDATE ecg_features_annotatable
                        SET mistake_target = 'RHYTHM'
                        WHERE mistake_target IS NULL;
                    """)
                else:
                    print("  -> 'mistake_target' already exists.")

                conn.commit()
                print("✅ Migration checks and updates completed successfully.")

    except Exception as e:
        print(f"❌ Migration Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_migration()
