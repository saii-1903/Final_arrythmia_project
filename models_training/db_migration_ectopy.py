
import psycopg2
import sys

def run_migration_ectopy():
    print("Starting DB Migration for Ectopy Labels...")
    
    conn_params = {
        "host": "localhost",
        "database": "ecg_analysis",
        "user": "ecg_user",
        "password": "sais"
    }
    
    try:
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                # Add ectopy_label column
                print("Checking 'ectopy_label' column...")
                cur.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name='ecg_features_annotatable' AND column_name='ectopy_label';
                """)
                if not cur.fetchone():
                    print("  -> Adding 'ectopy_label' column...")
                    cur.execute("""
                        ALTER TABLE ecg_features_annotatable
                        ADD COLUMN ectopy_label VARCHAR(50) DEFAULT 'None';
                    """)
                else:
                    print("  -> 'ectopy_label' already exists.")

                # Add model_ectopy_label column
                print("Checking 'model_ectopy_label' column...")
                cur.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name='ecg_features_annotatable' AND column_name='model_ectopy_label';
                """)
                if not cur.fetchone():
                    print("  -> Adding 'model_ectopy_label' column...")
                    cur.execute("""
                        ALTER TABLE ecg_features_annotatable
                        ADD COLUMN model_ectopy_label VARCHAR(50);
                    """)
                else:
                    print("  -> 'model_ectopy_label' already exists.")

                conn.commit()
                print("✅ Ectopy Migration checks and updates completed successfully.")

    except Exception as e:
        print(f"❌ Migration Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_migration_ectopy()
