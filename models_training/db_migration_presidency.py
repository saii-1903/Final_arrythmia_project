
import psycopg2
import sys

PSQL_CONN_PARAMS = {
    "dbname": "ecg_analysis",
    "user": "ecg_user",
    "password": "sais",
    "host": "127.0.0.1",
    "port": "5432"
}

def migrate_presidency():
    print("Starting Presidency Column Migration...")
    
    try:
        conn = psycopg2.connect(**PSQL_CONN_PARAMS)
        conn.autocommit = True
        cur = conn.cursor()
        
        # 1. ecg_features_annotatable updates
        table = "ecg_features_annotatable"
        
        # Check and add is_verified
        cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}' AND column_name = 'is_verified'")
        if not cur.fetchone():
            print(f"  + Adding 'is_verified' to {table}...")
            cur.execute(f"ALTER TABLE {table} ADD COLUMN is_verified BOOLEAN DEFAULT FALSE;")
        else:
            print(f"  - 'is_verified' exists in {table}.")
            
        # Check and add mistake_target (just in case)
        cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}' AND column_name = 'mistake_target'")
        if not cur.fetchone():
             print(f"  + Adding 'mistake_target' to {table}...")
             cur.execute(f"ALTER TABLE {table} ADD COLUMN mistake_target TEXT;")
        else:
             print(f"  - 'mistake_target' exists in {table}.")

        # Check and add annotation_type
        cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}' AND column_name = 'annotation_type'")
        if not cur.fetchone():
             print(f"  + Adding 'annotation_type' to {table}...")
             cur.execute(f"ALTER TABLE {table} ADD COLUMN annotation_type TEXT;")
        else:
             print(f"  - 'annotation_type' exists in {table}.")

        # Check and add cardiologist_notes
        cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}' AND column_name = 'cardiologist_notes'")
        if not cur.fetchone():
             print(f"  + Adding 'cardiologist_notes' to {table}...")
             cur.execute(f"ALTER TABLE {table} ADD COLUMN cardiologist_notes TEXT;")
        else:
             print(f"  - 'cardiologist_notes' exists in {table}.")
             
        print("\nPresidency Migration Complete.")
            
    except Exception as e:
        print(f"Migration Error: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()

if __name__ == "__main__":
    migrate_presidency()
