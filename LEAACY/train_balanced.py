
import sys
import subprocess

def main():
    print("="*60)
    print("⚠️  DEPRECATION NOTICE: train_balanced.py is legacy logic.")
    print("Please use 'retrain.py' for the unified Presidency Pipeline.")
    print("="*60)
    
    print("\nStarting retrain.py instead...")
    try:
        subprocess.check_call([sys.executable, "models_training/retrain.py"] + sys.argv[1:])
    except Exception as e:
        print(f"Error launching retrain.py: {e}")

if __name__ == "__main__":
    main()
