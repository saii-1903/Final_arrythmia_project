import sys
import os

# Add project root and dashboard to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'dashboard'))

from app import app as flask_app
import json

def debug_api():
    print("=== Debug: /api/xai/47789 Response Structure ===")
    with flask_app.test_client() as client:
        response = client.get('/api/xai/47789')
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.get_json()
            print("Response Keys:", data.keys())
            if "background_rhythm" not in data:
                print("[MISSING] background_rhythm")
            if "explanation" not in data:
                print("[MISSING] explanation")
            if "segment_state" not in data:
                print("[MISSING] segment_state")
            print("\nFull Response:")
            print(json.dumps(data, indent=2)[:1000]) # Truncate if too long
        else:
            print("Error Data:", response.data.decode())

if __name__ == "__main__":
    debug_api()
