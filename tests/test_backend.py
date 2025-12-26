import requests
import json

BASE_URL = "https://rag-native.onrender.com"

def test_health():
    """Test health endpoint"""
    print("=" * 60)
    print("1. Testing Health Check")
    print("=" * 60)
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_root():
    """Test root endpoint"""
    print("\n" + "=" * 60)
    print("2. Testing Root Endpoint")
    print("=" * 60)
    try:
        response = requests.get(f"{BASE_URL}/", timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_documents_list():
    """Test documents list endpoint"""
    print("\n" + "=" * 60)
    print("3. Testing Documents List")
    print("=" * 60)
    try:
        response = requests.get(f"{BASE_URL}/documents", timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_conversations_list():
    """Test conversations list endpoint"""
    print("\n" + "=" * 60)
    print("4. Testing Conversations List")
    print("=" * 60)
    try:
        response = requests.get(f"{BASE_URL}/conversations", timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_search():
    """Test search endpoint"""
    print("\n" + "=" * 60)
    print("5. Testing Search Endpoint")
    print("=" * 60)
    try:
        payload = {
            "query": "test query",
            "top_k": 3
        }
        response = requests.post(
            f"{BASE_URL}/search",
            json=payload,
            timeout=10
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print(f"\n🧪 Testing Backend: {BASE_URL}\n")
    
    results = {
        "Health": test_health(),
        "Root": test_root(),
        "Documents List": test_documents_list(),
        "Conversations List": test_conversations_list(),
        "Search": test_search(),
    }
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")
