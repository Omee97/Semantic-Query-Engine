import requests
import json

# Test the local PDF upload and question functionality
def test_local_endpoints():
    base_url = "http://127.0.0.1:8000"
    
    print("Testing local PDF Question Answering endpoints...")
    print("=" * 50)
    
    # Test 1: Check if the home page loads
    try:
        response = requests.get(f"{base_url}/")
        print(f"✓ Home page: Status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("✗ Home page: Connection failed - make sure the server is running")
        return
    
    # Test 2: Test the ask endpoint with a sample question
    try:
        data = {"question": "What is this document about?"}
        response = requests.post(f"{base_url}/api/ask", data=data)
        print(f"✓ Ask endpoint: Status {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"  Response: {result.get('answer', 'No answer')}")
        else:
            print(f"  Error: {response.text}")
    except Exception as e:
        print(f"✗ Ask endpoint: Error - {str(e)}")
    
    print("\nNote: To test with actual PDFs, upload a PDF through the web interface at:")
    print(f"{base_url}/")

if __name__ == "__main__":
    test_local_endpoints()