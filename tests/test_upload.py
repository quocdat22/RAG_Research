"""Upload test document to verify Supabase integration."""
import requests
import json

API_URL = "http://localhost:8000"

print("🚀 Testing document upload to Supabase...\n")

# Upload document
print("📤 Uploading test_document.txt...")
with open("test_document.txt", "rb") as f:
    files = {"file": ("test_document.txt", f, "text/plain")}
    response = requests.post(f"{API_URL}/documents/upload", files=files)

if response.status_code == 200:
    data = response.json()
    print(f"✅ Upload successful!")
    print(f"   Document ID: {data['document_id']}")
    print(f"   Filename: {data['filename']}")
    print(f"   Chunks: {data['chunk_count']}")
else:
    print(f"❌ Upload failed: {response.status_code}")
    print(f"   Error: {response.text}")
    exit(1)

# List documents
print("\n📋 Listing all documents...")
response = requests.get(f"{API_URL}/documents")
if response.status_code == 200:
    data = response.json()
    print(f"✅ Found {data['total']} documents:")
    for doc in data['documents']:
        print(f"   - {doc['filename']} (ID: {doc['document_id'][:8]}...)")
else:
    print(f"❌ Failed to list documents: {response.status_code}")

print("\n✅ Test completed! Check Supabase dashboard to verify.")
