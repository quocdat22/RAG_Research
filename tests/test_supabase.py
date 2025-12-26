"""Test Supabase Storage integration."""
import os

# Set environment for testing
os.environ["SUPABASE_URL"] = ""
os.environ["SUPABASE_KEY"] = ""

try:
    from src.storage.supabase_client import get_supabase_storage
    
    print("🔄 Connecting to Supabase...")
    storage = get_supabase_storage()
    print("✅ Connected successfully!")
    
    # Test upload
    print("\n🔄 Testing upload...")
    test_content = b"This is a test document for Supabase Storage."
    
    doc = storage.upload_document(
        file_path="test/sample.txt",
        file_content=test_content,
        metadata={"test": True}
    )
    print(f"✅ Uploaded: {doc['filename']} (ID: {doc['id']})")
    
    # List documents
    print("\n📁 Documents in Supabase:")
    docs = storage.list_documents()
    print(f"Total: {len(docs)} documents")
    for d in docs:
        print(f"  - {d['filename']} ({d['file_size']} bytes) - Processed: {d['processed']}")
    
    # Get stats
    print("\n📊 Storage Statistics:")
    stats = storage.get_stats()
    if stats:
        print(f"  Total documents: {stats.get('total_documents', 0)}")
        print(f"  Total size: {stats.get('total_size', 0)} bytes")
        print(f"  Processed: {stats.get('processed_count', 0)}")
        print(f"  Pending: {stats.get('pending_count', 0)}")
    
    print("\n✅ All tests passed!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
