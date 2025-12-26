"""
Test complete Supabase integration - Upload document và xem kết quả.
"""
import os
import sys

# Setup environment
os.environ["USE_SUPABASE_STORAGE"] = "true"
os.environ["SUPABASE_URL"] = ""
os.environ["SUPABASE_KEY"] = ""
from src.storage.supabase_client import get_supabase_storage

print("=" * 60)
print("📊 SUPABASE STORAGE STATUS")
print("=" * 60)

try:
    storage = get_supabase_storage()
    
    # List all documents
    print("\n📁 Documents in Supabase:")
    docs = storage.list_documents(limit=100)
    
    if not docs:
        print("   ⚠️  No documents found!")
        print("\n💡 Try uploading a document:")
        print("   1. Start backend: uv run uvicorn src.api.main:app --reload")
        print("   2. Upload via: http://localhost:8000/docs")
        print("   3. Or use curl:")
        print('      curl -X POST "http://localhost:8000/documents/upload" -F "file=@test.pdf"')
    else:
        print(f"   Total: {len(docs)} documents\n")
        
        for i, doc in enumerate(docs, 1):
            print(f"   {i}. {doc['filename']}")
            print(f"      📊 Size: {doc['file_size']:,} bytes")
            print(f"      📅 Uploaded: {doc['upload_date']}")
            print(f"      ✅ Processed: {doc['processed']}")
            print(f"      📦 Chunks: {doc['chunk_count']}")
            print(f"      🆔 ID: {doc['id']}")
            
            # Get chunks for this document
            chunks = storage.get_document_chunks(doc['id'])
            if chunks:
                print(f"      📝 Chunk details:")
                for chunk in chunks[:3]:  # Show first 3 chunks
                    content_preview = chunk['content'][:80] + "..." if len(chunk['content']) > 80 else chunk['content']
                    print(f"         - Chunk {chunk['chunk_index']}: {content_preview}")
                if len(chunks) > 3:
                    print(f"         ... and {len(chunks) - 3} more chunks")
            print()
    
    # Storage files in buckets
    print("\n☁️  Files in Storage Buckets:")
    from supabase import create_client
    client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    
    for bucket in ["documents", "embeddings"]:
        try:
            files = client.storage.from_(bucket).list()
            if files:
                print(f"\n   📦 {bucket}:")
                for f in files:
                    size_kb = f.get('metadata', {}).get('size', 0) / 1024
                    print(f"      - {f['name']} ({size_kb:.2f} KB)")
            else:
                print(f"\n   📦 {bucket}: (empty)")
        except Exception as e:
            print(f"\n   📦 {bucket}: Error - {e}")
    
    # Statistics
    print("\n" + "=" * 60)
    print("📈 STATISTICS")
    print("=" * 60)
    
    total_docs = len(docs)
    total_size = sum(d['file_size'] for d in docs)
    total_chunks = sum(d['chunk_count'] for d in docs)
    processed = sum(1 for d in docs if d['processed'])
    
    print(f"   Documents: {total_docs}")
    print(f"   Total size: {total_size / 1024:.2f} KB")
    print(f"   Total chunks: {total_chunks}")
    print(f"   Processed: {processed}/{total_docs}")
    
    print("\n" + "=" * 60)
    print("✅ Supabase integration is working!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
