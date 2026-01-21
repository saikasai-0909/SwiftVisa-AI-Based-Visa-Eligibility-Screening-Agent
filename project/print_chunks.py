import json

# Path to chunks JSON file
CHUNKS_FILE = "project/chunks.json"

def load_chunks():
    # Load all chunks from JSON
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def print_chunk_by_index(index):
    # Print chunk details using index
    chunks = load_chunks()

    if index < 0 or index >= len(chunks):
        print("❌ Invalid index")
        return

    chunk = chunks[index]


    # Display chunk information
    print("\n" + "="*60)
    print(f"📌 Chunk Index : {index}")
    print(f"🆔 Chunk ID    : {chunk['chunk_id']}")
    print(f"📄 File        : {chunk['file']}")
    print(f"📂 Section     : {chunk['section_title']}")
    print(f"🔢 Token Count : {chunk['token_count']}")
    print("="*60)
    print(chunk["text"])
    print("="*60 + "\n")

def print_chunk_by_id(chunk_id):
    # Search and print chunk using chunk ID
    chunks = load_chunks()

    for chunk in chunks:
        if chunk["chunk_id"] == chunk_id:
            print("\n" + "="*60)
            print(f"🆔 Chunk ID    : {chunk['chunk_id']}")
            print(f"📄 File        : {chunk['file']}")
            print(f"📂 Section     : {chunk['section_title']}")
            print(f"🔢 Token Count : {chunk['token_count']}")
            print("="*60)
            print(chunk["text"])
            print("="*60 + "\n")
            return

    print("❌ Chunk ID not found")

# ✅ CHANGE THIS ONLY
if __name__ == "__main__":
    print_chunk_by_index(1)   # Change index here
    