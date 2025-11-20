"""
RAG (Retrieval-Augmented Generation) Implementation

This module handles the RAG implementation for the Hibiscus Bot, including:
- File processing (text, HTML, PDF, JSON)
- Document chunking
- ChromaDB vector database ingestion
- Relevant chunk retrieval

JSON files (like face_scan.json) are automatically parsed and converted to 
readable text format for embedding and retrieval.
"""
import glob
import json
import os

import chromadb
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()

def get_files(path, exts=('.txt', '.html', '.pdf', '.json')):
    """
    Recursively finds all files with specified extensions in a given directory.

    Args:
        path (str): The directory to search.
        exts (tuple, optional): A tuple of file extensions to look for.
                                Defaults to ('.txt', '.html', '.pdf', '.json').

    Returns:
        list: A list of file paths that match the specified extensions.
    """
    special_files = []
    for ext in exts:
        # This means search source and all subdirectories for the given extension
        special_files += glob.glob(f"{path}/**/*{ext}", recursive=True)
    return special_files

# Works for txt, html, pdf, json
def chunk_file(filepath, ext):
    """
    Reads a file and splits its content into smaller, manageable chunks.

    This function supports text, HTML, PDF, and JSON files. The chunking is done based on
    a fixed number of lines or at empty lines to maintain semantic coherence.
    
    For JSON files, the structured data is converted to readable text format before chunking.

    Args:
        filepath (str): The path to the file to be chunked.
        ext (str): The extension of the file (e.g., '.pdf', '.txt', '.json').

    Returns:
        list: A list of text chunks.
    """
    if ext == '.pdf':
        reader = PdfReader(filepath)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"
        lines = full_text.splitlines()
    elif ext == '.json':
        # Parse JSON and convert to readable text format
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                # Convert JSON to readable text format
                lines = json_to_text_lines(data)
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse JSON file {filepath}: {e}")
                # Fallback: read as text
                f.seek(0)
                lines = f.readlines()
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
    
    # Chunk container
    chunks = []
    # Aggregators
    chunk, count = "", 0
    for line in lines:
        chunk += line
        count += 1
        # Triggers for chunking, 20 lines or empty line
        if count >= 20 or line.strip() == "":
            # If not empty line, add to the chunk
            if chunk.strip():
                chunks.append(chunk.strip())
            # Reset chunk and count
            chunk, count = "", 0
    # For remainder, add to chunks
    if chunk.strip():
        chunks.append(chunk.strip())
    return chunks

def json_to_text_lines(data, prefix=""):
    """
    Converts JSON data structure into a list of readable text lines.
    
    Handles nested dictionaries, lists, and various data types in a way that
    preserves semantic meaning for embedding and retrieval.
    
    Args:
        data: The JSON data (dict, list, or primitive type)
        prefix (str): Prefix for nested structures (used recursively)
    
    Returns:
        list: A list of text lines representing the JSON data
    """
    lines = []
    
    if isinstance(data, dict):
        for key, value in data.items():
            current_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, (dict, list)):
                # For nested structures, add a header line and recurse
                lines.append(f"{current_key}:")
                lines.extend(json_to_text_lines(value, current_key))
            else:
                # For primitive values, create a readable line with proper formatting
                formatted_value = str(value)
                lines.append(f"{current_key}: {formatted_value}")
    
    elif isinstance(data, list):
        for i, item in enumerate(data):
            current_key = f"{prefix}[{i}]" if prefix else f"[{i}]"
            
            if isinstance(item, (dict, list)):
                lines.extend(json_to_text_lines(item, current_key))
            else:
                lines.append(f"{current_key}: {item}")
    
    else:
        # Primitive value
        key_label = prefix if prefix else "value"
        lines.append(f"{key_label}: {data}")
    
    return lines

def ingest_corpus(dir, collection, force_reingest=False):
    """
    Processes a directory of documents and ingests them into a ChromaDB collection with persistence.

    This function finds all supported files in the directory, chunks them, generates
    embeddings for each chunk using SentenceTransformer, and then adds the
    chunks and their embeddings to the specified ChromaDB collection.
    
    The function checks for existing documents to avoid duplicates unless force_reingest=True.

    Args:
        dir (str): The directory containing the corpus of documents.
        collection: The ChromaDB collection object to ingest the documents into.
        force_reingest (bool): If True, re-ingest files even if they already exist. Default: False.
    """
    files = get_files(dir)
    if not files:
        print(f"No files found in {dir}")
        return
    
    # Get existing IDs to avoid duplicates
    existing_ids = set()
    if not force_reingest:
        try:
            existing_data = collection.get()
            if existing_data and existing_data.get('ids'):
                existing_ids = set(existing_data['ids'])
        except Exception as e:
            print(f"Warning: Could not check existing documents: {e}")
    
    docs, ids, metadatas = [], [], []
    seen_ids = set()  # Track IDs we've seen in this batch to avoid duplicates
    
    # Chunks each file in the corpus and adds each chunk to docs
    for file in files:
        file_basename = os.path.basename(file)
        _, ext = os.path.splitext(file)
        chunks = chunk_file(file, ext)
        
        # Use full file path hash to ensure uniqueness across different directories
        import hashlib
        file_hash = hashlib.md5(file.encode()).hexdigest()[:8]
        
        for i, chunk in enumerate(chunks):
            # Create unique ID using file hash and chunk index
            chunk_id = f"{file_hash}_{file_basename}_{i}"
            
            # Skip if already exists in collection and not forcing reingest
            if chunk_id in existing_ids and not force_reingest:
                continue
            
            # Skip if duplicate in current batch
            if chunk_id in seen_ids:
                continue
            
            seen_ids.add(chunk_id)
            docs.append(chunk)
            ids.append(chunk_id)
            metadatas.append({
                "source": file_basename,
                "file_path": file,
                "chunk_index": i,
                "total_chunks": len(chunks)
            })
    
    if not docs:
        print(f"All documents from {dir} are already in the collection. Use force_reingest=True to re-ingest.")
        return
    
    print(f"Processing {len(docs)} new chunks from {len(files)} files...")
    
    # Initialize the model once for efficiency
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Generate embeddings in batches for better performance and memory management
    batch_size = 100
    all_embeddings = []
    
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i + batch_size]
        batch_embeddings = model.encode(batch_docs, show_progress_bar=True)
        all_embeddings.extend(batch_embeddings)
        print(f"Generated embeddings for {min(i + batch_size, len(docs))}/{len(docs)} chunks...")
    
    # Add to ChromaDB collection with persistence
    # ChromaDB has a maximum batch size limit, so we need to split into smaller batches
    try:
        chroma_batch_size = 5000  # Safe batch size for ChromaDB (under the 5461 limit)
        total_added = 0
        
        for i in range(0, len(docs), chroma_batch_size):
            batch_docs = docs[i:i + chroma_batch_size]
            batch_ids = ids[i:i + chroma_batch_size]
            batch_metadatas = metadatas[i:i + chroma_batch_size]
            batch_embeddings = all_embeddings[i:i + chroma_batch_size]
            
            collection.add(
                documents=batch_docs,
                ids=batch_ids,
                metadatas=batch_metadatas,
                embeddings=batch_embeddings
            )
            total_added += len(batch_docs)
            print(f"Added batch {i//chroma_batch_size + 1}: {len(batch_docs)} chunks (total: {total_added}/{len(docs)})")
        
        print(f"Successfully added {len(docs)} chunks to collection '{collection.name}' (persisted to ./chroma_db)")
    except Exception as e:
        print(f"Error adding documents to collection: {e}")
        raise

def embed_json_file(filepath, collection, force_reingest=False):
    """
    Embeds a single JSON file into a ChromaDB collection.
    
    This function reads a JSON file, converts it to readable text chunks,
    generates embeddings, and adds them to the specified collection.
    
    Args:
        filepath (str): Path to the JSON file to embed.
        collection: The ChromaDB collection to add the embeddings to.
        force_reingest (bool): If True, re-ingest even if already exists. Default: False.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return
    
    file_basename = os.path.basename(filepath)
    print(f"\n📄 Processing JSON file: {file_basename}")
    
    # Get existing IDs to avoid duplicates
    existing_ids = set()
    if not force_reingest:
        try:
            existing_data = collection.get()
            if existing_data and existing_data.get('ids'):
                existing_ids = set(existing_data['ids'])
        except Exception as e:
            print(f"Warning: Could not check existing documents: {e}")
    
    # Chunk the JSON file
    chunks = chunk_file(filepath, '.json')
    if not chunks:
        print(f"Warning: No chunks generated from {filepath}")
        return
    
    # Generate unique IDs for chunks
    import hashlib
    file_hash = hashlib.md5(filepath.encode()).hexdigest()[:8]
    
    docs, ids, metadatas = [], [], []
    
    for i, chunk in enumerate(chunks):
        chunk_id = f"{file_hash}_{file_basename}_{i}"
        
        # Skip if already exists and not forcing reingest
        if chunk_id in existing_ids and not force_reingest:
            continue
        
        docs.append(chunk)
        ids.append(chunk_id)
        metadatas.append({
            "source": file_basename,
            "file_path": filepath,
            "chunk_index": i,
            "total_chunks": len(chunks)
        })
    
    if not docs:
        print(f"All chunks from {file_basename} are already in the collection. Use force_reingest=True to re-ingest.")
        return
    
    print(f"Processing {len(docs)} chunks from {file_basename}...")
    
    # Initialize the model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Generate embeddings in batches
    batch_size = 100
    all_embeddings = []
    
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i + batch_size]
        batch_embeddings = model.encode(batch_docs, show_progress_bar=True)
        all_embeddings.extend(batch_embeddings)
        print(f"Generated embeddings for {min(i + batch_size, len(docs))}/{len(docs)} chunks...")
    
    # Add to ChromaDB collection
    # ChromaDB has a maximum batch size limit, so we need to split into smaller batches
    try:
        chroma_batch_size = 5000  # Safe batch size for ChromaDB (under the 5461 limit)
        total_added = 0
        
        for i in range(0, len(docs), chroma_batch_size):
            batch_docs = docs[i:i + chroma_batch_size]
            batch_ids = ids[i:i + chroma_batch_size]
            batch_metadatas = metadatas[i:i + chroma_batch_size]
            batch_embeddings = all_embeddings[i:i + chroma_batch_size]
            
            collection.add(
                documents=batch_docs,
                ids=batch_ids,
                metadatas=batch_metadatas,
                embeddings=batch_embeddings
            )
            total_added += len(batch_docs)
            print(f"Added batch {i//chroma_batch_size + 1}: {len(batch_docs)} chunks (total: {total_added}/{len(docs)})")
        
        print(f"Successfully added {len(docs)} chunks to collection '{collection.name}'")
    except Exception as e:
        print(f"Error adding documents to collection: {e}")
        raise

def get_collection_info(collection):
    """
    Get information about a persistent ChromaDB collection.
    
    Args:
        collection: The ChromaDB collection to inspect.
    
    Returns:
        dict: Dictionary with collection information including count, metadata, etc.
    """
    try:
        count = collection.count()
        metadata = collection.metadata or {}
        return {
            "name": collection.name,
            "count": count,
            "metadata": metadata,
            "persisted": True  # If we can access it, it's persisted
        }
    except Exception as e:
        return {
            "name": collection.name if hasattr(collection, 'name') else "Unknown",
            "error": str(e),
            "persisted": False
        }

def retrieve_relevant_chunks(query, collection, k, include_metadata=False):
    """
    Retrieves the most relevant document chunks from a persistent ChromaDB collection for a given query.

    This function generates an embedding for the user's query, searches the specified
    ChromaDB collection for the top-k most similar chunks, and returns them.

    Args:
        query (str): The user's query.
        collection: The ChromaDB collection to search (must be from a PersistentClient).
        k (int): The number of relevant chunks to retrieve.
        include_metadata (bool): If True, returns metadata along with documents and IDs. Default: False.

    Returns:
        list: A list of tuples. If include_metadata=False: [(chunk, id), ...]
              If include_metadata=True: [(chunk, id, metadata, distance), ...]
    """
    # Initialize model (could be cached for better performance)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding = model.encode(query)
    
    # Query the persistent collection
    results = collection.query(
        query_embeddings=[embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    
    if not results['documents'] or not results['documents'][0]:
        return []
    
    # Build results with optional metadata
    if include_metadata:
        return list(zip(
            results['documents'][0],
            results['ids'][0],
            results['metadatas'][0] if results.get('metadatas') else [{}] * len(results['documents'][0]),
            results['distances'][0] if results.get('distances') else [0.0] * len(results['documents'][0])
        ))
    else:
        return list(zip(results['documents'][0], results['ids'][0]))

def get_all_subdirectories(root_dir):
    """
    Recursively get all subdirectories in a root directory.
    
    Args:
        root_dir (str): The root directory to search.
    
    Returns:
        list: List of all subdirectory paths.
    """
    subdirs = []
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            subdirs.append(item_path)
            # Recursively get subdirectories
            subdirs.extend(get_all_subdirectories(item_path))
    return subdirs

def sanitize_collection_name(path, data_dir):
    """
    Convert a directory path to a valid collection name.
    
    Args:
        path (str): The full path to the directory.
        data_dir (str): The root data directory.
    
    Returns:
        str: A sanitized collection name.
    """
    # Get relative path from data_dir
    rel_path = os.path.relpath(path, data_dir)
    
    # Handle root data directory case
    if rel_path == "." or rel_path == "":
        return "data"
    
    # Replace path separators and special characters with hyphens
    collection_name = rel_path.replace(os.sep, "-").replace(" ", "_")
    # Remove any other invalid characters and make lowercase
    collection_name = "".join(c if c.isalnum() or c in ("-", "_") else "-" for c in collection_name)
    
    # Remove leading/trailing hyphens and dots
    collection_name = collection_name.strip("-._")
    
    # If empty after sanitization, use directory name
    if not collection_name:
        collection_name = os.path.basename(path)
        collection_name = "".join(c if c.isalnum() or c in ("-", "_") else "-" for c in collection_name)
        collection_name = collection_name.strip("-._")
    
    # Ensure it doesn't start with a number or hyphen
    if collection_name and (collection_name[0].isdigit() or collection_name[0] == "-"):
        collection_name = "data-" + collection_name
    
    # Ensure minimum length and valid characters
    if len(collection_name) < 3:
        collection_name = "data-" + collection_name
    
    # Ensure it starts and ends with alphanumeric
    while collection_name and not collection_name[0].isalnum():
        collection_name = collection_name[1:]
    while collection_name and not collection_name[-1].isalnum():
        collection_name = collection_name[:-1]
    
    return collection_name.lower()

# Main function for embedding docs
if __name__ == "__main__":
    """
    This block of code is executed when the script is run directly.
    It's used for ingesting a corpus of documents into the persistent ChromaDB.
    It processes all data in the 'data' folder and creates a separate collection
    for each subfolder.
    
    The embeddings and collections are persisted to disk, so they will be available
    in subsequent sessions without re-ingestion.
    """
    
    # Get the project root directory (parent of utils folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Path to financial analysis texts
    financial_texts_dir = os.path.join(project_root, "data", "FinAnalysisTexts")
    
    # Check if directory exists
    if not os.path.isdir(financial_texts_dir):
        print(f"Error: Directory not found: {financial_texts_dir}")
        print("Please ensure the 'data/FinAnalysisTexts' directory exists.")
        exit(1)

    print("=" * 60)
    print("Financial Analysis Texts Ingestion")
    print("=" * 60)
    print(f"Data Directory: {financial_texts_dir}")
    # Use CHROMA_DB_PATH env var if set (for persistent volumes in deployment), otherwise use ./chroma_db
    chroma_db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    print(f"ChromaDB Path: {chroma_db_path}")
    print("=" * 60)
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=chroma_db_path)
    
    # Get or create the financial analysis texts collection
    financial_collection = client.get_or_create_collection(
        name="financial-analysis-texts",
        metadata={"description": "Financial analysis textbooks and resources"}
    )
    
    # Check current count
    current_count = financial_collection.count()
    print(f"\nCurrent collection size: {current_count} document chunks")
    
    # Ask user if they want to re-ingest
    # In non-interactive environments (like deployment), auto-reingest if needed
    if current_count > 0:
        # Check if running in non-interactive mode (no TTY)
        import sys
        if sys.stdin.isatty():
            response = input(f"\nCollection already has {current_count} chunks. Re-ingest? (y/N): ")
            if response.lower() != 'y':
                print("Skipping ingestion. Exiting.")
                exit(0)
        else:
            # Non-interactive: auto-reingest in deployment
            print(f"Non-interactive mode: Auto-reingesting {current_count} existing chunks...")
        force_reingest = True
    else:
        force_reingest = False
    
    print(f"\nProcessing files from: {financial_texts_dir}")
    print("This may take several minutes depending on the number of PDFs...\n")
    
    try:
        # Ingest the corpus
        ingest_corpus(financial_texts_dir, financial_collection, force_reingest=force_reingest)
        
        # Show final count
        final_count = financial_collection.count()
        print("\n" + "=" * 60)
        print(f"Ingestion Complete!")
        print(f"Total document chunks in collection: {final_count}")
        print("=" * 60)
        print("\nYou can now start the chatbot with: chainlit run app.py")
        
    except Exception as e:
        print(f"\nError during ingestion: {e}")
        raise
