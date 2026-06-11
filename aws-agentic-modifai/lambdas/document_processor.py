import json
import boto3
import os
import uuid
import io
import PyPDF2

s3 = boto3.client('s3')

def chunk_text(text, target_words=384, overlap_words=48):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + target_words])
        if len(chunk.split()) >= 20: # Min words
            chunks.append(chunk)
        i += (target_words - overlap_words)
    return chunks

def extract_pdf_text(bucket, key):
    file_bytes = s3.get_object(Bucket=bucket, Key=key)['Body'].read()
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    
    text_pages = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_pages.append(page_text)
            
    return "\n".join(text_pages)

def extract_txt(bucket, key):
    file_bytes = s3.get_object(Bucket=bucket, Key=key)['Body'].read()
    return file_bytes.decode('utf-8', errors='ignore')

def lambda_handler(event, context):
    strategy = event.get('strategy', {}).get('chunking', {})
    document_uris = event.get('document_s3_uris', [])
    
    if not document_uris:
        raise ValueError("Missing document_s3_uris")
        
    all_text = []
    bucket = document_uris[0].split("/")[2] # Assuming all are in the same bucket
    
    # 1. Extract text from all documents
    for uri in document_uris:
        key = "/".join(uri.split("/")[3:])
        ext = key.lower().split('.')[-1]
        
        try:
            if ext in ['pdf']:
                text = extract_pdf_text(bucket, key)
            elif ext in ['txt', 'md', 'csv']:
                text = extract_txt(bucket, key)
            else:
                # Bypass unsupported formats (like images without Textract)
                print(f"Skipping unsupported file type: {ext}")
                continue
            all_text.append(text)
        except Exception as e:
            print(f"Failed to process {uri}: {e}")
            
    combined_text = "\n\n---\n\n".join(all_text)
    
    # 2. Chunk combined text based on strategy
    max_tokens = strategy.get("max_tokens", 512)
    overlap = strategy.get("overlap", 64)
    target_words = int(max_tokens * 0.75)
    overlap_words = int(overlap * 0.75)
    
    chunks = chunk_text(combined_text, target_words, overlap_words)
    
    # 3. Upload chunks to S3 for Map State
    run_id = str(uuid.uuid4())[:8]
    chunk_uris = []
    
    for i, chunk_data in enumerate(chunks):
        chunk_key = f"modifai-jobs/{run_id}/chunks/chunk_{i}.json"
        s3.put_object(
            Bucket=bucket,
            Key=chunk_key,
            Body=json.dumps({"chunk_id": i, "text": chunk_data})
        )
        chunk_uris.append(f"s3://{bucket}/{chunk_key}")
        
    return {
        "chunk_uris": chunk_uris,
        "total_chunks": len(chunk_uris),
        "run_id": run_id,
        "bucket": bucket
    }
