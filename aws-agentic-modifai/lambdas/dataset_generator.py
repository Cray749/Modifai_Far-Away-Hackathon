import json
import boto3
import os
import time
from google.genai import types
from google.genai.errors import APIError
from gemini_helper import get_gemini_client

s3 = boto3.client('s3')

def lambda_handler(event, context):
    chunk_uri = event
    bucket = chunk_uri.split("/")[2]
    key = "/".join(chunk_uri.split("/")[3:])
    
    # 1. Download chunk
    chunk_data = json.loads(s3.get_object(Bucket=bucket, Key=key)['Body'].read())
    chunk_text = chunk_data.get("text", "")
    
    # 2. Gemini generation
    system_prompt = """
    You are a training data generator. Generate 4 question-answer pairs based entirely on the provided chunk.
    Return ONLY a JSON array of objects with "instruction", "input" (empty), and "output" fields.
    """
    
    try:
        gemini_client = get_gemini_client()
        for attempt in range(5):
            try:
                response = gemini_client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=f"SOURCE CHUNK:\n{chunk_text}",
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.7,
                        max_output_tokens=1000
                    )
                )
                raw_response = response.text
                raw_response = raw_response.strip().strip("`").strip("json").strip()
                samples = json.loads(raw_response)
                break  # success
            except APIError as e:
                # 429 is rate limiting/quota exhausted
                if e.code != 429:
                    raise
                wait = (2 ** attempt) * 3  # 3, 6, 12, 24, 48 seconds
                print("USING GEMINI!!!"); print(f"Throttled, waiting {wait}s before retry (attempt {attempt+1}/5)...")
                time.sleep(wait)
                samples = []
        else:
            print("All retry attempts exhausted due to throttling.")
            samples = []
    except Exception as e:
        print("====== DEBUG: ENTERED EXCEPTION HANDLER IN DATASET_GENERATOR ======")
        print(f"Failed to generate samples: {e}")
        samples = []
        
    # Format and ensure chunk_id is present
    for s in samples:
        s["chunk_id"] = chunk_data.get("chunk_id", 0)
        
    # 3. Upload to S3
    sample_key = key.replace("chunks", "samples")
    s3.put_object(
        Bucket=bucket,
        Key=sample_key,
        Body=json.dumps(samples)
    )
    
    return {
        "sample_uri": f"s3://{bucket}/{sample_key}",
        "sample_count": len(samples)
    }
