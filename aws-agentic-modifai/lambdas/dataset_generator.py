import json
import boto3
import os

bedrock = boto3.client('bedrock-runtime', region_name=os.environ.get("AWS_REGION", "ap-south-1"))
s3 = boto3.client('s3')

def lambda_handler(event, context):
    chunk_uri = event
    bucket = chunk_uri.split("/")[2]
    key = "/".join(chunk_uri.split("/")[3:])
    
    # 1. Download chunk
    chunk_data = json.loads(s3.get_object(Bucket=bucket, Key=key)['Body'].read())
    chunk_text = chunk_data.get("text", "")
    
    # 2. Bedrock generation
    system_prompt = """
    You are a training data generator. Generate 4 question-answer pairs based entirely on the provided chunk.
    Return ONLY a JSON array of objects with "instruction", "input" (empty), and "output" fields.
    """
    
    try:
        response = bedrock.converse(
            modelId="amazon.nova-micro-v1:0",
            system=[{"text": system_prompt}],
            messages=[{"role": "user", "content": [{"text": f"SOURCE CHUNK:\n{chunk_text}"}]}],
            inferenceConfig={"temperature": 0.7, "maxTokens": 1000}
        )
        raw_response = response['output']['message']['content'][0]['text']
        raw_response = raw_response.strip().strip("`").strip("json").strip()
        samples = json.loads(raw_response)
    except Exception as e:
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
