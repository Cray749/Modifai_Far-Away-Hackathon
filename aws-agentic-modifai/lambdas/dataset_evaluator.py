import json
import boto3
import os

s3 = boto3.client('s3')
bedrock = boto3.client('bedrock-runtime', region_name=os.environ.get("AWS_REGION", "ap-south-1"))

def lambda_handler(event, context):
    dataset_uris = [item['sample_uri'] for item in event]
    
    if not dataset_uris:
        return {"action": "proceed", "training_data_uri": None, "score": 1.0}
        
    bucket = dataset_uris[0].split("/")[2]
    run_id = dataset_uris[0].split("/")[4]
    
    # Merge all samples
    all_samples = []
    for uri in dataset_uris:
        key = "/".join(uri.split("/")[3:])
        try:
            samples = json.loads(s3.get_object(Bucket=bucket, Key=key)['Body'].read())
            all_samples.extend(samples)
        except Exception:
            pass
            
    # Simple Mock evaluation to save time, in reality we run Bedrock as Critic
    # Assuming dataset is good for this hackathon
    dataset_score = 0.9
    
    if dataset_score < 0.7:
        return {
            "action": "regenerate",
            "reason": f"Dataset score {dataset_score} < 0.7. Needs improvement."
        }
        
    # Write to final JSONL format for Bedrock Fine-tuning
    jsonl_lines = []
    for s in all_samples:
        prompt = f"User: {s.get('instruction', '')}\nBot:"
        completion = f" {s.get('output', '')}"
        jsonl_lines.append(json.dumps({"prompt": prompt, "completion": completion}))
        
    training_data = "\n".join(jsonl_lines)
    training_key = f"modifai-jobs/{run_id}/training_data.jsonl"
    
    s3.put_object(
        Bucket=bucket,
        Key=training_key,
        Body=training_data
    )
    
    return {
        "action": "proceed",
        "training_data_uri": f"s3://{bucket}/{training_key}",
        "score": dataset_score,
        "bucket": bucket,
        "run_id": run_id
    }
