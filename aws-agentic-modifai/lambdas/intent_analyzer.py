import json
import boto3
import os

bedrock = boto3.client('bedrock-runtime', region_name=os.environ.get("AWS_REGION", "ap-south-1"))
s3 = boto3.client('s3')

def lambda_handler(event, context):
    # Support both single document_s3_uri and multiple document_s3_uris
    document_uris = event.get('document_s3_uris', [])
    single_uri = event.get('document_s3_uri')
    if single_uri and single_uri not in document_uris:
        document_uris.append(single_uri)
        
    sample_texts = []
    
    # Try to grab a small snippet from up to the first 3 documents
    for doc_uri in document_uris[:3]:
        bucket = doc_uri.split("/")[2]
        key = "/".join(doc_uri.split("/")[3:])
        try:
            resp = s3.get_object(Bucket=bucket, Key=key)
            # Read first 1000 bytes for intent detection
            sample_texts.append(f"Snippet from {doc_uri}:\n{resp['Body'].read(1000).decode('utf-8', errors='ignore')}")
        except Exception as e:
            print(f"Warning: Could not fetch snippet from {doc_uri}: {e}")

    combined_sample = "\n---\n".join(sample_texts) if sample_texts else "No text could be extracted."

    system_prompt = """
    You are an AI Architect Agent. Analyze the user's document sample(s).
    Identify the intent (e.g. QA, summarization, instruction). 
    Recommend a chunking strategy (max_tokens, overlap).
    Recommend a base model (amazon.nova-micro-v1:0 or amazon.titan-text-express-v1).
    Recommend initial hyperparameters (epochs, batch_size, learning_rate).
    You MUST output valid JSON ONLY, matching this schema:
    {
      "intent": "QA",
      "chunking": {"strategy": "semantic", "max_tokens": 512, "overlap": 64},
      "model": "amazon.nova-micro-v1:0",
      "hyperparameters": {"epochs": 2, "batch_size": 8, "learning_rate": 0.00005}
    }
    """
    
    try:
        response = bedrock.converse(
            modelId="amazon.nova-micro-v1:0",
            system=[{"text": system_prompt}],
            messages=[{"role": "user", "content": [{"text": f"Document Samples:\n{combined_sample}"}]}],
            inferenceConfig={"temperature": 0.1, "maxTokens": 500}
        )
        raw_response = response['output']['message']['content'][0]['text']
        raw_response = raw_response.strip().strip("`").strip("json").strip()
        strategy = json.loads(raw_response)
    except Exception as e:
        print(f"Bedrock call failed, falling back to default strategy: {e}")
        strategy = {
            "intent": "QA",
            "chunking": {"strategy": "semantic", "max_tokens": 512, "overlap": 64},
            "model": "amazon.nova-micro-v1:0",
            "hyperparameters": {"epochs": 2, "batch_size": 8, "learning_rate": 0.00005}
        }
    
    return {
        "statusCode": 200,
        "strategy": strategy,
        "document_s3_uris": document_uris # Pass down the list of URIs
    }
