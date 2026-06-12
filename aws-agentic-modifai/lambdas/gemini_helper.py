import os
import boto3
import json
from google import genai

def get_gemini_client():
    secret_name = "modifai/gemini"
    region_name = os.environ.get("AWS_REGION", "ap-south-1")
    
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )
    
    try:
        response = client.get_secret_value(SecretId=secret_name)
        api_key = response['SecretString']
        
        try:
            secret_dict = json.loads(api_key)
            if "api_key" in secret_dict:
                api_key = secret_dict["api_key"]
            elif "GEMINI_API_KEY" in secret_dict:
                api_key = secret_dict["GEMINI_API_KEY"]
        except json.JSONDecodeError:
            pass
            
        return genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Failed to retrieve Gemini API key from Secrets Manager: {e}")
        raise
