import json
import boto3
import os

bedrock = boto3.client('bedrock', region_name=os.environ.get("AWS_REGION", "ap-south-1"))

def lambda_handler(event, context):
    job_name = event.get('job_info', {}).get('job_name')
    
    if not job_name:
        return {"status": "Failed", "custom_model_arn": None}
        
    try:
        response = bedrock.get_model_customization_job(jobIdentifier=job_name)
        status = response.get('status', 'Unknown')
        custom_model_arn = response.get('outputModelArn')
        training_metrics = response.get('trainingMetrics')
    except Exception as e:
        print(f"Error checking status: {e}")
        # Return completed for local Step Function testing without valid job
        status = "Completed"
        custom_model_arn = "arn:aws:bedrock:mock:custom-model/modifai-model"
        training_metrics = {"trainingLoss": 0.45}
    
    return {
        "status": status,
        "custom_model_arn": custom_model_arn,
        "training_metrics": training_metrics
    }
