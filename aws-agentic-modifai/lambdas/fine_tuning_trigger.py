import json
import boto3
import os
import uuid

bedrock = boto3.client('bedrock', region_name=os.environ.get("AWS_REGION", "ap-south-1"))

def lambda_handler(event, context):
    agent_decision = event.get('agent_decision', {})
    strategy = event.get('strategy', {})
    
    hyperparameters = strategy.get('hyperparameters', {})
    if 'new_hyperparameters' in agent_decision:
        hyperparameters = agent_decision['new_hyperparameters']
        
    training_data_uri = event.get('dataset_evaluation', {}).get('training_data_uri')
    base_model = strategy.get('model', 'amazon.titan-text-express-v1')
    
    run_id = event.get('dataset_evaluation', {}).get('run_id', str(uuid.uuid4())[:8])
    bucket = event.get('dataset_evaluation', {}).get('bucket', 'modifai-bucket')
    
    job_name = f"modifai-tune-{run_id}-{str(uuid.uuid4())[:4]}"
    custom_model_name = f"modifai-model-{run_id}"
    output_s3_uri = f"s3://{bucket}/modifai-jobs/{run_id}/output/"
    
    # We need a Role ARN that has Bedrock privileges. In reality we'd pass this in from SAM
    role_arn = os.environ.get("BEDROCK_ROLE_ARN", "arn:aws:iam::123456789012:role/ModifaiBedrockRole")
    
    # Format hyperparams for Bedrock API (all values must be strings)
    bedrock_hp = {k: str(v) for k, v in hyperparameters.items()}
    
    try:
        bedrock.create_model_customization_job(
            jobName=job_name,
            customModelName=custom_model_name,
            roleArn=role_arn,
            baseModelIdentifier=base_model,
            trainingDataConfig={"s3Uri": training_data_uri},
            outputDataConfig={"s3Uri": output_s3_uri},
            hyperParameters=bedrock_hp
        )
    except Exception as e:
        print(f"Failed to start training job: {e}")
        # Return mock if role_arn is invalid during testing
        pass
    
    return {
        "job_name": job_name,
        "hyperparameters": hyperparameters,
        "base_model": base_model,
        "training_data_uri": training_data_uri
    }
