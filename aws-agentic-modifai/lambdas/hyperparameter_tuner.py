import json
import boto3
import os

bedrock = boto3.client('bedrock-runtime', region_name=os.environ.get("AWS_REGION", "ap-south-1"))

def lambda_handler(event, context):
    previous_decision = event.get('agent_decision', {})
    attempt_count = previous_decision.get('tuning_attempt', event.get('tuning_attempt', 0))
    evaluation = event.get('model_evaluation', {})
    
    metric_score = evaluation.get('training_metrics_score', 0.8)
    test_score = evaluation.get('test_prompts_score', 0.7)
    
    # Weights defined by user requirement
    weighted_score = (metric_score * 0.4) + (test_score * 0.6)
    
    THRESHOLD = 0.85
    MAX_ATTEMPTS = 3
    
    if weighted_score >= THRESHOLD:
        return {
            "action": "deploy",
            "reason": f"Model achieved weighted score {weighted_score} >= {THRESHOLD}"
        }
        
    if attempt_count >= MAX_ATTEMPTS:
        return {
            "action": "max_attempts_reached",
            "reason": f"Failed to reach {THRESHOLD} after {MAX_ATTEMPTS} attempts. Best score: {weighted_score}"
        }
        
    current_hp = event.get('job_info', {}).get('hyperparameters', {})
    
    system_prompt = """
    You are an AI Hyperparameter Tuning Agent.
    The current fine-tuning job failed to reach the required quality threshold.
    Current Hyperparameters: {current_hp}
    Current Score: {score}
    
    Generate new hyperparameters to try and improve the model.
    Output JSON ONLY:
    {{
      "epochs": 3,
      "batch_size": 8,
      "learning_rate": 0.00005
    }}
    """
    
    try:
        response = bedrock.converse(
            modelId="amazon.nova-micro-v1:0",
            system=[{"text": system_prompt.format(current_hp=current_hp, score=weighted_score)}],
            messages=[{"role": "user", "content": [{"text": "Generate better hyperparameters."}]}],
            inferenceConfig={"temperature": 0.2, "maxTokens": 300}
        )
        raw_response = response['output']['message']['content'][0]['text']
        raw_response = raw_response.strip().strip("`").strip("json").strip()
        new_hyperparameters = json.loads(raw_response)
    except Exception as e:
        print(f"Failed to generate new HP with Bedrock: {e}")
        new_hyperparameters = {
            "epochs": current_hp.get("epochs", 2) + 1,
            "batch_size": current_hp.get("batch_size", 8),
            "learning_rate": current_hp.get("learning_rate", 0.00005)
        }
    
    return {
        "action": "tune",
        "new_hyperparameters": new_hyperparameters,
        "tuning_attempt": attempt_count + 1,
        "reason": f"Score {weighted_score} < {THRESHOLD}. Tuning hyperparameters."
    }
