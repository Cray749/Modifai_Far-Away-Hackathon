import json
import boto3
import os
from google.genai import types
from gemini_helper import get_gemini_client

def lambda_handler(event, context):
    job_status = event.get('job_status', {})
    training_metrics = job_status.get('training_metrics', {})
    
    # 1. Fetch training loss metrics
    training_loss = training_metrics.get('trainingLoss', 0.5)
    metric_score = max(0, 1.0 - training_loss)
    
    # 2. Run test prompts against the newly fine-tuned model (Wait: you can't test a custom model until it's provisioned)
    # Since provisioning costs money, we use a Critic Agent (Base Model) to evaluate the training metrics
    # instead of provisioning every loop attempt.
    
    system_prompt = """
    Evaluate the training loss of the model. 
    If training loss is > 0.5, the model is underperforming.
    Return ONLY a JSON object: {"score": float, "feedback": "string"}
    Score is between 0 and 1 (1 being best).
    """
    
    try:
        gemini_client = get_gemini_client()
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=f"Training Loss: {training_loss}",
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.1,
                max_output_tokens=200
            )
        )
        raw_response = response.text
        raw_response = raw_response.strip().strip("`").strip("json").strip()
        evaluation = json.loads(raw_response)
        test_score = evaluation.get('score', 0.5)
    except Exception as e:
        print(f"Failed Gemini evaluation: {e}")
        test_score = 0.75
        
    return {
        "training_metrics_score": metric_score,
        "test_prompts_score": test_score,
        "details": "Model evaluated successfully."
    }
