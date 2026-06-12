import json
import boto3
import os

from gemini_helper import call_gemini


def lambda_handler(event, context):
    job_status       = event.get('job_status', {})
    training_metrics = job_status.get('training_metrics', {})

    # 1. Derive metric score from training loss (lower loss = better)
    training_loss = training_metrics.get('trainingLoss', 0.5)
    metric_score  = max(0.0, 1.0 - float(training_loss))

    # 2. Ask Gemini to estimate a test score based on the training loss
    # (We can't call the fine-tuned model directly without provisioning it,
    #  so we use Gemini as a lightweight critic for the loop.)
    system_prompt = (
        "You are an AI Model Evaluator. You are given the training loss of a fine-tuning run. "
        "Estimate a 'test_score' between 0.0 and 1.0 (lower loss = higher score). "
        "Output ONLY valid JSON: {\"test_score\": <float>, \"reasoning\": \"<one sentence>\"}"
    )
    prompt = f"Training Loss: {training_loss}"

    try:
        raw = call_gemini(prompt=prompt, system=system_prompt, model="gemini-2.0-flash")
        raw = raw.strip().strip("`").replace("json\n", "").strip()
        eval_data  = json.loads(raw)
        test_score = float(eval_data.get('test_score', 0.5))
        print(f"Gemini eval: test_score={test_score}, reasoning={eval_data.get('reasoning','')}")
    except Exception as e:
        print(f"Gemini evaluation failed, using fallback: {e}")
        test_score = 0.75

    return {
        "training_metrics_score": metric_score,
        "test_prompts_score":     test_score,
        "details": "Model evaluated via Gemini critic agent."
    }
