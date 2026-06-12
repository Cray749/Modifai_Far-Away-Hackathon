import json
import boto3
import os

from gemini_helper import call_gemini


def lambda_handler(event, context):
    previous_decision = event.get('agent_decision', {})
    attempt_count     = previous_decision.get('tuning_attempt', event.get('tuning_attempt', 0))
    evaluation        = event.get('model_evaluation', {})

    metric_score = evaluation.get('training_metrics_score', 0.8)
    test_score   = evaluation.get('test_prompts_score', 0.7)

    # Weights defined by user requirement (40% training metrics, 60% test performance)
    weighted_score = (metric_score * 0.4) + (test_score * 0.6)

    THRESHOLD    = float(os.environ.get("QUALITY_THRESHOLD", "0.85"))
    MAX_ATTEMPTS = int(os.environ.get("MAX_TUNING_ATTEMPTS", "3"))

    if weighted_score >= THRESHOLD:
        return {
            "action": "deploy",
            "reason": f"Model achieved weighted score {weighted_score:.3f} >= {THRESHOLD}"
        }

    if attempt_count >= MAX_ATTEMPTS:
        return {
            "action": "max_attempts_reached",
            "reason": (
                f"Failed to reach threshold {THRESHOLD} after {MAX_ATTEMPTS} attempts. "
                f"Best weighted score: {weighted_score:.3f}"
            )
        }

    current_hp = event.get('job_info', {}).get('hyperparameters', {})

    system_prompt = (
        "You are an AI Hyperparameter Tuning Agent for LLM fine-tuning. "
        "Given the current hyperparameters and the model's weighted quality score, "
        "suggest improved hyperparameters. "
        "Output ONLY valid JSON: {\"epochs\": <int>, \"batch_size\": <int>, \"learning_rate\": <float>}"
    )
    prompt = (
        f"Current hyperparameters: {json.dumps(current_hp)}\n"
        f"Current weighted score: {weighted_score:.3f} (target: {THRESHOLD})\n"
        "Suggest better hyperparameters to improve the score."
    )

    try:
        raw = call_gemini(prompt=prompt, system=system_prompt, model="gemini-2.0-flash")
        raw = raw.strip().strip("`").replace("json\n", "").strip()
        new_hyperparameters = json.loads(raw)
        print(f"Gemini suggested hyperparameters: {new_hyperparameters}")
    except Exception as e:
        print(f"Gemini hyperparameter suggestion failed, incrementing epochs: {e}")
        new_hyperparameters = {
            "epochs":        current_hp.get("epochs", 2) + 1,
            "batch_size":    current_hp.get("batch_size", 8),
            "learning_rate": current_hp.get("learning_rate", 0.00005)
        }

    return {
        "action":             "tune",
        "new_hyperparameters": new_hyperparameters,
        "tuning_attempt":     attempt_count + 1,
        "reason":             f"Score {weighted_score:.3f} < {THRESHOLD}. Tuning hyperparameters."
    }
