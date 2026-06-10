"""
finetuning.py — Starts and monitors an AWS Bedrock model customization (fine-tuning) job.

Uses the Bedrock fine-tuning API to fine-tune amazon.titan-text-express-v1
on the generated training dataset.

Prerequisites:
  - S3 bucket in the same region as Bedrock (us-east-1)
  - IAM role with permissions: AmazonBedrockFullAccess + S3 read access
  - Minimum 50 training samples (Bedrock requirement for fine-tuning)

Usage:
    from modifai.core.finetuning import start_finetuning_job, wait_for_job

    job_name = start_finetuning_job(
        training_data_s3_uri="s3://my-bucket/modifai-jobs/job123/training_data.jsonl",
        output_s3_uri="s3://my-bucket/modifai-jobs/job123/output/",
        custom_model_name="modifai-hr-policy-v1",
        role_arn="arn:aws:iam::123456789012:role/ModifaiBedrockRole",
    )
    model_arn = wait_for_job(job_name)
    print("Fine-tuned model ARN:", model_arn)
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

import boto3

logger = logging.getLogger(__name__)

_DEFAULT_REGION = os.environ.get("AWS_REGION", "us-east-1")
_BASE_MODEL_ID = "amazon.titan-text-express-v1"

# Bedrock fine-tuning job statuses
_TERMINAL_STATUSES = {"Completed", "Failed", "Stopped"}


def start_finetuning_job(
    training_data_s3_uri: str,
    output_s3_uri: str,
    custom_model_name: str,
    role_arn: str,
    job_name: Optional[str] = None,
    base_model_id: str = _BASE_MODEL_ID,
    region: Optional[str] = None,
    hyperparameters: Optional[dict] = None,
) -> str:
    """
    Start a Bedrock model customization (fine-tuning) job.

    Args:
        training_data_s3_uri: S3 URI of the training JSONL file.
                              e.g. "s3://my-bucket/modifai-jobs/job123/training_data.jsonl"
        output_s3_uri:        S3 URI prefix for the fine-tuned model output artifacts.
                              e.g. "s3://my-bucket/modifai-jobs/job123/output/"
        custom_model_name:    Name for the fine-tuned model (alphanumeric + hyphens, max 63 chars).
                              e.g. "modifai-hr-policy-v1"
        role_arn:             IAM role ARN with AmazonBedrockFullAccess + S3 read/write.
                              e.g. "arn:aws:iam::123456789012:role/ModifaiBedrockRole"
        job_name:             Optional unique job name. Auto-generated if not provided.
        base_model_id:        Bedrock base model to fine-tune (default: titan-text-express-v1).
        region:               AWS region override (default: us-east-1).
        hyperparameters:      Override default training hyperparameters. Defaults:
                              {"epochCount": "2", "batchSize": "8", "learningRate": "0.00005"}

    Returns:
        job_name (str) — use this to poll status with wait_for_job() or get_job_status().

    Raises:
        RuntimeError: If job creation fails.
    """
    region = region or _DEFAULT_REGION
    client = boto3.client("bedrock", region_name=region)

    import uuid
    job_name = job_name or f"modifai-ft-{str(uuid.uuid4())[:8]}"

    default_hyperparams = {
        "epochCount": "2",
        "batchSize": "8",
        "learningRate": "0.00005",
        "warmupSteps": "50",
    }
    if hyperparameters:
        default_hyperparams.update(hyperparameters)

    logger.info(
        "Starting Bedrock fine-tuning job: %s (base=%s)", job_name, base_model_id
    )
    logger.info("  Training data: %s", training_data_s3_uri)
    logger.info("  Output:        %s", output_s3_uri)
    logger.info("  Hyperparams:   %s", default_hyperparams)

    try:
        client.create_model_customization_job(
            jobName=job_name,
            customModelName=custom_model_name,
            roleArn=role_arn,
            baseModelIdentifier=base_model_id,
            trainingDataConfig={"s3Uri": training_data_s3_uri},
            outputDataConfig={"s3Uri": output_s3_uri},
            hyperParameters=default_hyperparams,
        )
        logger.info("Fine-tuning job created: %s", job_name)
        return job_name

    except Exception as exc:
        raise RuntimeError(f"Failed to start fine-tuning job '{job_name}': {exc}") from exc


def wait_for_job(
    job_name: str,
    region: Optional[str] = None,
    poll_interval_seconds: int = 60,
    max_wait_seconds: int = 7200,  # 2 hours max
) -> str:
    """
    Block until a Bedrock fine-tuning job completes and return the custom model ARN.

    Args:
        job_name:              Job name returned by start_finetuning_job().
        region:                AWS region override.
        poll_interval_seconds: Seconds between status checks (default 60).
        max_wait_seconds:      Max wait time (default 7200s = 2 hours).

    Returns:
        custom_model_arn (str) — ARN of the successfully fine-tuned model.

    Raises:
        RuntimeError: If the job fails, is stopped, or times out.
    """
    region = region or _DEFAULT_REGION
    client = boto3.client("bedrock", region_name=region)

    logger.info(
        "Waiting for fine-tuning job '%s' to complete (polling every %ds, max %ds)...",
        job_name, poll_interval_seconds, max_wait_seconds,
    )

    elapsed = 0
    while elapsed < max_wait_seconds:
        time.sleep(poll_interval_seconds)
        elapsed += poll_interval_seconds

        status_info = get_job_status(job_name, region=region)
        status = status_info["status"]
        logger.info(
            "Job '%s' status: %s (elapsed %ds / %ds)",
            job_name, status, elapsed, max_wait_seconds,
        )

        if status == "Completed":
            model_arn = status_info.get("custom_model_arn")
            if not model_arn:
                raise RuntimeError(
                    f"Job '{job_name}' completed but no custom_model_arn in response."
                )
            logger.info("Fine-tuning complete! Model ARN: %s", model_arn)
            return model_arn

        elif status in ("Failed", "Stopped"):
            failure_message = status_info.get("failure_message", "unknown")
            raise RuntimeError(
                f"Fine-tuning job '{job_name}' ended with status '{status}'. "
                f"Reason: {failure_message}"
            )
        # else: InProgress — keep polling

    raise RuntimeError(
        f"Fine-tuning job '{job_name}' did not complete within {max_wait_seconds}s."
    )


def get_job_status(job_name: str, region: Optional[str] = None) -> dict:
    """
    Get the current status of a fine-tuning job.

    Args:
        job_name: Job name from start_finetuning_job().
        region:   AWS region override.

    Returns:
        Dict with:
          - status (str): "InProgress" | "Completed" | "Failed" | "Stopped"
          - custom_model_arn (str | None): set when status == "Completed"
          - failure_message (str | None): set when status == "Failed"
    """
    region = region or _DEFAULT_REGION
    client = boto3.client("bedrock", region_name=region)

    response = client.get_model_customization_job(jobIdentifier=job_name)
    return {
        "status": response.get("status", "Unknown"),
        "custom_model_arn": response.get("outputModelArn"),
        "failure_message": response.get("failureMessage"),
        "job_name": response.get("jobName", job_name),
        "base_model_id": response.get("baseModelId"),
        "creation_time": str(response.get("creationTime", "")),
        "end_time": str(response.get("endTime", "")),
    }
