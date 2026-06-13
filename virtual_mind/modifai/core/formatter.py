"""
formatter.py — Converts pipeline output samples into training format
and saves the training file locally.

Usage:
    from modifai.core.formatter import format_and_save_locally

    local_path = format_and_save_locally(
        samples=state["final_samples"],
        output_dir="uploads",
        job_id="job_abc123",
    )
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from typing import List, Optional

logger = logging.getLogger(__name__)


def format_and_save_locally(
    samples: List[dict],
    output_dir: str = "uploads",
    job_id: Optional[str] = None,
) -> str:
    """
    Format samples into fine-tuning JSONL and save locally.

    Args:
        samples:    List of sample dicts from run_agentic_loop() final_samples.
                    Each must have: instruction (str), input (str), output (str).
        output_dir: Local directory to save the JSONL file.
        job_id:     Unique job identifier. Auto-generated if not provided.

    Returns:
        Local path of the saved training file.

    Raises:
        ValueError: If samples list is empty.
    """
    if not samples:
        raise ValueError("Cannot format empty samples list — nothing to save.")

    job_id = job_id or str(uuid.uuid4())[:8]

    # Create output dir if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert samples to JSONL format
    jsonl_content = _samples_to_bedrock_jsonl(samples)
    line_count = jsonl_content.count("\n") + 1

    # Save locally
    file_path = os.path.join(output_dir, f"{job_id}_training_data.jsonl")

    logger.info("Saving %d training samples to %s", line_count, file_path)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(jsonl_content)
        
    logger.info("Save complete: %s", file_path)
    return file_path


def format_to_jsonl_string(samples: List[dict]) -> str:
    """
    Convert samples to fine-tuning JSONL string without uploading.
    Useful for local inspection or saving to disk.

    Args:
        samples: List of sample dicts with instruction, input, output fields.

    Returns:
        JSONL string (one JSON object per line).
    """
    return _samples_to_bedrock_jsonl(samples)


def save_to_local_jsonl(samples: List[dict], output_path: str) -> str:
    """
    Save formatted samples to a local JSONL file.
    Useful for inspecting dataset before uploading.

    Args:
        samples:     List of sample dicts.
        output_path: Local file path to write to.

    Returns:
        output_path (echoed for convenience).
    """
    jsonl_content = _samples_to_bedrock_jsonl(samples)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(jsonl_content)
    logger.info("Saved %d samples to %s", len(samples), output_path)
    return output_path


# ── Private helpers ────────────────────────────────────────────────────────────

def _samples_to_bedrock_jsonl(samples: List[dict]) -> str:
    """
    Convert samples to custom model fine-tuning JSONL format.

    Each line: {"prompt": "...", "completion": "..."}

    The prompt combines instruction + input (if present).
    The completion is the expected output.
    """
    lines = []
    skipped = 0

    for i, sample in enumerate(samples):
        instruction = str(sample.get("instruction", "")).strip()
        input_text = str(sample.get("input", "")).strip()
        output_text = str(sample.get("output", "")).strip()

        if not instruction or not output_text:
            logger.warning(
                "Sample %d missing instruction or output — skipping.", i
            )
            skipped += 1
            continue

        # Build prompt
        if input_text:
            prompt = f"{instruction}\n\n{input_text}"
        else:
            prompt = instruction

        record = {
            "prompt": prompt,
            "completion": output_text,
        }
        lines.append(json.dumps(record, ensure_ascii=False))

    if skipped > 0:
        logger.warning("Skipped %d malformed samples during formatting.", skipped)

    return "\n".join(lines)
