"""
finetuning.py — Virtual Fine-Tuning.

Generates a specialized system prompt from a dataset instead of updating model weights,
allowing instant "deployment" as a custom agent on OpenRouter.

Usage:
    from modifai.core.finetuning import start_virtual_finetuning

    model_name, sys_prompt = start_virtual_finetuning(
        dataset_path="uploads/job123.jsonl",
        custom_model_name="my-hr-bot",
        base_model_id="openai/gpt-4o-mini",
    )
"""
from __future__ import annotations

import json
import logging
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

def list_supported_models(region: Optional[str] = None) -> dict[str, str]:
    """
    Return a mapping of short names to OpenRouter Model IDs that we will use
    for the virtual fine-tuned agent.
    """
    return {
        "gpt-4o-mini": "openai/gpt-4o-mini",
        "llama-3-8b": "meta-llama/llama-3-8b-instruct",
        "llama-3-70b": "meta-llama/llama-3-70b-instruct",
        "gemini-flash": "google/gemini-flash-1.5",
        "claude-3-haiku": "anthropic/claude-3-haiku",
    }

def start_virtual_finetuning(
    dataset_path: str,
    custom_model_name: str,
    base_model_id: str = "openrouter/free",
) -> Tuple[str, str]:
    """
    Simulates fine-tuning by compiling the dataset into a comprehensive System Prompt.

    Args:
        dataset_path: Path to the local JSONL training file.
        custom_model_name: The name for the resulting custom model.
        base_model_id: Full OpenRouter model ID to use as the base engine.

    Returns:
        job_name (str): The unique job name / model ID.
        system_prompt (str): The compiled knowledge prompt.
    """
    logger.info("Starting virtual fine-tuning for model: %s", custom_model_name)
    logger.info("  Base model: %s", base_model_id)
    
    # Read the dataset and extract knowledge
    knowledge_facts = []
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                prompt = data.get("prompt", "")
                completion = data.get("completion", "")
                if prompt and completion:
                    knowledge_facts.append(f"Q: {prompt}\nA: {completion}")
    except Exception as e:
        logger.error("Failed to read dataset: %s", e)
        knowledge_facts = ["Error reading dataset."]
    
    # Compress knowledge if it's too huge (simplified for demo)
    max_facts = 100
    if len(knowledge_facts) > max_facts:
        logger.warning("Dataset too large for prompt stuffing, truncating to %d facts", max_facts)
        knowledge_facts = knowledge_facts[:max_facts]
    
    compiled_knowledge = "\n\n".join(knowledge_facts)
    
    system_prompt = f"""You are '{custom_model_name}', a specialized AI model.
You have been fine-tuned on the following specific domain knowledge.
Always base your answers exclusively on this knowledge. If the answer is not contained here, state that you do not know.

--- DOMAIN KNOWLEDGE ---
{compiled_knowledge}
------------------------
"""
    
    job_name = f"{custom_model_name}-{int(time.time())}"
    logger.info("Virtual fine-tuning complete! System prompt length: %d chars.", len(system_prompt))
    
    return job_name, system_prompt
