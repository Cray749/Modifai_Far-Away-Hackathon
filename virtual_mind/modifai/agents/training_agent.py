"""
TrainingAgent — acts as the virtual fine-tuning orchestrator.

WHERE IT SITS IN THE PIPELINE:
    DatasetGeneration → Critic → [TrainingAgent] → Deployment

This version implements "Virtual Fine-Tuning" to run locally using OpenRouter.
Instead of updating model weights on SageMaker, it generates a highly specialized
System Prompt representing the fine-tuned knowledge and registers it as a custom endpoint.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional
import os

from modifai.core.finetuning import start_virtual_finetuning

logger = logging.getLogger(__name__)


class TrainingAgent:
    """
    Simulates fine-tuning by compiling the dataset into a comprehensive System Prompt.
    """

    def __init__(
        self,
        base_model_id: str = "openrouter/free",
        **kwargs
    ):
        """
        Args:
            base_model_id: Full OpenRouter model ID to use as the base engine.
        """
        self.base_model_id = base_model_id

    def run(
        self,
        samples: List[Dict[str, Any]],
        dataset_stats: Dict[str, Any],
        job_id: Optional[str] = None,
        custom_model_name: Optional[str] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes virtual fine-tuning.

        Args:
            samples: Final accepted/rewritten samples from the Critic loop.
            dataset_stats: Stats dict from CriticAgent.run_batch().
            job_id: Unique run identifier.
            custom_model_name: Human-readable name for the trained model.

        Returns:
            Dict containing the training metadata.
        """
        job_id = job_id or str(uuid.uuid4())[:8]
        custom_model_name = custom_model_name or f"modifai-model-{job_id}"

        logger.info("=" * 60)
        logger.info("TrainingAgent (Virtual) — Job: %s", job_id)
        logger.info("  Base model:   %s", self.base_model_id)
        logger.info("  Dataset size: %d samples", len(samples))
        logger.info("=" * 60)

        # In a real environment, we would first save the dataset to disk, but here we can 
        # compile it directly from the samples list!
        knowledge_facts = []
        for sample in samples:
            instruction = str(sample.get("instruction", "")).strip()
            input_text = str(sample.get("input", "")).strip()
            output_text = str(sample.get("output", "")).strip() or str(sample.get("response", "")).strip()

            if not instruction or not output_text:
                continue

            prompt = f"{instruction}\n\n{input_text}" if input_text else instruction
            knowledge_facts.append(f"Q: {prompt}\nA: {output_text}")

        # Compress knowledge
        max_facts = 100
        if len(knowledge_facts) > max_facts:
            logger.warning("Dataset too large for prompt stuffing, truncating to %d facts", max_facts)
            knowledge_facts = knowledge_facts[:max_facts]

        compiled_knowledge = "\n\n".join(knowledge_facts)

        system_prompt = f"""You are '{custom_model_name}', a highly specialized AI model.
You have been fine-tuned on the following specific domain knowledge.
Always base your answers exclusively on this knowledge. If the answer is not contained here, state that you do not know.

--- DOMAIN KNOWLEDGE ---
{compiled_knowledge}
------------------------
"""

        job_name = f"{custom_model_name}-{job_id}"
        logger.info("Virtual fine-tuning complete! System prompt length: %d chars.", len(system_prompt))
        
        # We will save this system prompt locally so the inference endpoint can read it.
        os.makedirs("uploads", exist_ok=True)
        model_prompt_path = f"uploads/{job_name}_prompt.txt"
        with open(model_prompt_path, "w", encoding="utf-8") as f:
            f.write(system_prompt)

        return {
            "job_id": job_id,
            "job_name": job_name,
            "hf_model_id": self.base_model_id,
            "instance_type": "virtual-inference",
            "hyperparameters": {"type": "in-context"},
            "hp_reasoning": "Virtual fine-tuning via system prompt generation.",
            "dataset_s3_uri": f"local://uploads/{job_id}_training_data.jsonl",
            "model_s3_uri": f"local://{model_prompt_path}",
            "status": "Completed",
        }
