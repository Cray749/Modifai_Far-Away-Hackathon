import os
import json
import boto3
from google import genai


def get_gemini_client():
    """
    Returns an authenticated google.genai.Client.

    API key resolution order:
    1. GEMINI_API_KEY env var (fast path for local / CI)
    2. AWS Secrets Manager secret  "modifai/gemini"  → {"api_key": "..."}
    """
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        secret_name = "modifai/gemini"
        region_name = os.environ.get("AWS_REGION", "ap-south-1")
        session = boto3.session.Session()
        sm = session.client(service_name="secretsmanager", region_name=region_name)
        try:
            response = sm.get_secret_value(SecretId=secret_name)
            secret = json.loads(response["SecretString"])
            api_key = secret.get("api_key") or secret.get("GEMINI_API_KEY")
        except Exception as e:
            raise RuntimeError(
                f"Could not retrieve Gemini API key from Secrets Manager "
                f"(secret '{secret_name}'): {e}"
            )

    if not api_key:
        raise RuntimeError(
            "Gemini API key not found. Set GEMINI_API_KEY env var or store it "
            "in AWS Secrets Manager under 'modifai/gemini' as {\"api_key\": \"...\"}"
        )

    return genai.Client(api_key=api_key)


def call_gemini(prompt: str, system: str = "", model: str = "gemini-2.0-flash") -> str:
    """
    Convenience wrapper: sends a prompt to Gemini and returns the text response.

    Args:
        prompt:  The user message / task description.
        system:  Optional system instruction.
        model:   Gemini model ID (default: gemini-2.0-flash).

    Returns:
        The model's text response as a plain string.
    """
    from google.genai import types

    client = get_gemini_client()

    config = types.GenerateContentConfig(
        temperature=0.7,
        max_output_tokens=2048,
        system_instruction=system if system else None,
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )

    return response.text
