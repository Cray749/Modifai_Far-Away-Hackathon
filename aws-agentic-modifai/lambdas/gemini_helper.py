"""
gemini_helper.py — Shared Gemini client for all pipeline Lambdas.

API-key resolution order
------------------------
1. GEMINI_API_KEY  environment variable  (local / CI fast-path)
2. AWS Secrets Manager  →  secret name from GEMINI_SECRET_NAME env var
   (default: "modifai/gemini")  →  JSON payload: {"api_key": "..."}

Public surface
--------------
call_gemini(prompt, system, model, temperature, max_output_tokens)
    → str

call_gemini_json(prompt, system, model, temperature, max_output_tokens)
    → dict   (parses the response as JSON; raises ValueError on failure)
"""

import json
import logging
import os
import re
import time

import boto3
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ── tuneable defaults (all overridable via env vars) ─────────────────────────
DEFAULT_MODEL             = os.environ.get("GEMINI_MODEL",             "gemini-2.0-flash")
DEFAULT_TEMPERATURE       = float(os.environ.get("GEMINI_TEMPERATURE", "0.3"))
DEFAULT_MAX_OUTPUT_TOKENS = int(os.environ.get("GEMINI_MAX_TOKENS",    "2048"))
MAX_RETRIES               = int(os.environ.get("GEMINI_MAX_RETRIES",   "3"))
RETRY_BASE_DELAY_S        = float(os.environ.get("GEMINI_RETRY_DELAY", "1.5"))


# ── key resolution ────────────────────────────────────────────────────────────

def _resolve_api_key() -> str:
    """Return the Gemini API key, or raise RuntimeError if unavailable."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if api_key:
        return api_key

    secret_name = os.environ.get("GEMINI_SECRET_NAME", "modifai/gemini")
    region      = os.environ.get("AWS_REGION", "ap-south-1")
    session     = boto3.session.Session()
    sm          = session.client("secretsmanager", region_name=region)
    try:
        payload = sm.get_secret_value(SecretId=secret_name)
        secret  = json.loads(payload["SecretString"])
        api_key = (secret.get("api_key") or secret.get("GEMINI_API_KEY", "")).strip()
    except Exception as exc:
        raise RuntimeError(
            f"Cannot retrieve Gemini API key from Secrets Manager "
            f"(secret='{secret_name}', region='{region}'): {exc}"
        ) from exc

    if not api_key:
        raise RuntimeError(
            "Gemini API key resolved to an empty string.  "
            "Set GEMINI_API_KEY env var or store it in Secrets Manager "
            f"under '{secret_name}' as {{\"api_key\": \"<key>\"}}."
        )
    return api_key


def get_gemini_client() -> genai.Client:
    """Return an authenticated google.genai.Client."""
    return genai.Client(api_key=_resolve_api_key())


# ── core call with retry ──────────────────────────────────────────────────────

def call_gemini(
    prompt:           str,
    system:           str  = "",
    model:            str  = DEFAULT_MODEL,
    temperature:      float = DEFAULT_TEMPERATURE,
    max_output_tokens: int  = DEFAULT_MAX_OUTPUT_TOKENS,
) -> str:
    """
    Send *prompt* to Gemini and return the response text.

    Retries up to MAX_RETRIES times on transient errors with exponential
    back-off.  Raises the last exception if all attempts fail.
    """
    client = get_gemini_client()
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        system_instruction=system or None,
    )

    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            text = response.text
            logger.info("Gemini call succeeded (attempt %d/%d)", attempt, MAX_RETRIES)
            return text
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            delay = RETRY_BASE_DELAY_S * (2 ** (attempt - 1))
            logger.warning(
                "Gemini call failed (attempt %d/%d): %s — retrying in %.1fs",
                attempt, MAX_RETRIES, exc, delay,
            )
            if attempt < MAX_RETRIES:
                time.sleep(delay)

    raise RuntimeError(
        f"Gemini call failed after {MAX_RETRIES} attempts: {last_exc}"
    ) from last_exc


def call_gemini_json(
    prompt:            str,
    system:            str   = "",
    model:             str   = DEFAULT_MODEL,
    temperature:       float = DEFAULT_TEMPERATURE,
    max_output_tokens: int   = DEFAULT_MAX_OUTPUT_TOKENS,
) -> dict:
    """
    Like call_gemini(), but parses the response as JSON and returns a dict.

    Strips leading/trailing whitespace, backtick fences, and the 'json'
    language tag before parsing so that models that ignore "output ONLY JSON"
    instructions still work.

    Raises ValueError if the response cannot be parsed as JSON.
    """
    raw = call_gemini(
        prompt=prompt,
        system=system,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    cleaned = _strip_json_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Gemini returned non-JSON content.\n"
            f"Raw response (first 500 chars): {raw[:500]!r}\n"
            f"Parse error: {exc}"
        ) from exc


# ── helpers ───────────────────────────────────────────────────────────────────

def _strip_json_fences(text: str) -> str:
    """Remove markdown code fences and leading/trailing whitespace."""
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$",          "", text)
    return text.strip()
