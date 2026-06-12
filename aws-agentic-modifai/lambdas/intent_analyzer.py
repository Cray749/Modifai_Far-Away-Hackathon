import json
import boto3
import os
import io
import PyPDF2

from gemini_helper import call_gemini

s3 = boto3.client('s3')


# ── Text extraction helpers ───────────────────────────────────────────────────

def extract_snippet(bucket: str, key: str, ext: str) -> str:
    """Download a file and return up to 1500 chars of text for intent analysis."""
    file_bytes = s3.get_object(Bucket=bucket, Key=key)['Body'].read()
    if ext == 'pdf':
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages[:2]:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text[:1500]
    return file_bytes[:1500].decode('utf-8', errors='ignore')


def lambda_handler(event, context):
    document_uris = event.get('document_s3_uris', [])
    single_uri    = event.get('document_s3_uri')
    if single_uri and single_uri not in document_uris:
        document_uris.append(single_uri)

    # ── Sample text from up to 3 documents ───────────────────────────────────
    sample_texts = []
    for doc_uri in document_uris[:3]:
        bucket = doc_uri.split("/")[2]
        key    = "/".join(doc_uri.split("/")[3:])
        ext    = key.lower().rsplit('.', 1)[-1]
        try:
            snippet = extract_snippet(bucket, key, ext)
            sample_texts.append(f"Snippet from {doc_uri}:\n{snippet}")
        except Exception as e:
            print(f"Warning: could not fetch snippet from {doc_uri}: {e}")

    combined_sample = "\n---\n".join(sample_texts) if sample_texts else "No text could be extracted."

    # ── Gemini: analyse intent and recommend strategy ─────────────────────────
    system_prompt = (
        "You are an AI Architect Agent. Analyse the provided document samples. "
        "Identify the intent (QA, summarization, or instruction). "
        "Recommend a chunking strategy and initial fine-tuning hyperparameters. "
        "Use 'meta.llama3-8b-instruct-v1:0' as the base model. "
        "You MUST output valid JSON ONLY matching this exact schema — no markdown, no explanation:\n"
        "{\n"
        "  \"intent\": \"QA\",\n"
        "  \"chunking\": {\"strategy\": \"semantic\", \"max_tokens\": 512, \"overlap\": 64},\n"
        "  \"model\": \"meta.llama3-8b-instruct-v1:0\",\n"
        "  \"hyperparameters\": {\"epochs\": 2, \"batch_size\": 8, \"learning_rate\": 0.00005}\n"
        "}"
    )
    prompt = f"Document Samples:\n{combined_sample}"

    try:
        raw      = call_gemini(prompt=prompt, system=system_prompt, model="gemini-2.0-flash")
        raw      = raw.strip().strip("`").replace("json\n", "").strip()
        strategy = json.loads(raw)
        print(f"Intent analysis complete: intent={strategy.get('intent')}")
    except Exception as e:
        print(f"Gemini call failed, falling back to default strategy: {e}")
        strategy = {
            "intent":   "QA",
            "chunking": {"strategy": "semantic", "max_tokens": 512, "overlap": 64},
            "model":    "meta.llama3-8b-instruct-v1:0",
            "hyperparameters": {"epochs": 2, "batch_size": 8, "learning_rate": 0.00005}
        }

    return {
        "statusCode":       200,
        "strategy":         strategy,
        "document_s3_uris": document_uris
    }
