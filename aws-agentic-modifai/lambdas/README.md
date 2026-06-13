# ModifAI Pipeline — Gemini-Powered, Bedrock-Free

A Step Functions pipeline of five Lambda functions that drives an
LLM fine-tuning loop.  All AI inference routes through **Gemini** via
a single shared helper (`gemini_helper.py`).  Amazon Bedrock has been
removed entirely.

---

## Architecture

```
document_s3_uris
      │
      ▼
┌─────────────────┐
│ intent_analyzer │  ← Gemini analyses docs, picks chunking & HPs
└────────┬────────┘
         │ strategy
         ▼
┌──────────────────────┐
│ fine_tuning_trigger  │  ← Gemini validates config; writes manifest to S3
└──────────┬───────────┘
           │ job_info
           ▼
┌────────────────┐
│ status_checker │  ← Reads S3 manifest (or Gemini simulation in demo mode)
└───────┬────────┘
        │ job_status
        ▼
┌─────────────────┐
│ model_evaluator │  ← Gemini critic estimates test_score from training_loss
└────────┬────────┘
         │ model_evaluation
         ▼
┌──────────────────────┐
│ hyperparameter_tuner │  ← Gemini suggests new HPs; decides deploy vs tune
└──────────────────────┘
         │
    ┌────┴────┐
  deploy    tune (loop back to fine_tuning_trigger)
```

---

## Shared helper: `gemini_helper.py`

| Function | Description |
|---|---|
| `call_gemini(prompt, system, ...)` | Returns raw text response |
| `call_gemini_json(prompt, system, ...)` | Parses response as JSON dict |
| `get_gemini_client()` | Returns authenticated `genai.Client` |

**API-key resolution** (in order):
1. `GEMINI_API_KEY` environment variable
2. AWS Secrets Manager → secret named by `GEMINI_SECRET_NAME` (default: `modifai/gemini`)
   — JSON payload: `{"api_key": "..."}`

All functions retry up to `GEMINI_MAX_RETRIES` times (default 3) with
exponential back-off.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | — | Gemini API key (fast-path) |
| `GEMINI_SECRET_NAME` | `modifai/gemini` | Secrets Manager secret name |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model ID |
| `GEMINI_TEMPERATURE` | `0.3` | Sampling temperature |
| `GEMINI_MAX_TOKENS` | `2048` | Max output tokens |
| `GEMINI_MAX_RETRIES` | `3` | Retry count on transient errors |
| `GEMINI_RETRY_DELAY` | `1.5` | Base retry delay (seconds) |
| `AWS_REGION` | `ap-south-1` | AWS region |
| `S3_BUCKET` | `modifai-bucket` | Bucket for manifests & data |
| `JOB_MANIFEST_PREFIX` | `modifai-jobs` | S3 key prefix for job manifests |
| `DEMO_MODE` | `false` | Skip real back-end; use Gemini simulation |
| `QUALITY_THRESHOLD` | `0.85` | Minimum weighted score to deploy |
| `MAX_TUNING_ATTEMPTS` | `3` | Maximum tune iterations |
| `METRIC_WEIGHT` | `0.4` | Weight for training-metrics score |
| `TEST_WEIGHT` | `0.6` | Weight for test-prompts score |
| `BASE_MODEL` | `meta.llama3-8b-instruct-v1:0` | Base model forwarded to trigger |

---

## Deployment

### Prerequisites
- AWS SAM CLI ≥ 1.100
- Docker (for SAM build)
- Python 3.12
- Gemini API key stored in Secrets Manager:

```bash
aws secretsmanager create-secret \
  --name modifai/gemini \
  --secret-string '{"api_key": "YOUR_GEMINI_KEY"}' \
  --region ap-south-1
```

### Build & deploy

```bash
# 1. Install deps into the layer directory
mkdir -p layer/python
pip install -r requirements.txt -t layer/python --upgrade

# 2. Copy Lambda source into src/
mkdir -p src
cp gemini_helper.py intent_analyzer.py fine_tuning_trigger.py \
   status_checker.py model_evaluator.py hyperparameter_tuner.py src/

# 3. SAM build + deploy
sam build
sam deploy \
  --guided \
  --parameter-overrides Environment=dev \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM
```

### Local testing (demo mode)

```bash
export GEMINI_API_KEY=your_key_here
export DEMO_MODE=true
export S3_BUCKET=my-local-test-bucket

# Test intent analyzer
python - <<'EOF'
import json, intent_analyzer
event = {"document_s3_uris": ["s3://my-bucket/sample.pdf"]}
print(json.dumps(intent_analyzer.lambda_handler(event, None), indent=2))
EOF
```

---

## Plugging in a real training back-end

Both `fine_tuning_trigger.py` and `status_checker.py` contain clearly
marked stub functions:

- `fine_tuning_trigger._submit_training_job(job_manifest)` — replace with
  your SDK call (SageMaker, Vertex AI, Modal, RunPod, etc.)
- `status_checker._fetch_job_status(job_manifest)` — replace with the
  corresponding status-polling call.

Everything else — Gemini validation, S3 manifest management, the tuning
loop — works unchanged regardless of which back-end you wire in.

---

## Removing Bedrock: what changed

| File | Change |
|---|---|
| `gemini_helper.py` | Added retry logic, `call_gemini_json()`, env-var config |
| `intent_analyzer.py` | Removed `boto3.client('bedrock')` entirely; kept S3 extraction |
| `fine_tuning_trigger.py` | Replaced `create_model_customization_job` with S3 manifest + Gemini validation |
| `status_checker.py` | Replaced `get_model_customization_job` with S3 manifest read + Gemini simulation |
| `model_evaluator.py` | Was already Gemini-only; cleaned up + structured JSON helper |
| `hyperparameter_tuner.py` | Was already Gemini-only; added `METRIC_WEIGHT`/`TEST_WEIGHT` env vars |
