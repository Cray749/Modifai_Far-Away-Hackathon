<p align="center">
  <img src="https://img.shields.io/badge/AWS-Step_Functions-FF9900?style=for-the-badge&logo=amazon-aws" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black" />
  <img src="https://img.shields.io/badge/SageMaker-FF9900?style=for-the-badge&logo=amazon-aws" />
  <img src="https://img.shields.io/badge/Bedrock-232F3E?style=for-the-badge&logo=amazon-aws&logoColor=white" />
</p>

# Modifai

**Transform your raw documents into fine-tuned AI models — all from one pipeline.**

Modifai is an end-to-end AI model factory that takes users from raw documents (PDFs, text files) through OCR, synthetic dataset generation, quality control, fine-tuning, and deployment — all orchestrated by AWS Step Functions with a premium React frontend.

---

## 🚀 What It Does

```
Documents → OCR → Chunking → Dataset Generation → Quality Control → Fine-Tuning → Deployment
```

1. **Upload** raw documents (PDF, text, images)
2. **Evaluate** data quality with AI before processing
3. **Generate** synthetic training datasets using LLMs (Bedrock Nova)
4. **Fine-tune** foundation models on SageMaker
5. **Deploy** as a real-time SageMaker endpoint
6. **Compare** base vs fine-tuned model responses side-by-side

---

## 🎯 Four Pipeline Modes

| Mode | What It Does |
|------|-------------|
| **Dataset Only** | OCR → Chunk → Generate → QC → Export JSONL |
| **Fine-Tune Only** | Upload a JSONL dataset → Fine-tune on SageMaker |
| **Dataset + Fine-Tune** | Full data pipeline + fine-tuning |
| **Full Pipeline** | Everything above + deploy as SageMaker endpoint |

---

## 🏗️ Architecture

```
┌─────────────────────┐     ┌──────────────────┐     ┌─────────────────────────┐
│  React + Vite       │────▶│  FastAPI Backend  │────▶│  AWS Step Functions     │
│  (Premium Dark UI)  │     │  (Thin API Layer) │     │  (Pipeline Orchestrator) │
└─────────────────────┘     └──────────────────┘     └─────────────────────────┘
        │                          │                          │
        │ Direct S3 Upload         │ Presigned URLs           ├── Lambda: OCR (Textract)
        │ via Presigned URLs       │ Start/Poll Pipeline      ├── Lambda: Chunking
        ▼                          ▼                          ├── Lambda: Dataset Gen (Bedrock)
   ┌──────────┐             ┌──────────┐                      ├── Lambda: Quality Control
   │  S3      │             │ Postgres │                      ├── Lambda: Fine-Tune (SageMaker)
   │  Bucket  │             │   (RDS)  │                      └── Lambda: Deploy (SageMaker)
   └──────────┘             └──────────┘
```

---

## 📁 Project Structure

```
modifai/
├── frontend/               # React + Vite + shadcn/ui
│   ├── src/
│   │   ├── pages/          # Dashboard, Projects, NewProject (6-step wizard),
│   │   │                   # ProjectDetail, DatasetReview, ModelComparison
│   │   ├── components/     # ProjectCard, PipelineTracker, StatsCard, FileUploadZone
│   │   └── api/            # API client (ky-based)
│   └── .env
│
├── backend/                # FastAPI (Python 3.12)
│   ├── app/
│   │   ├── api/v1/         # projects, evaluate, compare routers
│   │   ├── models/         # SQLAlchemy models (Project, PipelineStep)
│   │   ├── services/       # AWSService (S3, SFN, Bedrock, SageMaker)
│   │   └── main.py
│   ├── migrations/         # Alembic DB migrations
│   └── tests/              # 24 pytest tests (async, in-memory SQLite)
│
├── lambdas/                # AWS Lambda functions
│   ├── ocr/                # Textract PDF/image extraction
│   ├── chunking/           # Semantic text splitting
│   ├── dataset_generation/ # LLM-powered sample generation
│   ├── quality_control/    # Score + filter training examples
│   ├── fine_tune/          # SageMaker training job submission
│   ├── status_checker/     # Training job status polling
│   └── deploy/             # SageMaker endpoint creation
│
├── infra/                  # AWS infrastructure
│   ├── state_machine.json  # Step Functions ASL definition
│   └── deploy.sh           # Lambda + SFN deployment script
│
└── Arch.md                 # Architecture documentation
```

---

## ✨ Key Features

### 🧙 Smart Project Wizard
6-step adaptive wizard that changes based on selected mode — mode selection, intent, upload, evaluation, configuration, and review.

### 📊 Live Pipeline Tracking
Real-time status polling of Step Functions execution with a visual pipeline tracker showing per-step progress.

### 📝 Dataset Review & Editing
Inspect generated training examples, edit instructions/responses inline, delete bad examples, search, and export as JSONL.

### 🔬 Model Comparison
Side-by-side prompt testing — send the same prompt to both the base model (Bedrock Nova Micro) and your fine-tuned model (SageMaker), with typing animation, latency bars, and comparison history.

### 🛡️ Data Quality Evaluation
Before processing, AI evaluates your data quality and recommends whether it's suitable for fine-tuning or better suited for RAG.

### 📤 Secure Direct Upload
Files upload directly to S3 via presigned URLs — the backend never touches the file bytes.

---

## 🛠️ Getting Started

### Prerequisites

- **Node.js** ≥ 18
- **Python** ≥ 3.12
- **PostgreSQL** (local or RDS)
- **AWS Account** with configured credentials for:
  - S3, Step Functions, Lambda, Textract, Bedrock, SageMaker

### Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your DATABASE_URL and AWS credentials

# Run migrations
alembic upgrade head

# Start the server
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Configure environment
cp .env.example .env
# Edit .env: VITE_API_URL=http://localhost:8000/api/v1

# Start dev server
npm run dev
```

### Deploy Lambdas & Step Functions (AWS)

```bash
cd infra
chmod +x deploy.sh
./deploy.sh
```

---

## 🧪 Testing

```bash
cd backend
source venv/bin/activate

# Run all 24 tests
python -m pytest tests/ -v

# Expected output:
# ======================== 24 passed in 0.20s =========================
```

Tests cover: health check, project CRUD, presigned uploads, pipeline operations, data evaluation, dataset CRUD, and model comparison.

---

## 🔌 API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/api/v1/health` | Health check |
| `POST` | `/api/v1/projects/` | Create project |
| `GET` | `/api/v1/projects/` | List all projects |
| `GET` | `/api/v1/projects/{id}` | Get project details |
| `DELETE` | `/api/v1/projects/{id}` | Delete project + S3 data |
| `POST` | `/api/v1/projects/{id}/upload-url` | Get presigned S3 upload URL |
| `POST` | `/api/v1/projects/{id}/start` | Start pipeline execution |
| `GET` | `/api/v1/projects/{id}/status` | Poll pipeline status |
| `GET` | `/api/v1/projects/{id}/results` | Get pipeline results |
| `GET` | `/api/v1/projects/{id}/logs` | Get execution logs |
| `GET` | `/api/v1/projects/{id}/dataset` | List dataset examples |
| `PUT` | `/api/v1/projects/{id}/dataset/{idx}` | Edit example |
| `DELETE` | `/api/v1/projects/{id}/dataset/{idx}` | Delete example |
| `GET` | `/api/v1/projects/{id}/dataset/search` | Search examples |
| `GET` | `/api/v1/projects/{id}/dataset/export` | Export JSONL |
| `POST` | `/api/v1/evaluate/` | Evaluate data quality |
| `POST` | `/api/v1/compare/` | Compare base vs fine-tuned model |

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 19, Vite, shadcn/ui, Lucide Icons, React Router |
| **Backend** | FastAPI, SQLAlchemy (async), Pydantic, Alembic |
| **Database** | PostgreSQL (via asyncpg) |
| **Cloud** | AWS S3, Step Functions, Lambda, Textract, Bedrock, SageMaker |
| **Testing** | pytest + pytest-asyncio + httpx (24 tests) |

---

## 📄 License

Built for [AWS Hack2Skill](https://vision.hack2skill.com/event/ai-for-bharat/) hackathon.
