# Modifai — Automated LLM Fine-Tuning Pipeline

> **Hackathon project** · Automatically generates high-quality fine-tuning datasets from your documents using a self-improving multi-agent loop on AWS Bedrock.

---

## What does Modifai do?

You give it a document (chunked into text segments) and a goal.  
It automatically:
1. **Figures out the best strategy** for your document type
2. **Generates synthetic training samples** (question-answer pairs, instruction-output pairs, etc.)
3. **Judges the quality** of every sample
4. **Fixes the bad ones** by learning what went wrong and regenerating
5. **Hands you a clean dataset** ready for fine-tuning

No manual labelling. No prompt engineering. The agents do it themselves.

---

## The Three Agents — Simple Explanation

### 🎯 Orchestrator Agent
**"The strategist — decides the plan before any work starts"**

Looks at your document metadata (what domain is it? how many pages? how dense?) and your goal, then decides:
- **Intent** — Should it generate Q&A pairs? Instruction-following examples? Tutoring dialogues?
- **Quality bar** — How strict should the quality checker be? (0.5 = lenient, 0.95 = very strict)
- **Volume** — How many samples per chunk?

Real example from our smoke test:
```
Goal: "Generate a fine-tuning dataset for customer support Q&A"
→ intent=QA, quality_threshold=0.85, 4 samples per chunk
   Reasoning: "High quality threshold for customer support docs"
```

---

### 🔍 Critic Agent  
**"The quality inspector — scores every generated sample against the source"**

For each training sample, it reads the source chunk it was generated from and scores it on 3 dimensions:

| Dimension | What it checks | Example of failure |
|---|---|---|
| **Specificity** | Is the answer concrete or vague? | "It depends on the situation" = bad |
| **Grounding** | Does every fact come from the source? | Inventing facts not in the doc = bad |
| **Format** | Is it a complete, well-formed answer? | Incomplete sentences = bad |

Then it issues one of three verdicts:
- ✅ **accept** — sample is good, keep it
- ✏️ **rewrite** — salvageable, here's a corrected version
- ❌ **reject** — too broken to fix, discard

Real example from our smoke test:
```
12 samples evaluated → 11 accepted, 1 rewritten, 0 rejected (91.7% quality)
```

---

### 📚 Curriculum Agent  
**"The teacher — figures out WHY samples failed and writes better instructions"**

Only runs if the Critic says quality is below the threshold.

Takes all the rejection reasons, clusters them into **gap categories** (specific failure patterns), and writes a targeted improvement prompt. Example:

```
Gap found: "factual_drift" — samples introduce facts not in the source chunk
Gap found: "too_vague_on_entities" — answers don't name specific systems/tools
Gap found: "lacks_step_by_step_reasoning" — answers skip intermediate steps

→ Produces: "Each answer MUST enumerate all numbered steps from the source.
             Name every specific tool, system, or person referenced.
             Never introduce facts not explicitly stated in the chunk."
```

This prompt is injected into the next round of dataset generation — the loop self-corrects.

---

## The Full Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Document + Goal                                               │
│        │                                                        │
│        ▼                                                        │
│   ┌─────────────┐                                               │
│   │ Orchestrator│  → decides intent, threshold, samples/chunk  │
│   └──────┬──────┘                                               │
│          │                                                      │
│          ▼                                                      │
│   Dataset Generator  → creates N samples per chunk             │
│          │                                                      │
│          ▼                                                      │
│   ┌──────────────┐                                              │
│   │ Critic Agent │  → accept / rewrite / reject each sample    │
│   └──────┬───────┘                                              │
│          │                                                      │
│   Quality ≥ threshold? ──YES──→ Done ✅ return clean dataset   │
│          │                                                      │
│          NO                                                     │
│          │                                                      │
│          ▼                                                      │
│   ┌──────────────────┐                                          │
│   │ Curriculum Agent │  → identifies gaps, writes better prompt│
│   └──────┬───────────┘                                          │
│          │                                                      │
│   Dataset Generator (with improved prompt) → back to Critic    │
│                                                                 │
│   (Repeats up to max_iterations times)                          │
└─────────────────────────────────────────────────────────────────┘
```

**Exit conditions:**
- `threshold_met` — quality hit the bar (best case ✅)
- `all_accepted_first_pass` — perfect on first try (ideal ✅)
- `max_iterations` — hit the loop limit, returns best dataset so far

---

## Project Structure

```
modifai/
├── core/
│   ├── critic_agent.py         # The Critic's LLM evaluation functions
│   ├── dataset_generation.py   # Bedrock-powered sample generator
│   └── utils.py                # Logger helper
└── agents/
    ├── schemas.py              # All shared TypedDicts (locked — don't rename keys)
    ├── orchestrator.py         # OrchestratorAgent class
    ├── critic.py               # CriticAgent adapter class
    ├── curriculum.py           # CurriculumAgent class
    ├── logging_utils.py        # AgentEventLogger (writes JSONL for P3 dashboard)
    ├── pipeline_loop.py        # run_agentic_loop() — wires everything together
    └── tests/
        ├── test_orchestrator.py    # 19 unit tests
        ├── test_curriculum.py      # 7 unit tests
        └── test_pipeline_e2e.py    # 5 end-to-end tests
```

---

## Quick Start

### Prerequisites
- Python 3.8+
- AWS credentials with Bedrock access in `us-east-1`
- `pip install boto3 pytest`

### 1. Clone and install

```bash
git clone <repo-url>
cd <repo>
pip install boto3 pytest
```

### 2. Configure AWS credentials

```bash
# Option A — AWS CLI (recommended)
aws configure
# Enter your AWS Access Key ID, Secret, and set region to us-east-1

# Option B — Environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_REGION=us-east-1
```

> **Important:** Bedrock must be enabled in `us-east-1`. Model required: `amazon.nova-micro-v1:0`

### 3. Run the unit tests (no AWS needed)

```bash
python -m pytest modifai/agents/tests/ -v
```

Expected: **31 passed** in under 1 second. All AWS calls are mocked — no credits spent.

### 4. Run the smoke test (real AWS, ~1-3 minutes)

```bash
python smoke_test.py
```

Expected output:
```
Exit reason:   threshold_met   ← quality hit the bar
Iterations:    1               ← done in one pass
Final samples: 12              ← 12 clean training samples
Accept pct:    91.7%
Events logged: 2
Smoke test PASSED [OK]
```

---

## Using the Pipeline in Your Code

```python
from modifai.agents.pipeline_loop import run_agentic_loop

# Your document chunks (list of text strings)
chunks = [
    "Chunk 1 text...",
    "Chunk 2 text...",
    # ...
]

state = run_agentic_loop(
    goal="Build a Q&A bot for our HR policy documents",
    doc_metadata={
        "filename": "hr_policy.pdf",
        "page_count": 30,
        "domain": "HR policy",
        "estimated_chunk_count": len(chunks),
    },
    chunks=chunks,
    max_iterations=3,              # how many Critic→Curriculum loops to allow
    event_log_path="events.jsonl", # P3 dashboard reads this file
    region="us-east-1",
)

# Use the results
print(state["exit_reason"])        # "threshold_met" | "max_iterations" | "all_accepted_first_pass"
print(state["final_samples"])      # list of {instruction, input, output, chunk_id}
print(state["final_stats"])        # {total, accepted, rewritten, rejected, accept_pct}
print(state["events"])             # list of agent decision events (for P3 dashboard)
```

### Output: `final_samples` format

```json
[
  {
    "instruction": "What is step 2 of the ticket process?",
    "input": "Step 1: Open the portal. Step 2: Click on Tickets...",
    "output": "Step 2 is to click on Tickets in the support portal.",
    "chunk_id": 0
  }
]
```

### Output: `events` format (for P3 dashboard JSONL)

```json
{"event_id": "uuid", "timestamp": "2026-06-09T07:18:17Z", "agent": "orchestrator",
 "iteration": 0, "decision": "intent=QA, threshold=0.85, spc=4", ...}
{"event_id": "uuid", "timestamp": "2026-06-09T07:18:36Z", "agent": "critic",
 "iteration": 1, "decision": "accepted=11/12 (91.7%)...", ...}
```

---

## For Teammates — How to Contribute

### Branch naming convention
```
feature/<your-name>/<what-youre-adding>
# Examples:
feature/riya/pdf-chunking
feature/arjun/p3-dashboard
feature/lakshya/bedrock-finetuning
```

### Before pushing
```bash
# Always run this first — must be 31/31
python -m pytest modifai/agents/tests/ -v
```

### What NOT to change without a team sync
- Field names in `modifai/agents/schemas.py` — P2 (infra) and P3 (dashboard) depend on these exact keys
- `modifai/core/critic_agent.py` — the critic implementation is locked (owned by teammate)
- The `run_agentic_loop()` function signature — other services call this

### Adding new tests
Put them in `modifai/agents/tests/`. Mock all AWS calls using `@patch("modifai.agents.pipeline_loop.CriticAgent")`.

---

## What Teammates Need to Test

| What | What they need |
|---|---|
| **Unit tests only** (no AWS) | Python 3.8+, `pip install pytest boto3`, clone the repo |
| **Smoke test** (real pipeline) | Above + AWS credentials with Bedrock access in `us-east-1` |
| **P3 dashboard integration** | The `smoke_events.jsonl` file (generated by running smoke test) |

### Sharing AWS credentials (for testing only)
If a teammate doesn't have their own AWS access:
1. Create an IAM user in your AWS account
2. Attach policy: `AmazonBedrockFullAccess`
3. Share the access key + secret key
4. They run `aws configure` and paste them in

---

## Key Design Decisions

| Decision | Why |
|---|---|
| Bedrock region: `us-east-1` | Only confirmed region for `amazon.nova-micro-v1:0` |
| CriticAgent is an adapter class | Wraps the teammate's function-based implementation without modifying it |
| `accept_pct` counts only pure accepts | Rewrites indicate the generator still needs curriculum feedback |
| Events written to JSONL | P3 dashboard polls this file for real-time updates |
| All TypedDicts locked in `schemas.py` | P2 infrastructure depends on these exact field names |

---

## AWS Model Used

| Model | ID | Cost |
|---|---|---|
| Amazon Nova Micro | `amazon.nova-micro-v1:0` | Very low — text-only, fastest Nova model |

Estimated cost for one `smoke_test.py` run: **< $0.01**

---

## Team

| Person | Component |
|---|---|
| Lakshya | Orchestrator Agent, Pipeline Loop, Integration |
| [Teammate] | Critic Agent (core evaluation logic) |
| [Teammate] | Curriculum Agent |
| [Teammate] | P3 Dashboard / Frontend |

---

*Built for the FAR AWAY Hackathon · June 2026*
