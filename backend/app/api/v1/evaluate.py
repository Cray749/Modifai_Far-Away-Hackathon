from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.services.aws_service import AWSService

router = APIRouter()


class EvaluateRequest(BaseModel):
    text_sample: str
    intent: str


class EvaluateResponse(BaseModel):
    score: float
    explanation: str
    error: Optional[str] = None


@router.post("/", response_model=EvaluateResponse)
async def evaluate_text_sample(request: EvaluateRequest):
    """
    Evaluates a small sample of the user's uploaded document to see if it
    is high quality and relevant to their selected fine-tuning intent.
    Uses Bedrock (Nova Micro).
    """
    print(f"DEBUG [evaluate_text_sample]: Received request data: {request.model_dump()}")
    if not request.text_sample or not request.intent:
        print("DEBUG [evaluate_text_sample]: Missing text_sample or intent")
        raise HTTPException(status_code=400, detail="Missing text_sample or intent")

    print(f"DEBUG [evaluate_text_sample]: Calling AWSService.evaluate_text_sample for intent: {request.intent}")
    result = await AWSService.evaluate_text_sample(
        text=request.text_sample, 
        intent=request.intent
    )
    print(f"DEBUG [evaluate_text_sample]: Received result from AWS: {result}")

    return EvaluateResponse(**result)
