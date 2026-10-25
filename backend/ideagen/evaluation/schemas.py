from enum import Enum
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone


class EvaluationType(str, Enum):
    """How an evaluation result was obtained."""
    FIXTURE = "fixture"    # recorded/hardcoded values for a controlled fixture; not a measured run
    EXECUTED = "executed"  # produced by actually running the evaluation


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvaluationResult(BaseModel):
    evaluation_name: str
    fixture_version: str
    evaluation_type: EvaluationType = EvaluationType.FIXTURE
    # Generated per instance (a plain default would be evaluated once at import time).
    timestamp: str = Field(default_factory=_utc_now_iso)
    model_provider: Optional[str]
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    denominators: Dict[str, int]
    errors: List[str]
    limitations: str
    is_engineering_validation: bool = False
