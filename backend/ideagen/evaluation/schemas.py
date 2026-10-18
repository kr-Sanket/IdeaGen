from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime

class EvaluationResult(BaseModel):
    evaluation_name: str
    fixture_version: str
    timestamp: str = datetime.utcnow().isoformat()
    model_provider: Optional[str]
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    denominators: Dict[str, int]
    errors: List[str]
    limitations: str
    is_engineering_validation: bool = False
