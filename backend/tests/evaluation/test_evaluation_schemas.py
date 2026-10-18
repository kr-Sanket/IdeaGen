from ideagen.evaluation.schemas import EvaluationResult
from ideagen.evaluation.baselines import get_baselines
from ideagen.evaluation.discovery_eval import evaluate_discovery_engineering
from ideagen.evaluation.grounding_eval import evaluate_grounding_engineering
from ideagen.evaluation.temporal_eval import evaluate_temporal_leakage
from ideagen.evaluation.classifier_eval import evaluate_classifier_engineering
import json

def test_evaluation_results_are_serializable():
    results = [
        evaluate_discovery_engineering(),
        evaluate_grounding_engineering(),
        evaluate_temporal_leakage(),
        evaluate_classifier_engineering()
    ]
    
    for res in results:
        assert isinstance(res, EvaluationResult)
        assert res.is_engineering_validation is False
        # Verify it can be dumped to json
        json_data = res.model_dump_json()
        assert isinstance(json_data, str)
        assert res.evaluation_name in json_data

def test_baselines_definition():
    baselines = get_baselines()
    assert len(baselines) == 4
    for b in baselines:
        assert b.is_implemented is False
