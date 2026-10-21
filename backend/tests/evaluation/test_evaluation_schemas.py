from datetime import datetime, timedelta, timezone

from ideagen.evaluation import schemas
from ideagen.evaluation.schemas import EvaluationResult, EvaluationType
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


def _fixture_records():
    return [
        evaluate_discovery_engineering(),
        evaluate_grounding_engineering(),
        evaluate_temporal_leakage(),
        evaluate_classifier_engineering(),
    ]


def test_all_hardcoded_records_are_marked_fixture_not_executed():
    for res in _fixture_records():
        assert res.evaluation_type is EvaluationType.FIXTURE
        assert res.evaluation_type is not EvaluationType.EXECUTED
        assert json.loads(res.model_dump_json())["evaluation_type"] == "fixture"


def _minimal_result(**overrides):
    fields = dict(
        evaluation_name="example",
        fixture_version="v0",
        model_provider=None,
        parameters={},
        metrics={},
        denominators={},
        errors=[],
        limitations="example",
    )
    fields.update(overrides)
    return EvaluationResult(**fields)


def test_evaluation_type_defaults_to_fixture_for_existing_callers():
    assert _minimal_result().evaluation_type is EvaluationType.FIXTURE
    assert _minimal_result(evaluation_type="executed").evaluation_type is EvaluationType.EXECUTED


def test_timestamp_is_generated_per_instance(monkeypatch):
    instants = iter([
        datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=1),
    ])

    class _FakeClock:
        @staticmethod
        def now(tz=None):
            assert tz is timezone.utc
            return next(instants)

    monkeypatch.setattr(schemas, "datetime", _FakeClock)
    first, second = _minimal_result(), _minimal_result()
    assert first.timestamp == "2026-01-01T12:00:00+00:00"
    assert second.timestamp == "2026-01-01T12:00:01+00:00"


def test_timestamp_is_timezone_aware_utc_iso_and_serializable():
    res = _minimal_result()
    parsed = datetime.fromisoformat(res.timestamp)
    assert parsed.utcoffset() == timedelta(0)
    payload = json.loads(res.model_dump_json())
    assert payload["timestamp"] == res.timestamp
    assert payload["evaluation_type"] == "fixture"
    assert EvaluationResult.model_validate(payload) == res


def test_baselines_definition():
    baselines = get_baselines()
    assert len(baselines) == 4
    tfidf_baseline = next(b for b in baselines if b.name == "TF-IDF Discovery")
    assert tfidf_baseline.is_implemented is True
    for b in baselines:
        if b.name != "TF-IDF Discovery":
            assert b.is_implemented is False
