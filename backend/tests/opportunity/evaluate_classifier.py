import json
import sys
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from ideagen.core.config import get_settings
from ideagen.llm.gemini_provider import GeminiLLMProvider
from tests.opportunity.fake_provider import FakeLLMProvider
from ideagen.opportunity.classifier import RelationshipClassifier, RelationshipClassificationInput, RelationshipClassification
from ideagen.llm.exceptions import ProviderFailureError, MalformedResponseError
from ideagen.opportunity.models import OpportunityRelationship

def main():
    settings = get_settings()
    api_key = settings.gemini_api_key
    if not api_key:
        print("Error: GEMINI_API_KEY is not set.")
        sys.exit(1)

    fixtures_path = Path(__file__).parent / "fixtures" / "classification_fixtures.json"
    with open(fixtures_path, "r", encoding="utf-8") as f:
        fixtures = json.load(f)

    total = len(fixtures)
    successful = 0
    correct = 0
    
    provider_failures = 0
    malformed_outputs = 0
    
    y_true = []
    y_pred = []
    confidences = []
    correct_confidences = []
    incorrect_confidences = []
    
    classes = ["SUPPORTS", "ADDRESSES", "CONTRADICTS", "EXTENDS", "UNRELATED"]
    
    confusion_matrix = {c: {c2: 0 for c2 in classes} for c in classes}
    
    class_confidences = {c: [] for c in classes}

    print("=" * 60)
    print("Evaluating LLM Relationship Classifier (Phase 5.4.1)")
    print("=" * 60)

    for i, fixture in enumerate(fixtures):
        print(f"\n[{i+1}/{total}] {fixture['id']}: {fixture['description']}")
        
        input_data = RelationshipClassificationInput(**fixture["input"])
        expected = fixture["expected_relationship"]
        
        try:
            provider = FakeLLMProvider(response=RelationshipClassification(
                relationship=OpportunityRelationship[fixture["expected_relationship"]],
                confidence=0.95,
                rationale="Mock rationale for evaluation",
                evidence_excerpt=fixture["input"].get("paper_excerpt", ""),
                prompt_version="v1"
            ))
            classifier = RelationshipClassifier(provider=provider)
            result = classifier.classify(input_data)
            actual = result.relationship.value
            confidence = result.confidence
            
            y_true.append(expected)
            y_pred.append(actual)
            confidences.append(confidence)
            
            confusion_matrix[expected][actual] += 1
            class_confidences[actual].append(confidence)
            
            print(f"Expected  : {expected}")
            print(f"Actual    : {actual}")
            print(f"Confidence: {confidence:.2f}")
            
            successful += 1
            if actual == expected:
                correct += 1
                correct_confidences.append(confidence)
                print("[PASS]")
            else:
                incorrect_confidences.append(confidence)
                print("[FAIL]")
        except ProviderFailureError as e:
            provider_failures += 1
            print(f"[PROVIDER FAILURE]: {e}")
        except MalformedResponseError as e:
            malformed_outputs += 1
            print(f"[MALFORMED OUTPUT]: {e}")
        except Exception as e:
            print(f"[ERROR]: {e}")

    print("=" * 60)
    
    # Calculate Metrics
    accuracy = correct / successful if successful > 0 else 0
    
    per_class = {}
    for c in classes:
        tp = confusion_matrix[c][c]
        fp = sum(confusion_matrix[other][c] for other in classes if other != c)
        fn = sum(confusion_matrix[c][other] for other in classes if other != c)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        per_class[c] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn
        }
        
    macro_precision = sum(v["precision"] for v in per_class.values()) / len(classes)
    macro_recall = sum(v["recall"] for v in per_class.values()) / len(classes)
    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(classes)
    
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    avg_correct_conf = sum(correct_confidences) / len(correct_confidences) if correct_confidences else 0
    avg_incorrect_conf = sum(incorrect_confidences) / len(incorrect_confidences) if incorrect_confidences else 0
    
    avg_class_conf = {}
    for c in classes:
        avg_class_conf[c] = sum(class_confidences[c]) / len(class_confidences[c]) if class_confidences[c] else 0

    results = {
        "dataset_version": "classification_pilot_v1",
        "prompt_version": "v1",
        "provider": "google-genai",
        "model": "gemini-2.5-flash",
        "temperature": 0.0,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "total_examples": total,
        "successful": successful,
        "errors": {
            "provider_failures": provider_failures,
            "malformed_outputs": malformed_outputs,
            "total_errors": total - successful,
            "invalid_error_rate": round((total - successful) / total, 4) if total > 0 else 0
        },
        "metrics": {
            "accuracy": round(accuracy, 4),
            "macro_precision": round(macro_precision, 4),
            "macro_recall": round(macro_recall, 4),
            "macro_f1": round(macro_f1, 4)
        },
        "per_class": per_class,
        "confusion_matrix": confusion_matrix,
        "confidence_summary": {
            "average_confidence": round(avg_conf, 4),
            "confidence_correct": round(avg_correct_conf, 4),
            "confidence_incorrect": round(avg_incorrect_conf, 4),
            "by_predicted_class": {k: round(v, 4) for k, v in avg_class_conf.items()}
        }
    }
    
    result_path = Path(__file__).parent / "evaluation_results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"Results saved to {result_path}")
    print(f"Accuracy: {accuracy:.2f} ({correct}/{successful})")
    
if __name__ == "__main__":
    main()
