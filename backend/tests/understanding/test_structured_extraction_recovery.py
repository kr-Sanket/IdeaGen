"""Tests for bounded structured-extraction recovery and Ollama truncation handling.

Validates:
1. Ollama provider raises MalformedResponseError on done_reason='length'.
2. ScientificPaperExtractor uses staged output budgets: 1024 for initial attempt, 2048 for recovery.
3. Custom constructor budgets and environment/config overrides are respected.
4. ScientificPaperExtractor performs exactly one bounded recovery attempt on truncation / malformed output.
5. ScientificPaperExtractor performs bounded recovery when a full-text paper yields 0 statements.
6. ScientificPaperExtractor does NOT retry on ProviderConnectionError or offline/timeout failures.
7. ScientificPaperExtractor handles double failures safely returning ExtractionStatus.FAILED without fabrication.
8. ScientificPaperExtractor skips extraction when passages are empty without calling the provider.
9. ScientificPaperExtractor retains backward compatibility with providers that do not accept num_predict.
"""

import uuid
from typing import List, Optional, Type
from unittest.mock import MagicMock, patch

import pytest
import requests

from ideagen.core.config import Settings
from ideagen.llm.exceptions import (
    MalformedResponseError,
    ProviderConnectionError,
    ProviderFailureError,
)
from ideagen.llm.ollama_provider import OllamaLLMProvider
from ideagen.llm.provider import LLMProvider
from ideagen.understanding.extractor import (
    ScientificPaperExtractor,
    RECOVERY_INSTRUCTION_EMPTY,
    RECOVERY_INSTRUCTION_TRUNCATED,
)
from ideagen.understanding.models import (
    ExtractionStatus,
    RawStatementExtraction,
    StatementType,
    StructuredPaperExtractionOutput,
)
from ideagen.understanding.passage_selector import SelectedPassage


def _make_dummy_passage(idx: int = 0) -> SelectedPassage:
    return SelectedPassage(
        passage_id=uuid.uuid4(),
        passage_index=idx,
        page_start=1,
        page_end=1,
        section="Abstract",
        text="Federated learning enables collaborative training without sharing raw private data.",
    )


def _make_dummy_statement(idx: int = 0) -> RawStatementExtraction:
    return RawStatementExtraction(
        statement_text="Federated learning enables collaborative training without sharing raw data.",
        statement_type=StatementType.METHOD,
        passage_index=idx,
        confidence=0.92,
    )


def test_ollama_provider_done_reason_length_raises_malformed_response_error():
    """Verify that OllamaLLMProvider raises MalformedResponseError when done_reason is 'length'."""
    provider = OllamaLLMProvider(
        base_url="http://localhost:11434",
        model_name="qwen3:8b",
        timeout=10.0,
        num_predict=1024,
    )

    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "message": {
            "content": '{"research_problem": "Federated learning privacy challenges'
        },
        "done": True,
        "done_reason": "length",
    }

    with patch("requests.post", return_value=mock_response):
        with pytest.raises(MalformedResponseError) as exc_info:
            provider.generate_structured(
                prompt="Extract knowledge",
                response_schema=StructuredPaperExtractionOutput,
            )

        assert "truncated" in str(exc_info.value).lower()
        assert "done_reason=length" in str(exc_info.value)


def test_extractor_staged_budgets_initial_1024_recovery_2048():
    """Verify that initial extraction receives 1024 tokens and recovery receives 2048 tokens by default."""
    mock_provider = MagicMock(spec=LLMProvider)

    valid_output = StructuredPaperExtractionOutput(
        research_problem="Privacy in distributed healthcare.",
        methodology="Federated averaging with differential privacy.",
        statements=[_make_dummy_statement(0)],
    )

    # First attempt fails with truncation; recovery succeeds
    mock_provider.generate_structured.side_effect = [
        MalformedResponseError("Ollama generation truncated (done_reason=length)"),
        valid_output,
    ]

    extractor = ScientificPaperExtractor(provider=mock_provider, model_name="qwen3:8b")
    passages = [_make_dummy_passage(0)]
    paper_id = uuid.uuid4()

    result = extractor.extract(paper_id=paper_id, selected_passages=passages)

    assert result.extraction_status == ExtractionStatus.SUCCESS
    assert mock_provider.generate_structured.call_count == 2

    # Check Attempt 1 num_predict is 1024
    first_call_kwargs = mock_provider.generate_structured.call_args_list[0].kwargs
    assert first_call_kwargs.get("num_predict") == 1024

    # Check Attempt 2 num_predict is 2048
    second_call_kwargs = mock_provider.generate_structured.call_args_list[1].kwargs
    assert second_call_kwargs.get("num_predict") == 2048


def test_extractor_custom_budgets_override():
    """Verify that explicit constructor token budgets override defaults for both attempts."""
    mock_provider = MagicMock(spec=LLMProvider)

    valid_output = StructuredPaperExtractionOutput(
        research_problem="Custom budget problem.",
        statements=[_make_dummy_statement(0)],
    )

    mock_provider.generate_structured.side_effect = [
        MalformedResponseError("Truncated on custom budget"),
        valid_output,
    ]

    extractor = ScientificPaperExtractor(
        provider=mock_provider,
        model_name="qwen3:8b",
        first_attempt_num_predict=512,
        recovery_num_predict=1536,
    )
    passages = [_make_dummy_passage(0)]

    result = extractor.extract(paper_id=uuid.uuid4(), selected_passages=passages)

    assert result.extraction_status == ExtractionStatus.SUCCESS
    assert mock_provider.generate_structured.call_count == 2
    assert mock_provider.generate_structured.call_args_list[0].kwargs.get("num_predict") == 512
    assert mock_provider.generate_structured.call_args_list[1].kwargs.get("num_predict") == 1536


def test_extractor_config_env_overrides_respected(monkeypatch):
    """Verify that environment variable overrides flow into extractor defaults."""
    monkeypatch.setenv("IDEAGEN_OLLAMA_NUM_PREDICT", "768")
    monkeypatch.setenv("IDEAGEN_OLLAMA_RECOVERY_NUM_PREDICT", "1792")

    mock_provider = MagicMock(spec=LLMProvider)
    valid_output = StructuredPaperExtractionOutput(
        research_problem="Env override test.",
        statements=[_make_dummy_statement(0)],
    )
    mock_provider.generate_structured.side_effect = [
        MalformedResponseError("Truncated"),
        valid_output,
    ]

    extractor = ScientificPaperExtractor(provider=mock_provider, model_name="qwen3:8b")
    assert extractor.first_attempt_num_predict == 768
    assert extractor.recovery_num_predict == 1792

    extractor.extract(paper_id=uuid.uuid4(), selected_passages=[_make_dummy_passage(0)])

    assert mock_provider.generate_structured.call_args_list[0].kwargs.get("num_predict") == 768
    assert mock_provider.generate_structured.call_args_list[1].kwargs.get("num_predict") == 1792


def test_extractor_bounded_recovery_on_truncation_success():
    """Verify that ScientificPaperExtractor executes exactly one recovery attempt on truncation and succeeds."""
    mock_provider = MagicMock(spec=LLMProvider)

    valid_output = StructuredPaperExtractionOutput(
        research_problem="Data heterogeneity in federated medical settings.",
        methodology="Personalized aggregation strategy.",
        statements=[_make_dummy_statement(0)],
    )

    mock_provider.generate_structured.side_effect = [
        MalformedResponseError("Ollama generation truncated: exceeded output token budget (done_reason=length)"),
        valid_output,
    ]

    extractor = ScientificPaperExtractor(provider=mock_provider, model_name="qwen3:8b")
    passages = [_make_dummy_passage(0)]
    paper_id = uuid.uuid4()

    result = extractor.extract(paper_id=paper_id, selected_passages=passages, paper_title="FL in Healthcare")

    assert result.extraction_status == ExtractionStatus.SUCCESS
    assert result.research_problem == "Data heterogeneity in federated medical settings."
    assert len(result.research_statements) == 1
    assert result.research_statements[0].statement_type == StatementType.METHOD
    assert result.research_statements[0].source_passage_id == passages[0].passage_id

    # Verify exactly 2 calls occurred
    assert mock_provider.generate_structured.call_count == 2
    # Verify the second call received the truncated recovery instruction
    second_prompt = mock_provider.generate_structured.call_args_list[1].kwargs.get("prompt")
    assert RECOVERY_INSTRUCTION_TRUNCATED.strip() in second_prompt


def test_extractor_bounded_recovery_both_fail_returns_failed_status():
    """Verify that if both initial extraction and recovery fail, extraction returns FAILED without raising uncaught errors."""
    mock_provider = MagicMock(spec=LLMProvider)
    mock_provider.generate_structured.side_effect = [
        MalformedResponseError("Truncated on attempt 1"),
        MalformedResponseError("Truncated on attempt 2"),
    ]

    extractor = ScientificPaperExtractor(provider=mock_provider, model_name="qwen3:8b")
    passages = [_make_dummy_passage(0)]
    paper_id = uuid.uuid4()

    result = extractor.extract(paper_id=paper_id, selected_passages=passages)

    assert result.extraction_status == ExtractionStatus.FAILED
    assert "Truncated on attempt 2" in result.error_message
    assert len(result.research_statements) == 0
    assert mock_provider.generate_structured.call_count == 2


def test_extractor_bounded_recovery_on_empty_statements_full_text():
    """Verify that if a full-text paper yields 0 statements initially, bounded recovery is attempted once."""
    mock_provider = MagicMock(spec=LLMProvider)

    # First attempt yields valid JSON but 0 statements
    empty_output = StructuredPaperExtractionOutput(
        research_problem="Federated learning overview.",
        methodology="Decentralized gradient averaging.",
        statements=[],
    )
    # Recovery attempt yields statements
    recovered_output = StructuredPaperExtractionOutput(
        research_problem="Federated learning overview.",
        methodology="Decentralized gradient averaging.",
        statements=[_make_dummy_statement(0)],
    )

    mock_provider.generate_structured.side_effect = [empty_output, recovered_output]

    extractor = ScientificPaperExtractor(provider=mock_provider, model_name="qwen3:8b")
    passages = [_make_dummy_passage(0)]
    paper_id = uuid.uuid4()

    result = extractor.extract(paper_id=paper_id, selected_passages=passages)

    assert result.extraction_status == ExtractionStatus.SUCCESS
    assert len(result.research_statements) == 1
    assert mock_provider.generate_structured.call_count == 2

    # Check recovery prompt instruction
    second_prompt = mock_provider.generate_structured.call_args_list[1].kwargs.get("prompt")
    assert RECOVERY_INSTRUCTION_EMPTY.strip() in second_prompt


def test_extractor_connection_error_skips_recovery():
    """Verify that provider connection / timeout failures do not trigger unnecessary duplicate retry."""
    mock_provider = MagicMock(spec=LLMProvider)
    mock_provider.generate_structured.side_effect = ProviderConnectionError(
        "Ollama connection error at http://localhost:11434: Connection refused"
    )

    extractor = ScientificPaperExtractor(provider=mock_provider, model_name="qwen3:8b")
    passages = [_make_dummy_passage(0)]
    paper_id = uuid.uuid4()

    result = extractor.extract(paper_id=paper_id, selected_passages=passages)

    assert result.extraction_status == ExtractionStatus.FAILED
    assert "Connection refused" in result.error_message
    assert len(result.research_statements) == 0
    # Must NOT retry connection failure
    assert mock_provider.generate_structured.call_count == 1


def test_extractor_timeout_message_in_provider_failure_skips_recovery():
    """Verify that timeout in generic ProviderFailureError is recognized as connection failure and not retried."""
    mock_provider = MagicMock(spec=LLMProvider)
    mock_provider.generate_structured.side_effect = ProviderFailureError(
        "Ollama request timed out after 300.0s: Read timed out"
    )

    extractor = ScientificPaperExtractor(provider=mock_provider, model_name="qwen3:8b")
    passages = [_make_dummy_passage(0)]
    paper_id = uuid.uuid4()

    result = extractor.extract(paper_id=paper_id, selected_passages=passages)

    assert result.extraction_status == ExtractionStatus.FAILED
    assert "timed out" in result.error_message
    assert mock_provider.generate_structured.call_count == 1


def test_extractor_no_passages_skips_without_calling_provider():
    """Verify that papers without passages are returned as FAILED without invoking LLM provider."""
    mock_provider = MagicMock(spec=LLMProvider)
    extractor = ScientificPaperExtractor(provider=mock_provider, model_name="qwen3:8b")
    paper_id = uuid.uuid4()

    result = extractor.extract(paper_id=paper_id, selected_passages=[])

    assert result.extraction_status == ExtractionStatus.FAILED
    assert "No passages available" in result.error_message
    assert mock_provider.generate_structured.call_count == 0


def test_extractor_backward_compatibility_with_legacy_provider():
    """Verify that provider stubs without num_predict in their signature work seamlessly."""
    class LegacyProviderStub:
        def __init__(self):
            self.calls = 0

        def generate_structured(self, prompt: str, response_schema: Type):
            self.calls += 1
            return StructuredPaperExtractionOutput(
                research_problem="Legacy provider compatibility test.",
                statements=[_make_dummy_statement(0)],
            )

    legacy_stub = LegacyProviderStub()
    extractor = ScientificPaperExtractor(provider=legacy_stub, model_name="legacy-test")
    passages = [_make_dummy_passage(0)]
    paper_id = uuid.uuid4()

    result = extractor.extract(paper_id=paper_id, selected_passages=passages)

    assert result.extraction_status == ExtractionStatus.SUCCESS
    assert result.research_problem == "Legacy provider compatibility test."
    assert len(result.research_statements) == 1
    assert legacy_stub.calls == 1
