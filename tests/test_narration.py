"""
test_narration.py — regression tests for src/narration/*.py

No real Gemini API calls are made — GeminiClient is mocked out via
pytest-mock everywhere except test_gemini_client_retry_backoff.py-equivalent
cases below, where the underlying google.genai.Client is mocked instead, to
also cover the retry/backoff behavior in gemini_client.py itself.
"""

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.narration.narrator import Narrator, NO_EVENTS_SUMMARY
from src.narration.gemini_client import GeminiClient, GeminiClientConfig
from google.genai import errors


def make_event(rule_id: str = "R1_RESTRICTED_ZONE_INTRUSION", frame: int = 10) -> dict:
    return {
        "rule_id": rule_id,
        "rule_name": "Restricted Zone Intrusion",
        "frame": frame,
        "timestamp": frame / 25.0,
        "severity": "warning",
        "track_ids": [1],
        "trigger": "Person 1 entered restricted zone 'aisle'.",
        "evidence": {"zone": "aisle"},
        "conclusion": "Person 1 was inside a restricted zone.",
    }


# --- Narrator: per-event narration ---

def test_narrate_events_attaches_narration_per_event(mocker):
    fake_client = Mock()
    fake_client.generate.return_value = "A person entered the restricted aisle."
    narrator = Narrator(fake_client)

    events = [make_event(frame=10), make_event(frame=20)]
    narrated = narrator.narrate_events(events)

    assert fake_client.generate.call_count == 2
    assert all(e["narration"] == "A person entered the restricted aisle." for e in narrated)
    # original dicts are untouched (narrate_events returns copies)
    assert "narration" not in events[0]


def test_narrate_events_passes_system_instruction(mocker):
    fake_client = Mock()
    fake_client.generate.return_value = "narrated"
    narrator = Narrator(fake_client)

    narrator.narrate_events([make_event()])

    prompt, system_instruction = fake_client.generate.call_args.args
    assert "do not add causal claims" in system_instruction.lower()
    assert "Person 1 entered restricted zone" in prompt


# --- Narrator: clip-level summary ---

def test_summarize_calls_client_once_when_events_exist(mocker):
    fake_client = Mock()
    fake_client.generate.return_value = "2 restricted-zone intrusions were observed."
    narrator = Narrator(fake_client)

    summary = narrator.summarize([make_event(frame=10), make_event(frame=20)])

    fake_client.generate.assert_called_once()
    assert summary == "2 restricted-zone intrusions were observed."


def test_summarize_skips_client_call_when_no_events(mocker):
    fake_client = Mock()
    narrator = Narrator(fake_client)

    summary = narrator.summarize([])

    fake_client.generate.assert_not_called()
    assert summary == NO_EVENTS_SUMMARY


# --- Narrator: end-to-end run() over a tmp rule_events.json ---

def test_run_loads_events_narrates_and_summarizes(tmp_path):
    import json
    events = [make_event(frame=10)]
    rule_events_path = tmp_path / "rule_events.json"
    rule_events_path.write_text(json.dumps(events))

    fake_client = Mock()
    fake_client.generate.return_value = "narrated text"
    narrator = Narrator(fake_client)

    result = narrator.run(str(rule_events_path))

    assert len(result["events"]) == 1
    assert result["events"][0]["narration"] == "narrated text"
    assert result["summary"] == "narrated text"


def test_run_raises_on_missing_file():
    narrator = Narrator(Mock())
    with pytest.raises(FileNotFoundError):
        narrator.run("does/not/exist.json")


# --- GeminiClient: retry/backoff on transient errors ---

def _api_error(code: int) -> errors.APIError:
    return errors.APIError(code, {"message": "transient failure", "status": "UNAVAILABLE"})


def test_gemini_client_retries_then_succeeds(mocker):
    mock_sleep = mocker.patch("src.narration.gemini_client.time.sleep")
    mock_response = Mock(text="generated narration")

    mock_genai_client = Mock()
    mock_genai_client.models.generate_content.side_effect = [
        _api_error(429), _api_error(503), mock_response,
    ]
    mocker.patch("src.narration.gemini_client.genai.Client", return_value=mock_genai_client)

    client = GeminiClient(GeminiClientConfig(api_key="fake-key", max_retries=3, retry_backoff_seconds=0.01))
    result = client.generate("prompt", "system instruction")

    assert result == "generated narration"
    assert mock_genai_client.models.generate_content.call_count == 3
    assert mock_sleep.call_count == 2


def test_gemini_client_raises_after_exhausting_retries(mocker):
    mocker.patch("src.narration.gemini_client.time.sleep")
    mock_genai_client = Mock()
    mock_genai_client.models.generate_content.side_effect = _api_error(429)
    mocker.patch("src.narration.gemini_client.genai.Client", return_value=mock_genai_client)

    client = GeminiClient(GeminiClientConfig(api_key="fake-key", max_retries=2, retry_backoff_seconds=0.01))

    with pytest.raises(errors.APIError):
        client.generate("prompt", "system instruction")

    assert mock_genai_client.models.generate_content.call_count == 3  # 1 initial + 2 retries


def test_gemini_client_does_not_retry_non_retryable_errors(mocker):
    mock_sleep = mocker.patch("src.narration.gemini_client.time.sleep")
    mock_genai_client = Mock()
    mock_genai_client.models.generate_content.side_effect = _api_error(400)
    mocker.patch("src.narration.gemini_client.genai.Client", return_value=mock_genai_client)

    client = GeminiClient(GeminiClientConfig(api_key="fake-key", max_retries=3))

    with pytest.raises(errors.APIError):
        client.generate("prompt", "system instruction")

    assert mock_genai_client.models.generate_content.call_count == 1
    mock_sleep.assert_not_called()


def test_gemini_client_requires_api_key(mocker):
    mocker.patch.dict("os.environ", {}, clear=True)
    with pytest.raises(ValueError):
        GeminiClient(GeminiClientConfig(api_key=None))
