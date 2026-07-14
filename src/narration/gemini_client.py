"""
gemini_client.py — ChainSight Narration Layer, Gemini API wrapper
Thin wrapper around google-genai's Client, used by narrator.py to turn a
prompt string into narration text.

Pipeline position:
    rules/engine.py (rule_events.json) -> narration/{prompt_templates,narrator}.py
        -> gemini_client.py (THIS, the only thing that talks to the network)

Uses the `google-genai` SDK (not the older `google-generativeai`, which
Google has fully deprecated — see requirements.txt) so this doesn't ship
already-obsolete code.

Design note: retries only cover transient failures (HTTP 429 rate limits,
5xx server errors) with exponential backoff — anything else (bad API key,
malformed request) fails immediately rather than being retried into a
longer, more confusing failure.
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

from google import genai
from google.genai import errors, types

logger = logging.getLogger("chainsight.narration.gemini")


@dataclass
class GeminiClientConfig:
    model: str = "gemini-flash-latest"
    temperature: float = 0.2
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0
    api_key: Optional[str] = None  # falls back to GEMINI_API_KEY env var if unset


_RETRYABLE_CODES = {429, 500, 502, 503, 504}


class GeminiClient:
    """Generates narration text for a single prompt, with retry/backoff on
    transient (rate-limit / server) errors."""

    def __init__(self, config: Optional[GeminiClientConfig] = None):
        self.config = config or GeminiClientConfig()
        api_key = self.config.api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "No Gemini API key found — set GEMINI_API_KEY (e.g. in .env) "
                "or pass GeminiClientConfig(api_key=...)."
            )
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str, system_instruction: str) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.config.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=self.config.temperature,
                    ),
                )
                return response.text
            except errors.APIError as e:
                last_error = e
                if e.code not in _RETRYABLE_CODES or attempt == self.config.max_retries:
                    raise
                backoff = self.config.retry_backoff_seconds * (2 ** attempt)
                logger.warning(
                    f"Gemini call failed ({e.code}), retrying in {backoff:.1f}s "
                    f"(attempt {attempt + 1}/{self.config.max_retries})"
                )
                time.sleep(backoff)
        raise last_error  # pragma: no cover — loop always returns or raises above
