"""LLM client wrapper with retry and rate limiting."""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import anthropic


@dataclass
class UsageStats:
    """Track API usage statistics."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_calls: int = 0
    total_errors: int = 0
    model: str = "claude-sonnet-4-20250514"

    def add_call(self, input_tokens: int, output_tokens: int):
        """Record a successful API call."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_calls += 1

    def add_error(self):
        """Record a failed API call."""
        self.total_errors += 1

    @property
    def estimated_cost(self) -> float:
        """Estimate cost in USD based on model."""
        if "opus" in self.model.lower():
            # Opus 4.5: $15/1M input, $75/1M output
            input_rate = 15
            output_rate = 75
        else:
            # Sonnet: $3/1M input, $15/1M output
            input_rate = 3
            output_rate = 15

        input_cost = self.total_input_tokens * input_rate / 1_000_000
        output_cost = self.total_output_tokens * output_rate / 1_000_000
        return input_cost + output_cost

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_calls": self.total_calls,
            "total_errors": self.total_errors,
            "model": self.model,
            "estimated_cost_usd": round(self.estimated_cost, 2),
        }


class LLMClient:
    """Claude API client with retry logic and rate limiting."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_delay: float = 0.5,
        max_retries: int = 3,
    ):
        """Initialize the LLM client.

        Args:
            model: Model ID to use.
            api_delay: Delay between API calls in seconds.
            max_retries: Maximum number of retries on failure.
        """
        self.model = model
        self.api_delay = api_delay
        self.max_retries = max_retries
        self.client = anthropic.Anthropic()
        self.usage = UsageStats(model=model)
        self._last_call_time: Optional[float] = None

    def _wait_for_rate_limit(self):
        """Wait if needed to respect rate limiting."""
        if self._last_call_time is not None:
            elapsed = time.time() - self._last_call_time
            if elapsed < self.api_delay:
                time.sleep(self.api_delay - elapsed)

    def call(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        system: Optional[str] = None,
    ) -> str:
        """Make an API call with retry logic.

        Args:
            prompt: The user message to send.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
            system: Optional system prompt.

        Returns:
            The text response from the model.

        Raises:
            Exception: If all retries fail.
        """
        self._wait_for_rate_limit()

        messages = [{"role": "user", "content": prompt}]
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(**kwargs)
                self._last_call_time = time.time()

                # Track usage
                self.usage.add_call(
                    response.usage.input_tokens,
                    response.usage.output_tokens,
                )

                return response.content[0].text

            except anthropic.RateLimitError as e:
                last_error = e
                wait_time = (2 ** attempt) * 10  # Exponential backoff: 10, 20, 40s
                print(f"Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)

            except anthropic.APIError as e:
                last_error = e
                self.usage.add_error()
                wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4s
                print(f"API error: {e}, retrying in {wait_time}s...")
                time.sleep(wait_time)

        self.usage.add_error()
        raise last_error

    def call_json(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        system: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Make an API call and parse JSON response.

        Args:
            prompt: The user message to send.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
            system: Optional system prompt.

        Returns:
            Parsed JSON response as dictionary.

        Raises:
            json.JSONDecodeError: If response is not valid JSON.
        """
        text = self.call(prompt, max_tokens, temperature, system)
        return self._parse_json(text)

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON from response text, handling markdown code blocks.

        Args:
            text: Raw response text.

        Returns:
            Parsed JSON dictionary.
        """
        text = text.strip()

        # Try to extract from markdown code blocks
        if "```json" in text:
            match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                text = match.group(1)
        elif "```" in text:
            match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                text = match.group(1)

        # Try to find JSON object boundaries
        if not text.startswith("{"):
            start = text.find("{")
            if start != -1:
                # Find matching closing brace
                depth = 0
                for i, c in enumerate(text[start:], start):
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            text = text[start:i + 1]
                            break

        return json.loads(text)

    def get_embedding(self, text: str) -> list:
        """Get text embedding using Claude's embedding endpoint.

        Note: Claude doesn't have a native embedding API, so we use
        a simple hash-based approach for diversity filtering.
        For production, consider using a dedicated embedding model.

        Args:
            text: Text to embed.

        Returns:
            List of floats representing the embedding.
        """
        # Simple character-based embedding for diversity filtering
        # In production, use a real embedding model like text-embedding-3-small
        import hashlib

        # Create a pseudo-embedding based on text hash
        hash_bytes = hashlib.sha256(text.encode()).digest()
        # Convert to list of floats in [-1, 1]
        embedding = [(b - 128) / 128.0 for b in hash_bytes]
        return embedding
