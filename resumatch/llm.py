"""
ResuMatch - LLM Client Module
================================
Async httpx client for any OpenAI-compatible endpoint (llama-server, LM
Studio, Ollama, vLLM). Configured by llm.base_url in config.yaml, or
RESUMATCH_LLM__BASE_URL.

Provides:
- LLMClient class with health_check, complete, rewrite_bullet,
  generate_feedback, and batch_rewrite methods
- Module-level convenience: get_client(), is_llm_available()
- Graceful degradation: never raises, returns fallback values

All calls go through the OpenAI-compatible /v1/chat/completions endpoint.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from resumatch.config import CONFIG, logger

# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

_REWRITE_SYSTEM = (
    "You are a professional resume writing assistant. "
    "Rewrite bullet points to use strong action verbs, include quantifiable metrics "
    "where possible, and be concise. Return ONLY the rewritten bullet point, "
    "no explanation or extra text."
)

_FEEDBACK_SYSTEM = (
    "You are a professional resume reviewer. Analyze the resume text and scores provided, "
    "then return actionable feedback as a JSON array. Each item must have exactly these keys: "
    '"category" (one of: impact, presentation, competencies, general), '
    '"priority" (one of: high, medium, low), '
    '"message" (short description of the issue), '
    '"suggestion" (concrete advice to fix it). '
    "Return ONLY valid JSON, no markdown fences or extra text."
)


class LLMClient:
    """Async client for the OpenAI-compatible llama-server API."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ) -> None:
        llm_cfg = CONFIG.get("llm", {})
        self.base_url: str = (base_url or llm_cfg.get("base_url", "http://localhost:8080")).rstrip("/")
        self.model: str = model or llm_cfg.get("model", "qwen3-coder-abliterated")
        self.timeout: int = timeout or llm_cfg.get("timeout_seconds", 120)
        self.max_tokens: int = llm_cfg.get("max_tokens", 2048)
        self.temperature: float = llm_cfg.get("temperature", 0.3)

    # -- Health ---------------------------------------------------------------

    async def health_check(self) -> bool:
        """GET {base_url}/health. Returns True if status 200."""
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                resp = await client.get(f"{self.base_url}/health")
                return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
            logger.warning("LLM health check failed: %s", exc)
            return False

    # -- Core completion ------------------------------------------------------

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """POST /v1/chat/completions. Returns assistant content or empty string."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
            logger.warning("LLM complete() failed: %s", exc)
            return ""
        except (KeyError, IndexError, TypeError) as exc:
            logger.warning("LLM complete() unexpected response format: %s", exc)
            return ""

    # -- High-level helpers ---------------------------------------------------

    async def rewrite_bullet(self, bullet: str, context: str = "") -> str:
        """Rewrite a weak bullet point with stronger action verbs and metrics.

        Returns the improved bullet, or the original on error.
        """
        prompt = f"Rewrite this resume bullet point:\n\n{bullet}"
        if context:
            prompt += f"\n\nContext about the role/industry: {context}"

        result = await self.complete(prompt, system=_REWRITE_SYSTEM)
        if not result:
            logger.warning("rewrite_bullet: LLM unavailable, returning original bullet")
            return bullet
        return result

    async def generate_feedback(
        self,
        resume_text: str,
        scores: dict[str, Any],
    ) -> list[dict[str, str]]:
        """Generate personalized feedback based on resume text and scores.

        Returns list of {category, priority, message, suggestion} dicts,
        or empty list on error.
        """
        prompt = (
            f"Resume text:\n{resume_text}\n\n"
            f"Current scores:\n{json.dumps(scores, indent=2)}\n\n"
            "Provide 3-5 actionable feedback items as a JSON array."
        )

        result = await self.complete(prompt, system=_FEEDBACK_SYSTEM)
        if not result:
            logger.warning("generate_feedback: LLM unavailable, returning empty list")
            return []

        try:
            # Strip markdown fences if the model wraps its output
            cleaned = result.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()

            parsed = json.loads(cleaned)
            if not isinstance(parsed, list):
                logger.warning("generate_feedback: expected list, got %s", type(parsed).__name__)
                return []
            return parsed
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("generate_feedback: failed to parse LLM response: %s", exc)
            return []

    async def batch_rewrite(self, bullets: list[str], context: str = "") -> list[str]:
        """Rewrite multiple bullets in one LLM call (more token-efficient).

        Returns list of rewritten bullets, or the originals on error.
        """
        if not bullets:
            return []

        numbered = "\n".join(f"{i + 1}. {b}" for i, b in enumerate(bullets))
        prompt = (
            f"Rewrite each of these resume bullet points. "
            f"Return exactly {len(bullets)} rewritten bullets, one per line, "
            f"numbered the same way:\n\n{numbered}"
        )
        if context:
            prompt += f"\n\nContext about the role/industry: {context}"

        result = await self.complete(prompt, system=_REWRITE_SYSTEM)
        if not result:
            logger.warning("batch_rewrite: LLM unavailable, returning original bullets")
            return list(bullets)

        # Parse numbered lines from the response
        lines = [ln.strip() for ln in result.strip().splitlines() if ln.strip()]
        rewritten: list[str] = []
        for line in lines:
            # Strip leading number + dot/paren, e.g. "1. " or "1) "
            cleaned = line
            for sep in [". ", ") ", ": "]:
                parts = line.split(sep, 1)
                if len(parts) == 2 and parts[0].strip().isdigit():
                    cleaned = parts[1].strip()
                    break
            rewritten.append(cleaned)

        # If the model returned a different count, fall back to originals
        if len(rewritten) != len(bullets):
            logger.warning(
                "batch_rewrite: expected %d bullets, got %d; returning originals",
                len(bullets),
                len(rewritten),
            )
            return list(bullets)

        return rewritten


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_client: LLMClient | None = None


def get_client() -> LLMClient:
    """Return a lazy singleton LLMClient instance."""
    global _client  # noqa: PLW0603
    if _client is None:
        _client = LLMClient()
    return _client


async def is_llm_available() -> bool:
    """Whether the LLM can be used: switched on in config *and* reachable.

    Reachability alone is not enough. Setting llm.enabled to false used to
    change nothing here, so /health still advertised the LLM as available and
    /improve still called it.
    """
    if not CONFIG.get("llm", {}).get("enabled", False):
        return False
    return await get_client().health_check()
