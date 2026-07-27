"""
Tests for resumatch.llm module.

All tests mock httpx — no real HTTP calls are made.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from resumatch.config import CONFIG
from resumatch.llm import LLMClient, get_client, is_llm_available


# =============================================================================
# LLMClient.__init__
# =============================================================================


class TestLLMClientInit:
    """Test that LLMClient reads defaults from CONFIG or accepts overrides."""

    def test_init_reads_from_config(self):
        """Defaults come from CONFIG['llm'] -- never a hardcoded address."""
        llm_cfg = CONFIG["llm"]
        client = LLMClient()
        assert client.base_url == llm_cfg["base_url"].rstrip("/")
        assert client.model == llm_cfg["model"]
        assert client.timeout == llm_cfg["timeout_seconds"]
        assert client.max_tokens == llm_cfg["max_tokens"]
        assert client.temperature == llm_cfg["temperature"]

    def test_init_with_custom_values(self):
        """Custom constructor args override config."""
        client = LLMClient(
            base_url="http://localhost:9999",
            model="test-model",
            timeout=30,
        )
        assert client.base_url == "http://localhost:9999"
        assert client.model == "test-model"
        assert client.timeout == 30

    def test_init_strips_trailing_slash(self):
        """base_url should have trailing slashes removed."""
        client = LLMClient(base_url="http://localhost:9999/")
        assert client.base_url == "http://localhost:9999"


# =============================================================================
# health_check
# =============================================================================


class TestHealthCheck:
    """Test the health_check method."""

    async def test_health_check_returns_true_on_200(self):
        """Should return True when server responds with 200."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("resumatch.llm.httpx.AsyncClient") as MockAsyncClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)
            MockAsyncClient.return_value = mock_client_instance

            client = LLMClient(base_url="http://test:8080")
            result = await client.health_check()

            assert result is True
            mock_client_instance.get.assert_awaited_once_with("http://test:8080/health")

    async def test_health_check_returns_false_on_connection_error(self):
        """Should return False when server is unreachable."""
        with patch("resumatch.llm.httpx.AsyncClient") as MockAsyncClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)
            MockAsyncClient.return_value = mock_client_instance

            client = LLMClient(base_url="http://test:8080")
            result = await client.health_check()

            assert result is False

    async def test_health_check_returns_false_on_timeout(self):
        """Should return False when request times out."""
        with patch("resumatch.llm.httpx.AsyncClient") as MockAsyncClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(
                side_effect=httpx.TimeoutException("Timed out")
            )
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)
            MockAsyncClient.return_value = mock_client_instance

            client = LLMClient(base_url="http://test:8080")
            result = await client.health_check()

            assert result is False


# =============================================================================
# complete
# =============================================================================


class TestComplete:
    """Test the complete method."""

    def _make_mock_client(self, response_content: str = "Hello world"):
        """Helper to build a mocked AsyncClient that returns a chat completion."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": response_content}}],
        }

        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        return mock_client_instance

    async def test_complete_sends_correct_request_format(self):
        """Should POST to /v1/chat/completions with proper payload."""
        mock_client_instance = self._make_mock_client("test response")

        with patch("resumatch.llm.httpx.AsyncClient") as MockAsyncClient:
            MockAsyncClient.return_value = mock_client_instance

            client = LLMClient(base_url="http://test:8080", model="test-model")
            await client.complete("Hello", system="Be helpful")

            mock_client_instance.post.assert_awaited_once()
            call_args = mock_client_instance.post.call_args
            assert call_args[0][0] == "http://test:8080/v1/chat/completions"

            payload = call_args[1]["json"]
            assert payload["model"] == "test-model"
            assert len(payload["messages"]) == 2
            assert payload["messages"][0] == {"role": "system", "content": "Be helpful"}
            assert payload["messages"][1] == {"role": "user", "content": "Hello"}

    async def test_complete_returns_assistant_content(self):
        """Should extract and return the assistant message content."""
        mock_client_instance = self._make_mock_client("  The answer is 42  ")

        with patch("resumatch.llm.httpx.AsyncClient") as MockAsyncClient:
            MockAsyncClient.return_value = mock_client_instance

            client = LLMClient(base_url="http://test:8080")
            result = await client.complete("What is the meaning of life?")

            assert result == "The answer is 42"

    async def test_complete_returns_empty_string_on_error(self):
        """Should return empty string when LLM is unavailable."""
        with patch("resumatch.llm.httpx.AsyncClient") as MockAsyncClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)
            MockAsyncClient.return_value = mock_client_instance

            client = LLMClient(base_url="http://test:8080")
            result = await client.complete("Hello")

            assert result == ""

    async def test_complete_without_system_message(self):
        """When system is None, messages should only have the user message."""
        mock_client_instance = self._make_mock_client("response")

        with patch("resumatch.llm.httpx.AsyncClient") as MockAsyncClient:
            MockAsyncClient.return_value = mock_client_instance

            client = LLMClient(base_url="http://test:8080")
            await client.complete("Hello")

            payload = mock_client_instance.post.call_args[1]["json"]
            assert len(payload["messages"]) == 1
            assert payload["messages"][0]["role"] == "user"


# =============================================================================
# rewrite_bullet
# =============================================================================


class TestRewriteBullet:
    """Test the rewrite_bullet method."""

    async def test_rewrite_bullet_sends_correct_prompt(self):
        """Should include the bullet in the prompt sent to complete()."""
        client = LLMClient(base_url="http://test:8080")
        with patch.object(client, "complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = "Led team of 5 engineers to deliver project 20% ahead of schedule"

            result = await client.rewrite_bullet(
                "Helped with the project",
                context="Software engineering",
            )

            mock_complete.assert_awaited_once()
            call_args = mock_complete.call_args
            prompt = call_args[0][0]
            assert "Helped with the project" in prompt
            assert "Software engineering" in prompt
            assert result == "Led team of 5 engineers to deliver project 20% ahead of schedule"

    async def test_rewrite_bullet_returns_original_on_error(self):
        """When LLM fails, should return the original bullet unchanged."""
        client = LLMClient(base_url="http://test:8080")
        with patch.object(client, "complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = ""

            result = await client.rewrite_bullet("Helped with the project")

            assert result == "Helped with the project"


# =============================================================================
# generate_feedback
# =============================================================================


class TestGenerateFeedback:
    """Test the generate_feedback method."""

    async def test_generate_feedback_returns_list_of_dicts(self):
        """Should parse LLM JSON response into a list of feedback dicts."""
        feedback_data = [
            {
                "category": "impact",
                "priority": "high",
                "message": "Bullet points lack quantifiable metrics",
                "suggestion": "Add numbers, percentages, or dollar amounts to each bullet",
            },
            {
                "category": "presentation",
                "priority": "medium",
                "message": "Missing skills section",
                "suggestion": "Add a dedicated skills section with relevant keywords",
            },
        ]

        client = LLMClient(base_url="http://test:8080")
        with patch.object(client, "complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = json.dumps(feedback_data)

            result = await client.generate_feedback("Some resume text", {"overall": 65})

            assert len(result) == 2
            assert result[0]["category"] == "impact"
            assert result[1]["priority"] == "medium"

    async def test_generate_feedback_returns_empty_list_on_error(self):
        """When LLM fails, should return empty list."""
        client = LLMClient(base_url="http://test:8080")
        with patch.object(client, "complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = ""

            result = await client.generate_feedback("Some resume text", {"overall": 65})

            assert result == []

    async def test_generate_feedback_handles_invalid_json(self):
        """When LLM returns invalid JSON, should return empty list."""
        client = LLMClient(base_url="http://test:8080")
        with patch.object(client, "complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = "This is not JSON at all"

            result = await client.generate_feedback("Some resume text", {"overall": 65})

            assert result == []


# =============================================================================
# batch_rewrite
# =============================================================================


class TestBatchRewrite:
    """Test the batch_rewrite method."""

    async def test_batch_rewrite_handles_multiple_bullets(self):
        """Should rewrite all bullets in a single call."""
        originals = [
            "Helped with projects",
            "Was responsible for tasks",
            "Assisted the team",
        ]
        rewritten_response = (
            "1. Spearheaded 3 cross-functional projects, delivering all on schedule\n"
            "2. Managed a portfolio of 12 tasks across 4 departments\n"
            "3. Mentored 5 junior team members, improving team velocity by 25%"
        )

        client = LLMClient(base_url="http://test:8080")
        with patch.object(client, "complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = rewritten_response

            result = await client.batch_rewrite(originals, context="Tech industry")

            assert len(result) == 3
            assert "Spearheaded" in result[0]
            assert "Managed" in result[1]
            assert "Mentored" in result[2]

    async def test_batch_rewrite_returns_originals_on_error(self):
        """When LLM fails, should return original bullets."""
        originals = ["Helped with projects", "Was responsible for tasks"]

        client = LLMClient(base_url="http://test:8080")
        with patch.object(client, "complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = ""

            result = await client.batch_rewrite(originals)

            assert result == originals

    async def test_batch_rewrite_empty_list(self):
        """Empty input should return empty output without calling LLM."""
        client = LLMClient(base_url="http://test:8080")
        with patch.object(client, "complete", new_callable=AsyncMock) as mock_complete:
            result = await client.batch_rewrite([])

            assert result == []
            mock_complete.assert_not_awaited()


# =============================================================================
# Module-level convenience
# =============================================================================


class TestModuleLevelConvenience:
    """Test get_client() and is_llm_available()."""

    def test_get_client_returns_singleton(self):
        """Repeated calls should return the same instance."""
        import resumatch.llm as llm_module

        # Reset singleton state
        llm_module._client = None

        client1 = get_client()
        client2 = get_client()

        assert client1 is client2
        assert isinstance(client1, LLMClient)

        # Clean up
        llm_module._client = None

    async def test_is_llm_available_delegates_to_health_check(self):
        """is_llm_available() should call health_check on the singleton."""
        import resumatch.llm as llm_module

        mock_client = MagicMock(spec=LLMClient)
        mock_client.health_check = AsyncMock(return_value=True)
        llm_module._client = mock_client

        result = await is_llm_available()

        assert result is True
        mock_client.health_check.assert_awaited_once()

        # Clean up
        llm_module._client = None

    async def test_disabled_in_config_is_unavailable_without_probing(self, monkeypatch):
        """llm.enabled=false must win even when a server is reachable.

        It didn't, so /health reported the LLM as available and /improve
        called it regardless of the setting.
        """
        import resumatch.llm as llm_module

        mock_client = MagicMock(spec=LLMClient)
        mock_client.health_check = AsyncMock(return_value=True)
        llm_module._client = mock_client
        monkeypatch.setitem(llm_module.CONFIG, "llm", {"enabled": False})

        result = await is_llm_available()

        assert result is False
        mock_client.health_check.assert_not_awaited()

        # Clean up
        llm_module._client = None
