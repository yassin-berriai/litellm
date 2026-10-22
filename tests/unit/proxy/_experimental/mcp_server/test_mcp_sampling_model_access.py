"""
Tests for MCP sampling handler model-access enforcement.

Verifies that handle_sampling_create_message and _check_model_access
enforce the same model-permission checks as regular /chat/completions
calls, preventing a malicious upstream MCP server from requesting
inference on models the caller's API key is not authorized to use.
"""

from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from mcp.types import ErrorData

from litellm.caching.dual_cache import DualCache
from litellm.models.team import LiteLLM_TeamTableCachedObj
from litellm.models.user import LiteLLM_UserTable
from litellm.proxy._experimental.mcp_server.sampling_handler import (
    _check_model_access,
    _run_budget_checks,
)
from litellm.proxy._types import UI_TEAM_ID, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import TeamNotFoundError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_user_api_key_auth(
    *,
    models=None,
    team_id=None,
    team_model_aliases=None,
    api_key="sk-test-key",
    token=None,
    user_role=None,
):
    """Build a minimal UserAPIKeyAuth-like object for tests."""
    auth = MagicMock()
    auth.models = models or []
    auth.team_id = team_id
    auth.team_model_aliases = team_model_aliases or {}
    auth.access_group_ids = []
    auth.api_key = api_key
    auth.token = token
    auth.user_role = user_role
    return auth


# ---------------------------------------------------------------------------
# _check_model_access
# ---------------------------------------------------------------------------


class TestCheckModelAccess:
    """Tests for the _check_model_access helper that gates sampling requests."""

    @pytest.mark.asyncio
    async def test_should_return_none_when_no_auth_context(self):
        """No auth context means no restriction — pass through."""
        result = await _check_model_access("gpt-4o", user_api_key_auth=None)
        assert result is None

    @pytest.mark.asyncio
    async def test_should_allow_model_when_key_has_access(self):
        """Key with explicit model access should be allowed."""
        auth = _make_user_api_key_auth(models=["gpt-4o", "gpt-3.5-turbo"])

        with patch(
            "litellm.proxy.auth.auth_checks.can_key_call_model",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_check:
            result = await _check_model_access("gpt-4o", user_api_key_auth=auth)

        assert result is None
        mock_check.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_should_deny_model_when_key_lacks_access(self):
        """Key without model access should be denied with ErrorData."""
        from litellm.proxy._types import ProxyException

        auth = _make_user_api_key_auth(models=["gpt-3.5-turbo"])

        with patch(
            "litellm.proxy.auth.auth_checks.can_key_call_model",
            new_callable=AsyncMock,
            side_effect=ProxyException(
                message="key not allowed to access model",
                type="key_model_access_denied",
                param="model",
                code=401,
            ),
        ):
            result = await _check_model_access("gpt-4o", user_api_key_auth=auth)

        # Should return ErrorData, not raise
        assert result is not None
        assert result.code == -1
        assert "Model access denied" in result.message
        assert "gpt-4o" in result.message

    @pytest.mark.asyncio
    async def test_should_allow_wildcard_model_access(self):
        """Key with wildcard model access should allow any model."""
        auth = _make_user_api_key_auth(models=["*"])

        with patch(
            "litellm.proxy.auth.auth_checks.can_key_call_model",
            new_callable=AsyncMock,
            return_value=True,
        ):
            result = await _check_model_access(
                "claude-3-opus-20240229", user_api_key_auth=auth
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_should_deny_expensive_model_requested_by_malicious_server(self):
        """Simulates the attack: malicious MCP server hints at an expensive model
        the caller's key is restricted from using."""
        from litellm.proxy._types import ProxyException

        # Key only has access to cheap models
        auth = _make_user_api_key_auth(models=["gpt-3.5-turbo"])

        with patch(
            "litellm.proxy.auth.auth_checks.can_key_call_model",
            new_callable=AsyncMock,
            side_effect=ProxyException(
                message="key not allowed to access model. This key can only access models=['gpt-3.5-turbo']. Tried to access claude-3-opus-20240229",
                type="key_model_access_denied",
                param="model",
                code=401,
            ),
        ):
            result = await _check_model_access(
                "claude-3-opus-20240229", user_api_key_auth=auth
            )

        assert result is not None
        assert result.code == -1
        assert "claude-3-opus-20240229" in result.message

    @pytest.mark.asyncio
    async def test_should_log_internal_denial_reason_and_hide_allowlist_from_client(self, caplog):
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.proxy.auth.model_access_denied import model_access_denied_client_message

        auth = UserAPIKeyAuth(api_key="sk-test-key", models=["gpt-3.5-turbo"])

        with caplog.at_level("WARNING", logger="LiteLLM"):
            result = await _check_model_access("gpt-4o\r\nforged", user_api_key_auth=auth)

        assert result is not None
        assert result.message == model_access_denied_client_message(model="gpt-4o\r\nforged")
        denial_records = [r for r in caplog.records if "gpt-3.5-turbo" in r.getMessage()]
        assert len(denial_records) == 1
        assert "Tried to access gpt-4oforged" in denial_records[0].getMessage()

    @pytest.mark.asyncio
    async def test_should_deny_empty_oauth_passthrough_placeholder(self):
        """Regression: process_mcp_request() returns an empty UserAPIKeyAuth()
        for OAuth2 upstream-token passthrough.  The None check alone is not
        sufficient — the empty placeholder is truthy but has no api_key, no
        token, and an empty models list.  can_key_call_model() would treat
        that as all-model access, letting an OAuth-only user trigger sampling
        calls on any proxy model without a LiteLLM key or budget."""
        # Simulate the empty placeholder from process_mcp_request()
        auth = _make_user_api_key_auth(
            models=[],
            api_key=None,
            token=None,
            user_role=None,
        )

        result = await _check_model_access("gpt-4o", user_api_key_auth=auth)

        # Must be denied — not passed through to can_key_call_model
        assert result is not None
        assert result.code == -1
        assert "sampling requires a valid LiteLLM" in result.message

    @pytest.mark.asyncio
    async def test_should_allow_proxy_admin_even_without_api_key(self):
        """Proxy admins may not have a traditional api_key but should still
        be allowed to use sampling."""
        auth = _make_user_api_key_auth(
            models=[],
            api_key=None,
            token=None,
            user_role="proxy_admin",
        )

        with patch(
            "litellm.proxy.auth.auth_checks.can_key_call_model",
            new_callable=AsyncMock,
            return_value=True,
        ):
            result = await _check_model_access("gpt-4o", user_api_key_auth=auth)

        assert result is None


# ---------------------------------------------------------------------------
# handle_sampling_create_message — auth + budget gating
# ---------------------------------------------------------------------------


class TestSamplingAuthAndBudgetGating:

    @pytest.mark.asyncio
    async def test_should_deny_when_no_auth_context(self):
        """Sampling must reject calls with no user_api_key_auth."""
        from litellm.proxy._experimental.mcp_server.sampling_handler import (
            handle_sampling_create_message,
        )

        params = MagicMock()
        params.model_preferences = None
        params.messages = []
        params.systemPrompt = None
        params.max_tokens = 100
        params.temperature = None
        params.stop_sequences = None
        params.tools = None
        params.tool_choice = None
        params.metadata = None

        result = await handle_sampling_create_message(
            context=MagicMock(),
            params=params,
            default_model="gpt-4o",
            user_api_key_auth=None,
        )

        assert result is not None
        assert result.code == -1
        assert "authenticated" in result.message.lower()

    @pytest.mark.asyncio
    async def test_should_run_budget_checks(self):
        """Sampling must call _run_budget_checks after model access check."""
        from litellm.proxy._experimental.mcp_server.sampling_handler import (
            handle_sampling_create_message,
        )

        auth = _make_user_api_key_auth(models=["gpt-4o"])
        params = MagicMock()
        params.model_preferences = None
        params.messages = []
        params.systemPrompt = None
        params.max_tokens = 100
        params.temperature = None
        params.stop_sequences = None
        params.tools = None
        params.tool_choice = None
        params.metadata = None

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.sampling_handler._check_model_access",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.sampling_handler._run_budget_checks",
                new_callable=AsyncMock,
                return_value=None,
            ) as mock_budget,
            patch(
                "litellm.proxy._experimental.mcp_server.sampling_handler._resolve_model_from_preferences",
                return_value="gpt-4o",
            ),
            patch(
                "litellm.proxy.proxy_server.llm_router",
                new=None,
            ),
            patch(
                "litellm.acompletion",
                new_callable=AsyncMock,
                return_value=MagicMock(
                    choices=[
                        MagicMock(
                            message=MagicMock(content="hi", tool_calls=None),
                            finish_reason="stop",
                        )
                    ],
                    model="gpt-4o",
                ),
            ),
        ):
            await handle_sampling_create_message(
                context=MagicMock(),
                params=params,
                default_model="gpt-4o",
                user_api_key_auth=auth,
            )

        mock_budget.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_should_deny_over_budget_caller(self):
        """When _run_budget_checks returns ErrorData, sampling must return it."""
        from litellm.proxy._experimental.mcp_server.sampling_handler import (
            handle_sampling_create_message,
        )

        auth = _make_user_api_key_auth(models=["gpt-4o"])
        params = MagicMock()
        params.model_preferences = None
        params.messages = []
        params.systemPrompt = None
        params.max_tokens = 100
        params.temperature = None
        params.stop_sequences = None
        params.tools = None
        params.tool_choice = None
        params.metadata = None

        budget_error = ErrorData(code=-1, message="ExceededBudget: over limit")

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.sampling_handler._check_model_access",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.sampling_handler._run_budget_checks",
                new_callable=AsyncMock,
                return_value=budget_error,
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.sampling_handler._resolve_model_from_preferences",
                return_value="gpt-4o",
            ),
        ):
            result = await handle_sampling_create_message(
                context=MagicMock(),
                params=params,
                default_model="gpt-4o",
                user_api_key_auth=auth,
            )

        assert result is budget_error
        assert "ExceededBudget" in result.message


async def _run_budget_checks_for_team(
    *,
    team_id: str | None = "t1",
    team_models: list[str] | None = None,
    team_max_budget: float | None = None,
    team_spend: float | None = None,
    team_lookup_error: Exception | None = None,
    team_object: LiteLLM_TeamTableCachedObj | None = None,
    general_settings: dict[str, object] | None = None,
) -> ErrorData | None:
    auth: Final = UserAPIKeyAuth(
        api_key="sk-sampling-test",
        token="hashed",
        user_id="u1",
        team_id=team_id,
        team_models=team_models if team_models is not None else [],
        team_max_budget=team_max_budget,
        team_spend=team_spend,
    )
    prisma_client: Final = MagicMock()
    prisma_client.db.litellm_teammembership.find_unique = AsyncMock(return_value=None)
    proxy_logging_obj: Final = MagicMock()
    proxy_logging_obj.budget_alerts = AsyncMock()

    with (
        patch("litellm.proxy.proxy_server.prisma_client", new=prisma_client),
        patch("litellm.proxy.proxy_server.user_api_key_cache", new=DualCache()),
        patch("litellm.proxy.proxy_server.proxy_logging_obj", new=proxy_logging_obj),
        patch(
            "litellm.proxy.proxy_server.general_settings",
            new=general_settings or {},
        ),
        patch("litellm.proxy.proxy_server.llm_router", new=None),
        patch(
            "litellm.proxy.auth.auth_checks.get_team_object",
            new_callable=AsyncMock,
            side_effect=team_lookup_error,
            return_value=team_object,
        ),
        patch(
            "litellm.proxy.auth.auth_checks.get_user_object",
            new_callable=AsyncMock,
            return_value=LiteLLM_UserTable(user_id="u1", max_budget=1.0, spend=5.0),
        ),
        patch(
            "litellm.proxy.proxy_server.get_current_spend",
            new_callable=AsyncMock,
            side_effect=lambda **kwargs: kwargs["fallback_spend"],
        ),
    ):
        return await _run_budget_checks("gpt-4o", auth)


@pytest.mark.asyncio
async def test_sampling_budget_uses_vouched_team_context_instead_of_personal_budget():
    result: Final = await _run_budget_checks_for_team(
        team_models=["gpt-4o"],
        team_max_budget=100.0,
        team_spend=0.0,
        team_lookup_error=HTTPException(status_code=404),
    )

    assert result is None


@pytest.mark.asyncio
async def test_sampling_budget_enforces_vouched_team_budget():
    result: Final = await _run_budget_checks_for_team(
        team_models=["gpt-4o"],
        team_max_budget=10.0,
        team_spend=50.0,
        team_lookup_error=HTTPException(status_code=404),
    )

    assert (
        result is not None
        and "Team=t1" in result.message
        and "Max budget: 10.0" in result.message
        and "User=u1" not in result.message
    )


@pytest.mark.asyncio
async def test_sampling_budget_rejects_team_not_found_without_personal_budget_denial():
    result: Final = await _run_budget_checks_for_team(
        team_models=["gpt-4o"],
        team_lookup_error=TeamNotFoundError(team_id="t1"),
    )

    assert result is not None and "t1" in result.message and "User=u1" not in result.message


@pytest.mark.asyncio
async def test_sampling_budget_denies_without_db_unavailable_fallback_setting():
    result: Final = await _run_budget_checks_for_team(
        team_lookup_error=HTTPException(status_code=404),
    )

    assert result is not None and "Sampling denied:" in result.message and "User=u1" not in result.message


@pytest.mark.asyncio
async def test_sampling_budget_allows_with_db_unavailable_fallback_setting():
    result: Final = await _run_budget_checks_for_team(
        team_lookup_error=HTTPException(status_code=404),
        general_settings={"allow_requests_on_db_unavailable": True},
    )

    assert result is None


@pytest.mark.asyncio
async def test_sampling_budget_vouches_for_ui_team_after_team_not_found():
    result: Final = await _run_budget_checks_for_team(
        team_id=UI_TEAM_ID,
        team_lookup_error=TeamNotFoundError(team_id=UI_TEAM_ID),
    )

    assert result is None


@pytest.mark.asyncio
async def test_sampling_budget_keeps_personal_key_budget_enforcement():
    result: Final = await _run_budget_checks_for_team(team_id=None)

    assert result is not None and "User=u1" in result.message


@pytest.mark.asyncio
async def test_sampling_budget_accepts_resolved_team_under_budget():
    result: Final = await _run_budget_checks_for_team(
        team_models=["gpt-4o"],
        team_object=LiteLLM_TeamTableCachedObj(
            team_id="t1",
            models=["gpt-4o"],
            max_budget=100.0,
            spend=0.0,
        ),
    )

    assert result is None
