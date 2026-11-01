"""Subscription classification from the public SDK's initialization response."""

from pydantic import JsonValue, TypeAdapter
from pydantic_ai.exceptions import UserError

_OBJECT = TypeAdapter(dict[str, JsonValue])
_PLANS = frozenset({'pro', 'max', 'ultra', 'team', 'enterprise'})


def subscription_account(info: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Reject unsupported accounts before a generation prompt is sent."""
    raw = info.get('account')
    account = _OBJECT.validate_python(raw, strict=True) if isinstance(raw, dict) else {}
    plan = account.get('subscriptionType')
    normalized = plan.casefold().removeprefix('claude ').strip() if isinstance(plan, str) else ''
    if account.get('apiProvider') != 'firstParty' or normalized not in _PLANS:
        raise UserError('An eligible first-party Claude subscription is required. SDK account: ' + str(account))
    return account
