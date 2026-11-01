"""Tests cannot invoke a model unless they explicitly opt into a fake transport."""

from collections.abc import Iterator

import pytest
from pydantic_ai.models import override_allow_model_requests


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture(autouse=True)
def disable_model_requests() -> Iterator[None]:
    with override_allow_model_requests(False):
        yield
