"""Run quota-free diagnostics with python -m pydantic_ai_claude_code."""

import json

import anyio

from .diagnostics import doctor


def main() -> None:
    """Print SDK diagnostics without sending a model request."""
    report = anyio.run(doctor)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['ok'] else 1)


if __name__ == '__main__':
    main()
