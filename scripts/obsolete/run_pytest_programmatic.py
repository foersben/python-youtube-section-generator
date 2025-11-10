"""Programmatically run pytest and print result code and captured output.

This helper is used by the CI assistant to determine pytest status reliably.
"""
import sys
import pytest

if __name__ == "__main__":
    # Run pytest with -q for concise output
    ret = pytest.main(["-q", "tests"])
    print(f"PYTEST_RETURN_CODE={ret}")
    sys.exit(ret)

