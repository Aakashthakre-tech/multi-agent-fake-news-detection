import pytest

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: marks tests as real-model integration tests (may require network/GPU)"
    )
