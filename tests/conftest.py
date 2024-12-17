import pytest
from bet_edge.logging_setup import setup_logging


@pytest.fixture(scope="session", autouse=True)
def configure_logging():
    setup_logging()
