import pytest
from bet_edge.setup_logging import setup_logging


@pytest.fixture(scope="session", autouse=True)
def configure_logging():
    setup_logging()
