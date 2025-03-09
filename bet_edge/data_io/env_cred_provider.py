"""
This module provides a concrete implementation of the ICredProvider interface
that loads credentials from environment variables or a .env file.

The EnvCredProvider dynamically assigns attributes based on a predefined
mapping of environment variable names to attribute names. This allows for easy expansion
and dynamic handling of various credential sets, such as AWS or database credentials.
"""

import os
import logging
from dotenv import load_dotenv
from bet_edge.data_io.interfaces import ICredProvider

logger = logging.getLogger(__name__)


class EnvCredProvider(ICredProvider):
    """
    A credential manager that retrieves credentials from environment variables or a .env file.

    The manager dynamically maps a predefined set of environment variables to internal attributes
    and validates the presence of required credentials. It is designed to be easily extensible
    by adding new entries to the `ENV_VARS` dictionary.
    """

    ENV_VARS = {
        #aws
        "aws_access_key_id": "AWS_ACCESS_KEY_ID",
        "aws_secret_access_key": "AWS_SECRET_ACCESS_KEY",
        "aws_session_token": "AWS_SESSION_TOKEN",
        "aws_default_region": "AWS_DEFAULT_REGION",
        #polygon
        "polygon_access_key_id" : "POLYGON_ACCESS_KEY_ID",
        "polygon_secret_access_key": "POLYGON_SECRET_ACCESS_KEY",
        "polygon_api_key": "POLYGON_API_KEY",
        #odds api
        "odds_api_key": "ODDS_API_KEY",
        #pg creds
        "postgres_host": "POSTGRES_HOST",
        "postgres_port": "POSTGRES_PORT",
        "postgres_db": "POSTGRES_DB",
        "postgres_user": "POSTGRES_USER",
        "postgres_password": "POSTGRES_PASSWORD",
    }

    def __init__(self, env_path=None):
        """
        Initializes the EnvCredProvider and loads credentials.

        Parameters:
            env_path (str, optional): The path to the .env file. If not provided, defaults
                                      to a standard relative path to the .env file.

        Notes:
            - If no `env_path` is provided, the module attempts to locate a .env file in a
              default relative path.
            - Environment variables take precedence over .env file values if both are set.
        """
        self.env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
        if env_path:
            self.env_path = env_path
        load_dotenv(dotenv_path=env_path, override=True)

        # Dynamically load credentials from the ENV_VARS mapping
        for attr, env_var in self.ENV_VARS.items():
            value = os.getenv(env_var)
            setattr(self, f"_{attr}", value)  # Store as _attr for consistency

        self._validate_credentials()

    def _validate_credentials(self):
        """
        Validates the presence of required credentials.

        Raises:
            ValueError: If `AWS_ACCESS_KEY_ID` or `AWS_SECRET_ACCESS_KEY` is missing.
        """
        if not getattr(self, "_aws_access_key_id") or not getattr(self, "_aws_secret_access_key"):
            logger.error("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set.")
            raise ValueError("Missing AWS credentials.")

    def get_credentials(self) -> dict:
        """
        Retrieves the loaded credentials as a dictionary.

        Returns:
            dict: A dictionary containing the credentials, with attribute names as keys and
                  their corresponding values. Example:
                  {
                      "aws_access_key_id": "your-access-key-id",
                      "aws_secret_access_key": "your-secret-access-key",
                      "aws_session_token": "your-session-token",
                      "aws_default_region": "your-region"
                  }
        """
        return {attr: getattr(self, f"_{attr}") for attr in self.ENV_VARS}
