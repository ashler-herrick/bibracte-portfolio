"""
API Client for Odds API

This module provides the `APIClient` class, which acts as a wrapper around the Odds API.
It facilitates fetching data for sports, events, and odds with a simple interface, and
handles date formatting and HTTP request execution.

Classes:
    APIClient: A client to interact with the Odds API using HTTP requests.

Typings:
    OddsAPIResult: Alias for a list of dictionaries representing the API response.
"""

import logging
import requests
from datetime import datetime
from typing import Optional, Dict, Any, List

from bet_edge.data_io.interfaces import ICredentialProvider

logger = logging.getLogger(__name__)

EventIdResult = List[Dict[str, str]]
SportResult = EventIdResult
EventResult = Dict[str, Any]
HistoricalEventResult = Dict[str, EventResult]
HistoricalEventIdResult = Dict[str, EventIdResult]


class APIClient:
    """
    A client for interacting with the Odds API to fetch sports data, current and historical events,
    and odds for specific sports or events.

    Attributes:
        credential_manager (ICredentialProvider): Provides API credentials.
        sport_code (str): Code representing the sport to query.
        regions (str): Region to query for odds data. Default is 'us'.
        odds_format (str): Format of the odds, either 'decimal' or 'american'. Default is 'decimal'.
        date_format (str): Format of the date, e.g., 'iso'. Default is 'iso'.
        headers (Dict[str, str]): Optional headers to include in API requests.
    """

    def __init__(
        self,
        credential_manager: ICredentialProvider,
        sport_code: str = "",
        regions: str = "us",
        odds_format: str = "decimal",
        date_format: str = "iso",
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Initializes the APIClient with the given credentials and configuration.

        Args:
            credential_manager (ICredentialProvider): Provides the API credentials.
            sport_code (str): Sport code to query for data. Default is ''.
            regions (str): Region to query for odds data. Default is 'us'.
            odds_format (str): Format of the odds, e.g., 'decimal' or 'american'. Default is 'decimal'.
            date_format (str): Format of the date, e.g., 'iso'. Default is 'iso'.
            headers (Dict[str, str], optional): Additional headers to include in requests.
        """
        self.credential_manager = credential_manager
        self.regions = regions
        self.odds_format = odds_format
        self.date_format = date_format
        self.headers = headers if headers else {}
        self.sport_code = sport_code

        creds = self.credential_manager.get_credentials()
        self.api_key = creds["odds_api_key"]
        self.current_events_url = "https://api.the-odds-api.com/v4/sports/{sport_code}/events"
        self.current_odds_url = "https://api.the-odds-api.com/v4/sports/{sport_code}/events/{event_id}/odds"
        self.historical_odds_url = (
            "https://api.the-odds-api.com/v4/historical/sports/{sport_code}/events/{event_id}/odds"
        )
        self.historical_events_url = "https://api.the-odds-api.com/v4/historical/sports/{sport_code}/events"
        self.sports_url = "https://api.the-odds-api.com/v4/sports/"

    def _format_date(self, date: str) -> str:
        """
        Formats a date string into ISO 8601 format.

        Args:
            date (str): Date string in 'YYYY-MM-DD' format.

        Returns:
            str: Date in ISO 8601 format with time set to '00:00:00Z'.

        Raises:
            ValueError: If the date is not in the correct format.
        """
        try:
            parsed_date = datetime.strptime(date, "%Y-%m-%d")
            date_iso = parsed_date.strftime("%Y-%m-%dT00:00:00Z")
        except ValueError as e:
            logger.error("Date must be in 'YYYY-MM-DD' format.")
            raise e

        return date_iso

    def get_response(self, url: str, params: Dict) -> Any:
        """
        Executes an HTTP GET request and handles errors.

        Args:
            url (str): URL for the API request.
            params (Dict): Query parameters to include in the request.

        Returns:
            Any: The JSON-decoded response data.

        Logs:
            Errors encountered during the request.
        """
        response = requests.Response()
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err} - Response: {response.text}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch events: {e}")

        data = response.json()
        return data

    def fetch_sports(self) -> SportResult:
        """
        Fetches a list of available sports from the Odds API.

        Returns:
            OddsAPIResult: A list of dictionaries representing available sports.
        """
        params = {"api_key": self.api_key}
        data = self.get_response(self.sports_url, params)
        return data

    def fetch_historical_event_ids(self, date: str) -> HistoricalEventIdResult:
        """
        Fetches historical events for a given sport and date.

        Args:
            date (str): Date of events to fetch in 'YYYY-MM-DD' format.

        Returns:
            OddsAPIResult: A list of dictionaries with historical event information.
        """
        date_iso = self._format_date(date)
        event_url = self.historical_events_url.format(sport_code=self.sport_code)
        params = {
            "api_key": self.api_key,
            "regions": self.regions,
            "oddsFormat": self.odds_format,
            "dateFormat": self.date_format,
            "date": date_iso,
        }

        data = self.get_response(event_url, params)
        return data

    def fetch_current_event_ids(self) -> EventIdResult:
        """
        Fetches current events for a given sport.

        Returns:
            OddsAPIResult: A list of dictionaries with event information.
        """
        event_url = self.current_events_url.format(sport_code=self.sport_code)
        params = {
            "api_key": self.api_key,
            "regions": self.regions,
            "oddsFormat": self.odds_format,
            "dateFormat": self.date_format,
        }

        data = self.get_response(event_url, params)
        return data

    def fetch_data_for_historical_event(self, event_id: str, date: str, markets: str) -> HistoricalEventResult:
        """
        Fetches historical odds data for a specific event.

        Args:
            event_id (str): ID of the event to fetch data for.
            date (str): Date of the event in 'YYYY-MM-DD' format.
            markets (str): Markets to fetch odds for.

        Returns:
            OddsAPIResult: A list of dictionaries with event odds data.
        """
        date_iso = self._format_date(date)
        event_url = self.historical_odds_url.format(event_id=event_id, sport_code=self.sport_code)
        params = {
            "api_key": self.api_key,
            "regions": self.regions,
            "oddsFormat": self.odds_format,
            "dateFormat": self.date_format,
            "markets": markets,
            "date": date_iso,
        }
        event_data = self.get_response(event_url, params)
        return event_data

    def fetch_data_for_current_event(self, event_id: str, markets: str) -> EventResult:
        """
        Fetches odds data for a current event.

        Args:
            event_id (str): ID of the event to fetch data for.
            markets (str): Markets to fetch odds for.

        Returns:
            OddsAPIResult: A list of dictionaries with event odds data.
        """
        event_url = self.current_odds_url.format(event_id=event_id, sport_code=self.sport_code)
        params = {
            "api_key": self.api_key,
            "regions": self.regions,
            "oddsFormat": self.odds_format,
            "dateFormat": self.date_format,
            "markets": markets,
        }

        event_data = self.get_response(event_url, params)
        return event_data
