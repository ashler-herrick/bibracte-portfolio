import pytest
from unittest.mock import patch, MagicMock
from bet_edge.odds_api.api_client import APIClient


class MockCredentialProvider:
    def get_credentials(self):
        return {"odds_api_key": "fake_api_key"}


@pytest.fixture
def mock_credentials():
    return MockCredentialProvider()


@pytest.mark.parametrize(
    "mocked_response",
    [
        ([{"key": "americanfootball_nfl", "active": True, "group": "American Football"}]),
        ([]),  # test empty list case
    ],
)
@patch("requests.get")
def test_fetch_sports(mock_get, mocked_response, mock_credentials):
    # Set up the mock response object
    mock_response = MagicMock()
    mock_response.json.return_value = mocked_response
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    # Instantiate your client
    client = APIClient(credential_manager=mock_credentials)

    # Call the method
    result = client.fetch_sports()

    # Assertions
    assert result == mocked_response
    mock_get.assert_called_once_with(
        "https://api.the-odds-api.com/v4/sports/", headers={}, params={"api_key": "fake_api_key"}
    )


@patch("requests.get")
def test_fetch_current_event_ids(mock_get, mock_credentials):
    # Mock response data
    mocked_response = [
        {
            "id": "event_1",
            "sport_key": "americanfootball_nfl",
            "commence_time": "2024-12-06T20:00:00Z",
            "home_team": "Team A",
            "away_team": "Team B",
        }
    ]

    # Set up the mock response object
    mock_response = MagicMock()
    mock_response.json.return_value = mocked_response
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    # Instantiate client with a given sport_code
    client = APIClient(credential_manager=mock_credentials, sport_code="americanfootball_nfl")
    result = client.fetch_current_event_ids()

    # Assertions
    assert result == mocked_response
    mock_get.assert_called_once_with(
        "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events",
        headers={},
        params={"api_key": "fake_api_key", "regions": "us", "oddsFormat": "decimal", "dateFormat": "iso"},
    )


@patch("requests.get")
def test_fetch_historical_event_ids(mock_get, mock_credentials):
    # Mock response data
    mocked_response = {"date": "2024-12-06T00:00:00Z", "events": []}

    # Set up mock
    mock_response = MagicMock()
    mock_response.json.return_value = mocked_response
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    client = APIClient(credential_manager=mock_credentials, sport_code="americanfootball_nfl")

    # The method under test formats the date before calling the API
    result = client.fetch_historical_event_ids("2024-12-06")

    # Assertions
    assert result == mocked_response
    # Check correct URL and params
    mock_get.assert_called_once_with(
        "https://api.the-odds-api.com/v4/historical/sports/americanfootball_nfl/events",
        headers={},
        params={
            "api_key": "fake_api_key",
            "regions": "us",
            "oddsFormat": "decimal",
            "dateFormat": "iso",
            "date": "2024-12-06T00:00:00Z",
        },
    )


@patch("requests.get")
def test_fetch_data_for_current_event(mock_get, mock_credentials):
    # Mock response data
    mocked_response = {
        "id": "event_1",
        "sport_title": "NFL",
        "commence_time": "2024-12-08T18:00:00Z",
        "home_team": "Minnesota Vikings",
        "away_team": "Atlanta Falcons",
        "bookmakers": [
            {
                "key": "draftkings",
                "title": "DraftKings",
                "markets": [
                    {
                        "key": "h2h",
                        "last_update": "2024-12-06T21:23:52Z",
                        "outcomes": [
                            {"name": "Atlanta Falcons", "price": 2.95},
                            {"name": "Minnesota Vikings", "price": 1.42},
                        ],
                    }
                ],
            }
        ],
    }

    # Set up mock
    mock_response = MagicMock()
    mock_response.json.return_value = mocked_response
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    client = APIClient(credential_manager=mock_credentials, sport_code="americanfootball_nfl")

    result = client.fetch_data_for_current_event("event_1", "h2h")

    # Assertions
    assert result == mocked_response
    # Check correct URL and params
    mock_get.assert_called_once_with(
        "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/event_1/odds",
        headers={},
        params={
            "api_key": "fake_api_key",
            "regions": "us",
            "oddsFormat": "decimal",
            "dateFormat": "iso",
            "markets": "h2h",
        },
    )


@patch("requests.get")
def test_fetch_data_for_historical_event(mock_get, mock_credentials):
    # Mock response data
    mocked_response = {
        "timestamp": "2024-12-06T00:00:00Z",
        "data": {
            "id": "event_1",
            "sport_title": "NFL",
            "commence_time": "2024-12-08T18:00:00Z",
            "home_team": "Minnesota Vikings",
            "away_team": "Atlanta Falcons",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "title": "DraftKings",
                    "markets": [
                        {
                            "key": "h2h",
                            "last_update": "2024-12-06T21:23:52Z",
                            "outcomes": [
                                {"name": "Atlanta Falcons", "price": 2.95},
                                {"name": "Minnesota Vikings", "price": 1.42},
                            ],
                        }
                    ],
                }
            ],
        },
    }

    # Set up mock
    mock_response = MagicMock()
    mock_response.json.return_value = mocked_response
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    client = APIClient(credential_manager=mock_credentials, sport_code="americanfootball_nfl")

    result = client.fetch_data_for_historical_event("event_1", "2024-12-06", "h2h")

    # Assertions
    assert result == mocked_response
    # Check correct URL and params
    mock_get.assert_called_once_with(
        "https://api.the-odds-api.com/v4/historical/sports/americanfootball_nfl/events/event_1/odds",
        headers={},
        params={
            "api_key": "fake_api_key",
            "regions": "us",
            "oddsFormat": "decimal",
            "dateFormat": "iso",
            "markets": "h2h",
            "date": "2024-12-06T00:00:00Z",
        },
    )
