# tests/test_processing_functions.py

import pytest
from typing import List, Dict, Any

# Import the dataclasses and processing functions
from bet_edge.odds_api.odds_processing import (
    Sport,
    EventId,
    Outcome,
    Market,
    Bookmaker,
    Event,
    HistoricalEvent,
    HistoricalEventIds,
    process_sport_response,
    process_event_id_response,
    process_historical_event_id_response,
    process_event_response,
    process_historical_event_response,
)

# Sample data for testing


@pytest.fixture
def sample_sport_response() -> List[Dict[str, Any]]:
    return [
        {
            "key": "soccer",
            "group": "football",
            "title": "Soccer",
            "description": "Association Football",
            "active": True,
            "has_outrights": False,
        },
        {
            "key": "basketball",
            "group": "basket",
            "title": "Basketball",
            "description": "Basketball Sport",
            "active": True,
            "has_outrights": True,
        },
    ]


@pytest.fixture
def sample_event_id_response() -> List[Dict[str, Any]]:
    return [
        {
            "id": "event1",
            "sport_key": "soccer",
            "sport_title": "Soccer",
            "commence_time": "2024-12-10T15:00:00Z",
            "home_team": "Team A",
            "away_team": "Team B",
        },
        {
            "id": "event2",
            "sport_key": "basketball",
            "sport_title": "Basketball",
            "commence_time": "2024-12-11T18:30:00Z",
            "home_team": "Team C",
            "away_team": "Team D",
        },
    ]


@pytest.fixture
def sample_event_response() -> Dict[str, Any]:
    return {
        "id": "event1",
        "sport_key": "soccer",
        "sport_title": "Soccer",
        "commence_time": "2024-12-10T15:00:00Z",
        "home_team": "Team A",
        "away_team": "Team B",
        "bookmakers": [
            {
                "key": "bookmaker1",
                "title": "Bookmaker One",
                "markets": [
                    {
                        "key": "h2h",
                        "last_update": "2024-12-09T12:00:00Z",
                        "outcomes": [
                            {"name": "Team A", "description": "Win Team A", "price": 1.5},
                            {"name": "Team B", "description": "Win Team B", "price": 2.5},
                        ],
                    }
                ],
            }
        ],
    }


@pytest.fixture
def sample_historical_event_response() -> Dict[str, Any]:
    return {
        "timestamp": "2024-12-09T12:00:00Z",
        "previous_timestamp": "2024-12-08T12:00:00Z",
        "next_timestamp": "2024-12-10T12:00:00Z",
        "data": {
            "id": "event1",
            "sport_key": "soccer",
            "sport_title": "Soccer",
            "commence_time": "2024-12-10T15:00:00Z",
            "home_team": "Team A",
            "away_team": "Team B",
            "bookmakers": [
                {
                    "key": "bookmaker1",
                    "title": "Bookmaker One",
                    "markets": [
                        {
                            "key": "h2h",
                            "last_update": "2024-12-09T12:00:00Z",
                            "outcomes": [
                                {"name": "Team A", "description": "Win Team A", "price": 1.5},
                                {"name": "Team B", "description": "Win Team B", "price": 2.5},
                            ],
                        }
                    ],
                }
            ],
        },
    }


@pytest.fixture
def sample_historical_event_id_response() -> Dict[str, Any]:
    return {
        "timestamp": "2024-12-09T12:00:00Z",
        "previous_timestamp": "2024-12-08T12:00:00Z",
        "next_timestamp": "2024-12-10T12:00:00Z",
        "data": [
            {
                "id": "event1",
                "sport_key": "soccer",
                "sport_title": "Soccer",
                "commence_time": "2024-12-10T15:00:00Z",
                "home_team": "Team A",
                "away_team": "Team B",
            },
            {
                "id": "event2",
                "sport_key": "basketball",
                "sport_title": "Basketball",
                "commence_time": "2024-12-11T18:30:00Z",
                "home_team": "Team C",
                "away_team": "Team D",
            },
        ],
    }


@pytest.fixture
def sample_historical_event_invalid_response() -> Dict[str, Any]:
    return {
        "timestamp": "2024-12-09T14:00:00Z",
        "previous_timestamp": "2024-12-08T14:00:00Z",
        "next_timestamp": "2024-12-10T14:00:00Z",
        "data": {},  # Invalid data, missing required fields
    }


# Test functions


def test_process_sport_response(sample_sport_response):
    sports = process_sport_response(sample_sport_response)
    assert isinstance(sports, list)
    assert len(sports) == 2
    for sport, expected in zip(sports, sample_sport_response):
        assert isinstance(sport, Sport)
        assert sport.key == expected["key"]
        assert sport.group == expected["group"]
        assert sport.title == expected["title"]
        assert sport.description == expected["description"]
        assert sport.active == expected["active"]
        assert sport.has_outrights == expected["has_outrights"]


def test_process_sport_response_empty():
    response = []
    sports = process_sport_response(response)
    assert isinstance(sports, list)
    assert len(sports) == 0


def test_process_event_id_response(sample_event_id_response):
    event_ids = process_event_id_response(sample_event_id_response)
    assert isinstance(event_ids, list)
    assert len(event_ids) == 2
    for event_id, expected in zip(event_ids, sample_event_id_response):
        assert isinstance(event_id, EventId)
        assert event_id.id == expected["id"]
        assert event_id.sport_key == expected["sport_key"]
        assert event_id.sport_title == expected["sport_title"]
        assert event_id.commence_time == expected["commence_time"]
        assert event_id.home_team == expected["home_team"]
        assert event_id.away_team == expected["away_team"]


def test_process_event_id_response_empty():
    response = []
    event_ids = process_event_id_response(response)
    assert isinstance(event_ids, list)
    assert len(event_ids) == 0


def test_process_event_response(sample_event_response):
    event = process_event_response(sample_event_response)
    assert isinstance(event, Event)
    expected = sample_event_response

    assert isinstance(event, Event)
    assert event.id == expected["id"]
    assert event.sport_key == expected["sport_key"]
    assert event.sport_title == expected["sport_title"]
    assert event.commence_time == expected["commence_time"]
    assert event.home_team == expected["home_team"]
    assert event.away_team == expected["away_team"]

    assert isinstance(event.bookmakers, list)
    assert len(event.bookmakers) == 1
    bookmaker = event.bookmakers[0]
    expected_bookmaker = expected["bookmakers"][0]

    assert isinstance(bookmaker, Bookmaker)
    assert bookmaker.key == expected_bookmaker["key"]
    assert bookmaker.title == expected_bookmaker["title"]

    assert isinstance(bookmaker.markets, list)
    assert len(bookmaker.markets) == 1
    market = bookmaker.markets[0]
    expected_market = expected_bookmaker["markets"][0]

    assert isinstance(market, Market)
    assert market.key == expected_market["key"]
    assert market.last_update == expected_market["last_update"]

    assert isinstance(market.outcomes, list)
    assert len(market.outcomes) == 2
    for outcome, expected_outcome in zip(market.outcomes, expected_market["outcomes"]):
        assert isinstance(outcome, Outcome)
        assert outcome.name == expected_outcome["name"]
        assert outcome.description == expected_outcome["description"]
        assert outcome.price == expected_outcome["price"]
        assert outcome.point == expected_outcome.get("point")  # May be None


def test_process_event_response_empty():
    response = {}
    with pytest.raises(KeyError):
        process_event_response(response)


def test_process_historical_event_id_response(sample_historical_event_id_response):
    historical_event_ids = process_historical_event_id_response(sample_historical_event_id_response)
    assert isinstance(historical_event_ids, HistoricalEventIds)
    for event_id, expected in zip(historical_event_ids.data, sample_historical_event_id_response["data"]):
        assert isinstance(event_id, EventId)
        assert event_id.id == expected["id"]
        assert event_id.sport_key == expected["sport_key"]
        assert event_id.sport_title == expected["sport_title"]
        assert event_id.commence_time == expected["commence_time"]
        assert event_id.home_team == expected["home_team"]
        assert event_id.away_team == expected["away_team"]


def test_process_historical_event_id_response_empty():
    response = {}
    with pytest.raises(KeyError):
        process_historical_event_id_response(response)


def test_process_historical_event_response(sample_historical_event_response):
    historical_event = process_historical_event_response(sample_historical_event_response)
    assert isinstance(historical_event, HistoricalEvent)

    event = historical_event.data
    expected_historical = sample_historical_event_response
    expected = expected_historical["data"]

    assert isinstance(event, Event)
    assert event.id == expected["id"]
    assert event.sport_key == expected["sport_key"]
    assert event.sport_title == expected["sport_title"]
    assert event.commence_time == expected["commence_time"]
    assert event.home_team == expected["home_team"]
    assert event.away_team == expected["away_team"]

    assert isinstance(event.bookmakers, list)
    assert len(event.bookmakers) == 1
    bookmaker = event.bookmakers[0]
    expected_bookmaker = expected["bookmakers"][0]

    assert isinstance(bookmaker, Bookmaker)
    assert bookmaker.key == expected_bookmaker["key"]
    assert bookmaker.title == expected_bookmaker["title"]

    assert isinstance(bookmaker.markets, list)
    assert len(bookmaker.markets) == 1
    market = bookmaker.markets[0]
    expected_market = expected_bookmaker["markets"][0]

    assert isinstance(market, Market)
    assert market.key == expected_market["key"]
    assert market.last_update == expected_market["last_update"]

    assert isinstance(market.outcomes, list)
    assert len(market.outcomes) == 2
    for outcome, expected_outcome in zip(market.outcomes, expected_market["outcomes"]):
        assert isinstance(outcome, Outcome)
        assert outcome.name == expected_outcome["name"]
        assert outcome.description == expected_outcome["description"]
        assert outcome.price == expected_outcome["price"]
        assert outcome.point == expected_outcome.get("point")  # May be None
    # Further assertions can be added to check bookmakers, markets, outcomes, etc.


def test_process_historical_event_response_empty():
    response = {}
    with pytest.raises(KeyError):
        process_historical_event_response(response)


def test_process_historical_event_response_missing_data(sample_historical_event_invalid_response):
    with pytest.raises(TypeError):
        process_historical_event_response(sample_historical_event_invalid_response)


def test_process_historical_event_response_invalid_data_type():
    response = [
        {
            "timestamp": "2024-12-09T14:00:00Z",
            "previous_timestamp": "2024-12-08T14:00:00Z",
            "next_timestamp": "2024-12-10T14:00:00Z",
            "data": {"invalid_field": "invalid_value"},
        }
    ]
    with pytest.raises(KeyError):  # Replace with the specific exception
        process_historical_event_response(response)  # type: ignore #
