from dataclasses import dataclass
from typing import List, Optional

from bet_edge.dataclasses.dataclass_from_dict import dataclass_from_dict
from bet_edge.odds_api.api_client import (
    EventResult,
    EventIdResult,
    HistoricalEventResult,
    HistoricalEventIdResult,
    SportResult,
)


@dataclass
class Sport:
    """
    Represents a sport in the API.

    Attributes:
        key (str): Unique identifier for the sport.
        group (str): The group to which the sport belongs.
        title (str): The name of the sport.
        description (str): A detailed description of the sport.
        active (bool): Whether the sport is currently active.
        has_outrights (bool): Indicates if the sport has outright markets.
    """

    key: str
    group: str
    title: str
    description: str
    active: bool
    has_outrights: bool


@dataclass
class EventId:
    """
    Represents an event identifier in the API.

    Attributes:
        id (str): Unique identifier for the event.
        sport_key (str): Key for the associated sport.
        sport_title (str): Title of the associated sport.
        commence_time (str): Start time of the event.
        home_team (str): Name of the home team.
        away_team (str): Name of the away team.
    """

    id: str
    sport_key: str
    sport_title: str
    commence_time: str
    home_team: str
    away_team: str


@dataclass
class Outcome:
    """
    Represents an outcome in a market.

    Attributes:
        name (str): Name of the outcome.
        description (str): Description of the outcome.
        price (float): The odds of the outcome.
        point (Optional[float]): An optional point spread or total, if applicable.
    """

    name: str
    description: str
    price: float
    point: Optional[float] = None


@dataclass
class Market:
    """
    Represents a market offered by a bookmaker.

    Attributes:
        key (str): Unique identifier for the market.
        last_update (str): Timestamp of the last update.
        outcomes (List[Outcome]): List of outcomes in the market.
    """

    key: str
    last_update: str
    outcomes: List[Outcome]


@dataclass
class Bookmaker:
    """
    Represents a bookmaker providing markets for an event.

    Attributes:
        key (str): Unique identifier for the bookmaker.
        title (str): Name of the bookmaker.
        markets (List[Market]): List of markets offered by the bookmaker.
    """

    key: str
    title: str
    markets: List[Market]


@dataclass
class Event:
    """
    Represents a sporting event.

    Attributes:
        id (str): Unique identifier for the event.
        sport_key (str): Key for the associated sport.
        sport_title (str): Title of the associated sport.
        commence_time (str): Start time of the event.
        home_team (str): Name of the home team.
        away_team (str): Name of the away team.
        bookmakers (List[Bookmaker]): List of bookmakers offering markets for the event.
    """

    id: str
    sport_key: str
    sport_title: str
    commence_time: str
    home_team: str
    away_team: str
    bookmakers: List[Bookmaker]


@dataclass
class HistoricalEvent:
    """
    Represents a historical event with timestamp data.

    Attributes:
        timestamp (str): Current timestamp of the historical data.
        previous_timestamp (str): Previous timestamp in the data sequence.
        next_timestamp (str): Next timestamp in the data sequence.
        data (Event): Event data for the timestamp.
    """

    timestamp: str
    previous_timestamp: str
    next_timestamp: str
    data: Event


@dataclass
class HistoricalEventIds:
    """
    Represents historical event identifiers.

    Attributes:
        timestamp (str): Current timestamp of the historical data.
        previous_timestamp (str): Previous timestamp in the data sequence.
        next_timestamp (str): Next timestamp in the data sequence.
        data (List[EventId]): List of event IDs for the timestamp.
    """

    timestamp: str
    previous_timestamp: str
    next_timestamp: str
    data: List[EventId]


def process_sport_response(response: SportResult) -> List[Sport]:
    """
    Converts a SportResult API response into a list of Sport instances.

    Args:
        response (SportResult): The API response containing sport data.

    Returns:
        List[Sport]: A list of Sport dataclass instances.
    """
    return [dataclass_from_dict(Sport, item) for item in response]


def process_event_id_response(response: EventIdResult) -> List[EventId]:
    """
    Converts an EventIdResult API response into a list of EventId instances.

    Args:
        response (EventIdResult): The API response containing event ID data.

    Returns:
        List[EventId]: A list of EventId dataclass instances.
    """
    return [dataclass_from_dict(EventId, item) for item in response]


def process_historical_event_id_response(response: HistoricalEventIdResult) -> HistoricalEventIds:
    """
    Converts a HistoricalResult API response into a HistoricalEventIds instance.

    Args:
        response (HistoricalResult): The API response containing historical event ID data.

    Returns:
        HistoricalEventIds: A HistoricalEventIds dataclass instance.
    """
    return dataclass_from_dict(HistoricalEventIds, response)


def process_event_response(response: EventResult) -> Event:
    """
    Converts an EventResult API response into an Event instance.

    Args:
        response (EventResult): The API response containing event data.

    Returns:
        Event: An Event dataclass instance.
    """
    return dataclass_from_dict(Event, response)


def process_historical_event_response(response: HistoricalEventResult) -> HistoricalEvent:
    """
    Converts a HistoricalEventResult API response into a HistoricalEvent instance.

    Args:
        response (HistoricalEventResult): The API response containing historical event data.

    Returns:
        HistoricalEvent: A HistoricalEvent dataclass instance.
    """
    return dataclass_from_dict(HistoricalEvent, response)
