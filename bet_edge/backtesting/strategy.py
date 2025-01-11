from abc import ABC, abstractmethod
from typing import Dict, Any

import polars as pl

class Strategy(ABC):
    def __init__(self, name: str, parameters:Dict[str, Any], data: Dict[str, pl.DataFrame], securities: Dict[str, str]):
        self.parameters = parameters
        self.data = data
        self.name = name
        self.securities = securities


    @abstractmethod
    def calculate_signals(self):
        """
        Use the current market data to decide if there's a signal.
        Return None if no signal.
        Otherwise, return SignalEvent(...) 
        """
        pass
