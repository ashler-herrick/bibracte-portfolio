from bet_edge.probabilistic_models import KFoldDPM
from bet_edge.odds_processing import OddsEntry

class BacktestEvent:
    
    def __init__(self, model: KFoldDPM, odds_entry: OddsEntry):
        self.model = model
        self.odds_entry = odds_entry



class Backtester:
