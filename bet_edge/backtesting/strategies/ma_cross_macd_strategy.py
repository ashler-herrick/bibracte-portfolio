from typing import Dict, Any

import polars as pl

from ..strategy import Strategy
from bet_edge.dataframes.ohlcv_time_series import calc_macd, calc_rsi, calc_crossover

df = pl.read_parquet(r"C:\Users\Ashle\OneDrive\Documents\bet_edge\data\es_cont_ohlcv_20160601_to_20250108.parquet")

class MACrossMACDStrategy(Strategy):

    def __init__(self, name: str, parameters:Dict[str, Any], data: Dict[str, pl.DataFrame], securities: Dict[str, str]):
        super().__init__(
             name=name
            ,parameters=parameters
            ,data=data
            ,securities=securities
        )

    def calculate_signals(self):
        df = self.data["ohlcv_1m"]
        df = calc_macd(df=df, col='close',fast_period=25, slow_period=50, signal_period=12)
        df = calc_rsi(df=df, col='close', window=40)
        df = calc_crossover(df, 'ema_fast', 'ema_slow', 'macd_cross_signal')
        df = calc_crossover(df, 'RSI', 58.0, 'high_rsi_signal')
        df = calc_crossover(df, 'RSI', 43.0, 'low_rsi_signal')
        df = df.with_columns([
            pl.when(pl.col("macd_cross_signal") != None)
            .then(pl.col("timestamp"))
            .otherwise(None)
            .alias("ma_cross_ts"),
            
            pl.when(pl.col("macd_signal") == 1)
            .then(pl.col("timestamp"))
            .otherwise(None)
            .alias("macd_ts"),
            
            pl.when(pl.col("rsi_signal") == 1)
            .then(pl.col("timestamp"))
            .otherwise(None)
            .alias("rsi_ts")
        ])