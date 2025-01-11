import polars as pl

from .data_handler import DataHandler
from .strategy import Strategy
from .portfolio import Portfolio
from .execution_handler import ExecutionHandler

def run_backtest(df: pl.DataFrame, strategy: Strategy):
    # 1. Initialize components
    data_handler = DataHandler(df, timestamp='ts_recv')
    portfolio = Portfolio(initial_capital=100000.0)
    execution_handler = ExecutionHandler()

    # 2. Main backtest loop
    while True:
        bar = data_handler.get_next_bar()
        if bar is None:
            # No more data
            break
        
        # Create MarketEvent and pass to strategy
        signal_event = strategy.calculate_signals(bar)
        
        if signal_event is not None:
            # Pass signal to portfolio
            order_event = portfolio.on_signal(signal_event)
            
            if order_event is not None:
                # Pass order to execution handler
                fill_event = execution_handler.execute_order(order_event, bar)
                if fill_event is not None:
                    # Update portfolio
                    portfolio.on_fill(fill_event)
    
    # 3. Final results
    print("Final portfolio value:", portfolio.total_equity)
    # Additional performance metrics...
