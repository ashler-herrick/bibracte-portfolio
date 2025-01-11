class ExecutionHandler:
    def execute_order(self, order_event, market_data):
        """
        Simulate or process the execution of an order.
        For a backtest, you typically fill instantly at the next bar's open/close, or same bar's close, etc.
        """
        # e.g. fill at current close price
        fill_price = market_data["close"]
        # build FillEvent
        # fill_event = FillEvent(symbol=order_event.symbol, fill_price=fill_price, ...)
        return None
