class Portfolio:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.current_cash = initial_capital
        self.positions = {}  # symbol -> position info
        self.total_equity = initial_capital

    def on_signal(self, signal_event):
        """
        Generate an OrderEvent (or multiple) based on the signal.
        """
        # Evaluate risk, position sizing, etc.
        # For example:
        # order = OrderEvent(symbol=signal_event.symbol, order_type='MKT', quantity=..., direction='BUY' or 'SELL')
        # return order
        return None

    def on_fill(self, fill_event):
        """
        Update your holdings, cash, and PnL based on the fill details.
        """
        # fill_event might have: symbol, quantity, fill_price, fill_cost, commission, etc.
        # Update positions[symbol], current_cash, and recalc total_equity.
        pass
