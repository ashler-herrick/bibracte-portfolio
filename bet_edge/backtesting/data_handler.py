import polars as pl

class DataHandler:
    def __init__(self, df: pl.DataFrame, timestamp: str):
        self.data = self.data.sort(by=timestamp)
        
        # Keep a pointer to the current row
        self.current_index = 0
        self.end_index = self.data.height  # number of rows

    def get_next_bar(self):
        """
        Returns the next bar (row) of data as a dictionary or a row object.
        """
        if self.current_index < self.end_index:
            bar = self.data[self.current_index, :]
            self.current_index += 1
            return bar
        else:
            return None  # signals end of data
