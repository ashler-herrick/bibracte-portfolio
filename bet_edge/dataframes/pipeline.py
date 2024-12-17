from typing import Callable, List, Any

from bet_edge.dataframes.dataframe_manager import DataFrameManager


class Pipeline:
    def __init__(self, functions: List[Callable[[Any], DataFrameManager]]):
        self.functions = functions

    def execute(self, x: DataFrameManager) -> DataFrameManager:
        for f in self.functions:
            x = f(x)
        return x
