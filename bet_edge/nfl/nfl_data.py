# nfl_data.py

import polars as pl
import numpy as np
import logging
from typing import List, Optional

from bet_edge.nfl.nfl_data_manager import DataManager
import bet_edge.nfl.weekly as nw
import bet_edge.nfl.injury as ni
from sports_data import SportsData

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class NFLData(SportsData):
    """
    NFL-specific implementation of SportsData.
    """

    def __init__(self, target: str):
        super().__init__(target)
        self.data_manager = DataManager()
        self.base_df = pl.from_pandas(self.data_manager.weekly_data)
        self.inj_df = pl.from_pandas(self.data_manager.injury_data)

    def _add_stats(
        self,
        df: pl.DataFrame,
        id_cols: List[str],
        cols: List[str],
        attr: str,
    ) -> pl.DataFrame:
        """
        Adds statistical features to the dataframe using utility functions from weekly.py.
        """
        if attr == "defense":
            roll_id = ["opponent_team"]
        elif attr == "player":
            roll_id = ["player_id"]
        else:
            logger.warning("Unknown attr passed to _add_stats. Output will possibly contain duplicates.")
        logger.info(f"Adding {attr} statistics")
        df = nw.add_prev_season_stats(df, id_cols, cols, attr)
        df = nw.add_curr_season_stats(df, id_cols, cols, attr)
        df = nw.fill_nan_curr_season_stats(df, cols, attr)
        df = nw.add_rolling_stats(df, id_cols, cols, attr, 3)
        df = nw.add_rolling_stats(df, roll_id, cols, attr, 10)
        df = nw.add_rolling_value_over_avg(df, roll_id, ["position"], cols, attr, window=12)
        logger.info(f"{attr} statistics added")
        return df

    def _add_defensive_data(self, df: pl.DataFrame, id_cols: List[str], cols: List[str], attr: str) -> pl.DataFrame:
        """
        Adds defensive statistics to the dataframe.
        """
        logger.info("Adding defensive data")
        df = self._add_stats(df, id_cols, cols, attr)
        return df

    def _add_player_data(self, df: pl.DataFrame, id_cols: List[str], cols: List[str], attr: str) -> pl.DataFrame:
        """
        Adds player-specific statistics to the dataframe.
        """
        logger.info("Adding player-specific data")
        df = self._add_stats(df, id_cols, cols, attr)
        return df

    def _add_injury_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Adds injury-related features to the dataframe using functions from injury.py.
        """
        logger.info("Adding injury data")
        try:
            df = ni.add_injury_to_wkly_df(df, self.inj_df)
            df = ni.add_offense_injury_indicators(df, self.inj_df)
            logger.info("Injury data added successfully")
        except Exception as e:
            logger.error("Failed to add injury data", exc_info=True)
            raise RuntimeError("Injury data addition failed") from e
        return df

    def _get_features(self, df: pl.DataFrame) -> List[str]:
        """
        Identifies new feature columns added during processing.
        """
        new_cols = nw._get_new_cols(self.base_df, df)
        return new_cols + ["age", "years_exp", "week"]

    def prepare_training_data(
        self,
        positions: List[str],
        player_cols: List[str],
        defense_cols: List[str],
        exclude_seasons: Optional[List[int]] = None,
    ) -> None:
        """
        Prepares the training data by filtering, adding necessary features, and scaling.
        """
        logger.info("Preparing training data")
        df = self.base_df.clone()
        df = df.filter(pl.col("position").is_in(positions))
        logger.info(f"Filtered positions: {positions}")

        # Add player-specific data
        df = self._add_player_data(df, ["player_id", "season"], player_cols, "player")

        # Add defensive data
        df = self._add_defensive_data(df, ["opponent_team", "season"], defense_cols, "defense")

        # Add injury data
        df = self._add_injury_data(df)

        # Create X and y
        exclude_conditions = {"rookie_flag": 1}
        self._create_X_y(df, exclude_conditions=exclude_conditions, target_min=1.0)

        # Scale the features
        self.scale_input()
        logger.info("Training data prepared successfully")

    def prepare_prediction_data(
        self,
        season: int,
        week: int,
    ) -> pl.DataFrame:
        """
        Prepares the prediction dataframe for all players for a specific season and week.
        """
        logger.info(f"Preparing prediction data for Season {season}, Week {week}")
        prediction_df = pl.from_pandas(self.data_manager.get_prediction_df(season, week))
        prediction_df = prediction_df.filter(pl.col("position").is_in(self.positions))
        # Add player-specific data
        prediction_df = self._add_player_data(prediction_df, ["player_id", "season"], self.player_cols, "player")

        # Add defensive data
        prediction_df = self._add_defensive_data(
            prediction_df, ["opponent_team", "season"], self.defense_cols, "defense"
        )

        # Add injury data
        prediction_df = self._add_injury_data(prediction_df)

        # Since it's prediction, target is not available
        prediction_df = prediction_df.with_column(pl.lit(np.nan).alias(self.target))

        # Identify features
        self.features = self._get_features(prediction_df)
        logger.debug(f"Features identified for prediction: {self.features}")

        # Scale the data using the fitted scaler
        if self.scaler is None:
            raise RuntimeError("Scaler has not been fitted. Please prepare training data first.")

        cols_to_scale = self._determine_scaling()
        if cols_to_scale:
            data_to_scale = prediction_df.select(cols_to_scale).to_numpy()
            scaled_data = self.scaler.transform(data_to_scale)
            # Assign scaled data back to DataFrame
            for idx, col in enumerate(cols_to_scale):
                prediction_df = prediction_df.with_column(pl.Series(col, scaled_data[:, idx]))
            logger.info(f"Scaled columns in prediction data: {cols_to_scale}")
        else:
            logger.info("No columns required scaling for prediction data based on variance threshold")

        # Extract feature matrix
        self.X_pred = prediction_df.select(self.features).to_numpy()
        logger.info("Prediction feature matrix created")

        # Return as DataFrame for compatibility
        return prediction_df.select(self.features)


# Now define the specific NFL data classes
class RushYardsData(NFLData):
    """
    Subclass for processing rushing yards data.
    """

    def __init__(self, target: str = "rushing_yards"):
        super().__init__(target)
        self.positions = ["RB", "QB"]
        self.player_cols = [
            "rushing_yards",
            "carries",
            "yards_per_carry",
            "scramble_pct",
        ]
        self.defense_cols = ["rushing_yards", "yards_per_carry"]

    def build_training_data(self) -> None:
        logger.info("Building training data for Rush Yards")
        self.prepare_training_data(self.positions, self.player_cols, self.defense_cols, exclude_seasons=[2009])
        logger.info("Training data for Rush Yards built successfully")


class RecYardsData(NFLData):
    """
    Subclass for processing receiving yards data.
    """

    def __init__(self, target: str = "receiving_yards"):
        super().__init__(target)
        self.positions = ["TE", "WR", "RB"]
        self.player_cols = [
            "receptions",
            "receiving_yards",
            "targets",
            "target_share",
            "air_yards_share",
            "racr",
            "wopr",
        ]
        self.defense_cols = ["receiving_yards", "comp_pct"]

    def build_training_data(self) -> None:
        logger.info("Building training data for Receiving Yards")
        self.prepare_training_data(self.positions, self.player_cols, self.defense_cols, exclude_seasons=[2009])
        logger.info("Training data for Receiving Yards built successfully")


class PassYardsData(NFLData):
    """
    Subclass for processing passing yards data.
    """

    def __init__(self, target: str = "passing_yards"):
        super().__init__(target)
        self.positions = ["QB"]
        self.player_cols = [
            "pacr",
            "dakota",
            "completions",
            "attempts",
            "sacks",
            "passer_rating",
            "comp_pct",
            "passing_yards",
        ]
        self.defense_cols = ["passing_yards", "comp_pct"]

    def build_training_data(self) -> None:
        logger.info("Building training data for Passing Yards")
        self.prepare_training_data(self.positions, self.player_cols, self.defense_cols, exclude_seasons=[2009])
        logger.info("Training data for Passing Yards built successfully")
