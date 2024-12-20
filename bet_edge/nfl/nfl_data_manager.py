# nfl_data_manager.py

import polars as pl
import numpy as np
from typing import Dict, List
import logging

import nfl_data_py as nfl
import bet_edge.nfl.weekly as nw

from sports_data_manager import (
    DataLoader,
    DataValidator,
    CacheManager,
    PredictionDataGenerator,
    DataManagerBase,
)

# Configure logging
logger = logging.getLogger(__name__)


class NFLDataLoader(DataLoader):
    def load_weekly_data(self, seasons: List[int]) -> pl.DataFrame:
        self.logger.info("Loading weekly data...")
        wkly_df = nfl.import_weekly_data(seasons)
        self.logger.debug(f"Weekly data loaded with shape: {wkly_df.shape}")
        return pl.from_pandas(wkly_df)

    def load_seasonal_rosters(self, seasons: List[int]) -> pl.DataFrame:
        self.logger.info("Loading seasonal rosters...")
        ssnl_roster = nfl.import_seasonal_rosters(seasons)
        self.logger.debug(f"Seasonal rosters loaded with shape: {ssnl_roster.shape}")
        return pl.from_pandas(ssnl_roster)

    def load_injury_data(self, seasons: List[int]) -> pl.DataFrame:
        self.logger.info("Loading injury data...")
        inj_df = nfl.import_injuries(seasons)
        inj_df = inj_df.rename(columns={"gsis_id": "player_id"})
        self.logger.debug(f"Injury data loaded with shape: {inj_df.shape}")
        return pl.from_pandas(inj_df)


class NFLPredictionDataGenerator(PredictionDataGenerator):
    def generate_prediction_df(self, season: int, week: int) -> pl.DataFrame:
        key = (season, week)
        cached_df = self.cache_manager.get_from_cache(key)
        if cached_df is not None:
            self.logger.info(f"Retrieving cached prediction dataframe for Season {season}, Week {week}")
            return cached_df

        self.logger.info(f"Generating prediction dataframe for Season {season}, Week {week}")
        self._validate_season_week(season, week)
        schedule = self._fetch_schedule(season, week)
        previous_week = week - 1
        players_historical = self._get_players_historical(season, previous_week)
        participating_teams = self._get_participating_teams(schedule)
        players_to_predict = self._select_players_to_predict(players_historical, participating_teams)
        team_opponent_map = self._create_team_opponent_map(schedule)
        players_with_opponents = self._assign_opponent_teams(players_to_predict, team_opponent_map)
        prediction_week_df = self._prepare_prediction_week_df(players_with_opponents, season, week)
        comprehensive_df = self._combine_historical_with_prediction(players_historical, prediction_week_df)
        self.cache_manager.save_to_cache(key, comprehensive_df)
        return comprehensive_df

    def _validate_season_week(self, season: int, week: int):
        if season not in self.dates:
            self.logger.error(f"Invalid season: {season}. Available seasons: {self.dates}")
            raise ValueError(f"Invalid season: {season}.")
        if week < 1 or week > 17:
            self.logger.error(f"Invalid week: {week}. Week must be between 1 and 17.")
            raise ValueError(f"Invalid week: {week}. Week must be between 1 and 17.")

    def _fetch_schedule(self, season: int, week: int) -> pl.DataFrame:
        try:
            schedule = nfl.import_schedules([season])
            schedule = schedule[schedule["week"] == week][["home_team", "season", "week", "away_team"]]
            if schedule.empty:
                self.logger.warning(f"No schedule data found for Season {season}, Week {week}.")
                raise ValueError(f"No schedule data found for Season {season}, Week {week}.")
            self.logger.debug(f"Schedule for Season {season}, Week {week}:\n{schedule.head()}")
            return pl.from_pandas(schedule)
        except Exception:
            self.logger.error("Error fetching schedule data", exc_info=True)
            raise

    def _get_players_historical(self, season: int, previous_week: int) -> pl.DataFrame:
        players_historical = self.data.filter(
            ((pl.col("season") == season) & (pl.col("week") <= previous_week)) | (pl.col("season") == season - 1)
        )
        if players_historical.is_empty():
            self.logger.warning(f"No historical player data found for Season {season} up to Week {previous_week}.")
            raise ValueError(f"No historical player data found for Season {season} up to Week {previous_week}.")
        self.logger.debug(
            f"Players' historical data up to Season {season}, Week {previous_week}:\n{players_historical.head()}"
        )
        return players_historical

    def _get_participating_teams(self, schedule: pl.DataFrame) -> set:
        home_teams = schedule.select("home_team").unique().to_series().to_list()
        away_teams = schedule.select("away_team").unique().to_series().to_list()
        participating_teams = set(home_teams).union(set(away_teams))
        self.logger.debug(f"Participating teams in schedule: {participating_teams}")
        return participating_teams

    def _select_players_to_predict(self, players_historical: pl.DataFrame, participating_teams: set) -> pl.DataFrame:
        players_to_predict = players_historical.filter(pl.col("recent_team").is_in(participating_teams)).clone()
        if players_to_predict.is_empty():
            self.logger.warning("No players found for participating teams.")
            raise ValueError("No players found for participating teams.")
        self.logger.debug(f"Players scheduled to play:\n{players_to_predict.head()}")
        return players_to_predict

    def _create_team_opponent_map(self, schedule: pl.DataFrame) -> Dict[str, str]:
        team_opponent_map = {}
        schedule_rows = schedule.to_dicts()
        for row in schedule_rows:
            home_team = row["home_team"]
            away_team = row["away_team"]
            if home_team in team_opponent_map or away_team in team_opponent_map:
                self.logger.error("Duplicate team entry found in schedule.")
                raise ValueError("Duplicate team entry found in schedule.")
            team_opponent_map[home_team] = away_team
            team_opponent_map[away_team] = home_team
        self.logger.debug(f"Team to Opponent mapping: {team_opponent_map}")
        return team_opponent_map

    def _assign_opponent_teams(self, players_df: pl.DataFrame, team_opponent_map: Dict[str, str]) -> pl.DataFrame:
        players_df = players_df.with_column(
            pl.col("recent_team").apply(lambda x: team_opponent_map.get(x, None)).alias("opponent_team")
        )
        missing_opponents = players_df.filter(pl.col("opponent_team").is_null()).shape[0]
        if missing_opponents > 0:
            self.logger.warning(f"{missing_opponents} players have no opponent team assigned. Excluding them.")
            players_df = players_df.filter(pl.col("opponent_team").is_not_null())
        self.logger.debug(f"Players with opponent teams assigned:\n{players_df.head()}")
        return players_df

    def _prepare_prediction_week_df(self, players_df: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
        prediction_df = players_df.clone()
        prediction_df = prediction_df.with_columns([pl.lit(week).alias("week"), pl.lit(season).alias("season")])
        # Reset statistical columns to NaN
        exclude_cols = self.pk_cols + [
            "recent_team",
            "opponent_team",
            "position",
            "age",
            "years_exp",
            "rookie_flag",
        ]
        stat_cols = [col for col in self.data.columns if col not in exclude_cols]
        for col in stat_cols:
            prediction_df = prediction_df.with_column(pl.lit(np.nan).alias(col))
        self.logger.debug(f"Prediction week dataframe:\n{prediction_df.head()}")
        return prediction_df

    def _combine_historical_with_prediction(
        self, historical_df: pl.DataFrame, prediction_df: pl.DataFrame
    ) -> pl.DataFrame:
        comprehensive_df = pl.concat([historical_df, prediction_df], how="vertical")
        self.logger.debug(f"Comprehensive prediction dataframe shape: {comprehensive_df.shape}")
        return comprehensive_df


class NFLDataManager(DataManagerBase):
    def _initialize(self):
        self.logger = logging.getLogger(__name__)
        self.dates = list(range(2009, 2025))
        self.pk_cols = ["player_id", "season", "week"]
        self.cache_manager = CacheManager()

        # Instantiate DataLoader and DataValidator
        data_loader = NFLDataLoader(logger=self.logger)
        data_validator = DataValidator(logger=self.logger)

        # Load data
        self.wkly_df = data_loader.load_weekly_data(self.dates)
        data_validator.validate_primary_keys(self.wkly_df, self.pk_cols)

        self.ssnl_roster = data_loader.load_seasonal_rosters(self.dates)
        data_validator.validate_primary_keys(self.ssnl_roster, ["player_id", "season"])

        # Process data
        self.logger.info("Applying general processing to weekly data...")
        self.data = nw.add_general_processing(self.wkly_df)
        self.logger.info("Adding roster information...")
        self.data = nw.add_roster_info(self.data, self.ssnl_roster)

        # Validate processed data
        self._validate_data()

        # Load injury data
        self.inj_df = data_loader.load_injury_data(self.dates)

        # Initialize PredictionDataGenerator
        self.prediction_generator = NFLPredictionDataGenerator(
            data=self.data,
            logger=self.logger,
            cache_manager=self.cache_manager,
            pk_cols=self.pk_cols,
            dates=self.dates,
        )

    def _validate_data(self):
        if not nw._is_pk(self.data.to_pandas(), self.pk_cols):
            self.logger.error(f"Processed data does not have unique primary keys {self.pk_cols}.")
            duplicates = nw._get_non_pk_rows(self.data.to_pandas(), self.pk_cols)
            self.logger.debug(f"Duplicate rows in processed data:\n{duplicates}")
            raise ValueError("Processed data does not have unique primary keys.")
        else:
            self.logger.info("Data validation passed. No duplicate primary keys found in processed data.")

    @property
    def weekly_data(self) -> pl.DataFrame:
        return self.data

    @property
    def injury_data(self) -> pl.DataFrame:
        return self.inj_df

    def get_prediction_df(self, season: int, week: int) -> pl.DataFrame:
        return self.prediction_generator.generate_prediction_df(season, week)
