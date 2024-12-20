import polars as pl


def add_passer_rating(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculates passer rating and adds it to the dataframe.
    """
    temp = df.select(["completions", "attempts", "passing_yards", "passing_tds", "interceptions"]).with_columns(
        [
            pl.when((pl.col("completions") / pl.col("attempts") - 0.3) * 5 < 0)
            .then(0)
            .when((pl.col("completions") / pl.col("attempts") - 0.3) * 5 > 2.375)
            .then(2.375)
            .otherwise((pl.col("completions") / pl.col("attempts") - 0.3) * 5)
            .alias("a"),
            pl.when((pl.col("passing_yards") / pl.col("attempts") - 3) * 0.25 < 0)
            .then(0)
            .when((pl.col("passing_yards") / pl.col("attempts") - 3) * 0.25 > 2.375)
            .then(2.375)
            .otherwise((pl.col("passing_yards") / pl.col("attempts") - 3) * 0.25)
            .alias("b"),
            pl.when((pl.col("passing_tds") / pl.col("attempts") * 20) < 0)
            .then(0)
            .when((pl.col("passing_tds") / pl.col("attempts") * 20) > 2.375)
            .then(2.375)
            .otherwise((pl.col("passing_tds") / pl.col("attempts") * 20))
            .alias("c"),
            pl.when(2.375 - (pl.col("interceptions") / pl.col("attempts") * 25) < 0)
            .then(0)
            .when(2.375 - (pl.col("interceptions") / pl.col("attempts") * 25) > 2.375)
            .then(2.375)
            .otherwise(2.375 - (pl.col("interceptions") / pl.col("attempts") * 25))
            .alias("d"),
        ]
    )
    df = df.with_columns(((temp["a"] + temp["b"] + temp["c"] + temp["d"]) / 6 * 100).alias("passer_rating"))
    return df


def add_roster_info(df: pl.DataFrame, roster_df: pl.DataFrame) -> pl.DataFrame:
    """
    Adds roster information such as years of experience and age to df.
    """
    roster_df = roster_df.select(["player_id", "season", "years_exp", "age", "team"])
    roster_df = roster_df.with_columns((pl.col("years_exp") == 0).cast(pl.Int8).alias("rookie_flag"))
    df = df.join(
        roster_df,
        left_on=["player_id", "season", "recent_team"],
        right_on=["player_id", "season", "team"],
        how="left",
    ).drop("team")
    assert_valid_pk(df)
    return df


def add_trade_flags(df: pl.DataFrame) -> pl.DataFrame:
    """
    Adds flags indicating whether a player was traded in-season or off-season.
    """
    df = df.sort(["player_id", "season", "week"])
    df = add_last_n_values(df, ["player_id", "season"], ["recent_team"], "week", 1)
    df = add_last_n_values(df, ["player_id"], ["recent_team"], "season", 1)
    df = df.with_columns(
        [
            ((pl.col("week_recent_team_1") != pl.col("recent_team")) & pl.col("week_recent_team_1").is_not_null())
            .cast(pl.Int8)
            .alias("inseason_trade_flag"),
            ((pl.col("season_recent_team_1") != pl.col("recent_team")) & pl.col("season_recent_team_1").is_not_null())
            .cast(pl.Int8)
            .alias("offseason_trade_flag"),
        ]
    )
    return df


def add_general_processing(df: pl.DataFrame) -> pl.DataFrame:
    """
    Applies general processing to a weekly NFL dataframe.
    """
    df = df.filter(pl.col("position").is_not_null())
    enc_df = pl.get_dummies(df["position"])
    df = df.with_columns(enc_df)
    df = add_passer_rating(df)
    df = add_trade_flags(df)
    df = df.with_columns(
        [
            (pl.col("rushing_yards") / pl.col("carries")).alias("yards_per_carry"),
            (pl.col("passing_yards") / pl.col("attempts")).alias("yards_per_att"),
            (pl.col("completions") / pl.col("attempts")).alias("comp_pct"),
            (pl.col("carries") / (pl.col("carries") + pl.col("attempts"))).alias("scramble_pct"),
        ]
    )
    return df
