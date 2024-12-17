import functools as ft
from typing import List
import pandas as pd

import bet_edge.nfl.weekly as nw


def get_inj_counts(inj_df: pd.DataFrame, ind: str, pos_list: List[str], pos: str) -> pd.DataFrame:
    temp = ind[:4].lower()
    return (
        inj_df[(inj_df["position"].isin(pos_list)) & (inj_df["report_status"] == ind)]
        .groupby(["season", "week", "team"])["player_id"]
        .count()
        .reset_index(name=f"count_{pos}_{temp}")
    )


def genr_inj_summary(inj_df: pd.DataFrame) -> pd.DataFrame:
    pos_lists = [
        (["DE", "DT"], "d_line"),
        (["LB"], "d_lb"),
        (["CB", "S"], "d_sec"),
        (["C", "G", "T"], "o_line"),
    ]
    ind_list = ["Out"]
    inj_base = inj_df[["season", "week", "team"]].drop_duplicates()
    dfs = [inj_base]
    for x in pos_lists:
        for ind in ind_list:
            dfs.append(get_inj_counts(inj_df, ind, x[0], x[1]))

    return ft.reduce(
        lambda left, right: pd.merge(left, right, on=["season", "week", "team"], how="left"),
        dfs,
    ).fillna(0)


def add_injury_to_wkly_df(wkly_df: pd.DataFrame, inj_df: pd.DataFrame) -> pd.DataFrame:
    inj_df = inj_df.rename(columns={"gsis_id": "player_id"})
    wkly_df = wkly_df.rename(columns={"recent_team": "team"})
    inj_summary = genr_inj_summary(inj_df)
    opp_cols = list(inj_summary.columns)
    opp_cols = [x for x in opp_cols if x not in ["count_o_line_out"]]
    data = pd.merge(
        wkly_df,
        inj_summary[["season", "week", "team", "count_o_line_out"]],
        on=["season", "week", "team"],
        how="left",
    ).fillna(0)
    data = pd.merge(
        data,
        inj_summary[opp_cols],
        left_on=["season", "week", "opponent_team"],
        right_on=["season", "week", "team"],
        how="left",
    ).fillna(0)
    data.drop(columns=["team_y", "team_x"])
    return data.drop(columns=["team_y", "team_x"])


def add_offense_injury_indicators(wkly_df: pd.DataFrame, inj_df: pd.DataFrame) -> pd.DataFrame:
    inj_df = inj_df.rename(columns={"gsis_id": "player_id"})
    inj_df = nw.add_last_n_values(inj_df, ["player_id", "season"], ["report_status"], "player", 2)
    id_cols = ["season", "week", "player_id"]
    inj_df["ques_flag"] = inj_df["report_status"].apply(lambda x: 1 if x == "Questionable" else 0)
    inj_df["prev_inj_flag"] = inj_df.apply(
        lambda row: 1 if row["report_status"] != "Out" and row["player_report_status_1"] == "Out" else 0,
        axis=1,
    )
    tdf = pd.merge(
        wkly_df,
        inj_df[id_cols + ["ques_flag", "prev_inj_flag"]],
        how="left",
        on=id_cols,
    )
    tdf[["ques_flag", "prev_inj_flag"]] = tdf[["ques_flag", "prev_inj_flag"]].fillna(0)
    return tdf
