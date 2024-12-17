import bet_edge.nfl.nfl_data as nfp


def main():
    qb_data = nfp.PassYardsData("passing_yards")
    pred_df = qb_data.prepare_prediction_data(2023, 1)
    pred_df.head()


if __name__ == "__main__":
    main()
