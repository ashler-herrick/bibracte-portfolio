import bet_edge.nfl.nfl_data as nfp
import bet_edge.probabilistic_models as pm
from bet_edge.probabilistic_models.torch_dpm import DeepNormalMixture, KFoldDPM


def main():
    # pass yards models
    qb_data = nfp.PassYardsData("passing_yards")
    qb_data.build_training_data()

    pass_model_dmm = DeepNormalMixture(
        n_inputs=qb_data.X.shape[1],
        n_hidden=256,
        learning_rate=1e-4,
        n_dist=3,
        p_dropout=0.5,
        batch_size=64,
    )
    pass_kfold_dmm = KFoldDPM(pass_model_dmm)
    pass_kfold_dmm.train_kfold(qb_data.X_scaled, qb_data.y)
    pm.utils.pickle_model(pass_kfold_dmm, "dmm_normal_pass_yds")

    # rush yards models
    rb_data = nfp.RushYardsData("rushing_yards")
    rb_data.build_training_data()

    rush_model_dmm = DeepNormalMixture(
        n_inputs=rb_data.X.shape[1],
        n_hidden=256,
        learning_rate=1e-4,
        n_dist=3,
        p_dropout=0.5,
        batch_size=64,
    )
    rush_kfold_dmm = KFoldDPM(rush_model_dmm)
    rush_kfold_dmm.train_kfold(rb_data.X_scaled, rb_data.y)
    pm.utils.pickle_model(rush_kfold_dmm, "dmm_normal_rush_yds")

    # rec_yards models
    rec_data = nfp.RecYardsData("receiving_yards")
    rec_data.build_training_data()

    rec_model_dmm = DeepNormalMixture(
        n_inputs=rec_data.X.shape[1],
        n_hidden=256,
        learning_rate=1e-4,
        n_dist=3,
        p_dropout=0.5,
        batch_size=64,
    )
    rec_kfold_dmm = KFoldDPM(rec_model_dmm)
    rec_kfold_dmm.train_kfold(rec_data.X_scaled, rec_data.y)
    pm.utils.pickle_model(rec_kfold_dmm, "dmm_normal_rec_yds")


if __name__ == "__main__":
    main()
