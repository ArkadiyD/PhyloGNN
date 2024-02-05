import sys
from pathlib import Path
import os
import yaml
from ray import tune
from ray.tune.search.basic_variant import BasicVariantGenerator

import repackage
repackage.up()
from baselines.tree_containment_baseline_xgboost import run_baseline_xgboost

num_seeds = 1
first_seed = 424242
num_train = 10000

datasets = [
    "train_5_10_test_11_20",
    "train_5_20_test_21_40",
    "train_5_30_test_31_60",
    "train_5_40_test_41_80",
    "train_5_50_test_51_100",
]

search_space = {
    "n_estimators": tune.grid_search([50, 300]),
    "max_depth": tune.grid_search([3, 5, 10, None]),
    "learning_rate": tune.grid_search([0.1, 0.01, 0.001]),
    "max_leaves": tune.grid_search([0, 10, 100]),
    "reg_lambda": tune.grid_search([None]),
    "subsample": tune.grid_search([0.8, 1.0]),
    "colsample_bytree": tune.grid_search([0.5, 1.0]),
    "dataset_names": datasets,
}


class Trainable(tune.Trainable):
    def setup(self, config):
        print(config)
        trial_id = self.trial_id

        full_config = {}
        full_config[
            "results_folder"
        ] = "/export/scratch1/home/arkadiy/CombineGNN/results/tune_baseline_xgboost"
        full_config[
            "out_name_template"
        ] = "dataset<{dataset[names]}>_model<{model[n_estimators]}_{model[max_depth]}_{model[learning_rate]}>"
        full_config["first_seed"] = first_seed
        full_config["num_seeds"] = num_seeds

        full_config["dataset"] = {
            "folder": "/export/scratch1/home/arkadiy/CombineGNN/datasets/",
            "names": config["dataset_names"],
        }

        full_config["training"] = {"num_samples": num_train}

        full_config["model"] = {
            "n_estimators": config["n_estimators"],
            "max_depth": config["max_depth"],
            "learning_rate": config["learning_rate"],
            "max_leaves": config["max_leaves"],
            "reg_lambda": config["reg_lambda"],
            "subsample": config["subsample"],
            "colsample_bytree": config["colsample_bytree"],
        }

        trial_num = int(trial_id.split("_")[1])
        self.config_path = os.path.join(
            "/export/scratch1/home/arkadiy/CombineGNN/configs/tunehparams_baseline_xgboost/",
            str(trial_num) + ".yaml",
        )
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, "w") as outfile:
            yaml.dump(full_config, outfile, default_flow_style=None, sort_keys=False)

    def save_checkpoint(self, checkpoint_dir):
        pass

    def step(self):
        val_acc, _ = run_baseline_xgboost(self.config_path)
        return {"score": val_acc, "done": True}


if __name__ == "__main__":
    import ray

    ray.init(_temp_dir="/export/scratch1/home/arkadiy/ray_results")
    tuner = tune.Tuner(
        tune.with_resources(Trainable, resources={"cpu": 1}),
        tune_config=tune.TuneConfig(
            search_alg=BasicVariantGenerator(max_concurrent=16),
        ),
        param_space=search_space,
    )
    tuner.fit()

    ray.shutdown()
