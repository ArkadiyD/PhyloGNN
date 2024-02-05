import sys
from pathlib import Path
import os
import yaml
from ray import tune
from ray.tune.search.basic_variant import BasicVariantGenerator

import repackage
repackage.up()
from tree_containment_CombineGNN import run_combine_gnn

num_seeds = 1
first_seed = 424242
num_train = 10000
batch_size = 200
num_cycles = 1
steps_in_cycle = 5000

datasets = [
    "train_5_50_test_51_100",
    "train_5_40_test_41_80",
    "train_5_30_test_31_60",
    "train_5_20_test_21_40",
    "train_5_10_test_11_20",
]

search_space = {
    "lr": tune.grid_search([1e-4, 1e-3, 1e-2]),
    "wd": tune.grid_search([0.0, 1e-4, 1e-3]),
    "hidden_size": tune.grid_search([32, 64, 128]),
    "conv": tune.grid_search(["GIN", "GCN", "GAT"]),
    "num_layers": tune.grid_search([3, 5]),
    "directed_graphs": tune.grid_search(["dirGNN"]),
    "conv_dropout": tune.grid_search([0.0]),
    "mlp_dropout": tune.grid_search([0.0, 0.2]),
    "pool": tune.grid_search(["add", "max", "mean"]),
    "dataset_names": datasets,
    "usenodefeatures": tune.grid_search([True]),
}


class Trainable(tune.Trainable):
    def setup(self, config):
        print(config)
        trial_id = self.trial_id

        full_config = {}
        full_config[
            "results_folder"
        ] = "/export/scratch1/home/arkadiy/CombineGNN/results/tune_ours_configs_new"
        full_config[
            "out_name_template"
        ] = "dataset<{dataset[names]}>_usenodefeatures{features[use_node_types]} _{directed_graphs}_network<{network[conv]}_{network[num_layers]}_{network[hidden_size]}_{network[pool]}_{network[conv_dropout]}_{network[mlp_dropout]}>_training<{training[num_samples]}_{training[lr]}_{training[wd]}"
        full_config["first_seed"] = first_seed
        full_config["num_seeds"] = num_seeds
        full_config["gpu"] = 0

        full_config["directed_graphs"] = config["directed_graphs"]

        full_config["dataset"] = {
            "folder": "/export/scratch1/home/arkadiy/CombineGNN/datasets/",
            "names": config["dataset_names"],
        }

        full_config["features"] = {"use_node_types": config["usenodefeatures"]}

        full_config["training"] = {
            "num_samples": num_train,
            "batch_size": batch_size,
            "num_cycles": num_cycles,
            "steps_in_cycle": steps_in_cycle,
            "wd": config["wd"],
            "lr": config["lr"],
        }

        conv = config["conv"]
        if config["directed_graphs"] == "dirGNN":
            conv = "Dir" + conv
        full_config["network"] = {
            "pool": config["pool"],
            "hidden_size": config["hidden_size"],
            "conv": conv,
            "num_layers": config["num_layers"],
            "mlp_dropout": config["mlp_dropout"],
            "conv_dropout": config["conv_dropout"],
        }

        trial_num = int(trial_id.split("_")[1])
        self.config_path = os.path.join(
            "/export/scratch1/home/arkadiy/CombineGNN/configs/tunehparams_CombineGNN/",
            str(trial_num) + ".yaml",
        )
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, "w") as outfile:
            yaml.dump(full_config, outfile, default_flow_style=None, sort_keys=False)

    def save_checkpoint(self, checkpoint_dir):
        pass

    def step(self):
        val_acc, _ = run_combine_gnn(
            self.config_path, early_terminate=0.9, cur_best=0.93
        )
        return {"score": val_acc, "done": True}


if __name__ == "__main__":
    import ray

    ray.init(_temp_dir="/export/scratch1/home/arkadiy/ray_results")
    tuner = tune.Tuner(
        # Trainable,
        tune.with_resources(Trainable, resources={"cpu": 1, "gpu": 0.25}),
        tune_config=tune.TuneConfig(
            search_alg=BasicVariantGenerator(max_concurrent=16),
            # num_samples=1
        ),
        param_space=search_space,
    )
    tuner.fit()

    ray.shutdown()