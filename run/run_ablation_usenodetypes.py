import sys
from pathlib import Path
import os
import yaml

from ray import tune
from ray.tune.search.basic_variant import BasicVariantGenerator

import repackage
repackage.up()
from tree_containment_CombineGNN import run_combine_gnn

num_seeds = 5
first_seed = 42
batch_size = 200
num_cycles = 1
steps_in_cycle = 5000

hparams = {
    "dataset_names": tune.grid_search(
        [
             ["train_5_10_test_11_20"],
             ["train_5_20_test_21_40"],
             ["train_5_30_test_31_60"],
             ["train_5_40_test_41_80"],
             ["train_5_50_test_51_100"],
        ],
    ),
    "lr": 0.001,
    "wd": 0.0,
    "hidden_size": 128,
    "conv": "GCN",
    "num_layers": 5,
    "directed_graphs": "dirGNN",
    "conv_dropout": 0.0,
    "mlp_dropout": 0.2,
    "pool": "max",
    "num_train": 10000,
    "use_node_types": False,
}


class Trainable(tune.Trainable):
    def setup(self, config):
        print(config)
        trial_id = self.trial_id
        dataset = config["dataset_names"]
        print(dataset)
        full_config = {}
        full_config[
            "results_folder"
        ] = "/export/scratch1/home/arkadiy/CombineGNN/results/CombineGNN_ablations"
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

        full_config["features"] = {"use_node_types": config["use_node_types"]}

        full_config["training"] = {
            "num_samples": config["num_train"],
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
            "/export/scratch1/home/arkadiy/CombineGNN/configs/CombineGNN_ablations",
            str(trial_num) + ".yaml",
        )
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, "w") as outfile:
            yaml.dump(full_config, outfile, default_flow_style=None, sort_keys=False)

    def save_checkpoint(self, checkpoint_dir):
        pass

    def step(self):
        _, test_acc = run_combine_gnn(self.config_path, early_terminate=None)
        return {"score": test_acc, "done": True}


if __name__ == "__main__":
    import ray

    ray.init(_temp_dir="/export/scratch1/home/arkadiy/ray_results")
    tuner = tune.Tuner(
        tune.with_resources(Trainable, resources={"cpu": 1, "gpu": 0.25}),
        tune_config=tune.TuneConfig(
            search_alg=BasicVariantGenerator(max_concurrent=16),
        ),
        param_space=hparams,
    )
    tuner.fit()

    ray.shutdown()
