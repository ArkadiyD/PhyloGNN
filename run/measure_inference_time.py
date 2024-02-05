import json
import sys
import time
from pathlib import Path
import numpy as np
import os
import yaml
import torch
import tqdm
import ray.tune as tune

import repackage
repackage.up()
from tree_containment_CombineGNN import only_evaluate_combine_gnn
import utils

num_seeds = 5
first_seed = 42
batch_size = 200
num_cycles = 1
steps_in_cycle = 5000

config = {
    "lr": 0.001,
    "wd": 0.0,
    "hidden_size": 128,
    "conv": "GCN",
    "num_layers": 5,
    "directed_graphs": "dirGNN",
    "conv_dropout": 0.0,
    "mlp_dropout": 0.2,
    "pool": "max",
    "use_node_types": True,
    "num_train": 10000
}


class Trainable(tune.Trainable):
    def setup(self, config):
        print(config)
        trial_id = self.trial_id
        dataset = config["dataset_names"]

        full_config = {}
        full_config[
            "results_folder"
        ] = "/export/scratch1/home/arkadiy/CombineGNN/results/time_measurements"
        full_config[
            "out_name_template"
        ] = "dataset<{dataset[names]}>_usenodefeatures{features[use_node_types]} _{directed_graphs}_network<{network[conv]}_{network[num_layers]}_{network[hidden_size]}_{network[pool]}_{network[conv_dropout]}_{network[mlp_dropout]}>_training<{training[num_samples]}_{training[lr]}_{training[wd]}"
        full_config["first_seed"] = first_seed
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
            "val_batch_size": 1,
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

        self.config_path = os.path.join(
            "/export/scratch1/home/arkadiy/CombineGNN/configs/tmp/",
            "tmp" + ".yaml",
        )
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        with open(self.config_path, "w") as outfile:
            yaml.dump(full_config, outfile, default_flow_style=None, sort_keys=False)

        self.full_config = full_config

    def train(self):
        val_acc, _ = run_combine_gnn(self.config_path)

    def evaluate(self):
        pass


if __name__ == "__main__":
    times_exact = []
    times_gnn = []
    dataframe = {"size": [], "algorithm": [], "time": [], "TC": []}

    all_data = []

    for dataset in [
       ["train_5_10_test_11_20"],
       ["train_5_20_test_21_40"],
       ["train_5_30_test_31_60"],
       ["train_5_40_test_41_80"],
       ["train_5_50_test_51_100"],
    ]:
        config["dataset_names"] = dataset
        trainer = Trainable(config)
        dataset_name = os.path.join(
            "/export/scratch1/home/arkadiy/CombineGNN/datasets/", dataset[0] + ".pkl"
        )

        train_data, val_data, test_data = utils.get_data_split(
            dataset_name, trainer.full_config
        )

        for data in val_data + test_data:
            for k in range(10):
                all_data.append((data, trainer, "exact"))
                all_data.append((data, trainer, "CombineGNN"))
                

    ind = np.arange(len(all_data))
    ind = np.random.permutation(ind)
    all_data = [all_data[i] for i in ind]

    for i, (data, trainer, algorithm) in enumerate(tqdm.tqdm(all_data)):
        N = data[0].number_of_nodes()
        TC = utils.network_is_treechild(data[0])
        if TC:
            continue
        
        if algorithm == "exact":
            start = time.time()
            net = data[0]
            tree = data[1]
            trees_contained = utils.check_tree_containment(net, tree)
            end = time.time()
            
            dataframe["size"].append(N)
            dataframe["algorithm"].append("exact")
            dataframe["time"].append(end - start)
            dataframe["TC"].append(TC)

            torch.cuda.empty_cache()
        
        else:
            data_time, model_time, inference_time = only_evaluate_combine_gnn(
                [data], trainer.full_config
            )

            dataframe["size"].append(N)
            dataframe["algorithm"].append("Combine-GNN")
            dataframe["time"].append(data_time + model_time + inference_time)
            dataframe["TC"].append(TC)

            dataframe["size"].append(N)
            dataframe["algorithm"].append("inference")
            dataframe["time"].append(inference_time)
            dataframe["TC"].append(TC)

            dataframe["size"].append(N)
            dataframe["algorithm"].append("model creation")
            dataframe["time"].append(model_time)
            dataframe["TC"].append(TC)

            dataframe["size"].append(N)
            dataframe["algorithm"].append("data processing")
            dataframe["time"].append(data_time)
            dataframe["TC"].append(TC)

            torch.cuda.empty_cache()
        
    #     if i % 1000 == 0:
    #         with open("times_new.json", "w") as f:
    #             json.dump(dataframe, f)

    # with open("times_new.json", "w") as f:
    #     json.dump(dataframe, f)
