import argparse
import gc
import json
import os
from types import SimpleNamespace

import networkx as nx
import numpy as np
import torch
import yaml
from sklearn.metrics import balanced_accuracy_score
from xgboost import XGBClassifier

import repackage
repackage.up()

import utils

def run_baseline_xgboost(config_path, threshold=0.9):
    config = yaml.safe_load(open(config_path))
    val_scores_all_datasets = []
    test_scores_all_datasets = []
    remain = len(config["dataset"]["names"])
    for name in config["dataset"]["names"]:
        dataset_name = os.path.join(config["dataset"]["folder"], name + ".pkl")

        train_data, val_data, test_data = utils.get_data_split(dataset_name, config)
        args = SimpleNamespace(**config)
        run_filename = args.out_name_template.format(**config)
        
        os.makedirs(config["results_folder"], exist_ok=True)

        val_score, test_score = run_multiple_seeds(
            config, (train_data, val_data, test_data), run_filename
        )
        val_scores_all_datasets.append(val_score)
        test_scores_all_datasets.append(test_score)

        config["first_seed"] += config["num_seeds"]

        del train_data, val_data, test_data
        gc.collect()

    return np.mean(val_scores_all_datasets), np.mean(test_scores_all_datasets)



def create_dataset(data):
    all_data = []
    for k in range(len(data)):
        target = float(data[k][2])

        network = data[k][0].copy()
        tree = data[k][1].copy()
        
        n1 = [x for x in network.nodes if network.out_degree(x) != 0]
        n2 = [x for x in tree.nodes if tree.out_degree(x) != 0]
        
        n3 = [x for x in tree.nodes if tree.out_degree(x) == 0]
        n3_perm = np.random.permutation(n3)

        network = nx.relabel_nodes(network, {x:y for x,y in zip(n1, np.random.permutation(n1))}, copy=True)
        tree = nx.relabel_nodes(tree, {x:y for x,y in zip(n2, np.random.permutation(n2))}, copy=True)

        network = nx.relabel_nodes(network, {x:y for x,y in zip(n3, n3_perm)}, copy=True)
        tree = nx.relabel_nodes(tree, {x:y for x,y in zip(n3, n3_perm)}, copy=True)

        network_leaves_list = sorted(
            [x for x in network.nodes if network.out_degree(x) == 0]
        )
        tree_leaves_list = sorted([x for x in tree.nodes if tree.out_degree(x) == 0])
        assert sorted(network_leaves_list) == sorted(tree_leaves_list)

        network_leaves_list = np.random.permutation(np.array(network_leaves_list))

        network_nodes = len(network.nodes)
        tree_nodes = len(tree.nodes)
        network_ret_nodes = utils.ret_number(network)
        number_leaves = len(network_leaves_list)
        
        network_root = [x for x in network.nodes if network.in_degree(x) == 0][0]
        tree_root = [x for x in tree.nodes if tree.in_degree(x) == 0][0]

        d1 = np.array([len(nx.shortest_path(network, network_root, l)) for l in network_leaves_list])
        d2 = np.array([len(nx.shortest_path(tree, tree_root, l)) for l in network_leaves_list])
        #print(d1, d2)
        f11 = np.min(d1)
        f12 = np.min(d2)

        network_depth = np.max(d1)
        tree_depth = np.max(d2)

        f31 = np.mean(d1)
        f32 = np.mean(d2)

        diff = d1 - d2

        all_data.append(
            (
                f11 / network_depth, f12 / tree_depth, f31 / network_depth, f32 / tree_depth,
                network_depth / network_nodes, tree_depth / tree_nodes,                
                np.min(d1) - np.max(d2), (np.min(d1) - np.max(d2)) / network_depth,
                np.mean(diff), np.min(diff), np.max(diff),
                tree_nodes / network_nodes,
                target,
            )
        )

    all_data = np.array(all_data)
    return all_data


def run_multiple_seeds(config, data, run_filename):
    train_data, val_data, test_data = data
    
    train_dataset = create_dataset(train_data)
    val_dataset = create_dataset(val_data)
    test_dataset = create_dataset(test_data)

    X_train, y_train = train_dataset[:, :-1], train_dataset[:, -1]
    X_val, y_val = val_dataset[:, :-1], val_dataset[:, -1]
    X_test, y_test = test_dataset[:, :-1], test_dataset[:, -1]

    test_acc_by_seed = []
    val_acc_by_seed = []
    results_filename = os.path.join(config["results_folder"], run_filename + ".json")
    
    for seed in range(config["first_seed"], config["first_seed"] + config["num_seeds"]):
        model = XGBClassifier(
            n_estimators=config["model"]["n_estimators"],
            max_depth=config["model"]["max_depth"],
            learning_rate=config["model"]["learning_rate"],
            max_leaves=config["model"]["max_leaves"],
            reg_lambda=config["model"]["reg_lambda"],
            objective="binary:logistic",
            verbosity=1,
            subsample=config["model"]["subsample"],
            colsample_bytree=config["model"]["colsample_bytree"],
            random_state=seed,
        )
        
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        print("train acc:", balanced_accuracy_score(y_train, train_pred))
        print("val acc:", balanced_accuracy_score(y_val, val_pred))
        print("test acc:", balanced_accuracy_score(y_test, test_pred))

        val_acc_by_seed.append(balanced_accuracy_score(y_val, val_pred))
        test_acc_by_seed.append(balanced_accuracy_score(y_test, test_pred))

        with open(results_filename, "w") as f:
            json.dump({"results_test": test_acc_by_seed, "config": config}, f)

    return np.mean(val_acc_by_seed), np.mean(test_acc_by_seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/baseline.yaml")
    args = parser.parse_args()
    config_path = args.config_path
    run_baseline_xgboost(config_path)
