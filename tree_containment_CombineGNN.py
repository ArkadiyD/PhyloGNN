import argparse
import gc
import json
import os
import time
from types import SimpleNamespace

import networkx as nx
import numpy as np
import torch
import torch_geometric
import yaml
from sklearn.metrics import balanced_accuracy_score
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import *
from tqdm import tqdm

import utils
from models import NetworkCombineGraphs


def run_combine_gnn(config_path, early_terminate=None, cur_best=1):
    config = yaml.safe_load(open(config_path))
    print(config)
    val_scores_all_datasets = []
    test_scores_all_datasets = []
    remain = len(config["dataset"]["names"])

    for name in config["dataset"]["names"]:
        dataset_name = os.path.join(config["dataset"]["folder"], name + ".pkl")

        train_data, val_data, test_data = utils.get_data_split(dataset_name, config)
        args = SimpleNamespace(**config)
        run_filename = args.out_name_template.format(**config)
        print(run_filename)

        os.makedirs(config["results_folder"], exist_ok=True)

        val_score, test_score = run_multiple_seeds(
            config, (train_data, val_data, test_data), run_filename
        )
        val_scores_all_datasets.append(val_score)
        test_scores_all_datasets.append(test_score)

        config["first_seed"] += config["num_seeds"]

        if early_terminate is not None:
            if val_score < early_terminate:
                break
            remain = len(config["dataset"]["names"]) - len(val_scores_all_datasets)
            cur_max = (np.sum(val_scores_all_datasets) + remain * 1.0) / len(
                config["dataset"]["names"]
            )
            if cur_max < cur_best:
                break
            print(
                np.mean(val_scores_all_datasets),
                "remain",
                remain,
                "cur_max",
                cur_max,
                "cur_best",
                cur_best,
            )

        del train_data, val_data, test_data
        gc.collect()

    print(val_scores_all_datasets, test_scores_all_datasets)
    if early_terminate is not None:
        return np.sum(val_scores_all_datasets), np.sum(test_scores_all_datasets)
    return np.mean(val_scores_all_datasets), np.mean(test_scores_all_datasets)


def only_evaluate_combine_gnn(test_data, config):
    s0 = time.time()

    mean, std = 0.5, 1
    # print(config)
    test_dataset = create_torch_dataset_combine(
        config, test_data, use_node_types=config["features"]["use_node_types"]
    )
    for d in test_dataset:
        d.x = (d.x - mean) / std

    full_dataset_test = GraphTreesDatasetCombine(test_dataset)

    device = "cuda:%d" % config["gpu"]
    sample = torch_geometric.data.Batch.from_data_list([full_dataset_test[0][0]]).to(
        device
    )

    s1 = time.time()

    model = NetworkCombineGraphs(
        full_dataset_test[0][0].x.shape[1],
        hidden_sizes=[
            config["network"]["hidden_size"]
            for i in range(config["network"]["num_layers"])
        ],
        pool=config["network"]["pool"],
        conv=config["network"]["conv"],
        conv_dropout=config["network"]["conv_dropout"],
        mlp_dropout=config["network"]["mlp_dropout"],
        print_info=False,
    ).to(device)
    # print(full_dataset_test)
    s2 = time.time()
    # print('time to create model', time.time()-s)
    model(sample)

    s3 = time.time()
    del model
    del sample
    del full_dataset_test
    del test_dataset
    # gc.collect()
    torch.cuda.empty_cache()

    return s1 - s0, s2 - s1, s3 - s2


class GraphTreesDatasetCombine(torch.utils.data.Dataset):
    def __init__(self, networks):
        self.graphs = []
        self.targets = []
        for b in networks:
            self.graphs.append(b)
            self.targets.append(b.y)
        self.targets = torch.tensor([int(y == 1) for y in self.targets])

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        target = self.targets[idx]
        return graph, target

    def collate(self, data_list):
        batch_network = Batch.from_data_list([data[0] for data in data_list])
        batch_target = torch.tensor([data[1] for data in data_list]).view(-1)
        return batch_network, batch_target


def convert_to_graph(
    config, network, tree, target, use_node_types
):

    #These relabel operation are simple sanity check
    # n1 = [x for x in network.nodes if network.out_degree(x) != 0]
    # n2 = [x for x in tree.nodes if tree.out_degree(x) != 0]   
    # n3 = [x for x in tree.nodes if tree.out_degree(x) == 0]
    # n3_perm = np.random.permutation(n3)
    # network = nx.relabel_nodes(network, {x:y for x,y in zip(n1, np.random.permutation(n1))}, copy=True)
    # tree = nx.relabel_nodes(tree, {x:y for x,y in zip(n2, np.random.permutation(n2))}, copy=True)
    # network = nx.relabel_nodes(network, {x:y for x,y in zip(n3, n3_perm)}, copy=True)
    # tree = nx.relabel_nodes(tree, {x:y for x,y in zip(n3, n3_perm)}, copy=True)

    cnt = np.max(list(network.nodes)) + 1
    tree_rename_nodes = {}
    for node in tree.nodes:
        if not tree.out_degree(node) == 0: #not a leaf node
            tree_rename_nodes[node] = cnt
            cnt += 1
        else:
            tree_rename_nodes[node] = node

    tree = nx.relabel_nodes(tree, tree_rename_nodes, copy=False)

    graph_combined = nx.DiGraph()

    graph_combined.add_nodes_from(network.nodes)
    graph_combined.add_nodes_from(tree.nodes)
    
    edges = []

    network_edges = tuple(network.edges)
    tree_edges = tuple(tree.edges)
    #assert set(network_edges) & set(tree_edges) == set()

    nodes_list = tuple(graph_combined.nodes)
    nodes_list_map = {x:i for i,x in enumerate(nodes_list)}

    #assert nodes_list[0] == 0 and nodes_list[-1] == len(nodes_list)-1

    for edge in list(network.edges) + list(tree.edges):
        x, y = edge
        x, y = nodes_list_map[x], nodes_list_map[y]

        graph_combined.add_edge(edge[0], edge[1])

        if config["directed_graphs"] == "undirected":
            edges.append((x, y))
            edges.append((y, x))

        elif config["directed_graphs"] == "dirGNN":
            edges.append((x, y))

        else:
            raise ValueError("directed_graphs must be one of: undirected, dirGNN")

    n1 = [x for x in network.nodes if network.out_degree(x) != 0]
    n2 = [x for x in tree.nodes if tree.out_degree(x) != 0]
    assert set(n1) & set(n2) == set()
    l1 = {x for x in network.nodes if network.out_degree(x) == 0}
    l2 = {x for x in tree.nodes if tree.out_degree(x) == 0}
    assert l1 == l2

    tree_nodes = set(tree.nodes)

    in_degree = np.array([graph_combined.in_degree(x) for x in nodes_list])
    out_degree = np.array([graph_combined.out_degree(x) for x in nodes_list])
    degree_features = torch.cat(
        [torch.tensor(in_degree).view(-1, 1), torch.tensor(out_degree).view(-1, 1)],
        dim=1,
    )

    edges = torch.tensor(edges).T.contiguous()
    #print(edges)
    if not use_node_types:
        features = degree_features.float()
    else:
        node_types = []
        for x in nodes_list:
            if x in l1:  # leaves
                type = [0, 0, 1]
            elif x in tree_nodes and x not in l1:  # tree nodes
                type = [0, 1, 0]
            else:
                type = [1, 0, 0]  # network nodes

            node_types.append(type)
        node_types = torch.tensor(np.array(node_types))
        #print(node_types)
        features = torch.concat([degree_features, node_types], dim=1)

    cur_data = Data(x=torch.tensor(features).float(), edge_index=edges, y=target)
    return cur_data


def create_torch_dataset_combine(config, data, use_node_types=False):
    all_data = []
    for k in range(len(data)):
        target = float(data[k][2])

        network = data[k][0].copy()
        tree = data[k][1].copy()

        cur_data = convert_to_graph(
            config, network, tree, target, use_node_types=use_node_types
        )
        all_data.append(cur_data)

    return all_data


def train_epoch(
    config,
    model,
    device,
    optimizer,
    train_loader,
    criterion,
    scheduler=None,
    logger=None,
    epoch=0,
):
    model.train()

    target_list, output_list = [], []
    
    for data in train_loader:
        optimizer.zero_grad()

        additional_features = None
        if len(data) == 2:
            batch, target = data
        else:
            batch, target, additional_features = data
            additional_features = additional_features.to(device)

        batch = batch.to(device)
        target = target.to(device)

        output = model(batch, additional_features=additional_features)

        loss = criterion(output.view(-1), target.view(-1).float())
        loss.backward()
        optimizer.step()

        if len(target) == 1:
            target_list += [list(target.cpu().detach().numpy().flatten())]
            output_list += [list(output.cpu().detach().numpy().flatten())]

        else:
            target_list += list(target.cpu().detach().numpy().flatten())
            output_list += list(output.cpu().detach().numpy().flatten())

        scheduler.step()

    output_list = np.array(output_list)
    target_list = np.array(target_list)
    with torch.no_grad():
        loss = criterion(
            torch.tensor(output_list).view(-1),
            torch.tensor(target_list).view(-1).float(),
        )
    preds = (output_list >= 0.5).astype(np.int32)
    targets = np.array(target_list).astype(np.int32)
    train_acc = balanced_accuracy_score(
        target_list, (output_list >= 0.5).astype(np.int32)
    )

    logger.add_scalar("train/loss", loss, epoch)
    logger.add_scalar("train/acc", train_acc, epoch)
    
    for param_group in optimizer.param_groups:
        logger.add_scalar("train/lr", param_group["lr"], epoch)

    return train_acc


def evaluate(config, model, device, val_loader):
    model.eval()
    target_list, output_list = [], []
    for data in val_loader:
        additional_features = None
        if len(data) == 2:
            batch, target = data
        else:
            batch, target, additional_features = data
            additional_features = additional_features.to(device)

        batch = batch.to(device)
        target = target.to(device)

        with torch.no_grad():
            output = model(batch, additional_features=additional_features)
            target_list += list(target.cpu().numpy().flatten())
            output_list += list(output.cpu().numpy().flatten())

    output_list = np.array(output_list)
    target_list = np.array(target_list)

    preds = (output_list >= 0.5).astype(np.int32)
    targets = np.array(target_list).astype(np.int32)
    val_acc = balanced_accuracy_score(targets, preds)

    return val_acc


def run_multiple_seeds(config, data, run_filename):
    train_data, val_data, test_data = data

    train_batch_size = config["training"]["batch_size"]
    if "val_batch_size" in config["training"]:
        val_batch_size = config["training"]["val_batch_size"]
    else:
        val_batch_size = train_batch_size * 2

    test_acc_by_seed = []
    val_acc_by_seed = []

    add_positional_features = False

    train_dataset = create_torch_dataset_combine(
        config, train_data, use_node_types=config["features"]["use_node_types"]
    )
    X = torch.concat([d.x for d in train_dataset])
    mean, std = X.mean(dim=0), X.std(dim=0)
    std[std == 0] = 1

    for d in train_dataset:
        d.x = (d.x - mean) / std
    val_dataset = create_torch_dataset_combine(
        config, val_data, use_node_types=config["features"]["use_node_types"]
    )
    for d in val_dataset:
        d.x = (d.x - mean) / std
    test_dataset = create_torch_dataset_combine(
        config, test_data, use_node_types=config["features"]["use_node_types"]
    )
    for d in test_dataset:
        d.x = (d.x - mean) / std

    full_dataset_train = GraphTreesDatasetCombine(train_dataset)
    full_dataset_val = GraphTreesDatasetCombine(val_dataset)
    full_dataset_test = GraphTreesDatasetCombine(test_dataset)

    del train_dataset, val_dataset, test_dataset
    del train_data, val_data, test_data
    gc.collect()

    results_filename = os.path.join(config["results_folder"], run_filename + ".json")
    print(results_filename)

    for seed in range(config["first_seed"], config["first_seed"] + config["num_seeds"]):
        device = "cuda:%d" % config["gpu"]

        logger = SummaryWriter(
            log_dir=os.path.join(
                config["results_folder"], run_filename, "seed_%d" % seed
            )
        )

        utils.set_random_seeds(seed)
        print("seed", seed)

        train_loader = torch.utils.data.DataLoader(
            full_dataset_train,
            batch_size=train_batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=full_dataset_train.collate,
            num_workers=4,
            worker_init_fn=utils.worker_init_fn,
        )
        val_loader = torch.utils.data.DataLoader(
            full_dataset_val,
            batch_size=val_batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            collate_fn=full_dataset_train.collate,
            num_workers=4,
            worker_init_fn=utils.worker_init_fn,
        )
        test_loader = torch.utils.data.DataLoader(
            full_dataset_test,
            batch_size=val_batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            collate_fn=full_dataset_train.collate,
            num_workers=4,
            worker_init_fn=utils.worker_init_fn,
        )

        model = NetworkCombineGraphs(
            full_dataset_train[0][0].x.shape[1],
            hidden_sizes=[
                config["network"]["hidden_size"]
                for i in range(config["network"]["num_layers"])
            ],
            pool=config["network"]["pool"],
            conv=config["network"]["conv"],
            conv_dropout=config["network"]["conv_dropout"],
            mlp_dropout=config["network"]["mlp_dropout"],
            print_info=True,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["training"]["lr"]),
            weight_decay=float(config["training"]["wd"]),
        )

        epochs = (
            config["training"]["num_cycles"]
            * config["training"]["steps_in_cycle"]
            // len(train_loader)
        )
        print("epochs", epochs, "train loader", len(train_loader))
        check_val = epochs // 5

        model.train()
        criterion = torch.nn.BCELoss()

        train_accs, val_accs, test_accs = [], [], []
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config["training"]["steps_in_cycle"] + 1,
            eta_min=1e-6,
            last_epoch=-1,
        )

        for epoch in tqdm(range(epochs)):
            train_acc = train_epoch(
                config,
                model,
                device,
                optimizer,
                train_loader,
                criterion,
                scheduler=scheduler,
                logger=logger,
                epoch=epoch,
            )

            if epoch % check_val == 0 or epoch == epochs - 1:
                val_acc = evaluate(config, model, device, val_loader)
                test_acc = evaluate(config, model, device, test_loader)
                logger.add_scalar("val/acc", val_acc, epoch)
                logger.add_scalar("test/acc", test_acc, epoch)

            train_accs.append(train_acc)
            val_accs.append(val_acc)
            test_accs.append(test_acc)

            # print(
            #     np.round(np.array(train_accs), 2),
            #     np.round(np.array(val_accs), 2),
            #     np.round(np.array(test_accs), 2),
            # )

        best_epoch = np.argmax(val_accs)
        print(
            f"performance at best epoch {best_epoch}:",
            train_accs[best_epoch],
            val_accs[best_epoch],
            test_accs[best_epoch],
        )
        val_acc_by_seed.append(val_accs[best_epoch])
        test_acc_by_seed.append(test_accs[best_epoch])
        print(test_acc_by_seed)

        with open(results_filename, "w") as f:
            json.dump(
                {
                    "results_val": val_acc_by_seed,
                    "results_test": test_acc_by_seed,
                    "config": config,
                },
                f,
            )

    return np.mean(val_acc_by_seed), np.mean(test_acc_by_seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(f"--config", type=str)
    parsed_args = parser.parse_args()
    config_path = parsed_args.config

    run_combine_gnn(config_path)
