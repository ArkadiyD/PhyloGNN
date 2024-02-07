import argparse
import gc
import json
import os
from types import SimpleNamespace

import networkx as nx
import numpy as np
import torch
import yaml

from sklearn.metrics import balanced_accuracy_score, r2_score
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import *
from tqdm import tqdm

import repackage
repackage.up()

import utils
from models import *



def run_baseline_gnn(config_path, early_terminate=None, cur_best=None):
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

        leaves_count = [utils.get_leaves_count(x[0]) for x in train_data]
        num_leaves_train = max(leaves_count)

        val_score, test_score = run_multiple_seeds(
            config, (train_data, val_data, test_data), run_filename, num_leaves_train
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

    print('all results val:', val_scores_all_datasets, 'all results test:', test_scores_all_datasets)
    if early_terminate is not None:
        return np.sum(val_scores_all_datasets), np.sum(test_scores_all_datasets)
    return np.mean(val_scores_all_datasets), np.mean(test_scores_all_datasets)


def convert_to_data(
    config,
    graph,
    target,
    nodes_list,
    leaves_map,
    num_leaves_train,
    add_ohe_leaves,
):

    n1 = [x for x in nodes_list if graph.out_degree(x) != 0]
    graph = nx.relabel_nodes(graph, {x:y for x,y in zip(n1, np.random.permutation(n1))}, copy=True)

    in_degree = np.array([graph.in_degree(x) for x in nodes_list])
    out_degree = np.array([graph.out_degree(x) for x in nodes_list])
    degree_features = torch.cat(
        [torch.tensor(in_degree).view(-1, 1), torch.tensor(out_degree).view(-1, 1)],
        dim=1,
    )
    leaves_features = torch.zeros(len(nodes_list), num_leaves_train)
    ood_leaves_features = torch.zeros(len(nodes_list), 1)

    for node_id in nodes_list:
        if graph.out_degree(node_id) != 0:
            continue

        if node_id in leaves_map:        
            leaves_features[nodes_list.index(node_id), leaves_map[node_id]] = 1
        else:
            ood_leaves_features[nodes_list.index(node_id), 0] = 1
    
    leaves_features = torch.cat([leaves_features, ood_leaves_features], dim=1)

    edges = []
    #print(graph.edges)
    for edge in graph.edges:
        x, y = edge
        x, y = nodes_list.index(x), nodes_list.index(y)

        if config["directed_graphs"] == "undirected":
            edges.append((x, y))
            edges.append((y, x))
            
        elif config["directed_graphs"] == "dirGNN":
            edges.append((x, y))

        else:
            raise ValueError("Unknown graph type")

    edge_index = torch.tensor(edges).T.contiguous()
    features = degree_features
    
    if add_ohe_leaves:
        features = torch.cat([features, leaves_features], dim=1)
    #print(features)
    cur_data = Data(
        x=torch.tensor(features).float(),
        edge_index=edge_index,
        y=target,
        edge_attr=None,
    )

    return cur_data


class GraphTreesDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.targets = torch.tensor([int(x[0].y == 1) for x in self.data])

        self.additional_features = None
        if len(self.data[0]) == 3:
            self.additional_features = [x[2] for x in self.data]
            self.additional_features = torch.tensor(self.additional_features).float()
            print(self.additional_features.shape, self.additional_features)
        # print(self.targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.additional_features is not None:
            return (
                self.data[idx][0],
                self.data[idx][1],
                self.targets[idx],
                self.additional_features[idx],
            )
        return self.data[idx][0], self.data[idx][1], self.targets[idx]

    def collate(self, data_list):
        batch_network = Batch.from_data_list([data[0] for data in data_list])
        batch_tree = Batch.from_data_list([data[1] for data in data_list])
        batch_target = torch.tensor([data[2] for data in data_list]).view(-1)

        if len(data_list[0]) == 4:
            batch_additional_features = torch.stack([data[3] for data in data_list])
            return batch_network, batch_tree, batch_target, batch_additional_features
        return batch_network, batch_tree, batch_target


def create_torch_dataset(
    config, data, num_leaves_train, add_ohe_leaves
):
    all_data = []
    print('num_leaves_train',num_leaves_train)
    for k in range(len(data)):
        target = float(data[k][2])

        network = data[k][0].copy()
        tree = data[k][1].copy()

        network_leaves_list = sorted([x for x in network.nodes if network.out_degree(x) == 0])
        tree_leaves_list = sorted([x for x in tree.nodes if tree.out_degree(x) == 0])
        assert network_leaves_list == tree_leaves_list

        # leaves_list = np.copy(network_leaves_list)       
        # n3 = np.random.permutation(leaves_list)
        # network = nx.relabel_nodes(network, {x:y for x,y in zip(leaves_list, n3)}, copy=True)
        # tree = nx.relabel_nodes(tree, {x:y for x,y in zip(leaves_list, n3)}, copy=True)

        network_nodes = sorted(list(network.nodes))
        tree_nodes = sorted(list(tree.nodes))
        leaves_list = sorted([x for x in network.nodes if network.out_degree(x) == 0])        
        leaves_ind = np.random.choice(num_leaves_train, min(len(leaves_list), num_leaves_train), replace=False)
        leaves_map = {x:y for x,y in zip(leaves_list, leaves_ind)}
        
        network_data = convert_to_data(
            config,
            network,
            target,
            network_nodes,
            leaves_map,
            num_leaves_train,
            add_ohe_leaves,
        )
        tree_data = convert_to_data(
            config,
            tree,
            target,
            tree_nodes,
            leaves_map,
            num_leaves_train,
            add_ohe_leaves,
        )

        all_data.append((network_data, tree_data))

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
        if len(data) == 3:
            batch, batch_tree, target = data
        else:
            batch, batch_tree, target, additional_features = data
            additional_features = additional_features.to(device)

        batch = batch.to(device)
        batch_tree = batch_tree.to(device)
        target = target.to(device)

        output = model(batch, batch_tree, additional_features=additional_features)

        loss = criterion(
            output.view(-1), target.view(-1).float())
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        if len(target) == 1:
            target_list += [list(target.cpu().detach().numpy().flatten())]
            output_list += [list(output.cpu().detach().numpy().flatten())]

        else:
            target_list += list(target.cpu().detach().numpy().flatten())
            output_list += list(output.cpu().detach().numpy().flatten())

        scheduler.step()

    output_list = np.array(output_list)
    target_list = np.array(target_list)

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
        if len(data) == 3:
            batch, batch_tree, target = data
        else:
            batch, batch_tree, target, additional_features = data
            additional_features = additional_features.to(device)

        batch = batch.to(device)
        batch_tree = batch_tree.to(device)
        target = target.to(device)

        with torch.no_grad():
            output = model(
                batch, batch_tree, additional_features=additional_features
            )
            target_list += list(target.cpu().numpy().flatten())
            output_list += list(output.cpu().numpy().flatten())

    output_list = np.array(output_list)
    target_list = np.array(target_list)

    preds = (output_list >= 0.5).astype(np.int32)
    targets = np.array(target_list).astype(np.int32)
    val_acc = balanced_accuracy_score(targets, preds)

    return val_acc


def run_multiple_seeds(config, data, run_filename, num_leaves_train):
    train_data, val_data, test_data = data
    train_batch_size = config["training"]["batch_size"]
    val_batch_size = train_batch_size * 2
    test_acc_by_seed = []
    val_acc_by_seed = []

    train_dataset = create_torch_dataset(
        config,
        train_data,
        num_leaves_train,
        add_ohe_leaves=config["features"]["add_ohe_leaves"],
    )
    X0 = torch.concat([d[0].x for d in train_dataset])
    X1 = torch.concat([d[1].x for d in train_dataset])
    mean0, std0 = X0[:,:2].mean(dim=0), X0[:,:2].std(dim=0)
    mean1, std1 = X1[:,:2].mean(dim=0), X1[:,:2].std(dim=0)
    std0[std0 == 0] = 1
    std1[std1 == 0] = 1

    for d in train_dataset:
        d[0].x[:,:2] = (d[0].x[:,:2] - mean0) / std0
        d[1].x[:,:2] = (d[1].x[:,:2] - mean1) / std1

    val_dataset = create_torch_dataset(
        config,
        val_data,
        num_leaves_train,
        add_ohe_leaves=config["features"]["add_ohe_leaves"],
    )
    for d in val_dataset:
        d[0].x[:,:2] = (d[0].x[:,:2] - mean0) / std0
        d[1].x[:,:2] = (d[1].x[:,:2] - mean1) / std1

    test_dataset = create_torch_dataset(
        config,
        test_data,
        num_leaves_train,
        add_ohe_leaves=config["features"]["add_ohe_leaves"],
    )
    for d in test_dataset:
        d[0].x[:,:2] = (d[0].x[:,:2] - mean0) / std0
        d[1].x[:,:2] = (d[1].x[:,:2] - mean1) / std1

    full_dataset_train = GraphTreesDataset(train_dataset)
    full_dataset_val = GraphTreesDataset(val_dataset)
    full_dataset_test = GraphTreesDataset(test_dataset)

    del train_dataset, val_dataset, test_dataset
    del train_data, val_data, test_data
    gc.collect()

    results_filename = os.path.join(config["results_folder"], run_filename + ".json")
    
    device = "cuda:%d" % config["gpu"]
    for seed in range(config["first_seed"], config["first_seed"] + config["num_seeds"]):
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

        epochs = (
            config["training"]["num_cycles"]
            * config["training"]["steps_in_cycle"]
            // len(train_loader)
        )
        print("epochs", epochs, "train loader", len(train_loader))
        check_val = epochs // 5
        
        model = BaselineSiameseGNN(
            full_dataset_train[0][0].x.shape[1],
            hidden_sizes=[
                config["network"]["hidden_size"]
                for i in range(config["network"]["num_layers"])
            ],
            conv=config["network"]["conv"],
            pool=config["network"]["pool"],
            mlp_dropout=config["network"]["mlp_dropout"],
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["training"]["lr"]),
            weight_decay=float(config["training"]["wd"]),
        )

        model.train()
        criterion = torch.nn.BCELoss()

        train_accs, val_accs, test_accs = [], [], []
        # scheduler = PolynomialLR(optimizer, total_iters=epochs)
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
            
        best_epoch = np.argmax(val_accs)
        print(
            f"performance at best epoch {best_epoch}: train {train_accs[best_epoch]}, val {val_accs[best_epoch]}, test {test_accs[best_epoch]}",
        )
        val_acc_by_seed.append(val_accs[best_epoch])
        test_acc_by_seed.append(test_accs[best_epoch])
        
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

    run_baseline_gnn(config_path)
