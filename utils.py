import pickle
import time
from copy import deepcopy

import networkx as nx
import numpy as np

import random
import torch
from TreeWidthTreeContainment import BOTCH

import repackage
repackage.up()


def get_legal_moves(net_, ensure_treechild=False):
    # t = time.time()
    net = net_.copy()
    legal_moves = []
    n = len(net.nodes)
    edges = np.random.permutation(net.edges)
    for e1 in edges:
        u, v = e1
        if check_if_move_is_legal(net, [u, v], ensure_treechild=ensure_treechild):
            for e2 in edges:
                u, v = e1
                x, y = e2
                if check_if_move_is_legal(
                    net, [u, v, x, y], ensure_treechild=ensure_treechild
                ):
                    legal_moves.append((u, v, x, y))
    return legal_moves


def check_is_tree_node(net, u):
    if net.in_degree(u) == 1 and net.out_degree(u) == 2:
        return True
    return False


def check_is_movable(net, u, v):
    parent = net.predecessors(u)
    children = net.successors(u)
    for p in parent:
        break
    for c in children:
        if c != v:
            break
    # print(u,v,p,c)
    if (p, c) in net.edges:
        return False
    return True


def check_is_not_above(net, u, v, s):
    is_above = nx.has_path(net, v, s)
    if is_above:
        return False
    return True


def network_is_treechild(network):
    for node in network.nodes:
        if network.out_degree(node) > 0:
            has_needed_child = False
            for child in network.successors(node):
                has_needed_child = (
                    has_needed_child
                    or is_tree_node(network, child)
                    or is_leaf_node(network, child)
                )
            if not has_needed_child:
                return False
    return True


def reticulations(network):
    return [
        network.in_degree(v) - 1 for v in network.nodes() if network.in_degree(v) >= 2
    ]

def ret_number(network):
    return np.sum(reticulations(network))

def is_ret_node(net, node):
    return net.in_degree(node) >= 2 and net.out_degree(node) == 1

def is_tree_node(net, node):
    return net.in_degree(node) == 1 and net.out_degree(node) == 2

def is_leaf_node(net, node):
    return net.out_degree(node) == 0

def get_ret_nodes(net):
    return [x for x in net.nodes if is_ret_node(net, x)]

def get_leaves(net):
    return [x for x in net.nodes if is_leaf_node(net, x)]

def get_leaves_count(net):
    return len(get_leaves(net))

def check_if_move_is_legal(net, picked_nodes, ensure_treechild=False):
    start = time.time()

    if len(set(picked_nodes)) != len(picked_nodes):
        return False

    if not check_is_tree_node(net, picked_nodes[0]):
        return False

    u, v = picked_nodes[0], picked_nodes[1]

    if (u, v) not in net.edges:
        return False

    is_movable = check_is_movable(net, picked_nodes[0], picked_nodes[1])
    if not is_movable:
        return False

    if len(picked_nodes) <= 2:
        return True
    
    if net.out_degree(picked_nodes[2]) == 0:
        return False
    elif (
        check_is_not_above(net, picked_nodes[0], picked_nodes[1], picked_nodes[2])
        == False
    ):
        return False

    if len(picked_nodes) == 3:
        return True

    s, t = picked_nodes[2], picked_nodes[3]

    if (s, t) not in net.edges:
        return False

    
    # print(net.edges())
    if ensure_treechild:
        net, p, c = tail_move(net, u, v, s, t, return_p_c=True)
        for node in net.nodes:
            if net.out_degree(node) > 0:
                has_right_child = False
                for child in net.successors(node):
                    # print(node, child, is_tree_node(net, child), is_leaf_node(net, child))
                    # child = False
                    has_right_child = (
                        has_right_child
                        or is_tree_node(net, child)
                        or is_leaf_node(net, child)
                    )
                    # print(right_child)
                if not has_right_child:
                    net = reverse_tail_move(net, u, v, s, t, p, c)
                    return False

        net = reverse_tail_move(net, u, v, s, t, p, c)

    return True


def is_tree_child(net):
    for node in net.nodes:
        if net.out_degree(node) > 0:
            has_right_child = False
            for child in net.successors(node):
                has_right_child = (
                    has_right_child
                    or is_tree_node(net, child)
                    or is_leaf_node(net, child)
                )
            if not has_right_child:
                return False
    return True


def tail_move(network, u, v, s, t, return_p_c=False):
    # print('doing tail move', u,v,s,t)
    parent = network.predecessors(u)
    children = network.successors(u)
    for p in parent:
        break
    for c in children:
        if c != v:
            break
    
    # pruning (p,u), (u,c) -> (p,c)
    network.remove_edge(p, u)
    network.remove_edge(u, c)
    network.add_edge(p, c)

    # moving u to (s,t)
    network.add_edge(s, u)
    network.add_edge(u, t)
    network.remove_edge(s, t)

    if not return_p_c:
        return network
    else:
        return network, p, c


def reverse_tail_move(network, u, v, s, t, p, c):
    network.add_edge(s, t)
    network.remove_edge(s, u)
    network.remove_edge(u, t)

    network.remove_edge(p, c)
    network.add_edge(p, u)
    network.add_edge(u, c)

    return network


def worker_init_fn(worker_id):
    torch_seed = torch.initial_seed() + worker_id
    torch_seed = torch_seed % 2**30
    random.seed(torch_seed)
    np.random.seed(torch_seed)


def set_random_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def check_tree_containment(network, tree):
    res = BOTCH.tc_brute_force(deepcopy(tree), deepcopy(network))
    return int(res)


def get_data_split(dataset_name, config):
    np.random.seed(42)
    train_data, val_test_data = pickle.load(open(dataset_name, "rb"))
    ind = np.random.permutation(len(train_data))[
        : int(config["training"]["num_samples"])
    ]
    train_data = [train_data[i] for i in ind]
    np.random.seed(42)
    ind = np.random.permutation(len(val_test_data))
    val_test_data = [val_test_data[i] for i in ind]
    val_data, test_data = (
        val_test_data[: len(val_test_data) // 2],
        val_test_data[len(val_test_data) // 2 :],
    )
    return train_data, val_data, test_data
