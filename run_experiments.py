import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import argparse
import os
from collections import defaultdict
import glob
from copy import deepcopy
import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger
import os
import random
from outcome_correlation import *
from adapter import *


def neighboring_node_idx(adj_matrix):
    new_added_train_idx = pd.read_csv("cora_splits/new_added_train_idx.csv")
    # print("new_added_train_idx",new_added_train_idx)
    new_added_neighbors = []
    
    for idx in range(len(new_added_train_idx)):
        row_i = new_added_train_idx.iloc[idx]
        # print("row_i = ",row_i)
        row = adj_matrix[row_i][0]
        # print("row =",  row)
        indexs=[]
        for i in range(len(row)):
            # try:
            if row[i]==1:
                indexs.append(i)
            # except:
            #     pass


        print(indexs)
        # break
        new_added_neighbors.extend(indexs)
    new_added_neighbors = list(set(new_added_neighbors))
    return new_added_neighbors


def main():
    parser = argparse.ArgumentParser(description='Outcome Correlations)')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--method', type=str)
    args = parser.parse_args()

    try:
        dataset = PygNodePropPredDataset(name=f'ogbn-{args.dataset}')
    except ValueError:
        dataset = dgl_to_ogbn(
            args.dataset, 'ogbn-cora-submission')
    data = dataset[0]

    adj, D_isqrt = process_adj(data, args.dataset not in ['arxiv', 'products'])
    normalized_adjs = gen_normalized_adjs(adj, D_isqrt)
    DAD, DA, AD = normalized_adjs
    adj_matrix = adj.to_dense()
    # print("adj_matrix = ",adj_matrix.shape)
    # # print("adj_matrix [0] = ",adj_matrix[0])
    # print("adj_matrix = ",adj_matrix)
    neighbor_nodes = neighboring_node_idx(adj_matrix)
    # print("neighbor_nodes = ",len(neighbor_nodes))
    # print("neighbor_nodes = ",neighbor_nodes)
    # while True:
    #     pass
    evaluator = Evaluator(name=f'ogbn-{args.dataset}')

    split_idx = dataset.get_idx_split()
    # print("train_split = ",len(split_idx["train"]))
    # print("train_split = ",type(split_idx["train"]))
    # print("train_split = ",split_idx["train"])

    new_train_id = pd.read_csv("cora_splits/new_train_idx.csv")
    new_train_id = new_train_id.to_numpy().flatten()
    split_idx["train"] = new_train_id
    # print("new_train_id = ",len(new_train_id))
    # print("new_train_id = ",type(new_train_id))
    # print("new_train_id = ",new_train_id)

    # while True:
    #     pass
    # split_idx["neighbor_nodes"] = neighbor_nodes
    def eval_test(result, idx=split_idx['test']):
        return evaluator.eval({'y_true': data.y[idx], 'y_pred': result[idx].argmax(dim=-1, keepdim=True), })['acc']

    if args.dataset == 'arxiv':
        lp_dict = {
            'idxs': ['train'],
            'alpha': 0.9,
            'num_propagations': 50,
            'A': AD,
        }
        plain_dict = {
            'train_only': True,
            'alpha1': 0.87,
            'A1': AD,
            'num_propagations1': 50,
            'alpha2': 0.81,
            'A2': DAD,
            'num_propagations2': 50,
            'display': False,
        }
        plain_fn = double_correlation_autoscale

        """
        If you tune hyperparameters on test set
        {'alpha1': 0.9988673963255859,
            'alpha2': 0.7942279952481052, 'A1': 'DA', 'A2': 'AD'}
        gets you to 72.64
        """
        linear_dict = {
            'train_only': True,
            'alpha1': 0.98,
            'alpha2': 0.65,
            'A1': AD,
            'A2': DAD,
            'num_propagations1': 50,
            'num_propagations2': 50,
            'display': False,
        }
        linear_fn = double_correlation_autoscale

        """
        If you tune hyperparameters on test set
        {'alpha1': 0.9956668128133523,
            'alpha2': 0.8542393515434346, 'A1': 'DA', 'A2': 'AD'}
        gets you to 73.35
        """
        mlp_dict = {
            'train_only': True,
            'alpha1': 0.9791632871592579,
            'alpha2': 0.7564990804200602,
            'A1': DA,
            'A2': AD,
            'num_propagations1': 50,
            'num_propagations2': 50,
            'display': False,
        }
        mlp_fn = double_correlation_autoscale

        gat_dict = {
            'labels': ['train'],
            'alpha': 0.8,
            'A': DAD,
            'num_propagations': 50,
            'display': False,
        }
        gat_fn = only_outcome_correlation
    elif args.dataset == 'products':
        lp_dict = {
            'idxs': ['train'],
            'alpha': 0.5,
            'num_propagations': 50,
            'A': DAD,
        }

        plain_dict = {
            'train_only': True,
            'alpha1': 1.0,
            'alpha2': 0.9,
            'scale': 20.0,
            'A1': DAD,
            'A2': DAD,
            'num_propagations1': 50,
            'num_propagations2': 50,
        }
        plain_fn = double_correlation_fixed

        linear_dict = {
            'train_only': True,
            'alpha1': 1.0,
            'alpha2': 0.9,
            'scale': 20.0,
            'A1': DAD,
            'A2': DAD,
            'num_propagations1': 50,
            'num_propagations2': 50,
        }
        linear_fn = double_correlation_fixed

        mlp_dict = {
            'train_only': True,
            'alpha1': 1.0,
            'alpha2': 0.8,
            'scale': 10.0,
            'A1': DAD,
            'A2': DA,
            'num_propagations1': 50,
            'num_propagations2': 50,
        }
        mlp_fn = double_correlation_fixed
    elif args.dataset == 'cora':
        # TODO tune hyper-parameters on ogbn-cora (presently, a copy of ogbn-arxiv)
        lp_dict = {
            'idxs': ['train'],
            'alpha': 0.9,
            'num_propagations': 50,
            'A': AD,
        }
        plain_dict = {
            'train_only': True,
            'alpha1': 0.87,
            'A1': AD,
            'num_propagations1': 50,
            'alpha2': 0.81,
            'A2': DAD,
            'num_propagations2': 50,
            'display': False,
        }
        plain_fn = double_correlation_autoscale


        plain_dict_gen_bound = {
            'train_only': True,
            'alpha1': 0.87,
            'A1': AD,
            'num_propagations1': 50,
            'alpha2': 0.81,
            'A2': DAD,
            'num_propagations2': 50,
            'display': False,
        }
        plain_fn_gen_bound = double_correlation_autoscale_gen_bound

        plain_dict_gen_bound_no_prop = {
            'train_only': True,
            'alpha1': 0.87,
            'A1': AD,
            'num_propagations1': 50,
            'alpha2': 0.81,
            'A2': DAD,
            'num_propagations2': 50,
            'display': False,
        }
        plain_fn_gen_bound_no_prop = double_correlation_autoscale_gen_bound

        """
        If you tune hyperparameters on test set
        {'alpha1': 0.9988673963255859,
            'alpha2': 0.7942279952481052, 'A1': 'DA', 'A2': 'AD'}
        gets you to 72.64
        """
        


        linear_dict = {
            'train_only': True,
            'alpha1': 0.98,
            'alpha2': 0.65,
            'A1': AD,
            'A2': DAD,
            'num_propagations1': 50,
            'num_propagations2': 50,
            'display': False,
        }
        linear_fn = double_correlation_autoscale



        linear_dict_gen_bound = {
            'train_only': True,
            'alpha1': 0.98,
            'alpha2': 0.65,
            'A1': AD,
            'A2': DAD,
            'num_propagations1': 50,
            'num_propagations2': 50,
            'display': False,
        }
        linear_fn_gen_bound = double_correlation_autoscale_gen_bound

        """
        If you tune hyperparameters on test set
        {'alpha1': 0.9956668128133523,
            'alpha2': 0.8542393515434346, 'A1': 'DA', 'A2': 'AD'}
        gets you to 73.35
        """
        mlp_dict = {
            'train_only': True,
            'alpha1': 0.9791632871592579,
            'alpha2': 0.7564990804200602,
            'A1': DA,
            'A2': AD,
            'num_propagations1': 50,
            'num_propagations2': 50,
            'display': False,
        }
        mlp_fn = double_correlation_autoscale


        mlp_dict_gen_bound = {
            'train_only': True,
            'alpha1': 0.9791632871592579,
            'alpha2': 0.7564990804200602,
            'A1': DA,
            'A2': AD,
            'num_propagations1': 50,
            'num_propagations2': 50,
            'display': False,
        }
        mlp_fn_gen_bound = double_correlation_autoscale_gen_bound

        gat_dict = {
            'labels': ['train'],
            'alpha': 0.8,
            'A': DAD,
            'num_propagations': 50,
            'display': False,
        }
        gat_fn = only_outcome_correlation
    split_string = args.method.split("_")
    print(split_string)
    if len(split_string)==1:
        model_outs = glob.glob(f'models/{args.dataset}_{args.method}/*.pt')
    else:
        method_to_use = split_string[0]
        model_outs = glob.glob(f'models/{args.dataset}_{method_to_use}/*.pt')
        print(f'models/{args.dataset}_{method_to_use}/*.pt')
    
    # try:
    #     model_outs = glob.glob(f'models/{args.dataset}_{args.method}/*.pt')
    # except:
    #     split_string = args.method.split("_")
    #     method_to_use = split_string[0]
    #     model_outs = glob.glob(f'models/{args.dataset}_{method_to_use}/*.pt')
    #     print(f'models/{args.dataset}_{split_string}/*.pt')

    if args.method == 'lp':
        out = label_propagation(data, split_idx, **lp_dict)
        print('Valid acc: ', eval_test(out, split_idx['valid']))
        print('Test acc:', eval_test(out, split_idx['test']))
        return

    get_orig_acc(data, eval_test, model_outs, split_idx)
    while True:
        if args.method == 'plain':
            evaluate_params(data, eval_test, model_outs,
                            split_idx, plain_dict, fn=plain_fn)
            new_node_neighbor_acc = evaluate_params_new_nodes(data, eval_test, model_outs,
                            split_idx,neighbor_nodes, plain_dict, fn=plain_fn)
            print("new_node_neighbor_acc = ",new_node_neighbor_acc)
        elif args.method == 'plain_gen_bound':
            evaluate_params(data, eval_test, model_outs,
                            split_idx, plain_dict_gen_bound, fn=plain_fn_gen_bound)
        elif args.method == 'linear_gen_bound':
            evaluate_params(data, eval_test, model_outs,
                            split_idx, linear_dict_gen_bound, fn=linear_fn_gen_bound)
        elif args.method == 'mlp_gen_bound':
            evaluate_params(data, eval_test, model_outs,
                            split_idx, mlp_dict_gen_bound, fn=mlp_fn_gen_bound)
        elif args.method == 'linear':
            evaluate_params(data, eval_test, model_outs,
                            split_idx, linear_dict, fn=linear_fn)
            new_node_neighbor_acc = evaluate_params_new_nodes(data, eval_test, model_outs,
                            split_idx,neighbor_nodes, plain_dict, fn=plain_fn)
            print("new_node_neighbor_acc = ",new_node_neighbor_acc)
        elif args.method == 'mlp':
            evaluate_params(data, eval_test, model_outs,
                            split_idx, mlp_dict, fn=mlp_fn)
            new_node_neighbor_acc = evaluate_params_new_nodes(data, eval_test, model_outs,
                            split_idx,neighbor_nodes, plain_dict, fn=plain_fn)
            print("new_node_neighbor_acc = ",new_node_neighbor_acc)
        elif args.method == 'gat':
            evaluate_params(data, eval_test, model_outs,
                            split_idx, gat_dict, fn=gat_fn)
#         import pdb; pdb.set_trace()
        break

#     name = f'{args.experiment}_{args.search_type}_{args.model_dir}'
#     setup_experiments(data, eval_test, model_outs, split_idx, normalized_adjs, args.experiment, args.search_type, name, num_iters=300)

#     return


if __name__ == "__main__":
    main()
