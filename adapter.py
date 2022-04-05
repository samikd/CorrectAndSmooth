from curses.ascii import isspace
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset , DglNodePropPredDataset
from ogb.io import DatasetSaver
import numpy as np
from dgl.data import *


def dgl_to_ogbn(dataset_name, mapping_path, is_sparse=False):
    if 'cora' == dataset_name:
        dataset = CoraGraphDataset()
        dataset_name = 'ogbn-cora'

        print("#### cora type initial = ",type(dataset))
        

    saver = DatasetSaver(dataset_name=dataset_name, is_hetero=False, version=1)

    g = dataset[0].to_networkx()
    graph = dict()
    graph['edge_index'] = np.array(
        [(u, v) for (u, v, w) in g.edges]).transpose()
    num_nodes = len(g.nodes)
    graph['num_nodes'] = num_nodes
    graph['node_feat'] = np.array(dataset[0].ndata['feat'])
    saver.save_graph_list([graph])

    saver.save_target_labels(
        np.array(dataset[0].ndata['label']).reshape((num_nodes, 1)))

    split_idx = dict()
    split_idx['train'] = np.array(g.nodes)[dataset[0].ndata['train_mask']]
    split_idx['valid'] = np.array(g.nodes)[dataset[0].ndata['val_mask']]
    split_idx['test'] = np.array(g.nodes)[dataset[0].ndata['test_mask']]
    # split_idx['train'] = np.load("cora_splits/cora_train_split.npy")
    # split_idx['valid'] = np.load("cora_splits/cora_validate_split.npy")
    # split_idx['test'] = np.load("cora_splits/cora_test_split.npy")

    print("train len = ",len(split_idx['train']))
    
    print("valid len = ",len(split_idx['valid']))
    print("test len = ",len(split_idx['test']))
    print("train type = ",type(split_idx['train']))
    print("train  = ",split_idx['train'])
    saver.save_split(split_idx, split_name='std')

    saver.copy_mapping_dir(mapping_path)

    saver.save_task_info(task_type='classification',
                         eval_metric='acc', num_classes=dataset.num_classes)
    meta_dict = saver.get_meta_dict()

    if is_sparse:
        pyg_dataset = PygNodePropPredDataset(
            dataset_name, meta_dict=meta_dict, transform=T.ToSparseTensor())
    else:
        pyg_dataset = PygNodePropPredDataset(dataset_name, meta_dict=meta_dict)

    saver.zip()

    return pyg_dataset


def dgl_to_dgl_ogbn(dataset_name, mapping_path, is_sparse=False):
    if 'cora' == dataset_name:
        dataset = CoraGraphDataset()
        dataset_name = 'ogbn-cora'
        print("#### cora type initial = ",type(dataset))
    saver = DatasetSaver(dataset_name=dataset_name, is_hetero=False, version=1)


    print("dataset[0] = ",type(dataset[0]))
    print("dataset[0] = ",dataset[0])

    g = dataset[0].to_networkx()
    print("cora g = ",g)
    graph = dict()
    graph['edge_index'] = np.array(
        [(u, v) for (u, v, w) in g.edges]).transpose()
    num_nodes = len(g.nodes)
    graph['num_nodes'] = num_nodes
    graph['node_feat'] = np.array(dataset[0].ndata['feat'])
    saver.save_graph_list([graph])

    saver.save_target_labels(
        np.array(dataset[0].ndata['label']).reshape((num_nodes, 1)))

    split_idx = dict()
    split_idx['train'] = np.array(g.nodes)[dataset[0].ndata['train_mask']]
    split_idx['valid'] = np.array(g.nodes)[dataset[0].ndata['val_mask']]
    split_idx['test'] = np.array(g.nodes)[dataset[0].ndata['test_mask']]
    print("train len = ",len(split_idx['train']))
    print("valid len = ",len(split_idx['valid']))
    print("test len = ",len(split_idx['test']))
    saver.save_split(split_idx, split_name='std')

    saver.copy_mapping_dir(mapping_path)

    saver.save_task_info(task_type='classification',
                         eval_metric='acc', num_classes=dataset.num_classes)
    meta_dict = saver.get_meta_dict()

    if is_sparse:
        pyg_dataset = DglNodePropPredDataset(
            dataset_name, meta_dict=meta_dict, transform=T.ToSparseTensor())
    else:
        pyg_dataset = DglNodePropPredDataset(dataset_name, meta_dict=meta_dict)

    saver.zip()

    return pyg_dataset
