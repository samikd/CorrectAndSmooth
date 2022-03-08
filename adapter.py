from curses.ascii import isspace
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.io import DatasetSaver
import numpy as np
from dgl.data import *


def dgl_to_ogbn(dataset_name, mapping_path, is_sparse=False):
    if 'cora' == dataset_name:
        dataset = CoraGraphDataset()
        dataset_name = 'ogbn-cora'

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
