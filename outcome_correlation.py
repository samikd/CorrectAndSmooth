import pickle
import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import os
from collections import defaultdict
import glob
import random

from copy import deepcopy
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected, is_undirected
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import optuna

from logger import Logger
import random
import shutil
import glob
from collections.abc import Iterable
import joblib


class SimpleLogger(object):
    def __init__(self, desc, param_names, num_values=2):
        self.results = defaultdict(dict)
        self.param_names = tuple(param_names)
        self.used_args = list()
        self.desc = desc
        self.num_values = num_values

    def add_result(self, run, args, values):
        """Takes run=int, args=tuple, value=tuple(float)"""
        assert(len(args) == len(self.param_names))
        assert(len(values) == self.num_values)
        self.results[run][args] = values
        if args not in self.used_args:
            self.used_args.append(args)

    def get_best(self, top_k=1):
        all_results = []
        for args in self.used_args:
            results = [i[args] for i in self.results.values() if args in i]
            results = torch.tensor(results)*100
            results_mean = results.mean(dim=0)[-1]
            results_std = results.std(dim=0)

            all_results.append((args, results_mean))
        results = sorted(all_results, key=lambda x: x[1], reverse=True)[:top_k]
        return [i[0] for i in results]

    def prettyprint(self, x):
        if isinstance(x, float):
            return '%.2f' % x
        return str(x)

    def display(self, args=None):

        disp_args = self.used_args if args is None else args
        if len(disp_args) > 1:
            print(f'{self.desc} {self.param_names}, {len(self.results.keys())} runs')
        for args in disp_args:
            results = [i[args] for i in self.results.values() if args in i]
            results = torch.tensor(results)*100
            results_mean = results.mean(dim=0)
            results_std = results.std(dim=0)
            res_str = f'{results_mean[0]:.2f} ± {results_std[0]:.2f}'
            for i in range(1, self.num_values):
                res_str += f' -> {results_mean[i]:.2f} ± {results_std[1]:.2f}'
            print(f'Args {[self.prettyprint(x) for x in args]}: {res_str}')
        if len(disp_args) > 1:
            print()


def process_adj(data, is_ogb_submission=False):
    N = data.num_nodes

    # FIXME figure out what's going on with "cora"
    if not is_ogb_submission:
        data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    row, col = data.edge_index

    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    deg = adj.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    return adj, deg_inv_sqrt


def gen_normalized_adjs(adj, D_isqrt):
    DAD = D_isqrt.view(-1, 1)*adj*D_isqrt.view(1, -1)
    DA = D_isqrt.view(-1, 1) * D_isqrt.view(-1, 1)*adj
    AD = adj*D_isqrt.view(1, -1) * D_isqrt.view(1, -1)
    return DAD, DA, AD


def gen_normalized_adj(adj, pw):  # pw = 0 is D^-1A, pw=1 is AD^-1
    deg = adj.sum(dim=1).to(torch.float)
    front = deg.pow(-(1-pw))
    front[front == float('inf')] = 0
    back = deg.pow(-(pw))
    back[back == float('inf')] = 0
    return (front.view(-1, 1)*adj*back.view(1, -1))


def model_load(file, device='cpu'):
    result = torch.load(file, map_location='cpu')
    run = get_run_from_file(file)
    try:
        split = torch.load(f'{file}.split', map_location='cpu')
    except:
        split = None

    mx_diff = (result.sum(dim=-1) - 1).abs().max()
    if mx_diff > 1e-1:
        print(f'Max difference: {mx_diff}')
        print("model output doesn't seem to sum to 1. Did you remember to exp() if your model outputs log_softmax()?")
        raise Exception
    if split is not None:
        return (result, split), run
    else:
        return result, run


def get_labels_from_name(labels, split_idx):
    if isinstance(labels, list):
        labels = list(labels)
        if len(labels) == 0:
            return torch.tensor([])
        for idx, i in enumerate(list(labels)):
            if torch.is_tensor(split_idx[i]):
                labels[idx] = split_idx[i]
            else:
                labels[idx] = torch.tensor(split_idx[i])
        residual_idx = torch.cat(labels)
    else:
        residual_idx = split_idx[labels]
    return residual_idx


def pre_residual_correlation(labels, model_out, label_idx):
    """Generates the initial labels used for residual correlation"""
    # E = Z - Y (page 4, section 2.2 equation 1)
    labels = labels.cpu()
    labels[labels.isnan()] = 0
    labels = labels.long()
    model_out = model_out.cpu()
    if torch.is_tensor(label_idx):
        label_idx = label_idx.cpu()
    else:
        label_idx = torch.tensor(label_idx).cpu()
    # label_idx = label_idx.cpu()
    c = labels.max() + 1
    n = labels.shape[0]
    y = torch.zeros((n, c))
    
    F_one_hot_code = F.one_hot(labels[label_idx], c).float().squeeze(1)
    print("*** y shape = ",y.dtype)
    print("*** model_out shape = ",model_out.dtype)
    print("*** label_idx shape = ",label_idx.dtype)
    print("*** F_one_hot_code shape = ",F_one_hot_code.dtype)
    y[label_idx] =  F_one_hot_code- model_out[label_idx]
    print("y[7]",y[7])
    print("*** y[6]  = ",y[6])
    print("*** y[5]  = ",y[5])
    print("*** y[4]  = ",y[4])
    return y


def pre_residual_correlation_gen_bound(labels, model_out, label_idx):
    """Generates the initial labels used for residual correlation"""
    # E = Z - Y (page 4, section 2.2 equation 1)
    labels = labels.cpu()
    labels[labels.isnan()] = 0
    labels = labels.long()
    model_out = model_out.cpu()
    if torch.is_tensor(label_idx):
        label_idx = label_idx.cpu()
    else:
        label_idx = torch.tensor(label_idx).cpu()
    # label_idx = label_idx.cpu()
    c = labels.max() + 1
    n = labels.shape[0]
    file_path = "./coraCSbounds-2.pkl"
    gen_bound_error_dict = pickle.load(open(file_path, "rb"))
    # E_matrix_res = get_E_from_gen_bound_dict(gen_bound_error_dict,model_out.shape[1],model_out.shape[0])    
    E_matrix_res = get_E_from_gen_bound_dict_distributed(gen_bound_error_dict,model_out.shape[1],model_out.shape[0],model_out)    
    # E_matrix_res = get_E_from_gen_bound_random_error(gen_bound_error_dict,model_out.shape[1],model_out.shape[0],model_out)    
    
    # y = torch.zeros((n, c))
    y = E_matrix_res
    F_one_hot_code = F.one_hot(labels[label_idx], c).float().squeeze(1)
    # print("*** y[7]  = ",y[7])

    # print("*** y shape = ",y.shape)
    # print("*** model_out shape = ",model_out.shape)
    # print("*** label_idx shape = ",label_idx.shape)
    # print("*** F_one_hot_code shape = ",F_one_hot_code.shape)
    y[label_idx] = F_one_hot_code - model_out[label_idx]
    print("y",y)
    # print("*** model_out = ",model_out)
    # print("*** y = ",y)
    # print("*** y[7]  = ",y[7])
    # print("*** y[6]  = ",y[6])
    # print("*** y[5]  = ",y[5])
    # print("*** y[4]  = ",y[4])

    # while True:
    #     pass
    return y


def pre_outcome_correlation(labels, model_out, label_idx):
    """Generates the initial labels used for outcome correlation"""

    labels = labels.cpu()
    model_out = model_out.cpu()
    label_idx = label_idx.cpu()
    c = labels.max() + 1
    n = labels.shape[0]
    y = model_out.clone()
    if len(label_idx) > 0:

        try:
            # print("y[label_idx] = ",y[label_idx].shape)
            y[label_idx] = F.one_hot(labels[label_idx], c).float().squeeze(1)
            # print("x_one = ",y[label_idx].shape)
        except:
            # print("y[label_idx] = ",y[label_idx][0].shape)

            x_one = F.one_hot(labels[label_idx][0], c).float().squeeze(1)
            # print("x_one = ",x_one.shape)
            y[label_idx] = x_one
    # while True:
    #     pass
    return y

# FIXME un-pin "device" (currently pinned to "cpu" -- "cuda" earlier)
# def general_outcome_correlation(adj, y, alpha, num_propagations, post_step, alpha_term, device='cuda', display=True):


def general_outcome_correlation(adj, y, alpha, num_propagations, post_step, alpha_term, device='cpu', display=True):
    """general outcome correlation. alpha_term = True for outcome correlation, alpha_term = False for residual correlation"""
    # E = (1-a)E + aSE , page 4, section 2.2, paragraph after eq 2, line 4th 
    # adj = adj.to(device)
    adj = adj.cpu()
    orig_device = y.device
    # y = y.to(device)
    y = y.cpu()
    result = y.clone()
    for _ in tqdm(range(num_propagations), disable=not display):
        result = alpha * (adj @ result)
        if alpha_term:
            result += (1-alpha)*y
        else:
            result += y
        result = post_step(result)
    return result.to(orig_device)


def label_propagation(data, split_idx, A, alpha, num_propagations, idxs):
    labels = data.y.data
    c = labels.max() + 1
    n = labels.shape[0]
    y = torch.zeros((n, c))
    label_idx = get_labels_from_name(idxs, split_idx)
    y[label_idx] = F.one_hot(labels[label_idx], c).float().squeeze(1)

    return general_outcome_correlation(A, y, alpha, num_propagations, post_step=lambda x: torch.clamp(x, 0, 1), alpha_term=True)


def double_correlation_autoscale(data, model_out, split_idx, A1, alpha1, num_propagations1, A2, alpha2, num_propagations2, scale=1.0, train_only=False, device='cuda', display=True):
    train_idx, valid_idx, test_idx = split_idx
    print("*******************************************")

    print("split_idx['train'] = ",len(split_idx['train']))
    print("model_out = ",model_out.shape)
    print("model_out = ",model_out)

    try:
        if train_only:
            label_idx = torch.cat([split_idx['train']])
            residual_idx = split_idx['train']
        else:
            label_idx = torch.cat([split_idx['train'], split_idx['valid']])
            residual_idx = label_idx
    except:
        if train_only:
            label_idx = torch.tensor([split_idx['train']])
            # print("label_idx = ",label_idx)

            residual_idx = split_idx['train']
        else:
            label_idx = torch.cat([split_idx['train'], split_idx['valid']])
            residual_idx = label_idx
    
    y = pre_residual_correlation(
        labels=data.y.data, model_out=model_out, label_idx=residual_idx)
    resid = general_outcome_correlation(adj=A1, y=y, alpha=alpha1, num_propagations=num_propagations1, post_step=lambda x: torch.clamp(x, -1.0, 1.0), alpha_term=True, display=display, device=device)
    # resid = y 
    # we'll have E hat here in resid, 
    orig_diff = y[residual_idx].abs().sum()/residual_idx.shape[0]
    resid_scale = (orig_diff/resid.abs().sum(dim=1, keepdim=True))
    resid_scale[resid_scale.isinf()] = 1.0
    cur_idxs = (resid_scale > 1000)
    resid_scale[cur_idxs] = 1.0
    res_result = model_out + resid_scale*resid
    res_result[res_result.isnan()] = model_out[res_result.isnan()]

    
    #we'll have Z from autoscale page 4 3rd last line 

    #below is smoothing process, section 2.3 page 5 
    y = pre_outcome_correlation(
        labels=data.y.data, model_out=res_result, label_idx=label_idx)
    result = general_outcome_correlation(adj=A2, y=y, alpha=alpha2, num_propagations=num_propagations2,
                                         post_step=lambda x: torch.clamp(x, 0, 1), alpha_term=True, display=display, device=device)
    print("#############################################")
    print("double_correlation_autoscale")
    # print("double_correlation_autoscale")
    # print("res_result = ",res_result.shape)
    # print("res_result = ",res_result)
    # print("result = ",result.shape)
    # print("result = ",result)
    # print("^#############################################")
    # print("^*******************************************")

    return res_result, result


def double_correlation_autoscale_gen_bound(data, model_out, split_idx, A1, alpha1, num_propagations1, A2, alpha2, num_propagations2, scale=1.0, train_only=False, device='cuda', display=True):
    train_idx, valid_idx, test_idx = split_idx
    print("double_correlation_autoscale_with_gen_bound *******************************************")

    print("split_idx['train'] = ",len(split_idx['train']))
    # print("model_out = ",model_out.shape)
    # print("model_out = ",model_out)

    try:
        if train_only:
            label_idx = torch.cat([split_idx['train']])
            residual_idx = split_idx['train']
        else:
            label_idx = torch.cat([split_idx['train'], split_idx['valid']])
            residual_idx = label_idx
    except:
        if train_only:
            label_idx = torch.tensor([split_idx['train']])
            # print("label_idx = ",label_idx)

            residual_idx = split_idx['train']
        else:
            label_idx = torch.cat([split_idx['train'], split_idx['valid']])
            residual_idx = label_idx
    
    y = pre_residual_correlation_gen_bound(
        labels=data.y.data, model_out=model_out, label_idx=residual_idx)
    resid = general_outcome_correlation(adj=A1, y=y, alpha=alpha1, num_propagations=num_propagations1, post_step=lambda x: torch.clamp(x, -1.0, 1.0), alpha_term=True, display=display, device=device)
    # resid = y
    # we'll have E hat here in resid, 

    orig_diff = y[residual_idx].abs().sum()/residual_idx.shape[0]
    resid_scale = (orig_diff/resid.abs().sum(dim=1, keepdim=True))
    resid_scale[resid_scale.isinf()] = 1.0
    cur_idxs = (resid_scale > 1000)
    resid_scale[cur_idxs] = 1.0
    res_result = model_out + resid_scale*resid
    res_result[res_result.isnan()] = model_out[res_result.isnan()]

    # res_result = resid
    #we'll have Z from autoscale page 4 3rd last line 

    #below is smoothing process, section 2.3 page 5 
    y = pre_outcome_correlation(
        labels=data.y.data, model_out=res_result, label_idx=label_idx)
    result = general_outcome_correlation(adj=A2, y=y, alpha=alpha2, num_propagations=num_propagations2,
                                         post_step=lambda x: torch.clamp(x, 0, 1), alpha_term=True, display=display, device=device)
    # print("double_correlation_autoscale")
    # print("res_result = ",res_result.shape)
    # print("res_result = ",res_result)
    # print("result = ",result.shape)
    # print("result = ",result)
    # print("^#############################################")
    # print("^*******************************************")

    return res_result, result



def double_correlation_fixed(data, model_out, split_idx, A1, alpha1, num_propagations1, A2, alpha2, num_propagations2, scale=1.0, train_only=False, device='cuda', display=True):
    train_idx, valid_idx, test_idx = split_idx
    if train_only:
        label_idx = torch.cat([split_idx['train']])
        residual_idx = split_idx['train']

    else:
        label_idx = torch.cat([split_idx['train'], split_idx['valid']])
        residual_idx = label_idx

    y = pre_residual_correlation(
        labels=data.y.data, model_out=model_out, label_idx=residual_idx)

    fix_y = y[residual_idx].to(device)

    def fix_inputs(x):
        x[residual_idx] = fix_y
        return x

    resid = general_outcome_correlation(adj=A1, y=y, alpha=alpha1, num_propagations=num_propagations1,
                                        post_step=lambda x: fix_inputs(x), alpha_term=True, display=display, device=device)
    res_result = model_out + scale*resid

    y = pre_outcome_correlation(
        labels=data.y.data, model_out=res_result, label_idx=label_idx)

    result = general_outcome_correlation(adj=A2, y=y, alpha=alpha2, num_propagations=num_propagations2,
                                         post_step=lambda x: x.clamp(0, 1), alpha_term=True, display=display, device=device)

    return res_result, result


def only_outcome_correlation(data, model_out, split_idx, A, alpha, num_propagations, labels, device='cuda', display=True):
    res_result = model_out.clone()
    label_idxs = get_labels_from_name(labels, split_idx)
    y = pre_outcome_correlation(
        labels=data.y.data, model_out=model_out, label_idx=label_idxs)
    result = general_outcome_correlation(adj=A, y=y, alpha=alpha, num_propagations=num_propagations,
                                         post_step=lambda x: torch.clamp(x, 0, 1), alpha_term=True, display=display, device=device)
    return res_result, result


def evaluate_params(data, eval_test, model_outs, split_idx, params, fn=double_correlation_autoscale):
    logger = SimpleLogger('evaluate params', [], 2)

    for out in model_outs:
        model_out, run = model_load(out)
        if isinstance(model_out, tuple):
            model_out, t = model_out
            split_idx = t
        res_result, result = fn(data, model_out, split_idx, **params)
        
        # while True:
        #     pass
        valid_acc, test_acc = eval_test(
            result, split_idx['valid']), eval_test(result, split_idx['test'])
        print(f"Valid: {valid_acc}, Test: {test_acc}")
        logger.add_result(run, (), (valid_acc, test_acc))
    print("RESULTS = ",result)
    print('Valid acc -> Test acc')
    logger.display()
    return logger

def evaluate_params_new_nodes(data, eval_test, model_outs, split_idx,neighbor_nodes, params, fn=double_correlation_autoscale):
    logger = SimpleLogger('evaluate params', [], 2)

    for out in model_outs:
        model_out, run = model_load(out)
        if isinstance(model_out, tuple):
            model_out, t = model_out
            split_idx = t
        res_result, result = fn(data, model_out, split_idx, **params)
        
        # while True:
        #     pass
        new_node_acc = eval_test(result, neighbor_nodes)
        print(f"new_node_acc: {new_node_acc}")
        logger.add_result(run, (), (new_node_acc,0))
    print("new_node_acc RESULTS = ",result)
    print('new_node_acc , len = ',len(neighbor_nodes))
    logger.display()
    return logger


def get_run_from_file(out):
    return int(os.path.splitext(os.path.basename(out))[0])


def get_orig_acc(data, eval_test, model_outs, split_idx):
    logger_orig = Logger(len(model_outs))
    for out in model_outs:
        model_out, run = model_load(out)
        if isinstance(model_out, tuple):
            model_out, split_idx = model_out
        test_acc = eval_test(model_out, split_idx['test'])
        logger_orig.add_result(run, (eval_test(model_out, split_idx['train']), eval_test(
            model_out, split_idx['valid']), test_acc))
    print('Original accuracy')
    logger_orig.print_statistics()


def prepare_folder(name, model):
    model_dir = f'models/{name}'

    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    with open(f'{model_dir}/metadata', 'w') as f:
        f.write(f'# of params: {sum(p.numel() for p in model.parameters())}\n')
    return model_dir


def get_E_from_gen_bound_dict(errors_dict,number_of_classes,num_nodes):
    E_matrix = [[0]*number_of_classes]*num_nodes
    
    for key in errors_dict.keys():
        e = errors_dict[key]
        row = [e]*number_of_classes
        E_matrix[key]= row
    E_matrix = torch.tensor(E_matrix, dtype=torch.float32)   
    return E_matrix

def get_E_from_gen_bound_dict_distributed(errors_dict,number_of_classes,num_nodes,model_out):
    E_matrix = [[0]*number_of_classes]*num_nodes
    
    for key in errors_dict.keys():
        # print("model_out[key]",model_out[key])
        pred_class_from_model = model_out[key].argmax(dim=-1, keepdim=True)
        # print("pred_class_from_model = ",pred_class_from_model)
        # print("pred_class_from_model",pred_class_from_model[0])
        e = errors_dict[key]
        # print("pred_class_from_model E = ",e)

        pred_class_error = 1-e 
        rest_class_error = e/(number_of_classes-1)
        row = [rest_class_error]*number_of_classes
        row[pred_class_from_model] = pred_class_error
        E_matrix[key]= row
        # print("row",row)
    E_matrix = torch.tensor(E_matrix, dtype=torch.float32)   
    return E_matrix


def get_E_from_gen_bound_random_error(errors_dict,number_of_classes,num_nodes,model_out):
    E_matrix = [[0]*number_of_classes]*num_nodes
    
    for key in errors_dict.keys():
        row = [0]*number_of_classes
        for k in range(number_of_classes):
            num = random.randint(100,500)
            num = num/1000
            row[k] = num
        
        
        E_matrix[key]= row
        # print("row",row)
    E_matrix = torch.tensor(E_matrix, dtype=torch.float32)   
    return E_matrix




