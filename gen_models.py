import argparse
import pickle
import pandas as pd

import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import math

from copy import deepcopy
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
import numpy as np


from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from outcome_correlation import prepare_folder
from diffusion_feature import preprocess
import glob
import os
import shutil

from logger import Logger

from adapter import *

select_top = 0.10
additional_traning_nodes_class ={}

def get_train_weights(num_of_nodes,split_idx):
    train_idx_weights = [0]*num_of_nodes
    for i in range(len(split_idx['train'])):
        index = split_idx['train'][i]
        train_idx_weights[index] = 1

    file_path = "./coraCSbounds.pkl"
    gen_bound_error_dict = pickle.load(open(file_path, "rb"))

    gen_bound_error_dict = {k: v for k, v in sorted(gen_bound_error_dict.items(), key=lambda item: item[1])}
    # print(gen_bound_error_dict)
    # print("___________________________________________________________")
    min_gen_key  = list(gen_bound_error_dict.keys())[0]
    min_gen_value  = gen_bound_error_dict[min_gen_key]

    for key in gen_bound_error_dict.keys():
        new_val = gen_bound_error_dict[key]
        new_val = (1/new_val)*min_gen_value
        gen_bound_error_dict[key] = new_val
        train_idx_weights[key] = new_val
    return train_idx_weights
    

class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, relu_first=True):
        super(MLP, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.relu_first = relu_first

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            if self.relu_first:
                x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            if not self.relu_first:
                x = F.relu(x, inplace=True)

            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return F.log_softmax(x, dim=-1)


class MLPLinear(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLPLinear, self).__init__()
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x):
        return F.log_softmax(self.lin(x), dim=-1)


def train_pre(model, x, y_true, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(x[train_idx])
    loss = F.nll_loss(out, y_true.squeeze(1)[train_idx]) 
    loss.backward()
    optimizer.step()

    return loss.item()


  
def train(model, x, y_true, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(x[train_idx])
    loss = F.nll_loss(out, y_true.squeeze(1)[train_idx]) 
    loss.backward()
    optimizer.step()

    return loss.item()

def train_with_weights(model, x, y_true, train_idx, optimizer,weights):
    model.train()

    optimizer.zero_grad()
    out = model(x[train_idx])
    loss = F.nll_loss(out, y_true.squeeze(1)[train_idx],reduction='none') 
    # print("loss = ",loss.shape)
    # print("loss = ",type(loss))
    # print("loss = ",loss)
    loss = torch.mul(loss,weights)
    loss = loss.mean()
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test_retrain(model, x, y, split_idx, evaluator):
    model.eval()

    out = model(x)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']

    y_true_valid = y[split_idx['valid']]
    y_idx_valid = split_idx['valid']
    # print("y_idx_valid = ",y_idx_valid)
    y_pred_valid = y_pred[split_idx['valid']]
    # print("y_pred_valid = ",y_pred_valid)

    top_selected_value = math.floor(select_top * len(y_true_valid) )
    # print("top_selected_value = ",top_selected_value)
    file_path = "/media/harsh/forUbuntu/Common_Drive/IIT_KGP/Subgroup_generalization/c and s/samik/CorrectAndSmooth/coraCSbounds.pkl"
    gen_bound_error_dict = pickle.load(open(file_path, "rb"))

    gen_bound_error_dict = {k: v for k, v in sorted(gen_bound_error_dict.items(), key=lambda item: item[1])}
    # print("gen_bound_error_dict = ",gen_bound_error_dict)
    # count=0
    # for key in gen_bound_error_dict.keys():
    #     key_index = np.where(y_idx_valid==key)
    #     # print("key_index1",key_index)
    #     key_index = key_index[0]
    #     if len(key_index>0):
    #         key_index = key_index[0]
    #         # print("key_index2",key_index)
    #         # print("y_pred_valid[key_index]",y_pred_valid[key_index][0].item())
    #         additional_traning_nodes_class[key] = y_pred_valid[key_index][0].item()
    #         count+=1
    #         if count> top_selected_value:
    #             break
    
    # print(additional_traning_nodes_class)

    # while True:
    #     pass
    test_acc = evaluator.eval({
        'y_true': y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']
    y_idx_test = split_idx['test']
    y_pred_test = y_pred[split_idx['test']]
    count=0
    for key in gen_bound_error_dict.keys():
        key_index = np.where(y_idx_valid==key)
        # print("key_index1",key_index)
        key_index = key_index[0]
        if len(key_index>0):
            key_index = key_index[0]
            # print("key_index2",key_index)
            # print("y_pred_valid[key_index]",y_pred_valid[key_index][0].item())
            additional_traning_nodes_class[key] = y_pred_valid[key_index][0].item()
            count+=1
            if count> top_selected_value:
                break
        else:
            key_index = np.where(y_idx_test==key)
            # print("key_index1",key_index)
            key_index = key_index[0]
            if len(key_index>0):
                key_index = key_index[0]
                # print("key_index2",key_index)
                # print("y_pred_valid[key_index]",y_pred_valid[key_index][0].item())
                ans = y_pred_test[key_index][0].item()
                additional_traning_nodes_class[key] = ans
                count+=1
                if count> top_selected_value:
                    break


    return (train_acc, valid_acc, test_acc), out


@torch.no_grad()
def test(model, x, y, split_idx, evaluator):
    model.eval()

    out = model(x)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return (train_acc, valid_acc, test_acc), out

def main():
    parser = argparse.ArgumentParser(description='gen_models')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='arxiv')
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--use_embeddings', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--runs', type=int, default=10)

    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    try:
        dataset = PygNodePropPredDataset(
            name=f'ogbn-{args.dataset}', transform=T.ToSparseTensor())
    except ValueError:
        dataset = dgl_to_ogbn(
            args.dataset, 'ogbn-cora-submission', is_sparse=True)
    data = dataset[0]

    data = dataset[0]
    # print("data = ",data)
    # print("data.x = ",data.x)
    data.adj_t = data.adj_t.to_symmetric()

    x = data.x
    # while True:
    #     pass
    split_idx = dataset.get_idx_split()
    # print("train_split = ",split_idx["train"])
    
    
    # while True:
    #     pass
    try:
        preprocess_data = PygNodePropPredDataset(
            name=f'ogbn-{args.dataset}')[0]
    except ValueError:
        preprocess_data = dgl_to_ogbn(
            args.dataset, 'ogbn-cora-submission')[0]

    if args.dataset == 'arxiv':
        embeddings = torch.cat([preprocess(preprocess_data, 'diffusion', post_fix=args.dataset),
                                preprocess(preprocess_data, 'spectral', post_fix=args.dataset)], dim=-1)
    elif args.dataset == 'products':
        embeddings = preprocess(
            preprocess_data, 'spectral', post_fix=args.dataset)
    elif args.dataset == 'cora':
        embeddings = preprocess(
            preprocess_data, 'diffusion', post_fix=args.dataset, is_ogb_submission=True)

    if args.use_embeddings:
        x = torch.cat([x, embeddings], dim=-1)

    if args.dataset == 'arxiv':
        x = (x-x.mean(0))/x.std(0)

    if args.model == 'mlp':
        # model = MLP(x.size(-1), args.hidden_channels, dataset.num_classes,
        #             args.num_layers, 0.5, args.dataset == 'products').cuda()
        model = MLP(x.size(-1), args.hidden_channels, dataset.num_classes,
                    args.num_layers, 0.5, args.dataset == 'products').to(device)
    elif args.model == 'linear':
        # model = MLPLinear(x.size(-1), dataset.num_classes).cuda()
        model = MLPLinear(x.size(-1), dataset.num_classes).to(device)
    elif args.model == 'plain':
        # model = MLPLinear(x.size(-1), dataset.num_classes).cuda()
        model = MLPLinear(x.size(-1), dataset.num_classes).to(device)

    x = x.to(device)
    y_true = data.y.to(device)
    
    print("## x = ",x.shape)
    print("## x = ",x)
    
    print("## y_true = ",y_true.shape)
    print("## y_true = ",y_true)
    if torch.is_tensor(split_idx['train']):
        train_idx = split_idx['train'].to(device)
    else:
        train_idx = torch.tensor(split_idx['train']).to(device)

    print("## train_split = ",len(train_idx))
    print("## train_split = ",train_idx)
    model_dir = prepare_folder(f'{args.dataset}_{args.model}', model)

    evaluator = Evaluator(name=f'ogbn-{args.dataset}')
    logger = Logger(args.runs, args)
    


    for run in range(args.runs):
        import gc
        gc.collect()
        print(sum(p.numel() for p in model.parameters()))
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_valid = 0
        best_out = None
        for epoch in range(1, args.epochs):
            loss = train_pre(model, x, y_true, train_idx, optimizer)
            result, out = test_retrain(model, x, y_true, split_idx, evaluator)
            train_acc, valid_acc, test_acc = result
            if valid_acc > best_valid:
                best_valid = valid_acc
                best_out = out.cpu().exp()

            print(f'Run: {run + 1:02d}, '
                  f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * train_acc:.2f}%, '
                  f'Valid: {100 * valid_acc:.2f}% '
                  f'Test: {100 * test_acc:.2f}%')
            logger.add_result(run, result)

        logger.print_statistics(run)
        # torch.save(best_out, f'{model_dir}/{run}.pt')

    logger.print_statistics()


    temp_new_added_node = []
    for key in additional_traning_nodes_class.keys():
        pred_class = additional_traning_nodes_class[key]
        y_true[key][0] = pred_class
        # x[key][pred_class] = 1
        train_node = torch.tensor([key])
        temp_new_added_node.append(train_node)
        train_idx = torch.cat((train_idx,train_node), dim=-1)
    print("## y_true = ",y_true.shape)
    print("## y_true = ",y_true)
    t_np = train_idx.numpy() #convert to Numpy array
    df = pd.DataFrame(t_np) #convert to a dataframe
    df.to_csv("cora_splits/new_train_idx.csv",index=False) #save to file
    temp_new_added_node = torch.tensor(temp_new_added_node)
    t_np = temp_new_added_node.numpy() #convert to Numpy array
    df = pd.DataFrame(t_np) #convert to a dataframe
    if select_top>0.0:
        df.to_csv("cora_splits/new_added_train_idx.csv",index=False) #save to file
    # print("## x = ",x.shape)
    # print("## x = ",x)
    # print("## x[0] = ",x[0][3])
    print("## train_split = ",len(train_idx))
    print("## train_split = ",train_idx)

    train_idx_weights = get_train_weights(data.num_nodes,split_idx)
    print(train_idx_weights)
    # train_idx_weights = torch.tensor(train_idx_weights)
    train_idx_weights_temp =[]
    for id in train_idx:
        train_idx_weights_temp.append(train_idx_weights[id])
    train_idx_weights_temp = torch.tensor(train_idx_weights_temp, dtype=torch.long)  
    train_idx_weights = train_idx_weights_temp
    # for id in range(data.num_nodes):
    #     if id not in train_idx:
    #         train_idx_weights[id] = 0 

    # print(train_idx_weights)

    # while True:
    #     pass
    for run in range(args.runs):
            import gc
            gc.collect()
            print(sum(p.numel() for p in model.parameters()))
            model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            best_valid = 0
            best_out = None
            for epoch in range(1, args.epochs):
                # loss = train_with_weights(model, x, y_true, train_idx, optimizer,train_idx_weights)
                loss = train(model, x, y_true, train_idx, optimizer)
                result, out = test(model, x, y_true, split_idx, evaluator)
                train_acc, valid_acc, test_acc = result
                if valid_acc > best_valid:
                    best_valid = valid_acc
                    best_out = out.cpu().exp()

                print(f'Run: {run + 1:02d}, '
                    f'Epoch: {epoch:02d}, '
                    f'Loss: {loss:.4f}, '
                    f'Train: {100 * train_acc:.2f}%, '
                    f'Valid: {100 * valid_acc:.2f}% '
                    f'Test: {100 * test_acc:.2f}%')
                logger.add_result(run, result)

            logger.print_statistics(run)
            torch.save(best_out, f'{model_dir}/{run}.pt')

    logger.print_statistics()



if __name__ == "__main__":
    main()



