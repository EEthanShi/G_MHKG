#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 10:48:48 2023

@author: daishi

generalized Heat kernel GCN (G_MHKG) for homophily graphs. 

"""

import numpy as np
from scipy import sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import get_laplacian
from torch_geometric.utils import to_undirected
import argparse
import os.path as osp


torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.FloatTensor)



class G_MHKG(nn.Module):
    def __init__(self, in_features, out_features, num_nodes, bias=True):
        super(G_MHKG, self).__init__()
        if torch.cuda.is_available():
            self.W = nn.Parameter(torch.Tensor(in_features, out_features).cuda())
            self.filter_1 = nn.Parameter(torch.Tensor(num_nodes, 1)).cuda()
            self.filter_2 = nn.Parameter(torch.Tensor(num_nodes, 1)).cuda()
        else:
            self.W = nn.Parameter(torch.Tensor(in_features, out_features))
            self.filter_1 = nn.Parameter(torch.Tensor(num_nodes, 1))
            self.filter_2 = nn.Parameter(torch.Tensor(num_nodes, 1))
        if bias:
            if torch.cuda.is_available():
                self.bias = nn.Parameter(torch.Tensor(out_features)).cuda()
            else:
                self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):

        nn.init.uniform_(self.filter_1, 0.9, 1.1)
        nn.init.uniform_(self.filter_2, 0.6, 0.8) # in general for homo, filter 1 shall be double for filter 2 and vice versa.
        nn.init.xavier_uniform_(self.W)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, eigen_vectors, lp, hp, x): #hp,lp high pass, low pass determined L.
        eigen_vectors = eigen_vectors.to(device)
        x = torch.matmul(x, self.W)
        x_1 = torch.mm(lp,eigen_vectors.T) # fourier transform
        x_1 = torch.mm(x_1, x) # heat kernel filtering
        x_1 = self.filter_1*x_1
        x_1 = torch.mm(eigen_vectors, x_1) # reconstruct

        x_2 = torch.mm(hp,eigen_vectors.T) # fourier transform
        x_2 = torch.mm(x_2, x) # heat kernel filtering
        x_2 = self.filter_2*x_2
        x_2 = torch.mm(eigen_vectors, x_2) # reconstruct
        x = x_1+x_2
        if self.bias is not None:
            x += self.bias
        return x

class Net(nn.Module):
    def __init__(self, num_features, nhid, num_classes,  num_nodes,num_layers = 2, activation = F.relu, dropout_prob=0.7):
        super(Net, self).__init__()
        self.GConv1 = G_MHKG(num_features, nhid, num_nodes)
        self.layers = num_layers
        if num_layers >2:
            self.hidden_layers = nn.ModuleList([
                G_MHKG(nhid,nhid,num_nodes)
                for ii in range(self.layers-2)
            ])
        self.GConv2 = G_MHKG(nhid, num_classes, num_nodes)
        self.drop1 = nn.Dropout(dropout_prob)
        self.act = activation

    def forward(self,eigen_vectors,lp,hp,data):
        x = torch.tensor(data.x).float().to(device)
        x = self.GConv1(eigen_vectors,lp,hp,x)
        x = self.act(x)
        x = self.drop1(x)
        if self.layers>2:
            for ii in range(self.layers - 2):
                x = self.act(self.hidden_layers[ii](eigen_vectors,lp,hp,x))
                x = self.drop1(x)

        x = self.GConv2(eigen_vectors,lp,hp,x)


        return F.log_softmax(x, dim=1)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f')
    parser.add_argument('--dataset', type=str, default='cora',
                        help='name of dataset (default: Cora)')
    parser.add_argument('--reps', type=int, default=10,
                        help='number of repetitions (default: 10)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers of the model (default: 2)')
    parser.add_argument('--initial_dynmaic_coefficient', type=float, default=1.1,
                        help='coefficient on L >1 for homo, less than 1 or 0 for hetero')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--wd', type=float, default=0.01,
                        help='weight decay (default: 1e-2)')
    parser.add_argument('--nhid', type=int, default=96, # 16,32,64,96 for arxiv is 64. for small graphs, nhid can be 96.
                        help='number of hidden units (default: 32)')
    parser.add_argument('--activation', type=str, default= 'sigmoid', # try different activations
                        help='activation function (default: relu): None, elu, sigmoid, relu, tanh')
    parser.add_argument('--dropout', type=float, default=0.9,
                        help='dropout probability (default: 0.5)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
    parser.add_argument('--filename', type=str, default='results',
                        help='filename to store results and the model (default: results)')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Training on CPU/GPU device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # load dataset
    dataname = args.dataset
    rootname = osp.join(osp.abspath(''), 'data', dataname)
    dataset = Planetoid(root=rootname, name=dataname,split = 'public')
    num_nodes = dataset[0].x.shape[0]
    data= dataset[0]
    edge_index = to_undirected(data.edge_index)
    L = get_laplacian(data.edge_index, num_nodes=num_nodes, normalization='sym')
    L = sparse.coo_matrix((L[1].numpy(), (L[0][0, :].numpy(), L[0][1, :].numpy())), shape=(num_nodes, num_nodes))
    
    
    
    '''spectral filtering. 
    1. assign filtering matrices; 
    2.(optional) ensure the filtering larger than 2 (+- 1) for the next step.       
    3. rescale the filteirng results back to [0,2]; 
    '''
    lambdas, eigenvecs = np.linalg.eigh(L.todense())  # currently only use eigendecomposition, slow.
    lambdas = torch.tensor(lambdas)
    lambdas[lambdas <= 0.0] = 0.0
    lambdas[lambdas > 2.0] = 2.0
    print(num_nodes -np.count_nonzero(lambdas))
    eigenvecs = torch.tensor(eigenvecs).to(device)
    lp_eigen = torch.exp(args.initial_dynmaic_coefficient*-lambdas+1) #ensure the output of initial filtering is >=2
    lp_eigen = 2*(lp_eigen-min(lp_eigen))/(max(lp_eigen)-min(lp_eigen)) # rearrage low pass to [0,2]
    hp_eigen = torch.exp(lambdas-1) #ensure the output of initial filtering is >=2
    hp_eigen = 2*(hp_eigen-min(hp_eigen))/(max(hp_eigen)-min(hp_eigen)) # rearrage high pass to [0,2]
    # """ optional: delayed HFD, with first k component of thetas = 0, which is equivalent to set first k images of lp and hp 0.
    # """
    # for i in range(num_nodes -np.count_nonzero(lambdas)):
    #     lp_eigen[i] = 0
    #     hp_eigen[i] = 0
    lp =torch.diag(lp_eigen).to(device)
    hp = torch.diag(hp_eigen).to(device)


    '''
    Training Scheme
    '''
    # data = dataset[0]
    # data.x = torch.tensor(np.array(normalize_features(data.x))).to(device)
    data = data.to(device)

    # Hyper-parameter Settings
    learning_rate = args.lr
    weight_decay = args.wd
    nhid = args.nhid
    if args.activation == 'None':
        activation = None
    else:
        activation = eval('F.'+ args.activation)

    # extract the data
    # data = dataset[0].to(device)

    # create result matrices
    num_epochs = args.epochs
    num_reps = args.reps
    epoch_loss = dict()
    epoch_acc = dict()
    epoch_loss['train_mask'] = np.zeros((num_reps, num_epochs))
    epoch_acc['train_mask'] = np.zeros((num_reps, num_epochs))
    epoch_loss['val_mask'] = np.zeros((num_reps, num_epochs))
    epoch_acc['val_mask'] = np.zeros((num_reps, num_epochs))
    epoch_loss['test_mask'] = np.zeros((num_reps, num_epochs))
    epoch_acc['test_mask'] = np.zeros((num_reps, num_epochs))
    saved_model_val_acc = np.zeros(num_reps)
    saved_model_test_acc = np.zeros(num_reps)
for i in range(10):
    for rep in range(num_reps):
        print('****** Rep {}: training start ******'.format(rep + 1))
        max_acc = 0.0

        # initialize the model
        model = Net(dataset.num_node_features, nhid, dataset.num_classes, num_nodes,
                dropout_prob=args.dropout, num_layers = args.num_layers,activation = activation).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


        # training
        for epoch in range(num_epochs):
            # training mode
            model.train()
            optimizer.zero_grad()
            out = model(eigenvecs,lp,hp,data)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            # evaluation mode
            model.eval()
            out = model(eigenvecs,lp,hp,data)
            for i, mask in data('train_mask', 'val_mask', 'test_mask'):
                pred = out[mask].max(dim=1)[1]
                correct = float(pred.eq(data.y[mask]).sum().item())
                e_acc = correct / mask.sum().item()
                epoch_acc[i][rep, epoch] = e_acc

                e_loss = F.nll_loss(out[mask], data.y[mask])
                epoch_loss[i][rep, epoch] = e_loss

            # scheduler.step(epoch_loss['val_mask'][rep, epoch])

            # print out results
            print('Epoch: {:3d}'.format(epoch + 1),
                  'train_loss: {:.4f}'.format(epoch_loss['train_mask'][rep, epoch]),
                  'train_acc: {:.4f}'.format(epoch_acc['train_mask'][rep, epoch]),
                  'val_loss: {:.4f}'.format(epoch_loss['val_mask'][rep, epoch]),
                  'val_acc: {:.4f}'.format(epoch_acc['val_mask'][rep, epoch]),
                  'test_loss: {:.4f}'.format(epoch_loss['test_mask'][rep, epoch]),
                  'test_acc: {:.4f}'.format(epoch_acc['test_mask'][rep, epoch]))

            # save model
            if epoch_acc['val_mask'][rep, epoch] > max_acc:
                torch.save(model.state_dict(), args.filename + '.pth')
                print('=== Model saved at epoch: {:3d}'.format(epoch + 1))
                max_acc = epoch_acc['val_mask'][rep, epoch]
                record_test_acc = epoch_acc['test_mask'][rep, epoch]

        saved_model_val_acc[rep] = max_acc
        saved_model_test_acc[rep] = record_test_acc
        print('#### Rep {0:2d} Finished! val acc: {1:.4f}, test acc: {2:.4f} ####\n'.format(rep + 1, max_acc, record_test_acc))

    print('***************************************************************************************************************************')
    print('Average test accuracy over {0:2d} reps: {1:.4f} with stdev {2:.4f}'.format(num_reps, np.mean(saved_model_test_acc), np.std(saved_model_test_acc)))
    print('dataset:', args.dataset, '; epochs:', args.epochs, '; reps:', args.reps, '; learning_rate:', args.lr,
          '; weight_decay:', args.wd, '; nhid:', args.nhid,'; dynamic_coefficient:', args.initial_dynmaic_coefficient,';activation:',args.activation
          )
    print( 'dropout:', args.dropout, '; seed:', args.seed, '; filename:', args.filename)
    print('\n')
    print(args.filename + '.pth', 'contains the saved model and ', args.filename + '.npz', 'contains all the values of loss and accuracy.')
    print('***************************************************************************************************************************')

    # save the results
    np.savez(args.filename + '.npz',
             epoch_train_loss=epoch_loss['train_mask'],
             epoch_train_acc=epoch_acc['train_mask'],
             epoch_valid_loss=epoch_loss['val_mask'],
             epoch_valid_acc=epoch_acc['val_mask'],
             epoch_test_loss=epoch_loss['test_mask'],
             epoch_test_acc=epoch_acc['test_mask'],
             saved_model_val_acc=saved_model_val_acc,
             saved_model_test_acc=saved_model_test_acc)




