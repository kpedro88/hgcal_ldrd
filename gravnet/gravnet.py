# torch geometric implementation of network introduced in arXiv:1902.07987

import torch
import torch.nn as nn
import numpy as np
from gravnetlayer import GravNetLayer, PassThroughNetwork, DoubleOutputNetwork

# a single block for the network
class GravBlock(nn.Module):
    def __init__(self, input_dim = 10, dense_dim = 64, spatial_dim = 4, learned_dim = 22, out_dim = 48, n_neighbors = 40, aggrs = ['add','mean','max']):
        self.layers = nn.Sequential(
            # first section: 3 dense layers w/ 64 nodes, tanh activation
            # add 1 to input_dim b/c concatenation of mean
            nn.Linear(in_features=input_dim+1,out_features=dense_dim),
            nn.Tanh(),
            nn.Linear(in_features=dense_dim,out_features=dense_dim),
            nn.Tanh(),
            nn.Linear(in_features=dense_dim,out_features=dense_dim),
            nn.Tanh(),
            # second section: GravNetLayer
            GravNetLayer(
                first_dense = DoubleOutputNetwork(
                    spatial = nn.Linear(in_features=dense_dim,out_features=spatial_dim),
                    learned = nn.Linear(in_features=dense_dim,out_features=learned_dim),
                ),
                n_neighbors = n_neighbors,
                aggrs = aggrs,
                second_dense = nn.Sequential(
                    nn.Linear(in_features=learned_dim,out_features=out_dim),
                    nn.Tanh(),
                ),
            ),
        )
        # keep track of this in order to chain blocks together
        self.out_dim = out_dim
        
    def forward(self, x):
        # concatenate mean of features
        x = torch.cat([x, np.mean(x)], dim=1)
        
        # apply layers
        x = self.layers(x)
        return x
        
# the full network, with multiple blocks
class GravNet(nn.Module):
    # kwargs passed to GravBlocks
    def __init__(self, n_blocks = 4, final_dim = 128, n_clusters = 2, **kwargs):
        # first block just takes kwargs
        self.blocks = [GravBlock(**kwargs)]
        # subsequent blocks need to know the first block's output
        self.blocks.extend([GravBlock(input_dim = self.blocks[0].out_dim, **kwargs) for n in range(1, n_blocks)]
        # final set of layers: dense ReLU, input from all blocks -> small dense ReLU -> small dense softmax
        self.final = nn.Sequential(
            nn.Linear(in_features=n_blocks*self.blocks[0].out_dim,out_features=final_dim),
            nn.ReLU(),
            nn.Linear(in_features=final_dim,out_features=n_clusters+1),
            nn.ReLU(),
            nn.Linear(in_features=n_showers+1,out_features=n_clusters),
            nn.Softmax(),
        )
        self.batchnorm = nn.BatchNorm1d(n_showers)
        
    def forward(self, x):
        # apply batch norm to input (and then to all block outputs)
        x = self.batchnorm(x)
        # concatenate output from all blocks, while feeding each block's output to the next
        all_output = [self.batchnorm(self.blocks[0](x))]
        for block in self.blocks[1:]:
            all_output.append(self.batchnorm(block(all_output[-1])))
        all_output = torch.cat(all_output, dim=1)
        return self.final(all_output)

# loss function to be used in training
class EnergyFractionLoss(nn.Module):
    def forward(self, energy, pred, truth):
        # used for per-sensor energy weighting w/in cluster
        total_energy_cluster = np.sqrt(energy*truth)
        # get numer and denom terms for each shower
        numers = np.sum(total_energy_cluster*(pred-truth)**2,axis=1)
        denoms = np.sum(total_energy_cluster,axis=1)
        # sum of weighted differences
        loss = np.sum(numers/denoms)
        return loss
