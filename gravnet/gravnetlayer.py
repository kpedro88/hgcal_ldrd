# torch geometric implementation of network introduced in arXiv:1902.07987

import torch.nn as nn
import numpy as np

# this network has two outputs with different numbers of features
# the two outputs are called "spatial" and "learned" based on usage in GravNet
class DoubleOutputNetwork(nn.Module):
    def __init__(self, spatial, learned):
        super(DoubleOutputNetwork, self).__init__()        
        self.spatial = spatial
        self.learned = learned

    def forward(self, x):
        return self.spatial(x), self.learned(x)

# performs message passing with input vector of weights that modify the features
from torch_geometric.nn import MessagePassing
class WeightedMessagePassing(MessagePassing):        
    def message(self, x_j, weights):
        return np.multiply(x_j, weights)

# potential functions for GravNet
# (essentially kernel functions of distance in latent space)
def gaussian_kernel(d_ij):
    return np.exp(-d_ij**2)
def exponential_kernel(d_ij):
    return np.exp(-np.abs(d_ij))
_allowed_kernels = {
    'gaussian': gaussian_kernel,
    'exponential': exponential_kernel,
}

# the full GravNet layer
# this is a sandwich of dense NN + neighbor assignment & message passing + dense NN
# the first dense NN should be a DoubleOutputNetwork or similar (or a sequence that ends in such)
from torch_geometric.nn import knn_graph
from torch import cdist, index_select
class GravNetLayer(nn.Module):
    def __init__(self,first_dense,n_neighbors,aggrs,second_dense,kernel='gaussian'):
        self.first_dense = first_dense
        self.n_neighbors = n_neighbors
        self.second_dense = second_dense

        if kernel not in _allowed_kernels:
            raise ValueError("Unrecognized kernel "+kernel+" (allowed values: "+', '.join(allowed_kernels)+")")
        self.kernel = _allowed_kernels[kernel]
        
        self.messengers = []
        for aggr in aggrs:
            self.messengers.append(WeightedMessagePassing(aggr=aggr,flow="target_to_source"))
        
    def forward(self, x, batch=None):
        # apply first dense NN to derive spatial and learned features
        spatial, learned = self.first_dense(x)
        
        # use spatial to generate edge index
        edge_index = knn_graph(spatial, self.n_neighbors, batch, loop=False)
        
        # make the vector of distance weights using kernel
        neighbors = index_select(spatial,0,edge_index[1])
        distances = cdist(spatial,neighbors,metric='euclidean')
        weights = self.kernel(distances)
        
        # use learned for message passing
        messages = [x]
        for messenger in self.messengers:
            messages.append(messenger(learned,weights))
            
        # concatenate features, keep input
        all_features = torch.cat(messages, dim=1)
        
        # apply second dense to get final set of features
        final = self.second_dense(all_features)
        
        return final
