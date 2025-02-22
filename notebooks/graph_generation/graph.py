import numpy as np
from collections import namedtuple
from scipy.sparse import csr_matrix, find
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree

Graph = namedtuple('Graph', ['X', 'Ri', 'Ro', 'y','simmatched'])

SparseGraph = namedtuple('SparseGraph',
        ['X', 'Ri_rows', 'Ri_cols', 'Ro_rows', 'Ro_cols', 'y', 'simmatched'])

def make_sparse_graph(X, Ri, Ro, y,simmatched=None):
    Ri_rows, Ri_cols = Ri.nonzero()
    Ro_rows, Ro_cols = Ro.nonzero()
    return SparseGraph(X, Ri_rows, Ri_cols, Ro_rows, Ro_cols, y, simmatched)

def save_graph(graph, filename):
    """Write a single graph to an NPZ file archive"""
    np.savez(filename, **graph._asdict())
    #np.savez(filename, X=graph.X, Ri=graph.Ri, Ro=graph.Ro, y=graph.y)

def save_graphs(graphs, filenames):
    for graph, filename in zip(graphs, filenames):
        save_graph(graph, filename)

def load_graph(filename, graph_type=SparseGraph):
    """Reade a single graph NPZ"""
    with np.load(filename) as f:
        return graph_type(**dict(f.items()))

def load_graphs(filenames, graph_type=SparseGraph):
    return [load_graph(f, graph_type) for f in filenames]

def graph_from_sparse(sparse_graph, dtype=np.uint8):
    n_nodes = sparse_graph.X.shape[0]
    n_edges = sparse_graph.Ri_rows.shape[0]
    mat_shape = (n_nodes,n_edges)
    data = np.ones(n_edges)
    Ri = csr_matrix((data,(sparse_graph.Ri_rows,sparse_graph.Ri_cols)),mat_shape,dtype=dtype)
    Ro = csr_matrix((data,(sparse_graph.Ro_rows,sparse_graph.Ro_cols)),mat_shape,dtype=dtype)
    return Graph(sparse_graph.X, Ri, Ro, sparse_graph.y, sparse_graph.simmatched)

feature_names = ['x','y','layer','t','E']
n_features = len(feature_names)

#thanks Steve :-)
def draw_sample(X, Ri, Ro, y, 
                cmap='bwr_r', 
                skip_false_edges=True,
                alpha_labels=False, 
                sim_list=None): 
    # Select the i/o node features for each segment    
    feats_o = X[find(Ro)[0]]
    feats_i = X[find(Ri)[0]]    
    # Prepare the figure
    fig, (ax0,ax1) = plt.subplots(1, 2, figsize=(20,12))
    cmap = plt.get_cmap(cmap)
    
    if sim_list is None:    
        # Draw the hits (layer, x, y)
        ax0.scatter(X[:,0], X[:,2], c='k')
        ax1.scatter(X[:,1], X[:,2], c='k')
    else:        
        #ax0.scatter(X[:,0], X[:,2], c='k')
        #ax1.scatter(X[:,1], X[:,2], c='k')
        ax0.scatter(X[sim_list,0], X[sim_list,2], c='b')
        ax1.scatter(X[sim_list,1], X[sim_list,2], c='b')
    
    # Draw the segments
    for j in range(y.shape[0]):
        if not y[j] and skip_false_edges: continue
        if alpha_labels:
            seg_args = dict(c='k', alpha=float(y[j]))
        else:
            seg_args = dict(c=cmap(float(y[j])))
        ax0.plot([feats_o[j,0], feats_i[j,0]],
                 [feats_o[j,2], feats_i[j,2]], '-', **seg_args)
        ax1.plot([feats_o[j,1], feats_i[j,1]],
                 [feats_o[j,2], feats_i[j,2]], '-', **seg_args)
    # Adjust axes
    ax0.set_xlabel('$x$ [cm]')
    ax1.set_xlabel('$y$ [cm]')
    ax0.set_ylabel('$layer$ [arb]')
    ax1.set_ylabel('$layer$ [arb]')
    plt.tight_layout()

def draw_sample3d(X, Ri, Ro, y, 
                  cmap='bwr_r', 
                  skip_false_edges=True,
                  alpha_labels=False, 
                  sim_list=None):
    # Select the i/o node features for each segment
    feats_o = X[find(Ri)[0]]
    feats_i = X[find(Ro)[0]]
    # Prepare the figure
    fig = plt.figure()
    ax2 = fig.add_subplot(111, projection='3d')
    cmap = plt.get_cmap(cmap)
    
    if sim_list is None:    
        # Draw the hits (layer, x, y)
        ax2.scatter(X[:,0], X[:,1], X[:,2], c='k')
    else:        
        ax2.scatter(X[sim_list,0], X[sim_list,1], X[sim_list,2], c='k')
    
    # Draw the segments
    for j in range(y.shape[0]):
        if not y[j] and skip_false_edges: continue
        if alpha_labels:
            seg_args = dict(c='k', alpha=float(y[j]))
        else:
            seg_args = dict(c=cmap(float(y[j])))        
        ax2.plot([feats_o[j,0], feats_i[j,0]],
                 [feats_o[j,1], feats_i[j,1]],
                 [feats_o[j,2], feats_i[j,2]],'-',**seg_args)
    # Adjust axes
    ax2.set_xlabel('$x$ [cm]')
    ax2.set_ylabel('$y$ [cm]')
    ax2.set_zlabel('$layer$ [arb]')
    
def make_graph_kdtree(coords,layers,sim_indices,r=2.5):
    #setup kd tree for fast processing
    the_tree = cKDTree(coords)
    
    #define the pre-processing (all layer-adjacent hits in ball R < r)
    #and build a sparse matrix representation, then blow it up 
    #to the full R_in / R_out definiton
    pairs = the_tree.query_pairs(r=r,output_type='ndarray')
    first,second = pairs[:,0],pairs[:,1]  
    #selected index pair list that we label as connected
    pairs_sel  = pairs[( (np.abs(layers[(second,)]-layers[(first,)]) <= 1)  )]
    data_sel = np.ones(pairs_sel.shape[0])
    
    #prepare the input and output matrices (already need to store sparse)
    r_shape = (coords.shape[0],pairs.shape[0])
    eye_edges = np.arange(pairs_sel.shape[0])
    
    R_i = csr_matrix((data_sel,(pairs_sel[:,1],eye_edges)),r_shape,dtype=np.uint8)
    R_o = csr_matrix((data_sel,(pairs_sel[:,0],eye_edges)),r_shape,dtype=np.uint8)
        
    #now make truth graph y (i.e. both hits are sim-matched)    
    y = (np.isin(pairs_sel,sim_indices).astype(np.int8).sum(axis=-1) == 2)
    
    return R_i,R_o,y