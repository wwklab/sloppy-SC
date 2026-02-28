
#----------Graph Guassian embedding with Fisher distance as energy loss


import argparse
import random

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph
from scipy.sparse import csr_matrix
import sklearn.linear_model as sklm
import sklearn.metrics as skm
import sklearn.model_selection as skms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
import umap
from sklearn.datasets import make_swiss_roll,make_s_curve
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm



class CompleteKPartiteGraph:
    """A complete k-partite graph
    """

    def __init__(self, partitions):
        """
        Parameters
        ----------
        partitions : [[int]]
            List of node partitions where each partition is list of node IDs
        """

        self.partitions = partitions
        self.counts = np.array([len(p) for p in partitions])
        self.total = self.counts.sum()

        assert len(self.partitions) >= 2
        assert np.all(self.counts > 0)

        # Enumerate all nodes so that we can easily look them up with an index
        # from 1..total
        self.nodes = np.array([node for partition in partitions for node in partition])

        # Precompute the partition count of each node
        self.n_i = np.array(
            [n for partition, n in zip(self.partitions, self.counts) for _ in partition]
        )

        # Precompute the start of each node's partition in self.nodes
        self.start_i = np.array(
            [
                end - n
                for partition, n, end in zip(
                    self.partitions, self.counts, self.counts.cumsum()
                )
                for node in partition
            ]
        )

        # Each node has edges to every other node except the ones in its own
        # level set
        self.out_degrees = np.full(self.total, self.total) - self.n_i

        # Sample the first nodes proportionally to their out-degree
        self.p = self.out_degrees / self.out_degrees.sum()

    def sample_edges(self, size=1):
        """Sample edges (j, k) from this graph uniformly and independently

        Returns
        -------
        ([j], [k])
        j will always be in a lower partition than k
        """

        # Sample the originating nodes for each edge
        j = np.random.choice(self.total, size=size, p=self.p, replace=True)

        # For each j sample one outgoing edge uniformly
        #
        # Se we want to sample from 1..n \ start[j]...(start[j] + count[j]). We
        # do this by sampling from 1..#degrees[j] and if we hit a node

        k = np.random.randint(self.out_degrees[j])
        filter = k >= self.start_i[j]
        k += filter.astype(int) * self.n_i[j]

        # Swap nodes such that the partition index of j is less than that of k
        # for each edge
        wrong_order = k < j
        tmp = k[wrong_order]
        k[wrong_order] = j[wrong_order]
        j[wrong_order] = tmp

        # Translate node indices back into user configured node IDs
        j = self.nodes[j]
        k = self.nodes[k]

        return j, k


class AttributedGraph: ##
    def __init__(self, A, X, z, K, directed=True, Input_matrix='Adjacency'):  ##
        self.A = A   ##近邻图
        self.X = torch.tensor(X)#torch.tensor(X.toarray())  ##数据
        self.z = z          ##label
        self.directed = directed
        self.Input_matrix = Input_matrix
        self.level_sets = level_sets(A, K, directed=directed, Input_matrix=Input_matrix)  ##每个点的0到k hop 邻居 数据类型为dict

        # Precompute the cardinality of each level set for every node
        self.level_counts = {
            node: np.array(list(map(len, level_sets)))
            for node, level_sets in self.level_sets.items()
        }

        # Precompute the weights of each node's expected value in the loss
        N = self.level_counts  ##每个点的0到k hop 邻居的个数
        self.loss_weights = 0.5 * np.array(
            [N[i][1:].sum() ** 2 - (N[i][1:] ** 2).sum() for i in self.nodes()]
        )

        n = self.A.shape[0]
        self.neighborhoods = [None] * n
        for i in range(n):
            ls = self.level_sets[i]
            if len(ls) >= 3:
                self.neighborhoods[i] = CompleteKPartiteGraph(ls[1:])

    def nodes(self):
        return range(self.A.shape[0])

    def eligible_nodes(self): ##返回起码有两阶邻居的点
        """Nodes that can be used to compute the loss"""
        N = self.level_counts

        # If a node only has first-degree neighbors, the loss is undefined
        return [i for i in self.nodes() if len(N[i]) >= 3]

    def sample_two_neighbors(self, node, size=1):
        """Sample to nodes from the neighborhood of different rank"""

        level_sets = self.level_sets[node]
        if len(level_sets) < 3:
            raise Exception(f"Node {node} has only one layer of neighbors")

        return self.neighborhoods[node].sample_edges(size)


class GraphDataset(IterableDataset):
    """A dataset that generates all necessary information for one training step

    Sampling the edges is actually the most expensive part of the whole training
    loop and by putting it in the dataset generator, we can parallelize it
    independently from the training loop.
    """

    def __init__(self, graph, nsamples, iterations):
        self.graph = graph
        self.nsamples = nsamples
        self.iterations = iterations

    def __iter__(self):
        graph = self.graph
        nsamples = self.nsamples

        eligible_nodes = list(graph.eligible_nodes())
        nrows = len(eligible_nodes) * nsamples

        weights = torch.empty(nrows)

        for _ in range(self.iterations):
            i_indices = torch.empty(nrows, dtype=torch.long)
            j_indices = torch.empty(nrows, dtype=torch.long)
            k_indices = torch.empty(nrows, dtype=torch.long)
            for index, i in enumerate(eligible_nodes):
                start = index * nsamples
                end = start + nsamples
                i_indices[start:end] = i

                js, ks = graph.sample_two_neighbors(i, size=nsamples)
                j_indices[start:end] = torch.tensor(js)
                k_indices[start:end] = torch.tensor(ks)

                weights[start:end] = graph.loss_weights[i]

            yield graph.X, i_indices, j_indices, k_indices, weights, nsamples


def gather_rows(input, index):
    """Gather the rows specificed by index from the input tensor"""
    return torch.gather(input, 0, index.unsqueeze(-1).expand((-1, input.shape[1])))


class Encoder(nn.Module):
    def __init__(self, D, L):
        """Construct the encoder

        Parameters
        ----------
        D : int
            Dimensionality of the node attributes
        L : int
            Dimensionality of the embedding

        """
        super().__init__()

        self.D = D
        self.L = L

        def xavier_init(layer):
            nn.init.xavier_normal_(layer.weight)
            # TODO: Initialize bias with xavier but pytorch cannot compute the
            # necessary fan-in for 1-dimensional parameters

        self.linear1 = nn.Linear(D, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear_mu = nn.Linear(128, L)
        self.linear_sigma = nn.Linear(128, L)

        xavier_init(self.linear1)
        xavier_init(self.linear2)
        xavier_init(self.linear_mu)
        xavier_init(self.linear_sigma)

    def forward(self, node):
        h = F.relu(self.linear1(node))
        h = F.relu(self.linear2(h))
        mu = self.linear_mu(h)
        sigma = F.elu(self.linear_sigma(h)) + 1 + 1e-10
        return mu, sigma

    def compute_loss(self, X, i, j, k, w, nsamples):
        """Compute the energy-based loss from the paper
        """

        mu, sigma = self.forward(X)

        mu_i = gather_rows(mu, i)
        sigma_i = gather_rows(sigma, i)
        mu_j = gather_rows(mu, j)
        sigma_j = gather_rows(sigma, j)
        mu_k = gather_rows(mu, k)
        sigma_k = gather_rows(sigma, k)

        diff_ij = mu_i - mu_j
        ratio_ji = sigma_j / sigma_i
        ss_ij=sigma_i+sigma_j
        ds_ij=sigma_i-sigma_j
        
        closer = 2*((torch.log (
            (torch.sqrt(0.5*diff_ij**2+ss_ij**2)+torch.sqrt(0.5*diff_ij**2+ds_ij**2))/\
            (torch.sqrt(0.5*diff_ij**2+ss_ij**2)-torch.sqrt(0.5*diff_ij**2+ds_ij**2)+1e-10)
             ))**2).sum(axis=-1)

        diff_ik = mu_i - mu_k
        ratio_ki = sigma_k / sigma_i
        ss_ik=sigma_i+sigma_k
        ds_ik=sigma_i-sigma_k
        
        apart = 2*((torch.log (
            (torch.sqrt(0.5*diff_ik**2+ss_ik**2)+torch.sqrt(0.5*diff_ik**2+ds_ik**2))/\
            (torch.sqrt(0.5*diff_ik**2+ss_ik**2)-torch.sqrt(0.5*diff_ik**2+ds_ik**2)+1e-10)
             ))**2).sum(axis=-1)

        E = closer  + torch.exp(-torch.sqrt(apart)) 

        loss = E.dot(w) / nsamples

        return loss


def getkhop(DP, K):
    n = DP.shape[0]
    sets = {i: [set() for _ in range(K + 2)] for i in range(n)}
    
    for i in tqdm(range(n)):
        sets[i][0].add(i)
        sets[i][-1] = set(range(n)) - {i}
        
        for k in range(1, K + 1):
            current_set = sets[i][k-1]
            next_set = sets[i][k]
            remaining_set = sets[i][-1]
            
            for j in remaining_set.copy():  # Iterate over a copy to modify the set
                if DP[i][j] >= 0 and DP[i][j] in current_set:
                    next_set.add(j)
                    remaining_set.remove(j)
                    
    return {i: [list(s) for s in sets[i]] for i in range(n)}
def level_sets(A, K, directed=True, Input_matrix ='Adjacency'):
    """Enumerate the level sets for each node's neighborhood

    Parameters
    ----------
    A : np.array
        Adjacency matrix
    K : int?
        Maximum path length to consider

        All nodes that are further apart go into the last level set.
    directed : bool
        directed graph or not
    Input_matrix : str
        the type of input matrix with 'Adjacency', 'weighted' or 'distance'

    Returns
    -------
    { node: [i -> i-hop neighborhood] }
    """

    if A.shape[0] == 0 or A.shape[1] == 0:
        return {}
    if Input_matrix == 'Adjacency':
        # Compute the shortest path length between any two nodes
        D = scipy.sparse.csgraph.shortest_path(
            A, method="D", unweighted=True, directed=directed
        )
    
        # Cast to int so that the distances can be used as indices
        #
        # D has inf for any pair of nodes from different cmponents and np.isfinite
        # is really slow on individual numbers so we call it only once here
        D[np.logical_not(np.isfinite(D))] = -1.0
        D = D.astype(int)
    
        # Handle nodes farther than K as if they were unreachable
        if K is not None:
            D[D > K] = -1
    
        # Read the level sets off the distance matrix
        set_counts = D.max(axis=1)
        sets = {i: [[] for _ in range(1 + set_counts[i] + 1)] for i in range(D.shape[0])}  ##
        for i in range(D.shape[0]):
            sets[i][0].append(i)  ##到自己的距离为0
            for j in range(D.shape[0]):
                d = D[i, j]
    
                # If a node is unreachable, add it to the outermost level set. This
                # trick ensures that nodes from different connected components get
                # pushed apart and is essential to get good performance.
                if d < 0:
                    sets[i][-1].append(j)
                else:
                    sets[i][d].append(j)

    elif Input_matrix == 'distance':
        D,DP = scipy.sparse.csgraph.shortest_path(
            A, method="D", unweighted=None, directed=directed,return_predecessors=True
        )
        sets = getkhop(DP,K)
    elif Input_matrix == 'weighted':
        A = A.tocoo()
        A_row = A.row
        A_col = A.col
        A_data = A.data
        A=csr_matrix((-np.log(A_data), (A_row, A_col)), shape=(A.shape[0], A.shape[0]))
        D,DP = scipy.sparse.csgraph.shortest_path(
            A, method="D", unweighted=None, directed=directed,return_predecessors=True
        )
        sets = getkhop(DP,K)
    return sets

def train_test_split(n, train_ratio=0.5):
    nodes = list(range(n))
    split_index = int(n * train_ratio)

    random.shuffle(nodes)
    return nodes[:split_index], nodes[split_index:]


def reset_seeds(seed=None):
    if seed is None:
        seed = get_worker_info().seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def training(encoder, optimizer, train_data, epochs=200, nsamples=5, learning_rate=1e-3):
    seed = 0
    if seed is not None:
        reset_seeds(seed)
    iterations = epochs
    dataset = GraphDataset(train_data, nsamples, iterations)
    loader = DataLoader(
        dataset,
        batch_size=1,
        #     num_workers=n_workers,
        worker_init_fn=reset_seeds,
        collate_fn=lambda args: args,
    )

    # for epoch in range(1, epochs + 1):
    #     print(epoch)

    for batch_idx, data in enumerate(loader):
        #     print(batch_idx)
        encoder.train()
        optimizer.zero_grad()
        loss = encoder.compute_loss(data[0][0], data[0][1], data[0][2], data[0][3], data[0][4], data[0][5])
        if batch_idx % 10 == 9:
            print(batch_idx, loss)
        loss.backward()
        optimizer.step()
    return encoder

# def embed_iter(X, z, A, k_nei=50, K=1, L=2, train_ratio=1, epochs=200, nsamples=5, iters=10, diff=0.2, learning_rate=1e-3):
#     n = X.shape[0]
#     if_conv = False
#     n_iter = 0

#     while not if_conv and n_iter<iters:
#         train_nodes, val_nodes = train_test_split(n, train_ratio=train_ratio)
#         X_train = X[train_nodes]
#         z_train = z[train_nodes]
#         A_train = A[train_nodes, :][:, train_nodes]
#         train_data = AttributedGraph(A_train, X_train, z_train, K)

#         encoder = Encoder(X.shape[1], L)
#         optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
#         encoder = training(encoder, optimizer, train_data, epochs=epochs, nsamples=nsamples, learning_rate=learning_rate)

#         mu, sigma = encoder(torch.tensor(X))
#         mu_learned = mu.detach().numpy()
#         sigma_learned = sigma.detach().numpy()
#         latent_z=np.hstack((mu_learned,sigma_learned))

# #         Af = np.zeros([n,n])
# #         ls = train_data.level_sets
# #         for i in range(n):
# #             lsi = []
# #             for j in range(K+1):
# #                 lsi += ls[i][j]
# #             lsi = np.array(lsi)
# #             fdi = [Fisher_dist(mu_learned[i],sigma_learned[i],mu_learned[j],sigma_learned[j]) for j in lsi]
# #             Af[i][lsi[np.argsort(fdi)[:k]]] = 1
# #         Af = sp.csr_matrix(Af)
#         Af = kneighbors_graph(latent_z, n_neighbors=k_nei, metric=Fisher_distz)
#         if ((Af!=A).count_nonzero()/A.count_nonzero()) < diff:
#             if_conv = True
#         print((Af!=A).count_nonzero()/A.count_nonzero())
#         plt.scatter(z,np.sum(1/mu_learned**2,axis=1))
#         A = Af
#         n_iter += 1
#     if not if_conv:
#         print("metric doesn't converge")
#     return encoder