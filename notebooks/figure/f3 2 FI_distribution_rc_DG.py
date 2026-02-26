# 2025.09.19 Compute the mean FI along rc for genes.

import scvelo as scv
import dynamo as dyn
import numpy as np
from anndata import AnnData
# import loompy
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from scipy.cluster.hierarchy import fcluster,leaders
from sklearn.decomposition import PCA
from scipy.linalg import inv
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from scipy.sparse import csr_matrix,issparse
import pandas as pd
import seaborn as sns

import argparse
import random
import networkx as nx

import scipy.sparse as sp
import scipy.sparse.csgraph
import sklearn.linear_model as sklm
import sklearn.metrics as skm
import sklearn.model_selection as skms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from ignite.engine import Engine, Events
#from ignite.handlers import ModelCheckpoint
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

CHECKPOINT_PREFIX = "g2g"

from utils import *
# from minepy import MINE
from sklearn.preprocessing import MinMaxScaler

from g2g_model_Fisher import *

data_path = 'data/'
data_name = 'DG_bin_ppt'
adata0 = scv.read(data_path+data_name+'.h5ad', cache=True)
jaccard_similarities = []

from itertools import product
 
k_nei_values = [10]
K_values = [2]
L_values = [10]
 
for k_nei1, K1, L1 in product(k_nei_values, K_values, L_values):
    print(f"k_nei1: {k_nei1}, K1: {K1}, L1: {L1}")


    pca_dim = 50
    [k_nei, K, L] = [k_nei1, K1, L1]


    result_path = 'FI_distribution_rc/'+data_name+'/'
    encoder_path = 'main_results/'+'DG_bin_ppt'+' '+str([k_nei,K,L])+'/'
    figure_path = result_path
    cmap = plt.colormaps['Spectral']

    import os

    folder = os.path.exists(result_path)
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(result_path)            #makedirs 创建文件时如果路径不存在会创建这个路径

    from scipy.sparse import csr_matrix
    import scanpy as sc
    adata = adata0.copy()
    sc.pp.pca(adata, n_comps=pca_dim)
    sc.pp.neighbors(adata, n_neighbors=k_nei)
    scv.pp.moments(adata, n_pcs=pca_dim, n_neighbors=k_nei)
    scv.tl.velocity(adata)

    gene_arr = adata.var.index.values
    X_pca = adata.obsm['X_pca']
    X_umap = adata.obsm['X_umap']
    Xs = adata.layers['Ms'] #adata.X.A#
    # Xs = adata.layers['M_s'] #如果是EG_ab_dyn
    X = Xs

    row = np.array([np.ones((k_nei,))*i for i in range(adata.shape[0])]).flatten()
    col = (adata.obsp['distances']+csr_matrix(np.eye(adata.obsp['distances'].shape[0]))).indices
    w_val = np.array([np.linalg.norm(X_pca[int(i),:]-X_pca[int(j),:]) for i,j in zip(row,col)])
    adj_val = np.ones(col.shape)
    A_mat = csr_matrix((adj_val, (row, col)), shape=(adata.shape[0], adata.shape[0]))
    W_mat = csr_matrix((w_val, (row, col)), shape=(adata.shape[0], adata.shape[0]))

    dc=np.amax(adata.obsp['distances'])
    cell_nei=adata.obsp['distances'].indices.reshape([-1,k_nei-1])
    nei_w=[]
    rho_arr=[]
    for i in range(cell_nei.shape[0]):
        dij=np.array([np.linalg.norm(X_pca[i,:]-X_pca[int(j),:]) for j in cell_nei[i]])
        
        rho=np.sum(np.exp(-dij**2/dc**2))
        nei_w.append(np.exp(-dij**2/dc**2)/rho)
        rho_arr.append(rho)
    rho_arr=np.array(rho_arr)/np.amax(rho_arr)
    nei_w=np.array(nei_w)


    #################################################
    #################################################
    encoder = torch.load(encoder_path+'encoder.pt')


    mu, sigma = encoder(torch.tensor(X))
    mu_learned = mu.detach().numpy()
    sigma_learned = sigma.detach().numpy()

    Fisher_g=np.zeros((X.shape[0],L*2,L*2))
    for i in range(X.shape[0]):
        for j in range(L):
            Fisher_g[i,j,j]=1/sigma_learned[i,j]**2
            Fisher_g[i,L+j,L+j]=2/sigma_learned[i,j]**2

    Fisher_g_diag = np.zeros([X.shape[0],L*2])
    for i in range(X.shape[0]):
        Fisher_g_diag[i] = np.diag(Fisher_g[i])

    def smooth_func(X_val,cell_nei=cell_nei,nei_w=nei_w):
        X_s=X_val.copy()
        for ci in range(len(X_val)):
            X_s[ci]=np.dot(X_val[cell_nei[ci,:]],nei_w[ci,:])
        return(X_s)

    def wasserstein_distance(mu1,sigma1,mu2,sigma2):
        dim=len(mu1)
        dmu=mu1-mu2
        W_dist2=0
        for i in range(dim):
            W_dist2+=dmu[i]**2+sigma1[i]**2+sigma2[i]**2-2*np.sqrt(sigma2[i]*sigma1[i]**2*sigma2[i])
        W_dist=np.sqrt(W_dist2+1e-12)
        return W_dist

    cRc_arr=[]
    cRc_arr_eu=[]
    A = csr_matrix(A_mat + np.eye(A_mat.shape[0]))
    for inds in np.split(A.indices, A.indptr)[1:-1]:
        self_ind=inds[0]
        cRc=0
        cRc_eu=0
        for nei_k in range(1,len(inds)):

            dEu=np.linalg.norm(X[self_ind,:]-X[inds[nei_k],:])
            dFi=Fisher_dist(mu_learned[self_ind,:],sigma_learned[self_ind,:],\
                            mu_learned[inds[nei_k],:],sigma_learned[inds[nei_k],:])
            dWa=wasserstein_distance(mu_learned[self_ind,:],sigma_learned[self_ind,:],\
                            mu_learned[inds[nei_k],:],sigma_learned[inds[nei_k],:])

            cRc+=1-dWa/dFi
            cRc_eu+=1-dWa/dEu

        cRc_arr.append(cRc/len(inds))
        cRc_arr_eu.append(cRc_eu/len(inds))
    crc = np.array(cRc_arr)
    crc_eu = np.array(cRc_arr_eu)
    crc_smooth = smooth_func(crc_eu)
    #------use a simple neural network to study dmu/dt and dsigma/dt
    pca_dim = 50
    from torch.optim import SGD
    reset_seeds(0)

    model = nn.Sequential(
        nn.Linear(pca_dim, 128),
        nn.ReLU(),
        nn.Linear(128,64),
        nn.ReLU(),
        nn.Linear(64,2*L),
    )

    # Define your loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)#, momentum=0.2

    latent_z = np.hstack((mu_learned,sigma_learned))

    x_in=torch.tensor(X_pca.astype(np.float32))
    x_out=torch.tensor(latent_z.astype(np.float32))
    # Train the model
    for epoch in range(1000):  # number of epochs
        # Forward pass
        output = model(x_in)
        loss = loss_fn(output,x_out) 
    #     if epoch% 10 == 9:
    #         print(epoch,loss)
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    pZ_pX = np.zeros([X.shape[0], L*2, X_pca.shape[1]])

    # Compute the gradients
    for i in range(X.shape[0]):
        x0=torch.tensor(X_pca[i,:],requires_grad=True)
        z=model(x0)
        for j in range(2*L):
            x0.grad = None       
            z[j].backward(retain_graph=True)
            pZ_pX[i,j,:] = x0.grad.detach()

    g_pc = np.zeros([X.shape[0],pca_dim,pca_dim])
    for i in range(X.shape[0]):
        g_pc[i] = pZ_pX[i].T @ Fisher_g[i] @ pZ_pX[i]

    

    ##########################################################################
    ##########################################################################
    # RC clusters
    average_path = np.load('DG_rc.npy')#[2:-1]
    # trans coordinates in average_path to nodes indices
    # trans coordinates in average_path to nodes indices
    distances = np.linalg.norm(X_pca[:, np.newaxis,:10] - average_path[:,:10], axis=2)

    nearest_indices = np.argmin(distances, axis=0)

    # cell_arr = [[] for i in range(average_path.shape[0])]
    # for j in range(adata.shape[0]):
    #     distances = np.linalg.norm(X_pca[j,:10] - average_path, axis=1)
    #     reaction_coordinate = np.argmin(distances, axis=0)
    #     cell_arr[reaction_coordinate].append(j)

    # for i in range(len(cell_arr)):
    #     cell_arr[i] = np.array(cell_arr[i])

    from scipy.spatial.distance import cdist

    def find_nearest_indices(average_path, X_umap, k=50):
        """
        找到X_umap中距离average_path每个点最近的k个点的索引
        
        参数:
        average_path : np.ndarray, shape (n, d)
            路径点数组
        X_umap : np.ndarray, shape (m, d)
            待搜索的点集
        k : int, 默认50
            要找到的最近邻数量
        
        返回:
        cell_arr : list of lists
            每个元素是一个包含最近邻索引的列表
        """
        # 计算所有点对之间的距离矩阵
        distances = cdist(average_path, X_umap, 'euclidean')
        
        # 对每一行(即每个average_path点)的距离进行排序，获取前k个最小距离的索引
        nearest_indices = np.argsort(distances, axis=1)[:, :k]
        
        # 转换为列表的列表
        cell_arr = [row.tolist() for row in nearest_indices]
        
        return cell_arr

    cell_arr = find_nearest_indices(average_path, X_pca[:,:10], k=50)

    ######################################################################
    ######################################################################
    # Mean FIM for each cluster

    # 创建 4x5 的子图结构
    fig, axs = plt.subplots(4, 5, figsize=(20, 16))  # 创建 4x5 网格，并调整大小

    # 迭代绘图
    mean_FIMs = []
    mean_FIMs_mean_cell = []
    for k in range(len(cell_arr)):
        cluster_cells = cell_arr[k]

        # 先Gaussian embedding再把FIM平均
        mean_FIM = np.mean(
            np.array([adata.varm['PCs'] @ g_pc[i] @ adata.varm['PCs'].T for i in cluster_cells]),
            axis=0  # 指定沿第 0 维度求均值，即求矩阵的逐元素平均
        )
        gene_FI = np.diag(mean_FIM)


        # # 先把细胞基因表达平均再Gaussian embedding得到FIM
        # mean_cell = np.mean(X[cluster_cells], axis=0)
        # mu_mean_cell, sigma_mean_cell = encoder(torch.tensor(mean_cell))
        # mu_mean_cell = mu_mean_cell.detach().numpy()
        # sigma_mean_cell = sigma_mean_cell.detach().numpy()

        # FIM_g=np.zeros((L*2,L*2))
        # for j in range(L):
        #     FIM_g[j,j]=1/sigma_mean_cell[j]**2
        #     FIM_g[L+j,L+j]=2/sigma_mean_cell[j]**2

        # # Compute the gradients for mean cell FIM
        # x00=torch.tensor(mean_cell@adata.varm['PCs'],requires_grad=True)
        # z00=model(x00)
        # pZ_pX0 = np.zeros([2*L, pca_dim])
        # for j in range(2*L):
        #     x00.grad = None       
        #     z00[j].backward(retain_graph=True)
        #     pZ_pX0[j,:] = x00.grad.detach()
        # g_pc0 = pZ_pX0.T @ FIM_g @ pZ_pX0

        # mean_FIMs_mean_cell.append(adata.varm['PCs'] @ g_pc0 @ adata.varm['PCs'].T)
        mean_FIMs.append(mean_FIM)

        ax = axs[k // 5, k % 5]  # 找到正确的子图位置
        ax.hist(np.log10(gene_FI), bins=100, alpha=0.5, color='black', label='Frequency Histogram')

        # 添加标题和标签
        ax.set_title(f'FI distribution of RC {k+1}', fontsize=16, weight='bold')
        ax.set_xlabel('log10 of mean FI', fontsize=12, weight='bold')
        ax.set_ylabel('Frequency', fontsize=12, weight='bold')

    # 调整整体图形布局
    plt.tight_layout()  # 自动调整子图之间的间距
    plt.savefig(result_path+'FI distribution rc.png')
    plt.show()
    np.save(result_path+'mean_FIMs.npy', mean_FIMs)
    # np.save(result_path+'mean_FIMs_mean_cell.npy', mean_FIMs_mean_cell)