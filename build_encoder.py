import scanpy as sc
import matplotlib.pyplot as plt
import scvelo as scv
import numpy as np
from utils import *
from g2g_model_Fisher import *

def main():
    cnt = 0
    for data_name in ['EG_ab_bin']:#['DG_bin_ppt', 'EG_ab_bin', 'zebrafish_dynamo_part']:
    # data_name = ''
        for k_nei in [10,20,30]:
            for K in [2,3]:
                for L in [3]:
                    print(cnt)
                    cnt += 1

                    result_path = 'main_results/'+data_name+' '+str([k_nei,K,L])+'/'
                    figure_path = result_path
                    cmap = plt.colormaps['Spectral']

                    adata0 = sc.read_h5ad('data/'+data_name+'.h5ad')
                    adata = adata0.copy()

                    import os

                    folder = os.path.exists(result_path)
                    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
                        os.makedirs(result_path)            #makedirs 创建文件时如果路径不存在会创建这个路径
                    else:
                        continue

                    from scipy.sparse import csr_matrix

                    sc.pp.pca(adata, n_comps=50)
                    sc.pp.neighbors(adata, n_neighbors=k_nei)
                    scv.pp.moments(adata, n_pcs=50, n_neighbors=k_nei)
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

                    epochs = 400
                    nsamples = 5
                    learning_rate = 1e-3
                    seed = 0
                    # n_workers = 4

                    if seed is not None:
                        reset_seeds(seed)

                    A = A_mat
                    A = A.astype(np.float32)
                    X = Xs.astype(np.float32)
                    z = np.zeros(adata.n_obs)

                    n = A.shape[0]
                    train_nodes, val_nodes = train_test_split(n, train_ratio=1.0)
                    A_train = A[train_nodes, :][:, train_nodes]
                    X_train = X[train_nodes]
                    z_train = z[train_nodes]
                    A_val = A[val_nodes, :][:, val_nodes]
                    X_val = X[val_nodes]
                    z_val = z[val_nodes]

                    train_data = AttributedGraph(A_train, X_train, z_train, K)
                    val_data = AttributedGraph(A_val, X_val, z_val, K)

                    encoder = Encoder(X.shape[1], L)

                    optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)

                    iterations = epochs #// n_workers
                    dataset = GraphDataset(train_data, nsamples, iterations)
                    loader = DataLoader(
                        dataset,
                        batch_size=1,
                    #     num_workers=n_workers,
                        worker_init_fn=reset_seeds,
                        collate_fn=lambda args: args,
                    )

                    for batch_idx, data in enumerate(loader):
                        encoder.train()
                        optimizer.zero_grad()

                        # compute weighted loss
                        _ ,i,j,k,w,nsamples = data[0][0],data[0][1],data[0][2],data[0][3],data[0][4],data[0][5]

                        mu, sigma = encoder.forward(data[0][0])

                        mu_i = gather_rows(mu, i)
                        sigma_i = gather_rows(sigma, i)
                        mu_j = gather_rows(mu, j)
                        sigma_j = gather_rows(sigma, j)
                        mu_k = gather_rows(mu, k)
                        sigma_k = gather_rows(sigma, k)

                        diff_ij = mu_i - mu_j
                        ss_ij = sigma_i + sigma_j
                        ds_ij = sigma_i - sigma_j
                        
                        closer = 2*((torch.log (
                            
                            (torch.sqrt(0.5*diff_ij**2+ss_ij**2+1e-10)+torch.sqrt(0.5*diff_ij**2+ds_ij**2+1e-10)+1e-10)/\
                            torch.abs((torch.sqrt(0.5*diff_ij**2+ss_ij**2+1e-10)-torch.sqrt(0.5*diff_ij**2+ds_ij**2+1e-10)+1e-10)
                            +1e-10)
                                ))**2).sum(axis=-1)

                        diff_ik = mu_i - mu_k
                        ss_ik = sigma_i + sigma_k
                        ds_ik = sigma_i - sigma_k
                        
                        apart = 2*((torch.log (
                            
                            (torch.sqrt(0.5*diff_ik**2+ss_ik**2+1e-10)+torch.sqrt(0.5*diff_ik**2+ds_ik**2+1e-10)+1e-10)/\
                            torch.abs((torch.sqrt(0.5*diff_ik**2+ss_ik**2+1e-10)-torch.sqrt(0.5*diff_ik**2+ds_ik**2+1e-10)+1e-10)
                            +1e-10)
                                ))**2).sum(axis=-1)

                        # E = closer*weight[i,j] + torch.exp(-torch.sqrt(apart))*weight[i,k] 
                        E = closer + torch.exp(-torch.sqrt(apart)) 

                        loss = E.dot(w) / nsamples
                        if batch_idx% 10 == 0:
                            print(batch_idx,loss)
                        loss.backward()
                        optimizer.step()

                    torch.save(encoder,result_path+'encoder.pt')
main()