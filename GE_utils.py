import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import SGD

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from scipy.cluster.hierarchy import fcluster,leaders,dendrogram,linkage
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd
import seaborn as sns

import tqdm


function_names = ['Get_GE_results','unkown']

def Jacobian_nn(x0,d,encoder):
    x = torch.tensor(x0,requires_grad=True)
    z = encoder(x)
    pMu_pX = np.zeros([d,x0.shape[0]])
    pSgm_pX = np.zeros([d,x0.shape[0]])
    for i in range(d):
        x.grad = None
        z[0][i].backward(retain_graph=True)
        pMu_pX[i] = x.grad.detach()
        x.grad = None
        z[1][i].backward(retain_graph=True)
        pSgm_pX[i] = x.grad.detach()
    return pMu_pX,pSgm_pX

#------use a simple neural network to study dmu/dt and dsigma/dt
def new_para(X,latent_z):
    x_in = torch.tensor(X, dtype=torch.float32)
    pca_dim = x_in.shape[1]
    L = int(latent_z.shape[1]/2)
    model = nn.Sequential(
        nn.Linear(pca_dim, 128),
        nn.ReLU(),
        nn.Linear(128,64),
        nn.ReLU(),
        nn.Linear(64,2*L),
    )

    # Define your loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    x_out = torch.tensor(latent_z.astype(np.float32), dtype=torch.float32)
    # Train the model
    for epoch in tqdm.tqdm(range(200)):  # number of epochs
        # Forward pass
        output = model(x_in)
        loss = loss_fn(output,x_out) 
    #     if epoch% 10 == 9:
    #         print(epoch,loss)
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    pZ_pX = np.zeros([X.shape[0], L*2, X.shape[1]])

    # Compute the gradients
    for i in range(X.shape[0]):
        x0=torch.tensor(X[i,:],requires_grad=True)
        z=model(x0)
        for j in range(2*L):
            x0.grad = None       
            z[j].backward(retain_graph=True)
            pZ_pX[i,j,:] = x0.grad.detach()
    return model,pZ_pX[:,:L,:],pZ_pX[:,L:,:]


def FIM_new(pMu_pX,pSgm_pX,Fisher_g):
    ###get FI and stiff module on new space
    
    n,L,m = pMu_pX.shape
    stiffnum = np.zeros(n)##number of stiffgene |(n,1)
    # tf_stiffnum = np.zeros(X.shape[0])
    pZ_pX = np.zeros([n, L*2, m])
    diagFIgene = np.zeros((n,m)) ##Fisher information of gene |(n,m)
    Eigenvec = np.zeros((n,m))   ##1st eigenvec for each cell |(n,m)
    Eigenval = []  ##前2特征值 | (n,2)

    print('Calculating Fisher Information Matrix of genes...')
    for i in tqdm.tqdm(range(n)):
        pZ_pX[i] = np.vstack((pMu_pX[i],pSgm_pX[i]))
        FIgene_i = pZ_pX[i].T@Fisher_g[i]@pZ_pX[i]
        Eigenval.append([np.linalg.eigh(FIgene_i)[0][-1],np.linalg.eigh(FIgene_i)[0][-2]])
        Eigenvec[i] = np.linalg.eigh(FIgene_i)[1][:,-1]
        diagFIgene[i] = np.diag(FIgene_i)
        FI_max = np.array([np.argmax(FIgene_i[j]) for j in range(m)])
        # tf_max = np.array([np.argmax(FIgene_i[tf_inds][:,tf_inds][j]) for j in range(len(tf_inds))])
        stiffnum[i] = len(np.unique(list(FI_max)))
        # tf_stiffnum[i] = len(np.unique(list(tf_max)))

    return stiffnum,diagFIgene,Eigenvec,np.array(Eigenval)

def EigenIF(Fisher_g):
    n,L,_ = Fisher_g.shape
    L=int(L/2)
    Eigenvalue = np.zeros((n,L))
    for i in range(n):
        for j in range(L):
            Eigenvalue[i,j] = Fisher_g[i,j,j]
    sorted_Eigenvalue = np.sort(Eigenvalue, axis=1)
    sorted_Eigenvalue = sorted_Eigenvalue[:, ::-1]
    relative_Eigenvalue = sorted_Eigenvalue[:,1:]/sorted_Eigenvalue[:,0].reshape(n,1)
    return Eigenvalue,sorted_Eigenvalue,relative_Eigenvalue

def Get_GE_results(encoder,X,GE_eigen=True):
    ### calculate GE results based on encoder and data X
    X = np.array(X)
    n,m = X.shape
    X = X.astype(np.float32)
    GE_results = {}
    mu, sigma = encoder(torch.tensor(X))
    mu_learned = mu.detach().numpy()
    sigma_learned = sigma.detach().numpy()
    latent_z = np.hstack((mu_learned,sigma_learned))
    n,L = latent_z.shape
    L = L//2

    GE_results['mu_learned'] = mu_learned
    GE_results['sigma_learned'] = sigma_learned
    GE_results['latent_z'] = latent_z

    diag_elements = np.concatenate([1 / GE_results['sigma_learned'] ** 2, 2 / GE_results['sigma_learned'] ** 2], axis=-1)
    Fisher_g = np.zeros((X.shape[0], 2 * L, 2 * L))
    for i in range(X.shape[0]):
        np.fill_diagonal(Fisher_g[i], diag_elements[i])

    pMu_pX = np.zeros([X.shape[0],L,X.shape[1]])
    pSgm_pX = np.zeros([X.shape[0],L,X.shape[1]])
    for i in range(X.shape[0]):
        pMu_pX[i],pSgm_pX[i] = Jacobian_nn(X[i],L,encoder)

    if GE_eigen:
        stiffnum,diagFIgene,Eigenvec,Eigenval = FIM_new(pMu_pX,pSgm_pX,Fisher_g)
        GE_results['stiffnum'] = stiffnum
        GE_results['diagFIgene'] = diagFIgene
        GE_results['Eigenvec'] = Eigenvec
        GE_results['Eigenval'] = Eigenval

    GE_results['pMu_pX'] = pMu_pX
    GE_results['pSgm_pX'] = pSgm_pX
    GE_results['Fisher_g'] = Fisher_g
    return GE_results




from scipy.integrate import quad
from scipy.optimize import bisect

def angle_cdf_cos(t0, n):
    # t0: cos(theta_0)，注意t0∈[-1,1]
    # 累积分布函数，给定余弦阈值返回概率
    # 分子：t从t0到1
    num = quad(lambda t: (1-t**2)**((n-3)/2), -t0, t0)[0]
    den = quad(lambda t: (1-t**2)**((n-3)/2), -1, 1)[0]
    return num/den

def find_cos_for_prob(n, p, tol=1e-8):
    # 二分查找 t0，使CDF达到p
    f = lambda t0: angle_cdf_cos(t0, n) - p
    # t0 ∈ [-1, 1], 探索左侧小概率
    return bisect(f, 0, 1, xtol=tol)

import hotspot
def eigen_gene(X_re,T_re):#return the weights of each gene in correponding eigen gene
    eigen_X_w = []
    pc = []
    for i in np.unique(T_re):
        pca=PCA(n_components=1).fit(X_re[:,T_re==i])
        print(pca.explained_variance_ratio_)
        pc.append(pca.transform(X_re[:,T_re==i]))
        eigen_X_w.append(pca.components_.T)
    #        eigen_gene_size.append(np.where(T_re==i)[0].shape[0])
    return np.array(pc),eigen_X_w

def eigen_gene_hs(X_re,T_re):#return the weights of each gene in correponding eigen gene
    eigen_X_w = []
    pc = []
    for i in np.unique(T_re):
        if i < 0:
            continue
        pca=PCA(n_components=1).fit(X_re[:,T_re==i])
        print(pca.explained_variance_ratio_)
        pc.append(pca.transform(X_re[:,T_re==i]))
        eigen_X_w.append(pca.components_[0,:])
#         eigen_gene_size.append(np.where(T_re==i)[0].shape[0])
    return np.array(pc),eigen_X_w

def compute_x_eigen(X,method='weighted'):
    global eigen_X,eigen_X_w
    X_corr=np.corrcoef(X, rowvar=False)
    plt.imshow(X_corr)
    plt.colorbar()
    plt.show()

    ###这里的method的设置
    Z=linkage((np.abs(1-X_corr))[np.triu_indices(X_corr.shape[0],k=1)],method=method)#method='weighted')
    dg=dendrogram(Z)
    X_re= X[:,dg['leaves']]#X_re reorder X by clustering

    X_corr_re=np.corrcoef(X_re, rowvar=False)
    plt.show()

    n = X.shape[0]
    td = 1 - find_cos_for_prob(n, 0.95)
    print('td:',td)
    T = fcluster(Z, t=td, criterion='distance')
    # T = fcluster(Z, t=L, criterion='maxclust')
    T_re = T[dg['leaves']]

    plt.imshow(X_corr_re, aspect='auto', cmap=plt.cm.coolwarm, interpolation='nearest',origin='lower')
    plt.show()

    X_corr_label = np.zeros(X_corr_re.shape)
    for i in range(X_corr_re.shape[0]):
        label_ind = np.where(T_re==T_re[i])[0]
        X_corr_label[i,label_ind] = 1

    plt.imshow(X_corr_label, aspect='auto', cmap=plt.cm.gray, interpolation='nearest',origin='lower')
    plt.colorbar()
    plt.show()

    eigen_X,eigen_X_w=eigen_gene(X_re,T_re)
    eigen_dim=len(np.unique(T_re))
    n_eigen = 1
    cell_eigen_X = eigen_X[0,:,:n_eigen]
    for i in range(eigen_dim-1):
        cell_eigen_X = np.hstack((cell_eigen_X,eigen_X[i+1,:,:n_eigen]))
    return cell_eigen_X,T

def compute_x_hs(X,adata,k_nei=10,threshold=20):
    global hs_X, hs_X_w
    # Create the Hotspot object and the neighborhood graph
    # hotspot works a lot faster with a csc matrix!
    hs = hotspot.Hotspot(
        adata, 
        model='danb',
        distances_obsp_key = 'distances'
    )

    hs.create_knn_graph(
        weighted_graph=False, n_neighbors=k_nei,
    )

    hs_results = hs.compute_autocorrelations(jobs=1)

    # Select the genes with significant lineage autocorrelation
    hs_genes = hs_results.loc[hs_results.FDR < 0.05].sort_values('Z', ascending=False).head(800).index

    # Compute pair-wise local correlations between these genes
    lcz = hs.compute_local_correlations(hs_genes, jobs=1)

    modules = hs.create_modules(
        min_gene_threshold=threshold, core_only=True, fdr_threshold=0.1
    )

    modules.value_counts()
    # np.save(result_path+'modules', modules.values)
    # np.save(result_path+'hs_genes', hs_genes)

    adata_hs = adata[:,hs_genes] # 只用hotspot有效的gene
    T_hs = np.array(hs.modules.tolist()) # 记录所有hs gene的module

    scaler = StandardScaler()
    X_hs = scaler.fit_transform(adata_hs.layers['Ms'])
    hs_X, hs_X_w = eigen_gene_hs(X_hs,T_hs)
    hs_dim=len(hs_X_w)

    print("T_hs模块编号合集:", np.unique(T_hs))
    print("hs_X_w个数:", len(hs_X_w))

    cell_hs_X=np.zeros((X_hs.shape[0],hs_dim))
    for j in range(X_hs.shape[0]):
        for k in range(len(hs_X_w)):
            cell_hs_X[j,k]=np.dot(hs_X_w[k],X_hs[j,T_hs==k+1])
    return cell_hs_X






def adjust_eigenvec_direction(eigenvec,sonnodes,clusters,X):
    ### adjust the direction of eigenvector according to the tree structure
    for i in eigenvec:
        if len(sonnodes[i])==0:
            continue
        center_i = np.mean(X[clusters==i],axis=0)
        sum_dot = 0
        for j in sonnodes[i]:
            center_j = np.mean(X[clusters==j],axis=0)
            dir_ij = center_j - center_i
            proj_ij = np.dot(eigenvec[i],dir_ij)/np.linalg.norm(dir_ij)
            sum_dot += np.sum(proj_ij)
        if sum_dot <0:
            eigenvec[i] = -eigenvec[i]
    return eigenvec

def flow_analyze(sonnodes,eigenvec,clusters,X):
    #### analyze the flow structure based on eigenvector and tree structure
    nodes,edges = [],[]
    for i in sonnodes:
        nodes.append(i)
        for j in sonnodes[i]:
            edges.append((i,j))
    node_size = [len(clusters[clusters==i]) for i in nodes]
    weight = {}
    for i in edges:
        weight[i] = 0
    clusters_eigenvec = []
    ktime = {}
    for k,i in enumerate(clusters):
        if i in ktime:
            ktime[i] = ktime[i]+1
        else:
            ktime[i] = 0
        if len(sonnodes[i])>1:
            Cors = []
            vec = eigenvec[i][ktime[i]]
            center = X[k]
            for j in sonnodes[i]:
                center_j = np.mean(X[clusters==j],axis=0)
                finite_diff = center_j - center
                cor = np.dot(vec,finite_diff)/(np.linalg.norm(vec)*np.linalg.norm(finite_diff))
                Cors.append(cor)
            chosen_son = sonnodes[i][np.argmax(Cors)]
            weight[i,chosen_son] += 1
            clusters_eigenvec.append(f'{i}_to_{chosen_son}')
        else:
            if len(sonnodes[i])==1:
                weight[i,sonnodes[i][0]] += 1
            clusters_eigenvec.append(f'{i}')
    for i in weight:
        weight[i] = weight[i]/len(clusters[clusters==i[0]])
    clusters_eigenvec = np.array(clusters_eigenvec)
    DAG = {}
    DAG['nodes'] = nodes
    DAG['edges'] = edges
    DAG['node_size'] = node_size
    DAG['weight'] = weight
    return DAG,clusters_eigenvec




### hierarchical clustering based on eigenvector correlation
from matplotlib.colors import Normalize                
def hierarchical_clustering(categories,eigenvec,t,result_path,method='average',save_name='Corr'):
    TT = {}
    for i in range(len(categories)):
        cl_cat = categories[i]
        vec_corr = eigenvec[cl_cat]@eigenvec[cl_cat].T
        
        #####这里的method的设置
        Z=linkage((np.abs(1-vec_corr))[np.triu_indices(vec_corr.shape[0],k=1)],method=method)#method='weighted')
        dg=dendrogram(Z)
        #X_re= X[:,dg['leaves']]#X_re reorder X by clustering
        plt.show()
        
        vec_corr_re = vec_corr[dg['leaves'],:][:,dg['leaves']]
        plt.imshow(vec_corr_re, aspect='equal', cmap=plt.cm.coolwarm, interpolation='nearest',norm=Normalize(-1,1))
        plt.title(categories[i],fontsize=20,weight='bold')
        cb=plt.colorbar()
        cb.ax.tick_params(labelsize=12)
        cb.ax.tick_params(width=2)  # 设置刻度线宽度
        cb.outline.set_linewidth(2)  # 设置 colorbar 边框宽度
    # 设置 colorbar 刻度字体和加粗
        for tick in cb.ax.get_yticklabels():
            tick.set_fontsize(18)  # 设置刻度标签字体大小
            tick.set_weight('bold')  # 设置刻度标签加粗
        # 设置 colorbar 的标题
        cb.set_ticklabels(['-1','', '-0.5', '','0','', '0.5','', '1'])  # 设置刻度标签
        cb.set_label('Correlation', fontsize=20, fontweight='bold')
        plt.xticks([])
        plt.yticks([])
        plt.savefig(result_path+f'{save_name} of {categories[i]}.png')
        plt.show()

        # xcos = np.linspace(0,1,100)
        # plt.plot(xcos,gaussian_kde(vec_corr.flatten())(xcos))
        # plt.title(categories[i],fontsize=15,weight='bold')
        # plt.show()
        
        # T = fcluster(Z, t=t, criterion='distance')
        T = fcluster(Z, t=t, criterion='distance')
        print(T)
        TT[categories[i]] = T
    return TT











### rearrange array according to frequency
from collections import Counter
def my_Rearrange(arr):
    flat = arr.ravel()
    counter = Counter(flat)
    # 按出现次数从多到少排序，次数相同按元素大小排序
    sorted_items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    # 构造数字到新编号的映射
    mapping = {num: i+1 for i, (num, _) in enumerate(sorted_items)}

    # 替换
    new_flat = np.array([mapping[x] for x in flat])
    new_arr = new_flat.reshape(arr.shape)
    return new_arr





##########Plotting functions

def PLT_box(clusters,matrix,save_path,ylabel,title_name='violin',xlabel='cell_type',cell_sort=None):
    categories = list(np.unique(clusters))
    if cell_sort is None:
        None
    else:
        categories.sort(key = list(clusters[np.argsort(cell_sort)]).index)
    eigenvalues = {}
    for i in categories:
        eigenvalues[i] = matrix[clusters==i]
    plt.figure()
    #plt.violinplot([eigenvalues[i].reshape(eigenvalues[i].shape[0]*eigenvalues[i].shape[1]) for i in list(eigenvalues)], showmeans=True, showmedians=True)
    for j in range(matrix.shape[1]):
        plt.boxplot([eigenvalues[i][:,j] for i in list(eigenvalues)])
    # 设置x轴的标签
    #plt.xticks([1, 2, 3], ['Group A', 'Group B', 'Group C'])
    plt.xticks(range(1,len(categories)+1),categories)
    if np.sum([len(str(i)) for i in categories])>50:
        plt.xticks(rotation=40)
    # 添加标题和标签
    plt.title(title_name,fontsize=16,weight='bold')
    plt.xlabel(xlabel,fontsize=14,weight='bold')
    plt.ylabel(ylabel,fontsize=14,weight='bold')
    plt.savefig(save_path,dpi=300, bbox_inches='tight')
    plt.show()

def plot3d(x1,x2,x3,color,color_bar_name='color_bar',save_name='3d.png'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x1, x2, x3,  s=10, c=color_label,alpha=0.8,cmap=plt.cm.jet)
    cmap = plt.colormaps['Spectral']
    sc = ax.scatter(x1, x2, x3,  s=10, c=color,alpha=0.8,cmap=cmap)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label(color_bar_name)
    plt.savefig(save_name)
    plt.show()

#### 庞加莱圆盘和坐标变换
def plain2disk(x,y):
    y = np.sqrt(2)*y
    down = x**2 + (y+1)**2
    u,v = x**2 - y**2 + 1, -2*x
    return u/down,v/down
def change_disk_center(u0,v0,u,v):
    A,B = 1-(u0*u+v0*v),u0*v-v0*u
    down = A**2 + B**2
    u1,v1 = (u-u0)*A - (v-v0)*B, (u-u0)*B + (v-v0)*A
    return u1/down,v1/down
def poincare_disk_distance_expand(X, lam):
    # X: n x d array, each row is a point in the disk (norm < 1)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    # Avoid division by zero for origin
    direction = np.zeros_like(X)
    direction[norm[:,0] > 0] = X[norm[:,0] > 0] / norm[norm[:,0] > 0]
    # Compute old r and new r
    r = np.log((1+norm)/(1-norm))
    r_new = lam * r
    norm_prime = np.tanh(r_new/2)
    X_prime = direction * norm_prime
    return X_prime