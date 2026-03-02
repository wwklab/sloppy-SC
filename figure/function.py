import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter
import torch
from scipy.spatial import distance
from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter
from scipy.sparse import csr_matrix
import scvelo as scv
import os
from matplotlib.ticker import MaxNLocator

def variance_graph_subplot(pca_svd_rc_variance_i, title_label, ax, y_lim, is_first_col, show_legend, share_y):
    """
    绘制单个方差图，使用均值±标准差作为阴影区域，并高亮标记前5个秩值点。
    Rank 1 (稳重): 蓝色；Rank 2 (欢快): 橙色；Rank 3 (火热): 红色。
    同时加粗所有坐标轴边框、标题和标签字体，并增大了字号。
    **图例根据 show_legend 参数决定是否显示。**
    """
    # L = 秩的数量
    size = 24
    L = pca_svd_rc_variance_i.shape[1]
    ranks = np.arange(1, L + 1)
    mean_variance = np.mean(pca_svd_rc_variance_i, axis=0)

    # --- 统计量计算 (改为标准差) ---
    std_variance = np.std(pca_svd_rc_variance_i, axis=0)
    upper_bound_std = mean_variance + std_variance
    lower_bound_std = mean_variance - std_variance
    # 确保下限不为负
    lower_bound_std[lower_bound_std < 0] = 0

    # --- 绘图 ---
    
    # 绘制均值实线 (主曲线颜色改为蓝色 'C0')
    ax.plot(ranks, mean_variance, color='C0', linewidth=2.5, label='Mean Variance')
    
    # 使用标准差 (std) 绘制阴影区域 (绑定到 ax)
    ax.fill_between(ranks, lower_bound_std, upper_bound_std, 
                     alpha=0.3, color='C0', label=r'Mean $\pm$ Std')

    # --- 标记前3个秩值点 ---
    
    # 定义高对比度颜色
    highlight_color_1 = 'blue'   
    highlight_color_2 = 'orange'  
    fire_color = 'red' 
    s = 200
    # 标记第1个点
    if L >= 1:
        ax.scatter(ranks[0], mean_variance[0], 
                   color=highlight_color_1, s=s, zorder=5, 
                   edgecolors='black', linewidths=0.5) 

    # 标记第2个点
    if L >= 2:
        ax.scatter(ranks[1], mean_variance[1], 
                   color=highlight_color_2, s=s, zorder=5, 
                   edgecolors='black', linewidths=0.5) 
                   
    # 标记第3个点（此处循环为 [2]，仅标记 Rank 3）
    rank_3_label = f'Rank 3'
    
    for j in [2]:
        if L > j:
            # 只有在 j=2 (即 Rank 3) 时显示标签
            current_label = rank_3_label if j == 2 else None
            
            ax.scatter(ranks[j], mean_variance[j], 
                    color=fire_color, s=s, zorder=5, 
                    edgecolors='black', linewidths=0.5) 
    
    # --- 轴和标签设置 (增大字体并加粗) ---
    
    # 标题字体：18
    ax.set_title(f'{title_label}', fontsize=size+1, fontweight='bold') 
    
    # X轴标签字体：16
    ax.set_xlabel('Rank', fontsize=size-2, fontweight='bold')
    ax.set_xlim(0.2, L + 0.2)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))
    # Y轴标签字体：16
    if is_first_col:
        ax.set_ylabel('Variance', fontsize=size-2, fontweight='bold')
    
    # **应用统一的 Y 轴范围**
    if share_y:
        ax.set_ylim(y_lim)

    # 调整边框和刻度线 (已加粗边框)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # 刻度标签字体：14
    ax.tick_params(width=1.5, labelsize=size-4, which='both', direction='inout', length=6)
    
    # 加粗刻度标签字体
    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')
    
    # --- 控制图例显示 ---
    if show_legend:
        # 1. 获取现有图例句柄和标签 (仅包含 Mean 和 Std)
        line_handles, line_labels = ax.get_legend_handles_labels()
        
        # 2. 创建 Rank 标记的自定义图例句柄
        rank_handles = []
        rank_labels = []
        marksize = 14

        if L >= 1:
            rank_handles.append(Line2D([0], [0], marker='o', color='w', 
                                       markerfacecolor=highlight_color_1, markersize=marksize, 
                                       markeredgecolor='black', markeredgewidth=0.5))
            rank_labels.append('Rank 1')
            
        if L >= 2:
            rank_handles.append(Line2D([0], [0], marker='o', color='w', 
                                       markerfacecolor=highlight_color_2, markersize=marksize, 
                                       markeredgecolor='black', markeredgewidth=0.5))
            rank_labels.append('Rank 2')
            
        if L >= 3:
            rank_handles.append(Line2D([0], [0], marker='o', color='w', 
                                       markerfacecolor=fire_color, markersize=marksize, 
                                       markeredgecolor='black', markeredgewidth=0.5))
            rank_labels.append('Rank 3')
            
        # 3. 合并所有句柄和标签
        all_handles = line_handles + rank_handles
        all_labels = line_labels + rank_labels
        
        # 4. 创建图例
        legend = ax.legend(all_handles, all_labels, 
                           frameon=False, fontsize=size-4, loc='upper right')
        
        # 5. 加粗图例字体
        for text in legend.get_texts():
            text.set_fontweight('bold')

def plot_all_variances(pca_svd_rc_variance_all, share_y = True):
    """
    创建子图集合并循环调用绘图函数，实现统一的 Y 轴范围，并确保所有子图显示 Y 轴刻度。
    **图例只在第一个子图显示。**
    """
    num_plots = len(pca_svd_rc_variance_all)
    
    # --- 步骤 1: 计算所有数据的最大/最小方差 (用于统一 Y 轴) ---
    max_variance = 0
    # 遍历所有数据集，找出方差均值 + 标准差的最大值
    for data_i in pca_svd_rc_variance_all:
        if data_i.size == 0 or data_i.shape[1] == 0:
            continue
            
        mean_variance = np.mean(data_i, axis=0)
        std_variance = np.std(data_i, axis=0)
        upper_bound_std = mean_variance + std_variance
        current_max = np.max(upper_bound_std)
        if current_max > max_variance:
            max_variance = current_max
            
    # 设置统一的 Y 轴范围 [0, max_variance * 1.05] (增加5%的边距)
    y_lim = (-0.03, max_variance * 0.95)
    
    # --- 步骤 2: 创建并配置子图 ---
    cols = min(5, num_plots)
    rows = math.ceil(num_plots / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5.5 * rows)) 
    
    # 扁平化 axes 数组
    if num_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten() 

    # --- 步骤 3: 循环调用绘图函数 ---
    for i in range(num_plots):
        # **修正点：标题序号从 1 开始**
        title_label = f'RC {i + 1}'
        
        # 计算当前子图是否为第一列
        current_col = i % cols
        is_first_col = (current_col == 0)
        
        # **关键改动: 仅在 i == 0 时设置 show_legend=True**
        show_legend = (i == 0)
        
        variance_graph_subplot(
            pca_svd_rc_variance_i=pca_svd_rc_variance_all[i], 
            title_label=title_label, # 传递修正后的标题
            ax=axes[i],
            y_lim=y_lim,  
            is_first_col=is_first_col,
            show_legend=show_legend, # 传递图例控制参数
            share_y= share_y
        )

    # 隐藏未使用的子图
    for j in range(num_plots, rows * cols):
        fig.delaxes(axes[j])  
    plt.tight_layout(w_pad=2.0, h_pad=2.5)
    plt.show()

def plot_all_variances_odd(pca_svd_rc_variance_all, share_y = True, odd = False):
    """
    创建子图集合并循环调用绘图函数，实现统一的 Y 轴范围，并确保所有子图显示 Y 轴刻度。
    **图例只在第一个子图显示。**
    """
    num_plots = len(pca_svd_rc_variance_all)
    
    # --- 步骤 1: 计算所有数据的最大/最小方差 (用于统一 Y 轴) ---
    max_variance = 0
    # 遍历所有数据集，找出方差均值 + 标准差的最大值
    for data_i in pca_svd_rc_variance_all:
        if data_i.size == 0 or data_i.shape[1] == 0:
            continue
            
        mean_variance = np.mean(data_i, axis=0)
        std_variance = np.std(data_i, axis=0)
        upper_bound_std = mean_variance + std_variance
        current_max = np.max(upper_bound_std)
        if current_max > max_variance:
            max_variance = current_max
            
    # 设置统一的 Y 轴范围 [0, max_variance * 1.05] (增加5%的边距)
    y_lim = (-0.03, max_variance * 0.95)
    
    # --- 步骤 2: 创建并配置子图 ---
    if num_plots > 5 and odd:
        cols = min(5, num_plots/2)
        rows = math.ceil(num_plots / (2 * cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5.5 * rows)) 
        if num_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten() 
        for i in range(num_plots):
            if i % 2 ==0:
                j = int(i / 2)
                # **修正点：标题序号从 1 开始**
                title_label = f'RC {i + 1}'
                
                # 计算当前子图是否为第一列
                current_col = i % cols
                is_first_col = (current_col == 0)
                
                # **关键改动: 仅在 i == 0 时设置 show_legend=True**
                show_legend = (i == 0)
                
                variance_graph_subplot(
                    pca_svd_rc_variance_i=pca_svd_rc_variance_all[i], 
                    title_label=title_label, # 传递修正后的标题
                    ax=axes[j],
                    y_lim=y_lim,  
                    is_first_col=is_first_col,
                    show_legend=show_legend, # 传递图例控制参数
                    share_y= share_y
                )
        for j in range(int(num_plots), rows * cols):
            fig.delaxes(axes[j])  
        plt.tight_layout(w_pad=2.0, h_pad=2.5)
        plt.show()
    if num_plots <= 5 or odd == False:
        cols = min(5, num_plots)
        rows = math.ceil(num_plots / (cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5.5 * rows)) 
        if num_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten() 
        for i in range(num_plots):
            # **修正点：标题序号从 1 开始**
            title_label = f'RC {i + 1}'
            
            # 计算当前子图是否为第一列
            current_col = i % cols
            is_first_col = (current_col == 0)
            
            # **关键改动: 仅在 i == 0 时设置 show_legend=True**
            show_legend = (i == 0)
            
            variance_graph_subplot(
                pca_svd_rc_variance_i=pca_svd_rc_variance_all[i], 
                title_label=title_label, # 传递修正后的标题
                ax=axes[i],
                y_lim=y_lim,  
                is_first_col=is_first_col,
                show_legend=show_legend, # 传递图例控制参数
                share_y= share_y
            )
        for j in range(int(num_plots), rows * cols):
            fig.delaxes(axes[j])  
        plt.tight_layout(w_pad=2.0, h_pad=2.5)
        plt.show()

def p_out_p_in(X, model, dim_out):
    #计算神经网络模型model输出对输入的导数
    X = torch.as_tensor(X, dtype=torch.float32)
    pZ_pX = np.zeros([X.shape[0], dim_out, X.shape[1]])
    for i in range(X.shape[0]):
        x0 = torch.tensor(X[i,:],requires_grad=True)
        z=model(x0)
        for j in range(dim_out):
            x0.grad = None       
            z[j].backward(retain_graph=True)
            pZ_pX[i,j,:] = x0.grad.detach()
    
    return pZ_pX

def find_nearest_indices_voronoi(adata, path0, n_nei, rc_distance = None, dim_n = None):
    cells_2d = adata.obsm[rc_distance]  # [N_cells, 2]
    path_2d = path0              # [N_rc_points, 2]
    if rc_distance == "X_pca":
        cells_2d = adata.obsm[rc_distance][:,:dim_n]
        path_2d = path0[:,:dim_n]
    dist_matrix = distance.cdist(cells_2d, path_2d)  # [N_cells, N_rc_points]
    closest_rc = np.argmin(dist_matrix, axis=1)      # 每个细胞对应的RC索引
    cells_in_voronoi = {} # cells_in_voronoi里存的是每个细胞归属于哪一个vosinoi格子。
    for rc_i in range(path_2d.shape[0]):
        idx_in_cell = np.where(closest_rc == rc_i)[0]
        cells_in_voronoi[rc_i] = idx_in_cell
    nearest_30_cells = []  # nearest_30_cells里存的是筛选到的对应每个RC紧挨着的30个细胞。
    nearest_cell = []
    for rc_i in range(path_2d.shape[0]):
        idx_in_cell = cells_in_voronoi[rc_i]
        print("idx_in_cell:", idx_in_cell.shape)
        if len(idx_in_cell) == 0:
            nearest_30_cells.append(np.array([]))
            nearest_cell.append(None)
            continue # 语句用来告诉Python跳过当前循环的剩余语句，然后继续进行下一轮循环。而break跳出整个循环
        dist_to_rc = dist_matrix[idx_in_cell, rc_i]  # dist_matrix存的是细胞到各RC点的距离。
        nearest_30_cells.append(idx_in_cell[np.argsort(dist_to_rc)[:n_nei]])
        print("len(nearest_30_cells[rc_i])", len(nearest_30_cells[rc_i]))
        nearest_cell.append(idx_in_cell[np.argsort(dist_to_rc)[0]])
    return nearest_30_cells, nearest_cell

def cell_arr_plot(average_path, cell_arr, X_umap):
    rc_num = average_path.shape[0]
    base_colors = ['orange', 'green', 'purple']
    plt.figure(figsize=(8,6))
    plt.scatter(X_umap[:,0], X_umap[:,1], s=5, color='lightgrey', label='All cells')
    plt.scatter(average_path[:,0], average_path[:,1], s=40, c='blue', label='Path points')

    for rc_i in range(rc_num):
        color = base_colors[rc_i % 3]  # 每3种颜色循环
        plt.scatter(average_path[rc_i,0], average_path[rc_i,1], s=100, c='red', marker='x')
        plt.scatter(X_umap[cell_arr[rc_i],0],
                    X_umap[cell_arr[rc_i],1],
                    s=25, c=[color], alpha=0.7, label=f'RC {rc_i}')

    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.title('')
    plt.tight_layout()
    plt.show()

def eigengene_value_sum(S_FI_Eg, X_pca, L, cell_arr, i = 0):

    if L > X_pca.shape[1]:
        L = X_pca.shape[1]/2
    # S_FI_Eg = (S_FI_Eg / S_FI_Eg.sum(axis=1, keepdims=True))
    # FI_Param1 = S_FI_Eg[:, i]
    FI_Param1 = np.sum(S_FI_Eg, axis=1)
    print("FI_Param1.shape:", FI_Param1.shape)
    data_for_plot = []
    for rc_index, cell_indices in enumerate(cell_arr):
        reaction_coordinate = rc_index + 1  # 反应坐标从 1 开始
        fi_values = FI_Param1[cell_indices]
        for value in fi_values:
            data_for_plot.append({
                'RC': reaction_coordinate,
                f'FI_P{i + 1}': value
            })
    df = pd.DataFrame(data_for_plot)
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        x='RC', 
        y=f'FI_P{i + 1}', 
        data=df, 
        inner='quartile', 
        color='lightgray',       # <--- 填充颜色改为浅灰色
        edgecolor='black',       # <--- 边界线设置为黑色
        bw=.2 
    )
    plt.xlabel('RC')
    plt.ylabel(f'sum FI')
    reaction_coordinates = np.arange(1, len(cell_arr) + 1)
    plt.xticks(np.arange(len(cell_arr)), reaction_coordinates) 
    plt.ylim(bottom=0) 
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def eigengene_value(S_FI_Eg, X_pca, L, cell_arr, norm = True, simple = False):
    
    if L > X_pca.shape[1] / 2:
        L = X_pca.shape[1] / 2
    if norm == True:
        S_FI_Eg = (S_FI_Eg / S_FI_Eg.sum(axis=1, keepdims=True))
    if simple:
        num_parameters = 3
    else:
        num_parameters = int(2 * L)
    data_for_plot = []
    # print("len(cell_arr)", len(cell_arr))
    # print("cell_arr[0]", cell_arr[0])
    for rc_index, cell_indices in enumerate(cell_arr):
        # if rc_index == 0:
        #     print("cell_indices", cell_indices)
        reaction_coordinate = rc_index + 1
        FI_values_all_params = S_FI_Eg[cell_indices, :]
        # print("FI_values_all_params.shape", FI_values_all_params.shape)
        # print("FI_values_all_params", FI_values_all_params) 
        for cell_row in FI_values_all_params:
            # print("cell_row.shape", cell_row.shape)
            # print(cell_row)
            for param_index in range(num_parameters): # 表示只考虑前num_parameters个eigengenevalue
                data_for_plot.append({
                    'RC': f'RC{reaction_coordinate}', 
                    'FI_Value': cell_row[param_index],
                    'Parameter_ID': f'P{param_index + 1}'
                })
                # print(cell_row[param_index])
                # break
                
    df = pd.DataFrame(data_for_plot)
    plt.figure(figsize=(15, 8))
    sns.violinplot(
        x='RC',
        y='FI_Value',
        hue='Parameter_ID',  # 按照 Parameter_ID 分组并着色
        data=df,
        inner='quart',         # 内部显示箱线图
        bw=.15, 
        linewidth=0.5,
        width=0.8,
        dodge=False
    )
    
    plt.yscale('log')
    if norm == True:
        plt.ylabel('FI/sum_FI (Log Scale)', fontsize=14, weight='bold')
    else:
        plt.ylabel('FI (Log Scale)', fontsize=14, weight='bold')
    plt.legend(title='Parameter ID', bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # 调整布局以容纳图例
    plt.show()

def compute_variances(U_FI_Eg, X_pca, cell_arr, k_rc, L, simple = False, cv = True):
    all_variances = []  # 存每个cell_arr[i]对应的方差结果
    all_means = []
    if simple == False:
        if L > X_pca.shape[1]:
            L = X_pca.shape[1]/2
    else:
        L = simple

    for idx_list in cell_arr:  # 遍历40个索引列表
        # 1. 取出A: shape [50, 50, 50]
        A = U_FI_Eg[idx_list]  # 索引50个矩阵
        
        # 2. 对每个A[j]计算B_j = np.einsum('jk,ij->ik', A[j], X_pca)
        #    输出B_j.shape = (3386, 50)
        B_all = np.stack([np.einsum('jk,ij->ik', A[j], X_pca).astype(np.float32) for j in range( len(idx_list) )], axis=0)
        # B_all.shape = (50, 3386, 50)
        
        # 3. 对每个B_all[j]取出cell_arr[i]中的细胞
        C = np.stack([B_all[j, idx_list, :] for j in range( len(idx_list) )], axis=0)  # [50, 50, 50]
        
        # 4. 你要的最终矩阵C是 [50, 50]
        #    根据你的描述，似乎要将第 j 个矩阵取对应行（即每个B_j对cell_arr[i]取的部分），再拼成 [50, 50]
        #    这时 C[j] 就是一个 [50, 50] 的矩阵，我们可以选择每个 j 对应的第 j 个切片
        # NOTE:这下面算的方差是样本的方差，所以有ddof=1。
        if cv == True:
            # C_final = np.stack([( np.var(C[j], axis = 0, ddof=1) / np.mean(np.abs(C[j]), axis = 0)**2 ) for j in range(C.shape[0])], axis=0)[:,:int(2*L)]  # [50, 50, 50]
            C_final = np.stack([( np.var(C[j], axis = 0, ddof=1) / np.mean(C[j], axis = 0)**2 ) for j in range(C.shape[0])], axis=0)[:,:int(2*L)]  # [50, 50, 50]
            C_final_mean = np.stack([np.mean(C[j], axis = 0)  for j in range(C.shape[0])], axis=0)[:,:int(2*L)]
        # else:
            # C_final = np.stack([ np.var(C[j], axis = 0, ddof=1)  for j in range(C.shape[0])], axis=0)[:,:int(2*L)] 
        all_variances.append(C_final)
        all_means.append(C_final_mean)

    return all_variances, all_means  # list长度为40，每个元素是长度50的一维数组

def compute_pca_mean_velocity(U_FI_Eg, X_pca_velocity, cell_arr, L, simple = False):
    all_variances = []  # 存每个cell_arr[i]对应的方差结果
    if simple == False:
        if L > X_pca_velocity.shape[1]:
            L = X_pca_velocity.shape[1]/2
    else:
        L = simple

    for idx_list in cell_arr:  # 遍历40个索引列表
        # 1. 取出A: shape [k_rc, 50, 50]
        A = U_FI_Eg[idx_list]  # 索引k_rc个矩阵
        
        # 2. 对每个A[j]计算B_j = np.einsum('jk,ij->ik', A[j], X_pca)
        #    输出B_j.shape = (3386, 50)
        B_all = np.stack([np.einsum('jk,ij->ik', A[j], X_pca_velocity).astype(np.float32) for j in range( len(idx_list) )], axis=0)
        # B_all.shape = (k_rc, 3386, 50)
        
        # 3. 对每个B_all[j]取出cell_arr[i]中的细胞
        C = np.stack([B_all[j, idx_list, :] for j in range( len(idx_list) )], axis=0)  # [k_rc, k_rc, 50]
        
        C_final = np.mean( np.stack([ np.mean(C[j], axis = 0)  for j in range(C.shape[0])], axis=0)[:,:int(2*L)] , axis=0 )
        all_variances.append(C_final)

    return all_variances  # list长度为40，每个元素是长度50的一维数组

# NOTE：这个是GPT生成的幂法求前k个特征值。算出的结果和np.linalg.svd是一样的。
def power_iteration(A, num_iter=5000, tol=1e-8):
    n, _ = A.shape
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)

    for _ in range(num_iter):
        Av = A @ v
        v_new = Av / np.linalg.norm(Av)
        
        if np.linalg.norm(v_new - v) < tol:
            break
        v = v_new

    # Rayleigh quotient = eigenvalue approximation
    eigenvalue = v.T @ A @ v
    return eigenvalue, v

def top_k_eigenpairs(A, k=10):
    A = A.copy().astype(float)

    eigenvalues = []
    eigenvectors = []

    for i in range(k):
        λ, v = power_iteration(A)
        eigenvalues.append(λ)
        eigenvectors.append(v)

        # Deflation: remove the found eigen component
        A = A - λ * np.outer(v, v)

    return np.array(eigenvalues), np.array(eigenvectors)

def wasserstein_distance(mu1,sigma1,mu2,sigma2):
    dim=len(mu1)
    dmu=mu1-mu2
    W_dist2=0
    for i in range(dim):
        W_dist2+=dmu[i]**2+sigma1[i]**2+sigma2[i]**2-2*np.sqrt(sigma2[i]*sigma1[i]**2*sigma2[i])
    W_dist=np.sqrt(W_dist2)
    return W_dist

def smooth_func(X_val, adata, k_nei):
    X_pca = adata.obsm['X_pca']
    row = np.array([np.ones((k_nei,))*i for i in range(adata.shape[0])]).flatten()
    col = adata.uns['neighbors']['indices'].flatten()
    w_val = np.array([np.linalg.norm(X_pca[int(i),:]-X_pca[int(j),:]) for i,j in zip(row,col)])
    dc=np.mean(w_val)
    cell_nei=adata.uns['neighbors']['indices']
    nei_w=[]
    rho_arr=[]
    for ni in adata.uns['neighbors']['indices']:
        dij=np.array([np.linalg.norm(X_pca[int(ni[0]),:]-X_pca[int(i),:]) for i in ni[1:]])
        rho=np.sum(np.exp(-dij**2/dc**2))
        nei_w.append(np.exp(-dij**2/dc**2)/np.sum(np.exp(-dij**2/dc**2)))
        rho_arr.append(rho)
        
    rho_arr=np.array(rho_arr)/np.amax(rho_arr)
    nei_w=np.array(nei_w)
    nei_w=np.hstack((np.ones((nei_w.shape[0],1)),nei_w))/2
    X_s=X_val.copy()
    for ci in range(len(X_val)):
        X_s[ci]=np.dot(X_val[cell_nei[ci,:]],nei_w[ci,:])
    return X_s

def crc_smooth(adata, mu_learned, sigma_learned,k_nei):
    X = torch.tensor(adata.layers['Ms'], dtype=torch.float32) 
    row = np.array([np.ones((k_nei,))*i for i in range(adata.shape[0])]).flatten()
    col = adata.uns['neighbors']['indices'].flatten()
    adj_val = np.ones(col.shape)
    A_mat = csr_matrix((adj_val, (row, col)), shape=(adata.shape[0], adata.shape[0]))
    A = A_mat
    # 算曲率
    cRc_arr_eu=[]
    for inds in np.split(A.indices, A.indptr)[1:-1]:
        self_ind=inds[0]
        cRc_eu=0
        for nei_k in range(1,len(inds)):
            dEu=np.linalg.norm(X[self_ind,:]-X[inds[nei_k],:])
            dWa=wasserstein_distance(mu_learned[self_ind,:],sigma_learned[self_ind,:],\
                            mu_learned[inds[nei_k],:],sigma_learned[inds[nei_k],:])
            cRc_eu+=1-dWa/dEu
        cRc_arr_eu.append(cRc_eu/len(inds))
    crc_eu = np.array(cRc_arr_eu)
    crc_smooth = smooth_func(crc_eu, adata, k_nei)
    return crc_smooth

def apply_plot_style1(ax, size = 14, length_y = 6, length_x = 6):
    """
    应用加粗、大字号和加粗坐标轴边框的通用样式。
    """
    
    # 调整边框和刻度线 (加粗边框)
    ax.spines['right'].set_visible(0)
    ax.spines['top'].set_visible(0)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    
    # 刻度标签字体
    ax.tick_params(
        axis='x', # 只针对 X 轴
        width=1.5, 
        labelsize=size-2, 
        which='both', 
        direction='out', 
        length=length_x, # 使用新的 length_x
        color = "black"
    )
    
    # 设置 Y 轴刻度参数 (使用 length_y)
    ax.tick_params(
        axis='y', # 只针对 Y 轴
        width=1.5, 
        labelsize=size-2, 
        which='both', 
        direction='out', 
        length=length_y, # 使用新的 length_y
        color = "black",

    )
    # 加粗刻度标签字体
    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')

    # X轴标签字体
    ax.set_xlabel(ax.get_xlabel(), fontsize=size-2, fontweight='bold')
    ax.set_ylabel(ax.get_ylabel(), fontsize=size-2, fontweight='bold')
    

    # 标题字体
    if ax.get_title():
        ax.set_title(ax.get_title(), fontsize=size, fontweight='bold')


def apply_plot_style2(ax, size = 14, length_y = 6, length_x = 6):
    """
    应用加粗、大字号和加粗坐标轴边框的通用样式。
    """
    
    # 调整边框和刻度线 (加粗边框)
    ax.spines['right'].set_visible(1)
    ax.spines['top'].set_visible(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    
    # 刻度标签字体
    ax.tick_params(
        axis='x', # 只针对 X 轴
        width=1.5, 
        labelsize=size-2, 
        which='both', 
        direction='out', 
        length=length_x, # 使用新的 length_x
        color = "black"
    )
    
    # 设置 Y 轴刻度参数 (使用 length_y)
    ax.tick_params(
        axis='y', # 只针对 Y 轴
        width=1.5, 
        labelsize=size-2, 
        which='both', 
        direction='out', 
        length=length_y, # 使用新的 length_y
        color = "black",

    )
    # 加粗刻度标签字体
    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')

    # X轴标签字体
    ax.set_xlabel(ax.get_xlabel(), fontsize=size-2, fontweight='bold')
    ax.set_ylabel(ax.get_ylabel(), fontsize=size-2, fontweight='bold')
    

    # 标题字体
    if ax.get_title():
        ax.set_title(ax.get_title(), fontsize=size, fontweight='bold')

def FI_i(ax, S_FI_Eg, cell_arr, i = 0, ymin=None, ymax=None):
    
    FI_Param1 = S_FI_Eg[:, i]

    # 1. 按 RC 分组，把值抽成列表
    data_groups = []
    for cell_indices in cell_arr:
        data_groups.append(FI_Param1[cell_indices])

    # 2. 画 violin
    parts = ax.violinplot(data_groups,
                          positions=np.arange(1, len(cell_arr)+1),
                          widths=0.7,
                          bw_method=0.2,      # 对应 seaborn 的 bw=.2
                          showmeans=False,
                          showmedians=False,
                          showextrema=False)

    # 3. 统一改颜色
    for pc in parts['bodies']:
        pc.set_facecolor('#0072b2')
        pc.set_edgecolor('#0072b2')
        pc.set_alpha(1)  
    # ax.set_xlabel('RC')
    ax.set_ylabel(f'FI (normalized)')
    
    reaction_coordinates = np.arange(1, len(cell_arr) + 1)
    ax.set(xticks=reaction_coordinates, xticklabels=reaction_coordinates)
    # ax.set_xticks(np.arange(len(cell_arr)), reaction_coordinates) 
    # ax.set_xticklabels(reaction_coordinates)
    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin=ymin, ymax=ymax)
    apply_plot_style2(ax, length_x=12, size = 16)
    ax.set_xlabel('')
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')
        
def variance_i(ax, pca_svd_rc_variance_all, i = 0, ymin=None, ymax=None):
    data_groups = [pca_svd_rc_variance_all[j][:,i] for j in range(pca_svd_rc_variance_all.shape[0])]

    # 2. 画 violin
    parts = ax.violinplot(data_groups,
                          positions=np.arange(1, pca_svd_rc_variance_all.shape[0] + 1),
                          widths=0.7,
                          bw_method=0.2,
                          showmeans=False,
                          showmedians=False,
                          showextrema=False)

    for pc in parts['bodies']:
        pc.set_facecolor('#0072b2')
        pc.set_edgecolor('#0072b2')
        pc.set_alpha(1)
        
    ax.set_yscale('log')
    ax.set_xlabel('RC')  # ax和plt调用函数的区别就是比plt多一个set_
    ax.set_ylabel(r"$\mathbf{CV}^{\mathbf{2}}$") 
    reaction_coordinates = np.arange(1, pca_svd_rc_variance_all.shape[0] + 1)

    ax.set(xticks=reaction_coordinates, xticklabels=reaction_coordinates)
    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin=ymin, ymax=ymax)
    apply_plot_style2(ax, size=16)

def FI_error_i(ax, S_FI_Eg, cell_arr, i=0, ymin=None, ymax=None, size=18):
    
    FI_Param1 = S_FI_Eg[:, i]
    data = [FI_Param1[cell_indices] for cell_indices in cell_arr]

    x = np.arange(1, len(cell_arr) + 1)
    y = np.array([np.mean(d) for d in data])
    max_vals = np.array([np.max(d) for d in data])
    min_vals = np.array([np.min(d) for d in data])
    error = np.array([y - min_vals, max_vals - y])

    color = '#0072b2'
    ax.errorbar(x, y, yerr=error, fmt='^', capsize=5,
                color=color, markersize=10, elinewidth = 2)

    ax.set_ylabel(r'$\mathbf{FI}$ (normalized)')
    ax.set_xticks(x)
    ax.set_xlabel('')

    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin=ymin, ymax=ymax)

        # 设置 x 轴刻度为整数
    if len(x) <= 7:  # 如果只有5个点
        ax.set_xticks(range(1, len(x) + 1))
        ax.set_xlim(0.5, len(x) + 0.5)
    elif 8<= len(x) <= 13:  # 如果有10个点
        ax.set_xticks(range(2, len(x) + 1, 2))
        ax.set_xlim(0.5, len(x) + 0.5)

    elif 14<= len(x):  # 如果有10个点
        ax.set_xticks(range(2, len(x) + 1, 3))
        ax.set_xlim(0.5, len(x) + 0.5)


    apply_plot_style2(ax, length_x=12, size=size)
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')

def variance_error_i(ax, pca_svd_rc_variance_all, i=0, ymin=None, ymax=None, size=18):
    
    data = [pca_svd_rc_variance_all[j][:, i]
            for j in range(pca_svd_rc_variance_all.shape[0])]

    x = np.arange(1, len(data) + 1)
    y = np.array([np.mean(d) for d in data])
    max_vals = np.array([np.max(d) for d in data])
    min_vals = np.array([np.min(d) for d in data])
    error = np.array([y - min_vals, max_vals - y])

    color = '#0072b2'
    ax.errorbar(x, y, yerr=error, fmt='^', capsize=5,
                color=color, markersize=10, elinewidth = 2)

    ax.set_yscale('log')
    ax.set_xlabel('RC')
    ax.set_ylabel(r"$\mathbf{CV}^{\mathbf{2}}$")
    ax.set_xticks(x)

    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin=ymin, ymax=ymax)

    # 设置 x 轴刻度为整数
    if len(x) <= 7:  # 如果只有5个点
        ax.set_xticks(range(1, len(x) + 1))
        ax.set_xlim(0.5, len(x) + 0.5)
    elif 8<= len(x) <= 13:  # 如果有10个点
        ax.set_xticks(range(2, len(x) + 1, 2))
        ax.set_xlim(0.5, len(x) + 0.5)

    elif 14<= len(x):  # 如果有10个点
        ax.set_xticks(range(2, len(x) + 1, 3))
        ax.set_xlim(0.5, len(x) + 0.5)

    apply_plot_style2(ax, size=size)

def FI_VAR_double_grid(S_FI_Eg, pca_svd_rc_variance_all, X_pca, L, cell_arr, norm=True, simple = True, share_y = False
                       ,fi_global_min = None, fi_global_max = None, variance_global_min = None, variance_global_max = None, var_norm = False,
                       save = False, figure_path = "/home/linux/桌面/Is single cell sloppy?/14_画图/figure/", branch_name  = None):

    if L > X_pca.shape[1]:
        L = X_pca.shape[1]/2
    n_params = int(2 *L)
    if simple:
        n_params = 1
    ncols = int(5)
    if n_params <= 12:
        ncols = n_params
    nrows = int(np.ceil(n_params / ncols))

    if norm:
        S_FI_Eg = (S_FI_Eg / S_FI_Eg.sum(axis=1, keepdims=True)) 

    pca_svd_rc_variance_all = np.array(pca_svd_rc_variance_all)
    if var_norm:
        pca_svd_rc_variance_all = (pca_svd_rc_variance_all / pca_svd_rc_variance_all.sum(axis=-1, keepdims=True))

    if share_y:
        fi_global_min = 0.0
        fi_global_max = np.percentile(S_FI_Eg.flatten(), 100)
        variance_data = np.array(pca_svd_rc_variance_all)[:, :, :n_params].flatten() 
        variance_global_min = 0.0
        variance_global_max = np.percentile(variance_data, 100)

    fig, axes = plt.subplots(nrows=nrows * 2, ncols=ncols, figsize=(ncols*6, nrows*6), squeeze=False)
    for i in range(n_params):
        row, col = divmod(i, ncols)
        ax_fi = axes[row * 2, col]
        ax_fi.set_title(fr'$\mathbf{{\theta_{{{i + 1}}}}}$') # 设置子图标题
        FI_error_i(ax_fi, S_FI_Eg, cell_arr, i=i, ymin=fi_global_min, ymax=fi_global_max)
        ax_var = axes[row * 2 + 1, col]
        variance_error_i(ax_var, pca_svd_rc_variance_all, i=i, ymin=variance_global_min, ymax=variance_global_max)
        # ax_var.set_title(f'Param {i+1} Projected Var') # 设置子图标题

    for j in range(n_params, nrows * ncols):
        row, col = divmod(j, ncols)
        axes[row * 2, col].remove()
        axes[row * 2 + 1, col].remove()
        
    fig.subplots_adjust(
        top=1,
        hspace=0
    )

    if save:
        os.makedirs(figure_path, exist_ok=True)
        fig.savefig(
            f"{figure_path}{branch_name}_rank1_FI_cv_2.svg",
            format="svg",         
            bbox_inches="tight"
            )
    plt.show()

def variance_along_rc(cell_arr, pca_svd_rc_variance_all, dim = 2, size = 18,
                      save = False, figure_path = "/home/linux/桌面/Is single cell sloppy?/14_画图/figure/", branch_name  = None):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    x = np.array(range(pca_svd_rc_variance_all.shape[0])) + 1

    legend_handles = []
    legend_labels = []
    markers = ['^', 's', 'o']
    for i in range(dim):
        data = [pca_svd_rc_variance_all[j][:,i] for j in range(pca_svd_rc_variance_all.shape[0])]
        y = np.array([np.mean(data[i]) for i in range(len(data))])
        max_vals = np.array([np.max(data[i]) for i in range(len(data))])
        min_vals = np.array([np.min(data[i]) for i in range(len(data))])
        upper_error = max_vals - y
        lower_error = y - min_vals 
        error = np.array([lower_error, upper_error])
        if i == 0:
            color = '#0072b2'
        elif i == 1:
            color = '#f0e442'
        else:
            color = '#d55e00'

        marker = markers[i]
        ax.errorbar(x, y, yerr=error, fmt=marker, capsize=5, 
                color = color, label=fr'$\mathbf{{\theta_{{{i + 1}}}}}$', markersize=8)
        custom_handle = Line2D([0], [0], 
                                marker=marker,          # 圆圈标记
                                color='w',           # 线条颜色设为白色/无色
                                markerfacecolor=color, # 标记填充颜色
                                markersize=12,       # 标记大小
                                linestyle='None')    # 不显示连接线

        # 3. 添加到列表
        legend_handles.append(custom_handle)
        legend_labels.append(fr'$\mathbf{{\theta_{{{i + 1}}}}}$',)
        
    ax.set_xlabel('RC')
    ax.set_ylabel(r"$\mathbf{CV}^{\mathbf{2}}$")
    ax.legend(
        handles=legend_handles, 
        labels=legend_labels,
        frameon=False,        # <--- 去除边框
        fontsize=size-2,          # 保持图例字体大一些
        handlelength=0,       # <--- 设置句柄长度为 0，隐藏线条
        handletextpad=1,    # 句柄与文本之间的距离
        loc='best',            # 或 'upper right' 等，根据需要调整位置
        ncols = len(legend_labels) - 1
    )
    ax.set_yscale('log')

    # 设置 x 轴刻度为整数
    if len(x) <= 7:  # 如果只有5个点
        ax.set_xticks(range(1, len(x) + 1))
        ax.set_xlim(0.5, len(x) + 0.5)
    elif 8<= len(x) <= 13:  # 如果有10个点
        ax.set_xticks(range(2, len(x) + 1, 2))
        ax.set_xlim(0.5, len(x) + 0.5)

    elif 14<= len(x):  # 如果有10个点
        ax.set_xticks(range(2, len(x) + 1, 3))
        ax.set_xlim(0.5, len(x) + 0.5)


    apply_plot_style2(ax, size=size)

    if save:
        os.makedirs(figure_path, exist_ok=True)
        fig.savefig(
            f"{figure_path}{branch_name}_cv_2.svg",
            format="svg",         
            bbox_inches="tight"
            )
    plt.show()

def rc_path(adata, mu_learned, sigma_learned, average_path, basis, k_nei, data_name):

    SIZE = 14 
    TICK_LENGTH = 4
    TICK_WIDTH = 1.5

    if "EG" in data_name or "zebrafish" in data_name:
        loc_path_graph = 'upper right'
        axis_name_1 = "Umap1"
        axis_name_2 = "Umap2"
    else:
        loc_path_graph = 'upper center'
        axis_name_1 = "PCA1"
        axis_name_2 = "PCA2"
        
    if basis == "umap":
        X_umap = adata.obsm['X_umap']
        X_plt = X_umap[:,0]
        Y_plt = X_umap[:,1]
        X_min = np.min(X_plt)
        X_max = np.max(X_plt)
        Y_min = np.min(Y_plt)
        Y_max = np.max(Y_plt)
        X_len = (X_max-X_min)/5
        Y_len = (Y_max-Y_min)/5
        wid = min(X_len,Y_len)/30
        X_ori = X_min-wid*10
        Y_ori = Y_min-wid*10
    elif basis == "pca":
        X_umap = adata.obsm['X_pca'][:, :2]
        if "DG" in data_name:
            X_plt = -X_umap[:,0]
        else:
            X_plt = X_umap[:,0]
        Y_plt = X_umap[:,1]
        X_min = np.min(X_plt)
        X_max = np.max(X_plt)
        Y_min = np.min(Y_plt)
        Y_max = np.max(Y_plt)
        X_len = (X_max-X_min)/5
        Y_len = (Y_max-Y_min)/5
        wid = min(X_len,Y_len)/30
        X_ori = X_min-wid*10
        Y_ori = Y_min-wid*10

    crc_smooth_ = crc_smooth(adata, mu_learned, sigma_learned, k_nei)
    fig, ax = plt.subplots(figsize=(6, 3.75))          
    
    # --- 刻度线参数调整 ---
    ax.tick_params(axis='both', which='major',
                labelsize=SIZE - 4, # 20 -> 10 (SIZE - 4)
                width=TICK_WIDTH,   # 2 -> 1.5
                length=TICK_LENGTH, # 4 -> 4
                direction='in')

    # --- 坐标轴边框处理 (保持原意，隐藏所有边框) ---
    for spine in ax.spines.values():
            spine.set_visible(False)
    
    # --- 绘制散点图 (保持不变) ---
    idx = ~np.isnan(crc_smooth_)
    cmap_bg = plt.colormaps['Spectral']

    im = ax.scatter(X_plt[idx], Y_plt[idx], c=crc_smooth_[idx],
                    s=30, cmap=cmap_bg, zorder=1)
    
    # --- 颜色条参数调整 ---
    clb = fig.colorbar(im, ax=ax, shrink=1, aspect=28) 
    clb.ax.set_ylabel('Curvature', 
                    fontsize=SIZE + 2, # 24 -> 16 (SIZE + 2)
                    weight='bold', 
                    labelpad=16) # labelpad 保持不变，因为它影响位置
    clb.ax.tick_params(labelsize=SIZE - 2, # 18 -> 12 (SIZE - 2)
                    width=TICK_WIDTH,    # 2 -> 1.5
                    length=2)
    for l in clb.ax.yaxis.get_majorticklabels():
        l.set_weight('bold')

    # --- 路径点绘制 (保持不变) ---
    n_points = len(average_path)
    colors = np.linspace(0, 1, n_points)
    path_sc = ax.scatter(average_path[1:-1, 0], average_path[1:-1, 1],
                        c=colors[1:-1], cmap='cool', s=150, zorder=3) # s=150 保持不变

    cool_cmap = plt.cm.get_cmap('cool')
    start_color = cool_cmap(0.0)
    end_color = cool_cmap(1.0)

    # 路径起点和终点大小
    ax.scatter(average_path[0, 0], average_path[0, 1],
            c=[start_color], s=180, label='Start', zorder=4) # 250 -> 180 (稍减小)
    ax.scatter(average_path[-1, 0], average_path[-1, 1],
            c=[end_color], s=180, label='End', zorder=4) # 250 -> 180 (稍减小)
    
    # --- 箭头和文本 (PCA 坐标轴指示) 参数调整 ---
    # 箭头宽度 wid 保持不变
    ax.arrow(X_ori - wid / 2, Y_ori, X_len, 0,
            width=wid * 1.5, color='black', head_width=5 * wid * 1.5, zorder=2)
    ax.arrow(X_ori, Y_ori - wid / 2, 0, Y_len,
            width=wid * 1.5, color='black', head_width=5 * wid * 1.5, zorder=2)
    
    # 轴标签文本大小
    ax.text(X_ori + X_len / 2, Y_ori - wid * 14,
            rf'{axis_name_1}', fontsize=SIZE - 2, ha='center', weight='bold') # 18 -> 12 (SIZE - 2)
    ax.text(X_ori - wid * 24, Y_ori + Y_len / 2,
            rf'{axis_name_2}', fontsize=SIZE - 2, ha='center', weight='bold') # 18 -> 12 (SIZE - 2)
            
    # --- 图例参数调整 ---
    handles, labels = ax.get_legend_handles_labels()

    ax.legend(handles, labels,
            loc=loc_path_graph,
            prop={'size': SIZE - 2, 'weight': 'bold'}, # 18 -> 12 (SIZE - 2)
            fontsize=SIZE - 2, # 18 -> 12 (SIZE - 2)
            markerscale=0.8, # 保持不变
            frameon=True, edgecolor='black')

    # --- 最终清理 ---
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('')
    

    fig.tight_layout()
    # plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.show()

def get_velocity(adata,pca_dim,k_nei):
    # get velocity of genes and PCAs.
    scv.pp.pca(adata, n_comps=pca_dim)
    scv.pp.neighbors(adata, n_neighbors=k_nei)
    scv.pp.moments(adata, n_pcs=pca_dim, n_neighbors=k_nei)
    scv.tl.velocity(adata)

    velo0 = np.array(adata.layers['velocity'])
    velo_g = np.zeros(velo0.shape)
    velo_g[:,adata.var['velocity_genes']] = velo0[:,adata.var['velocity_genes']]
    velo_pca = velo_g@adata.varm['PCs']

    return velo_g, velo_pca

def v_gene_v_pca_rc(velo_g, velo_pca, cell_arr):       
    velo_pca_sum = np.sqrt(np.sum(velo_pca**2, axis=1))
    velo_g_sum = np.sqrt(np.sum(velo_g**2, axis=1))

    print(velo_pca_sum.shape)
    print(velo_g_sum.shape)

    velo_pca_average = np.zeros([len(cell_arr)])
    velo_g_average = np.zeros([len(cell_arr)])

    for i in range(len(cell_arr)):
        velo_pca_path = velo_pca_sum[cell_arr[i]]
        velo_pca_average[i] = np.mean(velo_pca_path, axis=0)
        velo_g_path = velo_g_sum[cell_arr[i]]
        velo_g_average[i] = np.mean(velo_g_path, axis=0)

        if i == 0 :
            print(velo_pca_path.shape)
            print(velo_g_path.shape)

    fig, ax1 = plt.subplots(figsize=(8, 6))
    # 左边 y 轴（红色）
    ax1.scatter(range(velo_pca_average.shape[0]), 10 * velo_pca_average, color='purple', label='velo_pca', s=50)
    ax1.set_xlabel('Path', fontsize=14, fontweight='bold')
    ax1.set_ylabel(f'v pca', color='purple', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='both', labelsize=12, width=2, length=6)
    ax1.tick_params(axis='y', labelcolor='purple')
    # 右边 y 轴（蓝色）
    ax2 = ax1.twinx()
    ax2.scatter(range(velo_g_average.shape[0]),  10 * velo_g_average, color='blue', label='velo_g', s=50)
    ax2.set_ylabel(f'v gene', color='blue', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='y', labelsize=12, width=2, length=6, labelcolor='blue')
    # 加粗坐标轴边框
    for spine in ax1.spines.values():
        spine.set_linewidth(2)
    for spine in ax2.spines.values():
        spine.set_linewidth(2)
    plt.title(f'|v|', fontsize=16, fontweight='bold')
    fig.tight_layout()
    # plt.savefig(figure_path + f'V_of_pca_and_gene.png', dpi=600, bbox_inches='tight')
    plt.show()

def sort_FI(FI_m_s_average):
    fi_ranges = []
    for i in range(FI_m_s_average.shape[1]):
        fi_range = np.max(FI_m_s_average[:, i]) - np.min(FI_m_s_average[:, i])
        fi_ranges.append((i, fi_range))
    # 按FI范围降序排序
    fi_ranges.sort(key=lambda x: x[1], reverse=True)
    sorted_indices = [item[0] for item in fi_ranges]
    return sorted_indices

def rc_path_celltype(adata, average_path, basis, data_name, save=False, 
                     figure_path="/home/linux/桌面/Is single cell sloppy?/14_画图/figure/", branch_name = None):

    if "EG" in data_name:
        loc_path_graph = 'upper right'
        axis_name_1 = "Umap1"
        axis_name_2 = "Umap2"
        clusters_name = "clusters"
        clusters_color = "clusters_colors"
    elif "zebrafish" in data_name:
        loc_path_graph = 'upper right'
        axis_name_1 = "Umap1"
        axis_name_2 = "Umap2"
        clusters_name = "celltype"
        clusters_color = "Cell_type_colors"
    elif "DG" in data_name:
        loc_path_graph = 'upper center'
        axis_name_1 = "PCA1"
        axis_name_2 = "PCA2"
        clusters_name = "ClusterName"
        clusters_color = "ClusterName_colors"

    if basis == "umap":
        X_umap = adata.obsm['X_umap']
        X_plt = X_umap[:,0]
        Y_plt = X_umap[:,1]
        X_min = np.min(X_plt)
        X_max = np.max(X_plt)
        Y_min = np.min(Y_plt)
        Y_max = np.max(Y_plt)
        X_len = (X_max-X_min)/5
        Y_len = (Y_max-Y_min)/5
        wid = min(X_len,Y_len)/30
        X_ori = X_min-wid*10
        Y_ori = Y_min-wid*10
    elif basis == "pca":
        X_umap = adata.obsm['X_pca'][:, :2]
        if "DG" in data_name:
            X_plt = -X_umap[:,0]
        else:
            X_plt = X_umap[:,0]
        Y_plt = X_umap[:,1]
        X_min = np.min(X_plt)
        X_max = np.max(X_plt)
        Y_min = np.min(Y_plt)
        Y_max = np.max(Y_plt)
        X_len = (X_max-X_min)/5
        Y_len = (Y_max-Y_min)/5
        wid = min(X_len,Y_len)/30
        X_ori = X_min-wid*10
        Y_ori = Y_min-wid*10
        
    celltype = adata.obs[clusters_name]
    # 获取 scanpy 中细胞类型的颜色映射
    if clusters_color in adata.uns:
        # 从 adata.uns 中获取颜色列表
        cluster_colors = adata.uns[clusters_color]
        # 获取细胞类型顺序
        cluster_categories = adata.obs[clusters_name].cat.categories
        # 创建颜色映射字典
        color_dict = dict(zip(cluster_categories, cluster_colors))
    else:
        # 如果没有找到颜色信息，使用默认颜色映射
        unique_types = np.unique(celltype[~pd.isnull(celltype)])
        color_dict = {}
        # 使用 Spectral_r 颜色映射生成颜色
        cmap_bg = plt.colormaps['Spectral_r']
        for i, cell_type in enumerate(unique_types):
            color_dict[cell_type] = cmap_bg(i / (len(unique_types) - 1))


    print("用的color_dict是：", color_dict)

    fig, ax = plt.subplots(figsize=(7, 5))          
    ax.tick_params(axis='both', which='major',
                labelsize=20, width=2, length=4, direction='in')

    for spine in ax.spines.values():
        spine.set_visible(False)

    idx = ~pd.isnull(celltype)

    # 使用从 scanpy 获取的颜色
    celltype_colors = [color_dict[t] for t in np.array(celltype)[idx]]

    im = ax.scatter(
        X_plt[idx],
        Y_plt[idx],
        c=celltype_colors,  # 直接使用颜色列表
        s=20,
        alpha = 0.75,
        zorder=1
    )
    print("用的celltype_colors是：", celltype_colors)

    # === 以下部分保持不变 ===
    n_points = len(average_path)
    colors = np.linspace(0, 1, n_points)
    path_sc = ax.scatter(average_path[1:-1, 0], average_path[1:-1, 1],
                         c=colors[1:-1], cmap='cool', s=150, zorder=3)
    # === 以下部分保持不变 ===
    cool_cmap = plt.cm.get_cmap('cool')
    start_color = cool_cmap(0.0)
    end_color = cool_cmap(1.0)
    ax.scatter(average_path[0, 0], average_path[0, 1], c=[start_color], s=250, label='Start', zorder=4)
    ax.scatter(average_path[-1, 0], average_path[-1, 1], c=[end_color], s=250, label='End', zorder=4)
    ax.arrow(X_ori - wid / 2, Y_ori, X_len, 0, width=wid*1.5, color='black', head_width=5*wid*1.5, zorder=2)
    ax.arrow(X_ori, Y_ori - wid / 2, 0, Y_len, width=wid*1.5, color='black', head_width=5*wid*1.5, zorder=2)
    ax.text(X_ori + X_len / 2, Y_ori - wid * 14, rf"{axis_name_1}", fontsize=18, ha='center', weight='bold')
    ax.text(X_ori - wid * 24, Y_ori + Y_len / 2, rf"{axis_name_2}", fontsize=18, ha='center', weight='bold')
    handles, labels = ax.get_legend_handles_labels()


    ax.legend(handles, labels, loc=loc_path_graph,
              prop={'size': 18, 'weight': 'bold'},
              fontsize=18, markerscale=0.8,
              frameon=True, edgecolor='black')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('')
    fig.tight_layout()
    if save:
        os.makedirs(figure_path, exist_ok=True)
        fig.savefig(
            f"{figure_path}{branch_name}_rc_path_celltype.svg",
            format="svg",         
            bbox_inches="tight"
            )
    plt.show()

def rc_scatter(FI_parameter_average, v_parameter_average, newMLP,
               save=False, figure_path=None, branch_name = None,size = 30, mu = False): 
    n_params = FI_parameter_average.shape[1]
    if n_params <= 6:
        ncols = int(FI_parameter_average.shape[1])  
    elif n_params == 12:
        ncols = 6
    else:
        ncols = int(5)
    nrows = int(np.ceil(n_params / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 6, nrows * 5.8), squeeze=False)
    
    color1 = "#DC3220"
    color2 = "#005AB5"
    ax1_list, ax2_list = [], []  # 用于记录所有的左右轴
    for i in range(n_params):
        row, col = divmod(i, ncols)
        ax1 = axes[row, col]
        ax2 = ax1.twinx()
        ax1_list.append(ax1)
        ax2_list.append(ax2)

        # ---- 左侧：Fisher Information（红色）----
        ax1.scatter(range(1, FI_parameter_average.shape[0] + 1),
                    FI_parameter_average[:, i],
                    color=color1, 
                    # label=fr"$\mathbf{{g_{{{i+1},{i+1}}}(\theta)}}$", 
                    label=r"$\mathbf{FI}$",
                    s=120)
        ax1.set_xlabel('RC', fontsize=size, fontweight='bold')
        ax1.tick_params(axis='both', which='major', labelsize=size - 2, width=2, length=3, direction='in')
        ax1.tick_params(axis='y',labelsize=size -2, labelcolor=color1, color=color1, length=0)
        ax1.set_xticks(range(1, FI_parameter_average.shape[0] + 1, 3))
        ax1.set_xlim(-0.5, FI_parameter_average.shape[0] + 1.5)
        ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), fontweight='bold')
        plt.setp(ax1.yaxis.get_majorticklabels(), fontweight='bold')

        # ---- 右侧：Velocity（蓝色）----
        ax2.scatter(range(1, v_parameter_average.shape[0] + 1),
                    v_parameter_average[:, i],
                    color=color2, 
                    # label=fr"$\mathbf{{|V_{{\theta_{{{i+1}}}}}|}}$", 
                    label=r"$\mathbf{Velocity}$",
                    s=120)
        ax2.tick_params(axis='y', labelsize=size -2, width=2, length=0, labelcolor=color2, color=color2, direction='in')
        plt.setp(ax2.yaxis.get_majorticklabels(), fontweight='bold')


        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        if mu:
            if i <=5:
                ax1.set_title(
                fr'$\mathdefault{{\mu_{{{i + 1}}}}}$',
                fontsize=size + 8,
                fontweight='bold',
                pad = 15
            )
            else:
                # ---- 标题 ----
                # ax1.set_title(f'Param {i + 1}' if newMLP else f'param {i + 1}', fontsize=size +2, fontweight='bold')
                ax1.set_title(
                    fr'$\mathdefault{{\sigma_{{{i + 1-6}}}}}$',
                    fontsize=size + 8,
                    fontweight='bold',
                    pad = 15
                )
        else:
            ax1.set_title(
                    fr'$\mathdefault{{\sigma_{{{i + 1}}}}}$',
                    fontsize=size + 8,
                    fontweight='bold',
                    pad = 15
                )


        # ---- 边框样式 ----
        for spine in ax1.spines.values():
            spine.set_visible(False)
        for spine in ax2.spines.values():
            spine.set_visible(False)
        ax1.spines['left'].set_visible(True)
        ax1.spines['top'].set_visible(True)
        ax1.spines['bottom'].set_visible(True)
        ax1.spines['left'].set_color(color1)
        ax1.spines['left'].set_linewidth(2)
        ax2.spines['right'].set_visible(True)
        ax2.spines['right'].set_color(color2)
        ax2.spines['right'].set_linewidth(2)

        if i == 0:
            # ---- 图例 ----
            handles2, labels2 = ax1.get_legend_handles_labels()
            handles1, labels1 = ax2.get_legend_handles_labels()
            leg = ax1.legend(handles2 + handles1, labels2 + labels1, loc='best',ncols = 2,
                    prop={'size': size-6, 'weight': 'bold'}, markerscale=1.2, frameon=False, edgecolor='black', handletextpad=0)
            for handle in leg.legendHandles:
                if hasattr(handle, 'set_offset_position'):
                    handle.set_offset_position('data')
                    handle.set_offsets((0, 0))
        
    # ---- 统一Y轴范围 ----
    all_y1 = np.concatenate([ax1.collections[0].get_offsets()[:,1] for ax1 in ax1_list])
    all_y2 = np.concatenate([ax2.collections[0].get_offsets()[:,1] for ax2 in ax2_list])
    y1_min, y1_max = all_y1.min(), all_y1.max()
    y2_min, y2_max = all_y2.min(), all_y2.max()
    span1, span2 = y1_max - y1_min, y2_max - y2_min
    factor = 0.5  # 增高20%
    for ax1, ax2 in zip(ax1_list, ax2_list):
        ax1.set_ylim(y1_min - span1 * 0.05, y1_max + span1 * factor)
        ax2.set_ylim(y2_min - span1 * 0.05, y2_max + span2 * factor)

    if save:
        os.makedirs(figure_path, exist_ok=True)
        for i in range(n_params):
            fig_s, ax1_s = plt.subplots(figsize=(6, 5))
            ax2_s = ax1_s.twinx()

            # ---- 左侧：Fisher Information（红色）----
            ax1_s.scatter(range(1, FI_parameter_average.shape[0] + 1),
                        FI_parameter_average[:, i],
                        color=color1, 
                        # label=fr"$\mathbf{{g_{{{i+1},{i+1}}}(\theta)}}$", 
                        label=r"$\mathbf{FI}$",
                        s=60)
            ax1_s.set_xlabel('Path', fontsize=size, fontweight='bold')
            ax1_s.tick_params(axis='both', which='major', labelsize=size -2, width=2, length=3, direction='in')
            ax1_s.tick_params(axis='y', labelsize=size -2, labelcolor=color1, color=color1, length=0)
            ax1_s.set_xticks(range(1, FI_parameter_average.shape[0] + 1, 3))
            ax1_s.set_xlim(-0.5, FI_parameter_average.shape[0] + 1.5)
            ax1_s.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            plt.setp(ax1_s.xaxis.get_majorticklabels(), fontweight='bold')
            plt.setp(ax1_s.yaxis.get_majorticklabels(), fontweight='bold')

            # ---- 右侧：Velocity（蓝色）----
            ax2_s.scatter(range(1, v_parameter_average.shape[0] + 1),
                        v_parameter_average[:, i],
                        color=color2, 
                        # label=fr"$\mathbf{{|V_{{\theta_{{{i+1}}}}}|}}$", 
                        label =r"$\mathbf{Velocity}$",
                        s=60)
            ax2_s.tick_params(axis='y', labelsize=size -2, width=2, length=0, labelcolor=color2, color=color2, direction='in')
            plt.setp(ax2_s.yaxis.get_majorticklabels(), fontweight='bold')


            # ---- 标题 ----
            # ax1_s.set_title(f'Param {i + 1}' if newMLP else f'param {i + 1}', fontsize=size+2, fontweight='bold')
            # 使用 \boldsymbol{...} 来实现 LaTeX 内部加粗
            if mu:
                if i <=5:
                    ax1_s.set_title(
                    fr'$\mathdefault{{\mu_{{{i + 1}}}}}$',
                    fontsize=size + 8,
                    fontweight='bold',
                    pad = 15
                )
                else:
                    # ---- 标题 ----
                    # ax1.set_title(f'Param {i + 1}' if newMLP else f'param {i + 1}', fontsize=size +2, fontweight='bold')
                    ax1_s.set_title(
                    fr'$\mathdefault{{\sigma_{{{i + 1 -6}}}}}$',
                    fontsize=size + 8,
                    fontweight='bold',
                    pad = 15
                    )
            else:
                ax1_s.set_title(
                fr'$\mathdefault{{\sigma_{{{i + 1}}}}}$',
                fontsize=size + 8,
                fontweight='bold',
                pad = 15
                )


            ax1_s.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax2_s.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            # ⚠️ 关键：使用和大图完全一样的 y-lim
            ax1_s.set_ylim(y1_min, y1_max + span1 * factor)
            ax2_s.set_ylim(y2_min, y2_max + span2 * factor)
            yticks1 = ax1_s.get_yticks()
            ax1_s.set_yticks([y for y in yticks1 if 0<= y <= 1.0])

            yticks2 = ax2_s.get_yticks()
            ax2_s.set_yticks([y for y in yticks2 if 0<= y <= 1.0])
            # ---- 边框样式 ----
            for spine in ax1_s.spines.values():
                spine.set_visible(False)
            for spine in ax2_s.spines.values():
                spine.set_visible(False)
            ax1_s.spines['left'].set_visible(True)
            ax1_s.spines['top'].set_visible(True)
            ax1_s.spines['bottom'].set_visible(True)
            ax1_s.spines['left'].set_color(color1)
            ax1_s.spines['left'].set_linewidth(2)
            ax2_s.spines['right'].set_visible(True)
            ax2_s.spines['right'].set_color(color2)
            ax2_s.spines['right'].set_linewidth(2)

            if i == 0:
                # ---- 图例 ----
                handles2, labels2 = ax1_s.get_legend_handles_labels()
                handles1, labels1 = ax2_s.get_legend_handles_labels()
                leg = ax1_s.legend(handles2 + handles1, labels2 + labels1, loc='best',ncols = 2,
                        prop={'size': size-6, 'weight': 'bold'}, markerscale=1.2, frameon=False, edgecolor='black',handletextpad=0)
                
                for handle in leg.legendHandles:
                    if hasattr(handle, 'set_offset_position'):
                        handle.set_offset_position('data')
                        handle.set_offsets((0, 0))
            # fig_s.tight_layout(h_pad=2.5, w_pad=2.5)
            fig_s.tight_layout()
            fig_s.savefig(
                f"{figure_path}{branch_name}_v_FI_scatterplot_param_{i+1}_.svg",
                bbox_inches="tight",
                format="svg"
            )
            plt.close(fig_s)

    # ---- 移除空白子图 ----
    for j in range(i + 1, nrows * ncols):
        row, col = divmod(j, ncols)
        axes[row, col].remove()

    # fig.tight_layout(h_pad=2.5, w_pad=2.5)
    fig.tight_layout()
    if save:
        os.makedirs(figure_path, exist_ok=True)
        fig.savefig(
            f"{figure_path}{branch_name}_big_v_FI_scatterplot.svg",
            format="svg",         
            bbox_inches="tight"
            )
    plt.show()

def plot_single(data_name, ax, X_umap, Fisher_values_2, name, i, newMLP, clusters, t_list, use_mask, show_cbar_label, 
                ymin=None, ymax=None, size = 20, mu = False):
    if "EG" in data_name:
        axis_name_1 = "Umap1"
        axis_name_2 = "Umap2"
        X_plt = X_umap[:,0]
    elif "zebrafish" in data_name:
        axis_name_1 = "Umap1"
        axis_name_2 = "Umap2"
        X_plt = X_umap[:,0]
    elif "DG" in data_name:
        axis_name_1 = "PCA1"
        axis_name_2 = "PCA2"
        X_plt = -X_umap[:,0]


    
    Y_plt = X_umap[:,1]
    X_min, X_max = np.min(X_plt), np.max(X_plt)
    Y_min, Y_max = np.min(Y_plt), np.max(Y_plt)
    X_len = (X_max-X_min)/5
    Y_len = (Y_max-Y_min)/5
    wid = min(X_len,Y_len)/30
    X_ori = X_min-wid*10
    Y_ori = Y_min-wid*10

    ax.tick_params(axis='both', which='major',
                labelsize=size-2, width=2, length=4, direction='in')
    for spine in ax.spines.values():
        spine.set_visible(False)

    cmap_bg = plt.colormaps['Spectral_r']

    # mask: t_list 内的点用彩色，否则灰色
    mask = np.isin(clusters, t_list) if use_mask else np.ones_like(clusters, dtype=bool)
    # 灰色点（背景）
    ax.scatter(X_plt[~mask], Y_plt[~mask], s=10, c="lightgrey", alpha=0.6, zorder=0)
    c_values = np.sqrt(Fisher_values_2[mask])
    # 去掉最大 5% 和最小 5% 的影响（分位数截断）
    vmin = np.percentile(c_values, 2.5)
    vmax = np.percentile(c_values, 97.5)
    c_clip = np.clip(c_values, vmin, vmax)
    # 彩色点
    scatter = ax.scatter(X_plt[mask], Y_plt[mask], s=30, 
                            c=c_clip, vmin=ymin, vmax=ymax,
                            cmap=cmap_bg, zorder=1)

    # ax.arrow(X_ori - wid/2, Y_ori, X_len, 0, 
    #         width=wid*1.5, color='black', head_width=5*wid*1.5, zorder=2)
    # ax.arrow(X_ori, Y_ori - wid/2, 0, Y_len, 
    #         width=wid*1.5, color='black', head_width=5*wid*1.5, zorder=2)

    # ax.text(X_ori + X_len/2, Y_ori - wid*14, 
    #         rf'{axis_name_1}', fontsize=14, ha='center', weight='bold')
    # ax.text(X_ori - wid*30, Y_ori + Y_len/2, 
    #         rf'{axis_name_2}', fontsize=14, ha='center', weight='bold')

    clb = plt.colorbar(scatter, ax=ax, shrink=1, aspect=28)

    if show_cbar_label:
        pad = 15
        clb.ax.set_ylabel(
            f'{name}',
            fontsize=size + 4,
            weight='bold',
            labelpad=pad,
            rotation=90
        )
    else:
        clb.ax.set_ylabel('')

    clb.ax.tick_params(labelsize=size -2, width=2, length=3)
    for l in clb.ax.yaxis.get_majorticklabels():
        l.set_weight('bold')
    clb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_xticks([])
    ax.set_yticks([])
    if newMLP:
        ax.set_title(f'Param {i + 1}', fontsize=size + 2, fontweight='bold')
    else:
        if mu:
            if i <=5:
                ax.set_title(
                fr'$\mathdefault{{\mu_{{{i + 1}}}}}$',
                fontsize=size + 8,
                fontweight='bold',
                
            )
            else:
                # ---- 标题 ----
                # ax1.set_title(f'Param {i + 1}' if newMLP else f'param {i + 1}', fontsize=size +2, fontweight='bold')
                ax.set_title(
                    fr'$\mathdefault{{\sigma_{{{i + 1-6}}}}}$',
                    fontsize=size + 8,
                    fontweight='bold',
                )
        else:
            ax.set_title(
                    fr'$\mathdefault{{\sigma_{{{i + 1}}}}}$',
                    fontsize=size + 8,
                    fontweight='bold',
                    
                )


def FI_umap_double_grid(X_umap, adata, i, Fisher_g_diag_2, Z_velo_2, t_list,
                    cluster_key, k_nei, data_name,
                    newMLP=True, use_mask = True, share_y = True,
                    save=False, figure_path=None, branch_name = None, size = 30, mu = False):
    clusters = adata.obs[cluster_key].values
    n_params = Fisher_g_diag_2.shape[1]
    if n_params <= 6:
        ncols = int(Fisher_g_diag_2.shape[1])
    elif n_params==12:
        ncols = 6
    else:
        ncols = int(5)
    n_params = Fisher_g_diag_2.shape[1]
    nrows = int(np.ceil(n_params / ncols))
    
    if share_y == True:
        # 分别计算Fisher和Z_velo的全局范围
        fisher_global_min = np.percentile(np.sqrt(Fisher_g_diag_2), 2.5)
        fisher_global_max = np.percentile(np.sqrt(Fisher_g_diag_2), 97.5)
        zvelo_global_min = np.percentile(np.sqrt(Z_velo_2), 2.5)
        zvelo_global_max = np.percentile(np.sqrt(Z_velo_2), 97.5)
    else:
        fisher_global_min = None
        fisher_global_max = None
        zvelo_global_min = None
        zvelo_global_max = None
    
    # 建立 grid
    fig, axes = plt.subplots(nrows=nrows * 2, ncols=ncols, figsize=(ncols*5.5, nrows*6), squeeze=False)

    for i in range(n_params):
        row, col = divmod(i, ncols)
        is_last_row = (col == ncols - 1)
        ax = axes[row * 2, col]
        plot_single(data_name, ax, X_umap, smooth_func(Fisher_g_diag_2[:, i], adata, k_nei),
                    r"$\mathbf{FI}$",
                    i, newMLP, clusters, t_list, use_mask, show_cbar_label=is_last_row,
                    ymin=fisher_global_min, ymax=fisher_global_max, size=size, mu=mu)

        ax = axes[row * 2 + 1, col]
        plot_single(data_name, ax, X_umap, np.abs( smooth_func(Z_velo_2[:, i], adata, k_nei) ),
                    r"$\mathbf{Velocity}$",
                    i, newMLP, clusters, t_list, use_mask,show_cbar_label=is_last_row,
                    ymin=zvelo_global_min, ymax=zvelo_global_max, size=size, mu = mu)
        ax.set_title('')
        if save:
            os.makedirs(figure_path, exist_ok=True)

            fig_single, axes_single = plt.subplots(nrows=2, ncols=1, figsize=(5.5, 6))
            
            plot_single(data_name, axes_single[0], X_umap, smooth_func(Fisher_g_diag_2[:, i], adata, k_nei),
                        r"$\mathbf{FI}$", i, newMLP, clusters, t_list, use_mask,show_cbar_label=is_last_row,
                        ymin=fisher_global_min, ymax=fisher_global_max,mu=mu)
            
            plot_single(data_name, axes_single[1], X_umap, np.abs( smooth_func(Z_velo_2[:, i], adata, k_nei) ),
                        r"$\mathbf{Velocity}$", i, newMLP, clusters, t_list, use_mask,show_cbar_label=is_last_row,
                        ymin=zvelo_global_min, ymax=zvelo_global_max,mu=mu)
            axes_single[1].set_title('')

            fig_single.tight_layout()
            fig_single.savefig(
                f"{figure_path}/{branch_name}_v_FI_cell_param_{i+1}.svg", 
                format="svg", 
                bbox_inches="tight"
            )
            
            # 重要：绘图完成后关闭临时 Figure，防止内存泄漏
            plt.close(fig_single)

    # for j in range(n_params, nrows*ncols):
    #     row, col = divmod(j, ncols)
    #     axes[row, col].remove()

    for j in range(n_params, nrows * ncols):
        row, col = divmod(j, ncols)
        target_row_top = row * 2
        target_row_bottom = row * 2 + 1
        
        axes[target_row_top, col].remove()
        axes[target_row_bottom, col].remove()

    fig.tight_layout()
    if save:
        os.makedirs(figure_path, exist_ok=True)
        fig.savefig(
            f"{figure_path}{branch_name}_big_v_FI_cell.svg",
            format="svg",         
            bbox_inches="tight"
            )
    plt.show()
