import scanpy as sc

# 加载 .h5ad 文件
adata = sc.read_h5ad('../data/ms/filtered_ms_adata.h5ad')

# 查看数据集的形状
print(f"Data shape: {adata.shape}")

# 查看观测（cells）和变量（genes）的数量
num_cells = adata.n_obs
num_genes = adata.n_vars
print(f"Number of cells: {num_cells}")
print(f"Number of genes: {num_genes}")
# 假设细胞类型信息存储在 adata.obs['cell_type'] 中
# 你需要根据实际情况更新 'cell_type' 这一列名
if 'celltype' in adata.obs:
    # 查看所有细胞类型
    cell_types = adata.obs['celltype'].unique()
    
    # 打印细胞类型数量和每种细胞类型的数量
    print(f"总共 {len(cell_types)} 种细胞类型：")
    for cell_type in cell_types:
        count = (adata.obs['celltype'] == cell_type).sum()
        print(f"{cell_type}: {count} 个细胞")
else:
    print("未找到 'celltype' 列，请检查文件中实际存储细胞类型信息的列名。")
