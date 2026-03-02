# -*- coding: utf-8 -*-
import numpy as np
from pymatgen.core import Structure, Element
from pymatgen.analysis.local_env import VoronoiNN, CrystalNN
import mendeleev
import networkx as nx

# Define ATOMIC_PROPERTIES for fallback
ATOMIC_PROPERTIES = {
    'Al': {
        'group': 13, 'period': 3, 'mass': 26.98, 'IP1': 5.99, 'IP2': 18.83, 'IP3': 28.45,
        'ΔH_fusion': 10.71, 'ΔH_vapor': 294.0, 'λ': 237.0, 'V_molar': 10.0, 'r_cov': 1.21,
        'χ': 1.61, 'Z_eff': 3.0, 'ΔH_atom': 330.0, 'EA': 0.43, 'α_p': 6.8, 'r_vdW': 1.84,
        'C': 0.897, 'ρ': 2.70
    },
    'Ni': {
        'group': 10, 'period': 4, 'mass': 58.69, 'IP1': 7.64, 'IP2': 18.17, 'IP3': 35.19,
        'ΔH_fusion': 17.48, 'ΔH_vapor': 378.0, 'λ': 90.9, 'V_molar': 6.59, 'r_cov': 1.24,
        'χ': 1.91, 'Z_eff': 4.0, 'ΔH_atom': 430.0, 'EA': 1.16, 'α_p': 6.4, 'r_vdW': 1.63,
        'C': 0.444, 'ρ': 8.90
    },
    'Ti': {
        'group': 4, 'period': 4, 'mass': 47.87, 'IP1': 6.83, 'IP2': 13.58, 'IP3': 27.49,
        'ΔH_fusion': 14.15, 'ΔH_vapor': 425.0, 'λ': 21.9, 'V_molar': 10.64, 'r_cov': 1.60,
        'χ': 1.54, 'Z_eff': 3.0, 'ΔH_atom': 473.0, 'EA': 0.08, 'α_p': 8.0, 'r_vdW': 2.15,
        'C': 0.523, 'ρ': 4.51
    },
    #Lange's Handbook Record & EA/λ from pymatgen
    'Dy':{
        'group':3,'ΔH_fusion':11.06,'ΔH_vapor':280
        },
    'Er':{
        'group':3,'ΔH_fusion':19.90,'ΔH_vapor':280
        },
    'Eu':{
        'group':3,'ΔH_fusion':9.21,'ΔH_vapor':176,'χ':1.2
        },
    'Gd':{
        'group':3,'ΔH_fusion':10.05,'ΔH_vapor':301.3,'λ':10.6,'EA': 0.0, 'η': 3.0749#数据由grok访问dunning等文献查找，根据其摘要大致认为准确（EA在0.5~-0.3eV之间）
        },
    'Mg':{
        'EA': 0.0, 'η': 3.82312
        },
    'Ca':{
        'λ':200
        },
    'Sr':{
        'λ':35.3
        },
    'Ba':{
        'λ':18.4
        },    
    'Mn':{
        'λ':7.82,'EA': 0.0, 'η': 3.71701
        },
    'Zn':{
        'EA': 0.0, 'η': 4.6971
        },
    'Y':{
        'λ':17.2
        },
    'Ru':{
        'ΔH_vapor':591.6
        },
    'Cd':{
        'EA': 0.0, 'η': 4.4969
        },
    'Ce':{
        'group':3
        },
    'Pr':{
        'group':3
        },
    'Nd':{
        'group':3,'λ':16.5
        },
    'Sm':{
        'group':3,'λ': 13.3, 'EA': 0.15, 'η': 2.746855
        },
    'Tb':{
        'group':3,'ΔH_fusion':10.15,'χ':1.2
        },
    'Ho':{
        'group':3,'ΔH_fusion':16.8,'λ':16.2,'EA': 0.0, 'η': 3.01075
        },
    'Tm':{
        'group':3,'ΔH_fusion':16.84,'λ': 17.0
        },
    'Yb':{
        'group':3,'λ': 38.5, 'χ': 1.1
        },
    'Lu':{
        'group':3,'ΔH_fusion':22,'λ': 16.4
        },
    'Th':{
        'group':3,'λ': 54.0, 'EA': 0.608, 'η': 2.736
        },
    'U':{
        'group':3,'EA': 0.309, 'η': 2.94255
        }
}
def generate_fragments(structure, adjacency_matrix, distances, max_path_length=3):
    n_atoms = len(structure)
    species = [site.species_string for site in structure]
    
    # 1. 构建无向图
    G = nx.Graph()
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if adjacency_matrix[i, j] == 1:
                G.add_edge(i, j, weight=distances[i, j])  # 添加边和距离权重
    #1.生成路径片段
    # 化学种类去重
    fragments = {'path': [], 'circular': []}
    path_type_set = set()  # 基于种类序列去重
    path_index_set = set()  # 避免索引重复
    unique_starts = set()  # 限制起点
    print(n_atoms)
    for i in range(n_atoms):
        if species[i] not in unique_starts:  # 每个物种选一个代表起点
            unique_starts.add(species[i])
            for target in range(n_atoms):
                if target != i and nx.has_path(G, i, target):
                    paths = list(nx.all_shortest_paths(G, i, target))
                    for path in paths:
                        path_length = len(path) - 1
                        if 1 <= path_length <= max_path_length:
                            path_indices = tuple(path)
                            path_species = tuple(species[node] for node in path)  # 种类序列
                            if path_indices not in path_index_set and tuple(reversed(path)) not in path_index_set:
                                if path_species not in path_type_set:
                                    path_type_set.add(path_species)
                                    path_index_set.add(path_indices)
                                    fragments['path'].append((path_indices, list(path_species)))
                                    print(f"Path: {path_indices} -> {list(path_species)} (length={path_length})")
    # 2.生成环形片段
    for i in range(n_atoms):
        center_species = species[i]
        neighbors = [(species[j], distances[i, j]) for j in range(n_atoms) if adjacency_matrix[i, j] == 1]
        fragments['circular'].append({'center': center_species, 'neighbors': neighbors})
    
    neighbor_counts = np.array([len(frag['neighbors']) for frag in fragments['circular']])
    diameter = max(nx.shortest_path_length(G, source, target) for source in G for target in G if source != target)
    print(f"Graph diameter: {diameter}")
    return fragments, neighbor_counts
    # # 2. 生成路径片段 (l <= 3)
    # fragments = {'path': [], 'circular': []}
    # path_set = set()  # 基于索引去重
    # for source in range(n_atoms):
    #     for target in range(n_atoms):  # 包含 source == target 的情况
    #         if source != target and nx.has_path(G, source, target):
    #             paths = list(nx.all_shortest_paths(G, source, target))
    #             for path in paths:
    #                 if 1 <= len(path) - 1 <= max_path_length:  # l = 1, 2, 3
    #                     path_indices = tuple(path)  # 索引序列
    #                     path_species = [species[node] for node in path]
    #                     if path_indices not in path_set and tuple(reversed(path)) not in path_set:
    #                         path_set.add(path_indices)
    #                         fragments['path'].append((path_indices, path_species))
    #                         print(f"Path: {path_indices} -> {path_species}")
    
    # # 3. 生成环形片段 (l = 2)
    # for i in range(n_atoms):
    #     center_species = species[i]
    #     neighbors = [(j, distances[i, j]) for j in range(n_atoms) if adjacency_matrix[i, j] == 1]
    #     circular_frag = {'center': center_species, 'neighbors': []}
    #     for j, dist in neighbors:
    #         neighbor_species = species[j]
    #         circular_frag['neighbors'].append((neighbor_species, dist))
    #     fragments['circular'].append(circular_frag)
    #     print(f"Circular fragment for atom {i} ({center_species}): {circular_frag}")
    
    # # 4. 计算邻居数量
    # neighbor_counts = np.array([len(frag['neighbors']) for frag in fragments['circular']])
    
    # return fragments, neighbor_counts

def get_element_property(element_symbol, prop):
    # if element_symbol in ATOMIC_PROPERTIES and prop in ATOMIC_PROPERTIES[element_symbol]:
    #     return ATOMIC_PROPERTIES[element_symbol][prop]

    elem = mendeleev.element(element_symbol)
    value = None
    if prop == 'group':
        value = elem.group_id
    elif prop == 'period':
        value = elem.period
    elif prop == 'mass':
        value = elem.atomic_weight
    elif prop == 'IP1':
        value = elem.ionenergies.get(1, 0)
    elif prop == 'IP2':
        value = elem.ionenergies.get(2, 0)
    elif prop == 'IP3':
        value = elem.ionenergies.get(3, 0)
    elif prop == 'ΔH_fusion':
        value = elem.fusion_heat
    elif prop == 'ΔH_vapor':
        value = elem.evaporation_heat
    elif prop == 'λ':
        value = elem.thermal_conductivity
    elif prop == 'V_molar':
        value = elem.atomic_volume
    elif prop == 'r_cov':
        value = elem.covalent_radius_cordero
    elif prop == 'χ':
        value = elem.electronegativity_pauling()
    elif prop == 'Z_eff':
        value = elem.zeff()
    elif prop == 'ΔH_atom':
        value = elem.heat_of_formation 
    elif prop == 'EA':
        value = elem.electron_affinity
    elif prop == 'α_p':
        value = elem.dipole_polarizability
    elif prop == 'r_vdW':
        value = elem.vdw_radius
    elif prop == 'C':
        value = elem.specific_heat_capacity
    elif prop == 'ρ':
        value = elem.density
    elif prop == 'η':
        value = elem.hardness()
    elif prop == 'Nv':
        value = elem.nvalence()    

    # 如果 mendeleev 返回有效值（非 None 且非 0），直接返回
    if value is not None and value != 0:
        return value

    # 回退到 ATOMIC_PROPERTIES
    if element_symbol in ATOMIC_PROPERTIES and prop in ATOMIC_PROPERTIES[element_symbol]:
        value = ATOMIC_PROPERTIES[element_symbol][prop]
        if value is not None:
            return value

    # 如果都找不到，返回 0 并打印警告
    print(f"Warning: Property '{prop}' not available for element '{element_symbol}'. Returning 0.")
    return 0 if prop != 'r_cov' else 1.0  # r_cov 特殊处理，避免除以 0

def compute_galvez_matrix(structure, adjacency_matrix, species, element_properties):
    n_atoms = len(structure)
    D = np.zeros((n_atoms, n_atoms))
    distances = np.zeros((n_atoms, n_atoms))
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = structure.get_distance(i, j)
            distances[i, j] = dist
            distances[j, i] = dist
    
    D[adjacency_matrix == 1] = 1 / (distances[adjacency_matrix == 1] ** 2)
    D[distances == 0] = 0  # 避免除以 0
    
    M = adjacency_matrix * D
    ##只计算单个电负性的T_E
    # T_E = 0
    # T_E_bond = 0
    # for i in range(n_atoms):
    #     for j in range(i + 1, n_atoms):
    #         q_i = element_properties[species[i]]['χ']
    #         q_j = element_properties[species[j]]['χ']
    #         T_E += abs(q_i - q_j) * M[i, j]
    #         if adjacency_matrix[i, j] == 1:
    #             T_E_bond += abs(q_i - q_j) * M[i, j]
    #应该不仅仅是电负性，所有属性应该都有这么两个数
    # print("Adjacency Matrix:")
    # print(adjacency_matrix)
    # print("\nGalvez Matrix M:")
    # print(M)
    #return M, T_E, T_E_bond
    # 初始化结果列表
    T_E_list = []
    T_E_bond_list = []
    
    # 获取 element_properties 中的所有属性名称
    # 假设所有元素的属性名称相同，取第一个元素的属性键
    if not element_properties or not species:
        return M, T_E_list, T_E_bond_list, []

    property_names = list(element_properties[species[0]].keys())
    # 针对每个属性计算 T_E 和 T_E_bond
    for prop in property_names:
        T_E = 0
        T_E_bond = 0
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                # 获取原子 i 和 j 的属性值
                q_i = element_properties[species[i]][prop]
                q_j = element_properties[species[j]][prop]
                # 计算属性差的绝对值与 M[i, j] 的乘积
                delta = abs(q_i - q_j) * M[i, j]
                # T_E：所有原子对的加权和
                T_E += delta
                # T_E_bond：仅对相连原子对的加权和
                if adjacency_matrix[i, j] == 1:
                    T_E_bond += delta
        # 将结果添加到列表
        T_E_list.append(T_E)
        T_E_bond_list.append(T_E_bond)
    return M, T_E_list, T_E_bond_list

def generate_plmf(structure, cutoff_radius=5.0, max_path_length=3, distance_tolerance=0.5):
    if not isinstance(structure, Structure):
        raise TypeError(f"Expected a pymatgen Structure object, but got {type(structure)}")

    # 1. 提取晶体整体属性（固定10维）
    volume_per_atom = structure.volume / len(structure)
    lattice = structure.lattice
    a, b, c = lattice.a, lattice.b, lattice.c
    alpha, beta, gamma = lattice.alpha, lattice.beta, lattice.gamma
    crystal_features = [volume_per_atom, a, b, c, alpha, beta, gamma, a/b, b/c, a/c]
    
    # 预先计算所有元素的属性
    props = ['group', 'period', 'mass', 'IP1', 'IP2', 'IP3', 'ΔH_fusion', 'ΔH_vapor', 'λ',
             'V_molar', 'r_cov', 'χ', 'Z_eff', 'ΔH_atom', 'EA', 'α_p', 'r_vdW', 'C', 'ρ', 'η', 'Nv']
    elements = set(site.species_string for site in structure)
    element_properties = {elem: {} for elem in elements}
    for elem in elements:
        for prop in props:
            element_properties[elem][prop] = get_element_property(elem, prop)
    print(element_properties)
    r_covs = {elem: element_properties[elem]['r_cov'] for elem in elements}
    
# 构建邻接矩阵
    n_atoms = len(structure)
    species = [site.species_string for site in structure]
    distances = np.zeros((n_atoms, n_atoms))
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = structure.get_distance(i, j)
            distances[i, j] = dist
            distances[j, i] = dist

# 获取共价半径
    r_cov_array = np.array([r_covs[spec] for spec in species])
    r_cov_sum = r_cov_array[:, None] + r_cov_array[None, :]  # 两两共价半径之和
    distance_threshold = r_cov_sum + 0.6  # 加上 0.5 Å 容差

# 使用 VoronoiNN 获取近邻，并施加距离阈值
    voronoi_nn = VoronoiNN(cutoff=cutoff_radius)
    adjacency_matrix = np.zeros((n_atoms, n_atoms))
    for i in range(n_atoms):
        neighbors = voronoi_nn.get_nn_info(structure, i)
        for n in neighbors:
            j = n['site_index']
            distance = distances[i, j]
            chi_i = element_properties[species[i]]['χ']
            chi_j = element_properties[species[j]]['χ']
            if distance <= distance_threshold[i, j]:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1
    print(adjacency_matrix)
    # 3. 计算Galvez矩阵和拓扑特征（固定2维）
    _, T_E, T_E_bond = compute_galvez_matrix(structure, adjacency_matrix, species, element_properties)
    topo_features = T_E + T_E_bond
    
    # 4. 生成片段
    fragments, neighbor_counts = generate_fragments(structure, adjacency_matrix, distances, max_path_length)
    # 5. 属性标注与特征提取
    combo_props = [
        ('C', 'λ'), ('V_molar', 'C'), ('χ', 'Z_eff'), ('IP2','ΔH_atom'),
        ('mass', 'ΔH_atom'), ('mass', 'C'),('period','Z_eff'),('IP1','ΔH_atom'),
        ('group','ΔH_atom'),('mass', 'IP3')
    ]
    
    # 路径片段统计量
    path_lengths = np.array([len(frag[1]) - 1 for frag in fragments['path']])  # 基于种类序列计算长度
    path_stats = []
    for prop in props:
        prop_values = []
        for _, frag_species in fragments['path']:
            values = [element_properties[atom][prop] for atom in frag_species]
            prop_values.extend(values)
        path_stats.append(np.array(prop_values))
    #路径长度
    path_features = []
    if path_lengths.size:
        path_features.extend([
            np.min(path_lengths), np.max(path_lengths), np.sum(path_lengths),
            np.mean(path_lengths), np.std(path_lengths)
        ])
    else:
        path_features.extend([0, 0, 0, 0, 0])
    #原子属性
    for prop_values in path_stats:
        if prop_values.size:
            path_features.extend([
                np.min(prop_values), np.max(prop_values), np.sum(prop_values),
                np.mean(prop_values), np.std(prop_values)
            ])
        else:
            path_features.extend([0, 0, 0, 0, 0])
    #原子属性组合
    for prop1, prop2 in combo_props:
        values1 = []
        values2 = []
        for _, frag_species in fragments['path']:  # 解包
            for atom in frag_species:
                val1 = element_properties[atom][prop1]
                val2 = element_properties[atom][prop2]
                values1.append(val1 / val2 if val2 != 0 else 0)
                values2.append(val1 * val2)
        values1 = np.array(values1)
        values2 = np.array(values2)
        if values1.size and values2.size:
            path_features.extend([
                np.min(values1), np.max(values1), np.sum(values1),
                np.mean(values1), np.std(values1),
                np.min(values2), np.max(values2), np.sum(values2),
                np.mean(values2), np.std(values2)
            ])
        else:
            path_features.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    # 环形片段统计量
    neighbor_counts = np.array([len(frag['neighbors']) for frag in fragments['circular']])
    avg_dists = np.array([np.mean([d for _, d in frag['neighbors']]) if frag['neighbors'] else 0
                      for frag in fragments['circular']])
    
    circular_stats = []
    for prop in props:
        prop_values = []
        for frag in fragments['circular']:
            center_species = frag['center']
            values = [element_properties[n[0]][prop] for n in frag['neighbors']] + [element_properties[center_species][prop]]
            prop_values.extend(values)
        circular_stats.append(np.array(prop_values))
    #环形片段长度
    circular_features = []
    # neighbor_counts 统计量
    if neighbor_counts.size:
        circular_features.extend([
            np.min(neighbor_counts), np.max(neighbor_counts), np.sum(neighbor_counts),
            np.mean(neighbor_counts), np.std(neighbor_counts)
        ])
    else:
        circular_features.extend([0, 0, 0, 0, 0])
    print(f"Neighbor counts: min={np.min(neighbor_counts)}, max={np.max(neighbor_counts)}, sum={np.sum(neighbor_counts)}")
    # avg_dists 统计量
    if avg_dists.size:
        circular_features.extend([
            np.min(avg_dists), np.max(avg_dists), np.sum(avg_dists),
            np.mean(avg_dists), np.std(avg_dists)
        ])
    else:
        circular_features.extend([0, 0, 0, 0, 0])
    
    # props 统计量
    for prop_values in circular_stats:
        if prop_values.size:
            circular_features.extend([
                np.min(prop_values), np.max(prop_values), np.sum(prop_values),
                np.mean(prop_values), np.std(prop_values)
            ])
        else:
            circular_features.extend([0, 0, 0, 0, 0])
    
    # combo_props 统计量
    for prop1, prop2 in combo_props:
        values1 = []
        values2 = []
        for frag in fragments['circular']:
            center_species = frag['center']
            atoms = [(center_species, 0)] + frag['neighbors']
            for atom, _ in atoms:
                val1 = element_properties[atom][prop1]
                val2 = element_properties[atom][prop2]
                values1.append(val1 / val2 if val2 != 0 else 0)
                values2.append(val1 * val2)
        values1 = np.array(values1)
        values2 = np.array(values2)
        if values1.size and values2.size:
            circular_features.extend([
                np.min(values1), np.max(values1), np.sum(values1),
                np.mean(values1), np.std(values1),
                np.min(values2), np.max(values2), np.sum(values2),
                np.mean(values2), np.std(values2)
            ])
        else:
            circular_features.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    # 6. 拼接初始特征向量
    initial_descriptor = np.concatenate([path_features, circular_features, crystal_features, topo_features])
    print(f"Path features length: {len(path_features)}, Circular features length: {len(circular_features)}")
    return initial_descriptor
import os
import pandas as pd
from pymatgen.core import Structure
import numpy as np

# Assuming generate_plmf is already defined as in your previous code

def process_cif_files(folder_path, output_csv, cutoff_radius=5.0, max_path_length=3, distance_tolerance=0.5):
    # Dictionary to store results
    results = {}
    
    # Get all CIF files in the folder
    cif_files = [f for f in os.listdir(folder_path) if f.endswith('.cif')]
    
    # Process each CIF file
    for cif_file in cif_files:
        try:
            # Construct full path
            full_path = os.path.join(folder_path, cif_file)
            
            # Load structure and make supercell
            structure = Structure.from_file(full_path)
            structure.make_supercell([2, 2, 2])  # Expand to 54 atoms
            
            # Generate PLMF descriptor
            descriptor = generate_plmf(structure, 
                                     cutoff_radius=cutoff_radius, 
                                     max_path_length=max_path_length, 
                                     distance_tolerance=distance_tolerance)
            
            # Use filename (without extension) as index
            filename = os.path.splitext(cif_file)[0]
            results[filename] = descriptor
            
            print(f"Successfully processed: {cif_file}")
            
        except Exception as e:
            print(f"Error processing {cif_file}: {str(e)}")
            continue
    
    # Create DataFrame from results
    # Convert dictionary to DataFrame where columns are feature indices
    df = pd.DataFrame.from_dict(results, orient='index')
    
    # Generate column names (assuming all descriptors have same length)
    if len(results) > 0:
        n_features = len(next(iter(results.values())))
        df.columns = [f'feature_{i}' for i in range(n_features)]
    
    # Save to CSV
    df.to_csv(output_csv)
    print(f"Results saved to {output_csv}")
    
    return df

# Example usage
folder_path = "D:/毕设课题学习/毕设课题学习/毕设论文代码和数据集/Zn10Pd3/"#金属间化合物结构文件/删除样本(8个)/"
output_csv = "D:/毕设课题学习/毕设课题学习/毕设论文代码和数据集/Zn10Pd3.csv"

# Run the processing
descriptor_df = process_cif_files(folder_path, output_csv, 
                                cutoff_radius=5.0, 
                                max_path_length=3, 
                                distance_tolerance=0.5)

# Optional: Print some info about the results
print(f"Processed {len(descriptor_df)} structures")
print("First few rows of the DataFrame:")
print(descriptor_df.head())
# # 测试代码
# from pymatgen.core import Lattice, Structure
# import numpy as np

# cif_files = "ID_1999993.cif"
# structure = Structure.from_file(cif_files)
# structure.make_supercell([2, 2, 2])  # 扩展到 54 原子
# # 运行代码
# descriptor = generate_plmf(structure, cutoff_radius=5.0, max_path_length=3, distance_tolerance=0.5)
# print("Descriptor shape:", descriptor.shape)
# print("Descriptor:", descriptor)
# 示例使用
#cif_files = ["mp-622209.cif","mp-1057.cif"]
#cif_files = "mp-622209.cif"
# X = []
# for cif in cif_files:
#     try:
#         structure = Structure.from_file(cif)
#     except Exception as e:
#         print(f"Failed to load CIF file {cif}: {e}")
#         continue
#     plmf = generate_plmf(structure)
#     X.append(plmf)
# if not X:
#     raise ValueError("No valid structures loaded from CIF files.")
#X=generate_plmf(Structure.from_file(cif_files))
#print(X)
#print(len(X))
#X = np.array(X)
# X_filtered, _, _ = feature_selection(X, variance_threshold=0.001, correlation_threshold=0.95)
# print(X_filtered) 



