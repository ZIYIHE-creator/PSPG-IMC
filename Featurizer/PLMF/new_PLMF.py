import json
import os
import pandas as pd
from pymatgen.core import Structure, Element
import mendeleev
import numpy as np
from pymatgen.analysis.local_env import VoronoiNN, CrystalNN
import networkx as nx

# Unified properties list
props = [
    'group', 'period', 'mass', 'IP1', 'IP2', 'IP3', 'ΔH_fusion', 'ΔH_vapor', 'λ',
    'V_molar', 'r_cov', 'χ_pualing', 'Z_eff', 'ΔH_atom', 'Electron affinity', 'α_p',
    'r_vdW', 'C', 'ρ', 'η', 'Nv', 'Atomic radius', 'Melting point', 'Boiling point',
    'χ_MB', 'r_s', 'r_p', 'r_d'
]

# ATOMIC_PROPERTIES dictionary (same as provided in the first code)
ATOMIC_PROPERTIES = {
    'Al': {
        'group': 13, 'period': 3, 'mass': 26.98, 'IP1': 5.99, 'IP2': 18.83, 'IP3': 28.45,
        'ΔH_fusion': 10.71, 'ΔH_vapor': 294.0, 'λ': 237.0, 'V_molar': 10.0, 'r_cov': 1.21,
        'χ_pualing': 1.61, 'Z_eff': 3.0, 'ΔH_atom': 330.0, 'Electron affinity': 0.43, 'α_p': 6.8, 'r_vdW': 1.84,
        'C': 0.897, 'ρ': 2.70
    },
    'Ni': {
        'group': 10, 'period': 4, 'mass': 58.69, 'IP1': 7.64, 'IP2': 18.17, 'IP3': 35.19,
        'ΔH_fusion': 17.48, 'ΔH_vapor': 378.0, 'λ': 90.9, 'V_molar': 6.59, 'r_cov': 1.24,
        'χ_pualing': 1.91, 'Z_eff': 4.0, 'ΔH_atom': 430.0, 'Electron affinity': 1.16, 'α_p': 6.4, 'r_vdW': 1.63,
        'C': 0.444, 'ρ': 8.90
    },
    'Ti': {
        'group': 4, 'period': 4, 'mass': 47.87, 'IP1': 6.83, 'IP2': 13.58, 'IP3': 27.49,
        'ΔH_fusion': 14.15, 'ΔH_vapor': 425.0, 'λ': 21.9, 'V_molar': 10.64, 'r_cov': 1.60,
        'χ_pualing': 1.54, 'Z_eff': 3.0, 'ΔH_atom': 473.0, 'Electron affinity': 0.08, 'α_p': 8.0, 'r_vdW': 2.15,
        'C': 0.523, 'ρ': 4.51
    },
    'Dy':{
        'group':3,'ΔH_fusion':11.06,'ΔH_vapor':280
        },
    'Er':{
        'group':3,'ΔH_fusion':19.90,'ΔH_vapor':280
        },
    'Eu':{
        'group':3,'ΔH_fusion':9.21,'ΔH_vapor':176,'χ_pualing':1.2
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
        'group':3,'ΔH_fusion':10.15,'χ_pualing':1.2
        },
    'Ho':{
        'group':3,'ΔH_fusion':16.8,'λ':16.2,'EA': 0.0, 'η': 3.01075
        },
    'Tm':{
        'group':3,'ΔH_fusion':16.84,'λ': 17.0
        },
    'Yb':{
        'group':3,'λ': 38.5, 'χ_pualing': 1.1
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
def load_pseudo_radii(xlsx_file):
    """Load pseudopotential radii (r_s, r_p, r_d) from an XLSX file."""
    try:
        df = pd.read_excel(xlsx_file)
        required_columns = ['Atom', 'r_s', 'r_p', 'r_d']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"XLSX file {xlsx_file} must contain columns: {required_columns}")
        pseudo_radii = {row['Atom']: {'r_s': row['r_s'], 'r_p': row['r_p'], 'r_d': row['r_d']} for _, row in df.iterrows()}
        return pseudo_radii
    except Exception as e:
        print(f"Error reading XLSX file {xlsx_file}: {str(e)}")
        return {}

def get_element_property(element_symbol, prop, pseudo_radii=None, json_data=None):
    """Get a specified property for an element, including pseudopotential radii and JSON data."""
    # Check JSON data for specific properties
    if json_data and prop in ['Ionic radii', 'Shannon radii', 'Melting point', 'Boiling point', 'Electron affinity']:
        if element_symbol in json_data:
            element_data = json_data[element_symbol]
            if prop == 'Electron affinity':
                value = element_data.get('Electron affinity', None)
                if value is not None:
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        print(f"Warning: Invalid Electron affinity value '{value}' for element '{element_symbol}' in JSON data.")
                        return None
            elif prop == 'Shannon radii':
                return element_data.get(prop, {})
            else:
                return element_data.get(prop, None)

    # Handle pseudopotential radii
    if prop in ['r_s', 'r_p', 'r_d']:
        if pseudo_radii and element_symbol in pseudo_radii:
            return pseudo_radii[element_symbol].get(prop, None)
        return None

    # Use mendeleev for other properties
    try:
        elem = mendeleev.element(element_symbol)
        value = None
        if prop == 'group':
            value = elem.group_id
        elif prop == 'period':
            value = elem.period
        elif prop == 'Atomic radius':
            value = elem.atomic_radius_rahm
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
        elif prop == 'χ_pualing':
            value = elem.electronegativity_pauling()
        elif prop == 'Z_eff':
            value = elem.zeff()
        elif prop == 'ΔH_atom':
            value = elem.heat_of_formation
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
        elif prop == 'χ_MB':
            ip1 = elem.ionenergies.get(1, 0)
            if json_data and element_symbol in json_data:
                ea = json_data[element_symbol].get('Electron affinity', None)
                if ea is not None:
                    try:
                        ea = float(ea)
                    except (ValueError, TypeError):
                        print(f"Warning: Invalid Electron affinity value '{ea}' for element '{element_symbol}' in JSON data.")
                        ea = None
            else:
                ea = None
            if ea is None:
                ea = elem.electron_affinity
            if ip1 is not None and ea is not None and ip1 != 0:
                value = (ip1 + ea) / 2
            else:
                ip1 = ATOMIC_PROPERTIES.get(element_symbol, {}).get('IP1', 0)
                ea = ATOMIC_PROPERTIES.get(element_symbol, {}).get('Electron affinity', 0)
                if ip1 != 0 and ea is not None:
                    value = (ip1 + ea) / 2

        if value is not None and value != 0:
            return value
    except Exception as e:
        print(f"Warning: Error retrieving property '{prop}' for element '{element_symbol}' from mendeleev: {str(e)}")

    # Fallback to ATOMIC_PROPERTIES
    if element_symbol in ATOMIC_PROPERTIES and prop in ATOMIC_PROPERTIES[element_symbol]:
        value = ATOMIC_PROPERTIES[element_symbol][prop]
        if value is not None:
            return value

    print(f"Warning: Property '{prop}' not available for element '{element_symbol}'. Returning None.")
    return None



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
    
    M = np.dot(adjacency_matrix, D)

    T_E_list = []
    T_E_bond_list = []
    
    # Check if element_properties or species is empty
    if not element_properties or not species:
        print("Warning: Empty element_properties or species list, returning empty T_E and T_E_bond lists")
        return M, T_E_list, T_E_bond_list

    # Get property names (assume all elements have the same property keys)
    property_names = list(element_properties[species[0]].keys())
    
    # Calculate T_E and T_E_bond for each property
    for prop in property_names:
        # Check if the property is available for all atoms
        all_valid = True
        for i in range(n_atoms):
            if element_properties[species[i]][prop] is None:
                all_valid = False
                break
        
        if not all_valid:
            print(f"Warning: Property {prop} is missing for some elements, setting T_E and T_E_bond to 0")
            T_E_list.append(0)
            T_E_bond_list.append(0)
            continue
        
        T_E = 0
        T_E_bond = 0
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                # Get property values for atoms i and j
                q_i = element_properties[species[i]][prop]
                q_j = element_properties[species[j]][prop]
                # Calculate property difference multiplied by M[i,j] - M[j,i]
                delta = abs(q_i - q_j) * abs(M[i, j] - M[j, i])
                # T_E: Sum over all atom pairs
                T_E += delta
                # T_E_bond: Sum only over bonded atom pairs
                if adjacency_matrix[i, j] == 1:
                    T_E_bond += delta
        # Append results to lists
        T_E_list.append(T_E)
        T_E_bond_list.append(T_E_bond)
    
    return M, T_E_list, T_E_bond_list



def generate_plmf(structure, cutoff_radius=5.0, max_path_length=3, distance_tolerance=0.5, props=None):
    if not isinstance(structure, Structure):
        raise TypeError(f"Expected a pymatgen Structure object, but got {type(structure)}")

    # Initialize column names list
    column_names = []

    # 1. Crystal Features (10 features)
    volume_per_atom = structure.volume / len(structure)
    lattice = structure.lattice
    a, b, c = lattice.a, lattice.b, lattice.c
    alpha, beta, gamma = lattice.alpha, lattice.beta, lattice.gamma
    crystal_features = [volume_per_atom, a, b, c, alpha, beta, gamma, a/b, b/c, a/c]
    crystal_names = [
        'volume_per_atom', 'lattice_a', 'lattice_b', 'lattice_c',
        'lattice_alpha', 'lattice_beta', 'lattice_gamma',
        'lattice_a_b_ratio', 'lattice_b_c_ratio', 'lattice_a_c_ratio'
    ]


    # File paths (replace with actual paths)
    json_file_path = 'D:/毕设课题学习/毕设课题学习/毕设论文代码和数据集/数据集/property_of_periodic_table_from_pymatgen.json'
    xlsx_file = "D:/毕设课题学习/毕设课题学习/毕设论文代码和数据集/数据集/zunger pseudopotentials radii.xlsx"
    
    # Load JSON data
    json_data = {}
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
        print(f"Successfully loaded JSON file: {json_file_path}")
    except FileNotFoundError:
        print(f"Warning: JSON file {json_file_path} not found. Skipping JSON-based property extraction.")
    except json.JSONDecodeError:
        print("Warning: Invalid JSON format. Skipping JSON-based property extraction.")
    except Exception as e:
        print(f"Warning: Error loading JSON file: {str(e)}. Skipping JSON-based property extraction.")

    # Load pseudopotential radii
    pseudo_radii = load_pseudo_radii(xlsx_file)

    # Load structure and extract elements
    elements = set(site.species_string for site in structure)
    element_properties = {elem: {} for elem in elements}

    # Retrieve properties for each element
    for elem in elements:
        for prop in props:
            value = get_element_property(elem, prop, pseudo_radii, json_data)
            if prop == 'Shannon radii' and isinstance(value, dict):
                element_properties[elem][prop] = value
            else:
                element_properties[elem][prop] = value if value is not None else None

    # Extract r_cov values
    r_covs = {elem: element_properties[elem]['r_cov'] if element_properties[elem]['r_cov'] is not None else None for elem in elements}

    # Build adjacency matrix
    n_atoms = len(structure)
    species = [site.species_string for site in structure]
    distances = np.zeros((n_atoms, n_atoms))
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = structure.get_distance(i, j)
            distances[i, j] = dist
            distances[j, i] = dist

    r_cov_array = np.array([r_covs[spec] if r_covs[spec] is not None else 0 for spec in species])
    distance_threshold = r_cov_array[:, None] + r_cov_array[None, :] + 0.6

    voronoi_nn = VoronoiNN(cutoff=cutoff_radius)
    adjacency_matrix = np.zeros((n_atoms, n_atoms))
    for i in range(n_atoms):
        neighbors = voronoi_nn.get_nn_info(structure, i)
        for n in neighbors:
            j = n['site_index']
            distance = distances[i, j]
            if distance <= distance_threshold[i, j]:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1

    # Compute Galvez matrix and topological features
    _, T_E, T_E_bond = compute_galvez_matrix(structure, adjacency_matrix, species, element_properties)
    topo_features = T_E + T_E_bond


    # Generate fragments
    fragments, neighbor_counts = generate_fragments(structure, adjacency_matrix, distances, max_path_length)

    # Property combinations
    combo_props = [
        ('C', 'λ'), ('V_molar', 'C'), ('χ_pualing', 'Z_eff'), ('IP2', 'ΔH_atom'),
        ('mass', 'ΔH_atom'), ('mass', 'C'), ('period', 'Z_eff'), ('IP1', 'ΔH_atom'),
        ('group', 'ΔH_atom'), ('mass', 'IP3')
    ]

    # Path Features
    path_lengths = np.array([len(frag[1]) - 1 for frag in fragments['path']])
    path_features = []
    path_names = []

    # Path length statistics
    if path_lengths.size:
        path_features.extend([
            np.min(path_lengths), np.max(path_lengths), np.sum(path_lengths),
            np.mean(path_lengths), np.std(path_lengths)
        ])
        path_names.extend([
            'path_length_min', 'path_length_max', 'path_length_sum',
            'path_length_mean', 'path_length_std'
        ])
    else:
        path_features.extend([0, 0, 0, 0, 0])
        path_names.extend([
            'path_length_min', 'path_length_max', 'path_length_sum',
            'path_length_mean', 'path_length_std'
        ])

    # Path property statistics
    for prop in props:
        prop_values = []
        for _, frag_species in fragments['path']:
            values = [element_properties[atom][prop] for atom in frag_species if element_properties[atom][prop] is not None]
            prop_values.extend(values)
        prop_values = np.array(prop_values)
        if prop_values.size:
            path_features.extend([
                np.min(prop_values), np.max(prop_values), np.sum(prop_values),
                np.mean(prop_values), np.std(prop_values)
            ])
            path_names.extend([
                f'path_{prop}_min', f'path_{prop}_max', f'path_{prop}_sum',
                f'path_{prop}_mean', f'path_{prop}_std'
            ])
        else:
            print(f"Warning: No valid values for property {prop} in path fragments, setting features to 0")
            path_features.extend([0, 0, 0, 0, 0])
            path_names.extend([
                f'path_{prop}_min', f'path_{prop}_max', f'path_{prop}_sum',
                f'path_{prop}_mean', f'path_{prop}_std'
            ])

    # Path property combination statistics
    for prop1, prop2 in combo_props:
        values1 = []
        values2 = []
        for _, frag_species in fragments['path']:
            for atom in frag_species:
                val1 = element_properties[atom][prop1]
                val2 = element_properties[atom][prop2]
                if val1 is not None and val2 is not None and val2 != 0:
                    values1.append(val1 / val2)
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
            path_names.extend([
                f'path_{prop1}/{prop2}_min', f'path_{prop1}/{prop2}_max',
                f'path_{prop1}/{prop2}_sum', f'path_{prop1}/{prop2}_mean',
                f'path_{prop1}/{prop2}_std',
                f'path_{prop1}*{prop2}_min', f'path_{prop1}*{prop2}_max',
                f'path_{prop1}*{prop2}_sum', f'path_{prop1}*{prop2}_mean',
                f'path_{prop1}*{prop2}_std'
            ])
        else:
            print(f"Warning: No valid values for property pair ({prop1}, {prop2}) in path fragments, setting features to 0")
            path_features.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            path_names.extend([
                f'path_{prop1}/{prop2}_min', f'path_{prop1}/{prop2}_max',
                f'path_{prop1}/{prop2}_sum', f'path_{prop1}/{prop2}_mean',
                f'path_{prop1}/{prop2}_std',
                f'path_{prop1}*{prop2}_min', f'path_{prop1}*{prop2}_max',
                f'path_{prop1}*{prop2}_sum', f'path_{prop1}*{prop2}_mean',
                f'path_{prop1}*{prop2}_std'
            ])
    column_names.extend(path_names)

    # Circular Features
    neighbor_counts = np.array([len(frag['neighbors']) for frag in fragments['circular']])
    avg_dists = np.array([np.mean([d for _, d in frag['neighbors']]) if frag['neighbors'] else 0
                          for frag in fragments['circular']])
    circular_features = []
    circular_names = []

    # Neighbor count statistics
    if neighbor_counts.size:
        circular_features.extend([
            np.min(neighbor_counts), np.max(neighbor_counts), np.sum(neighbor_counts),
            np.mean(neighbor_counts), np.std(neighbor_counts)
        ])
        circular_names.extend([
            'circular_neighbor_count_min', 'circular_neighbor_count_max',
            'circular_neighbor_count_sum', 'circular_neighbor_count_mean',
            'circular_neighbor_count_std'
        ])
    else:
        circular_features.extend([0, 0, 0, 0, 0])
        circular_names.extend([
            'circular_neighbor_count_min', 'circular_neighbor_count_max',
            'circular_neighbor_count_sum', 'circular_neighbor_count_mean',
            'circular_neighbor_count_std'
        ])

    # Average distance statistics
    if avg_dists.size:
        circular_features.extend([
            np.min(avg_dists), np.max(avg_dists), np.sum(avg_dists),
            np.mean(avg_dists), np.std(avg_dists)
        ])
        circular_names.extend([
            'circular_avg_distance_min', 'circular_avg_distance_max',
            'circular_avg_distance_sum', 'circular_avg_distance_mean',
            'circular_avg_distance_std'
        ])
    else:
        circular_features.extend([0, 0, 0, 0, 0])
        circular_names.extend([
            'circular_avg_distance_min', 'circular_avg_distance_max',
            'circular_avg_distance_sum', 'circular_avg_distance_mean',
            'circular_avg_distance_std'
        ])

    # Circular property statistics
    for prop in props:
        prop_values = []
        for frag in fragments['circular']:
            center_species = frag['center']
            values = [element_properties[n[0]][prop] for n in frag['neighbors'] if element_properties[n[0]][prop] is not None]
            if element_properties[center_species][prop] is not None:
                values.append(element_properties[center_species][prop])
            prop_values.extend(values)
        prop_values = np.array(prop_values)
        if prop_values.size:
            circular_features.extend([
                np.min(prop_values), np.max(prop_values), np.sum(prop_values),
                np.mean(prop_values), np.std(prop_values)
            ])
            circular_names.extend([
                f'circular_{prop}_min', f'circular_{prop}_max', f'circular_{prop}_sum',
                f'circular_{prop}_mean', f'circular_{prop}_std'
            ])
        else:
            print(f"Warning: No valid values for property {prop} in circular fragments, setting features to 0")
            circular_features.extend([0, 0, 0, 0, 0])
            circular_names.extend([
                f'circular_{prop}_min', f'circular_{prop}_max', f'circular_{prop}_sum',
                f'circular_{prop}_mean', f'circular_{prop}_std'
            ])

    # Circular property combination statistics
    for prop1, prop2 in combo_props:
        values1 = []
        values2 = []
        for frag in fragments['circular']:
            center_species = frag['center']
            atoms = [(center_species, 0)] + frag['neighbors']
            for atom, _ in atoms:
                val1 = element_properties[atom][prop1]
                val2 = element_properties[atom][prop2]
                if val1 is not None and val2 is not None and val2 != 0:
                    values1.append(val1 / val2)
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
            circular_names.extend([
                f'circular_{prop1}/{prop2}_min', f'circular_{prop1}/{prop2}_max',
                f'circular_{prop1}/{prop2}_sum', f'circular_{prop1}/{prop2}_mean',
                f'circular_{prop1}/{prop2}_std',
                f'circular_{prop1}*{prop2}_min', f'circular_{prop1}*{prop2}_max',
                f'circular_{prop1}*{prop2}_sum', f'circular_{prop1}*{prop2}_mean',
                f'circular_{prop1}*{prop2}_std'
            ])
        else:
            print(f"Warning: No valid values for property pair ({prop1}, {prop2}) in circular fragments, setting features to 0")
            circular_features.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            circular_names.extend([
                f'circular_{prop1}/{prop2}_min', f'circular_{prop1}/{prop2}_max',
                f'circular_{prop1}/{prop2}_sum', f'circular_{prop1}/{prop2}_mean',
                f'circular_{prop1}/{prop2}_std',
                f'circular_{prop1}*{prop2}_min', f'circular_{prop1}*{prop2}_max',
                f'circular_{prop1}*{prop2}_sum', f'circular_{prop1}*{prop2}_mean',
                f'circular_{prop1}*{prop2}_std'
            ])
    column_names.extend(circular_names)
    
    column_names.extend(crystal_names)
    topo_names = []
    for prop in props:
        topo_names.extend([f'T_E_{prop}', f'T_E_bond_{prop}'])
    column_names.extend(topo_names)
    
    # Concatenate all features
    initial_descriptor = np.concatenate([path_features, circular_features, crystal_features, topo_features])

    print(f"Path features length: {len(path_features)}, Circular features length: {len(circular_features)}, "
          f"Crystal features length: {len(crystal_features)}, Topo features length: {len(topo_features)}")
    print(f"Total descriptor length: {len(initial_descriptor)}, Total column names: {len(column_names)}")

    return initial_descriptor, column_names

# def process_cif_files(folder_path, output_csv, cutoff_radius=5.0, max_path_length=3, distance_tolerance=0.5):
#     results = {}
#     column_names = None

#     cif_files = [f for f in os.listdir(folder_path) if f.endswith('.cif')]

#     for cif_file in cif_files:
#         try:
#             full_path = os.path.join(folder_path, cif_file)
#             structure = Structure.from_file(full_path)
#             structure.make_supercell([2, 2, 2])  # Expand to 54 atoms

#             descriptor, col_names = generate_plmf(
#                 structure, cutoff_radius, max_path_length, distance_tolerance, props=props
#             )
            
#             # Ensure column names are consistent across all structures
#             if column_names is None:
#                 column_names = col_names
#             elif col_names != column_names:
#                 print(f"Warning: Inconsistent column names for {cif_file}. Expected {len(column_names)} names, got {len(col_names)}")
#                 continue

#             filename = os.path.splitext(cif_file)[0]
#             results[filename] = descriptor

#             print(f"Successfully processed: {cif_file}")

#         except Exception as e:
#             print(f"Error processing {cif_file}: {str(e)}")
#             continue

#     # Create DataFrame with meaningful column names
#     df = pd.DataFrame.from_dict(results, orient='index', columns=column_names)

#     # Save to CSV
#     df.to_csv(output_csv)
#     print(f"Results saved to {output_csv}")

#     return df

#处理离子化合物
import os
import pandas as pd
from tqdm import tqdm
from pymatgen.core import Structure, Element
from pymatgen.core.periodic_table import Species   # ← 这里导入 Species（带电荷的）

def process_cif_files(folder_path, output_csv, cutoff_radius=5.0, max_path_length=3, distance_tolerance=0.5):
    results = {}
    column_names = None
    cif_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.cif')]
    
    for cif_file in tqdm(cif_files):
        try:
            full_path = os.path.join(folder_path, cif_file)
            structure = Structure.from_file(full_path)
            
            # === 关键修复：将带电 Species 转为中性 Element ===
            species_map = {}
            for sp in structure.types_of_specie:  # types_of_specie 返回 Element 或 Species
                if isinstance(sp, Species):       # ← 改成 Species（最新版类名）
                    neutral_element = sp.element  # Species 有 .element 属性，返回 Element
                    species_map[sp] = neutral_element
                    print(f"[{cif_file}] 已将离子 {sp} 转换为中性原子 {neutral_element.symbol}")
                else:
                    # 已经是 Element 或其他，无需转换
                    species_map[sp] = sp
            
            # 如果有任何离子需要转换，就应用替换
            if any(isinstance(k, Species) for k in species_map):
                structure = structure.replace_species(species_map)
                print(f"[{cif_file}] 完成离子剥离，现在所有物种均为中性元素")
            
            # 继续你的后续处理
            structure.make_supercell([2, 2, 2])  # 扩展超胞
            
            descriptor, col_names = generate_plmf(
                structure, cutoff_radius, max_path_length, distance_tolerance, props=props
            )
            
            if column_names is None:
                column_names = col_names
            elif col_names != column_names:
                print(f"警告: {cif_file} 列名不一致，已跳过")
                continue
            
            filename = os.path.splitext(cif_file)[0]
            results[filename] = descriptor
            print(f"成功处理: {cif_file}")
        
        except Exception as e:
            print(f"处理 {cif_file} 时出错: {str(e)}")
            continue
    
    # 保存结果
    if results:
        df = pd.DataFrame.from_dict(results, orient='index', columns=column_names)
        df.to_csv(output_csv, index_label='filename')
        print(f"结果保存到 {output_csv}")
        return df
    else:
        print("没有成功处理任何文件")
        return None


# Example usage
folder_path = "D:/博士生一年级/MP中没有理论计算结果的结构来自ICSD/structures/"#"D:/毕设课题学习/毕设课题学习/毕设论文代码和数据集/数据集/金属间化合物结构文件/Material Project(546个)/CIF/"
output_csv = "D:/博士生一年级/MP中没有理论计算结果的结构来自ICSD/new_PLMF.csv"#"D:/毕设课题学习/毕设课题学习/毕设论文代码和数据集/数据集/new_new4 PLMF for MP.csv"

#folder_path = "D:/毕设课题学习/毕设课题学习/毕设论文代码和数据集/数据集/模型准确性验证/测试结构/CIF/"#金属间化合物结构文件/OQMD(75个)/CIF/"
#output_csv = "D:/毕设课题学习/毕设课题学习/毕设论文代码和数据集/数据集/模型准确性验证/new PLMF for 9 test IMCs.csv"

descriptor_df = process_cif_files(
    folder_path, output_csv, cutoff_radius=5.0, max_path_length=3, distance_tolerance=0.5
)

print(f"Processed {len(descriptor_df)} structures")
print("First few rows of the DataFrame:")
print(descriptor_df.head())