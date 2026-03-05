import pandas as pd
import numpy as np
from pymatgen.core import Structure, Composition
from pymatgen.core.periodic_table import Element
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.structure import (DensityFeatures, 
                                          GlobalSymmetryFeatures,
                                          ChemicalOrdering)
from pymatgen.analysis.local_env import CrystalNN
import os
import warnings
import re

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen.core.structure")
warnings.filterwarnings("ignore", category=UserWarning, module="matminer.featurizers.composition")
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen.analysis.local_env")

# 定义价电子数（仅包括金属元素，非金属默认0）
valence_electrons = {
    'Li': 1, 'Na': 1, 'K': 1, 'Rb': 1, 'Cs': 1,
    'Be': 2, 'Mg': 2, 'Ca': 2, 'Sr': 2, 'Ba': 2,
    'Sc': 3, 'Ti': 4, 'V': 5, 'Cr': 6, 'Mn': 7, 'Fe': 8, 'Co': 9, 'Ni': 10, 'Cu': 11, 'Zn': 12,
    'Y': 3, 'Zr': 4, 'Nb': 5, 'Mo': 6, 'Tc': 7, 'Ru': 8, 'Rh': 9, 'Pd': 10, 'Ag': 11, 'Cd': 12,
    'Hf': 4, 'Ta': 5, 'W': 6, 'Re': 7, 'Os': 8, 'Ir': 9, 'Pt': 10, 'Au': 11, 'Hg': 12,
    'La': 3, 'Ce': 4, 'Pr': 5, 'Nd': 6, 'Pm': 7, 'Sm': 8, 'Eu': 9, 'Gd': 10, 'Tb': 11, 'Dy': 12,
    'Ac': 3, 'Th': 4, 'Pa': 5, 'U': 6,
    'Al': 3, 'Ga': 3, 'In': 3, 'Tl': 3, 'Sn': 4, 'Pb': 4, 'Bi': 5
}

# 定义所有金属元素的主量子数
principal_quantum_numbers = {
    'Li': 2, 'Na': 3, 'K': 4, 'Rb': 5, 'Cs': 6,
    'Be': 2, 'Mg': 3, 'Ca': 4, 'Sr': 5, 'Ba': 6,
    'Sc': 4, 'Ti': 4, 'V': 4, 'Cr': 4, 'Mn': 4, 'Fe': 4, 'Co': 4, 'Ni': 4, 'Cu': 4, 'Zn': 4,
    'Y': 5, 'Zr': 5, 'Nb': 5, 'Mo': 5, 'Tc': 5, 'Ru': 5, 'Rh': 5, 'Pd': 5, 'Ag': 5, 'Cd': 5,
    'Hf': 6, 'Ta': 6, 'W': 6, 'Re': 6, 'Os': 6, 'Ir': 6, 'Pt': 6, 'Au': 6, 'Hg': 6,
    'La': 6, 'Ce': 6, 'Pr': 6, 'Nd': 6, 'Pm': 6, 'Sm': 6, 'Eu': 6, 'Gd': 6, 'Tb': 6, 'Dy': 6,
    'Ac': 7, 'Th': 7, 'Pa': 7, 'U': 7,
    'Al': 3, 'Ga': 4, 'In': 5, 'Tl': 6, 'Sn': 5, 'Pb': 6, 'Bi': 6
}

def process_cif_files(cif_directory, excel_file, output_csv="D:/博士生一年级/MP中没有理论计算结果的结构来自ICSD/simple descriptors.csv"):
    features_data = []
    
    # 初始化特征提取器
    elem_prop = ElementProperty.from_preset("magpie", impute_nan=True)
    density_feat = DensityFeatures()
    symmetry_feat = GlobalSymmetryFeatures()
    chem_order = ChemicalOrdering()
    cnn = CrystalNN()
    
    # 检查 CIF 目录
    print(f"Checking CIF directory: {os.path.abspath(cif_directory)}")
    cif_files = [f for f in os.listdir(cif_directory) if f.endswith('.cif')]
    print(f"CIF files found: {cif_files}")
    if not cif_files:
        raise ValueError("No .cif files found in the specified directory.")
    
    for cif_file in cif_files:
        try:
            filename_base = cif_file.replace('.cif', '')
            full_path = os.path.join(cif_directory, cif_file)
            print(f"Attempting to process: {full_path}")
            structure = Structure.from_file(full_path)
            composition = structure.composition
            formula = str(composition.reduced_formula)
            
            # 提取 COD_ID 或 ICSD_ID（假设为文件名末尾的数字）
            cod_id_match = re.search(r'\d+$', filename_base)
            cod_id = cod_id_match.group(0) if cod_id_match else filename_base
            
            if '_' in filename_base:
                formula_from_file, _ = filename_base.rsplit('_', 1)
                if formula != formula_from_file and not formula_from_file.startswith('EntryWithCollCode'):
                    print(f"Warning: Formula mismatch in {cif_file}. File: {formula_from_file}, Structure: {formula}")
            else:
                print(f"Note: No explicit formula in filename {cif_file}, using structure-derived formula: {formula}")
            
            feature_dict = {'MP_ID': cod_id, 'Formula': formula}
            
            # A. 元素特性
            elem_stats = elem_prop.featurize(composition)
            feature_dict['Electronegativity_Diff'] = elem_stats[elem_prop.feature_labels().index('MagpieData range Electronegativity')]
            
            elements = composition.get_el_amt_dict()
            total_atoms = composition.num_atoms
            sorted_elements = sorted(elements.keys())
            a_element = sorted_elements[0]
            feature_dict['Fraction_A'] = elements[a_element] / total_atoms
            feature_dict['Fraction_B'] = sum(amt for el, amt in elements.items() if el != a_element) / total_atoms
            feature_dict['Element_A'] = a_element
            
            feature_dict['Atomic_Radius_Diff'] = elem_stats[elem_prop.feature_labels().index('MagpieData range CovalentRadius')]
            feature_dict['Max_Atomic_Number'] = max(Element(el).number for el in elements.keys())
            
            avg_principal_quantum = sum(principal_quantum_numbers.get(el, Element(el).row) * amt for el, amt in elements.items()) / total_atoms
            feature_dict['Avg_Principal_Quantum'] = avg_principal_quantum
            
            feature_dict['Single_Metal_Heat_Capacity'] = np.nan
            feature_dict['Atom_Count'] = total_atoms
            
            avg_mass = sum(Element(el).atomic_mass * amt for el, amt in elements.items()) / total_atoms
            feature_dict['Avg_Atomic_Mass'] = avg_mass
            feature_dict['Avg_Period'] = sum(Element(el).row * amt for el, amt in elements.items()) / total_atoms
            feature_dict['Avg_Group'] = sum(Element(el).group * amt for el, amt in elements.items()) / total_atoms
            
            # B. 结构环境
            density_features = density_feat.featurize(structure)
            feature_dict['Density'] = density_features[0]
            
            symmetry_data = symmetry_feat.featurize(structure)
            feature_dict['Space_Group'] = symmetry_data[0]
            feature_dict['Crystal_System'] = symmetry_data[1]
            feature_dict['N_Symmetry_Ops'] = symmetry_data[4]
            
            order_params = chem_order.featurize(structure)
            feature_dict['Warren_Cowley_Order'] = order_params[0]
            
            feature_dict['Lattice_a'] = structure.lattice.a
            feature_dict['Lattice_b'] = structure.lattice.b
            feature_dict['Lattice_c'] = structure.lattice.c
            feature_dict['Lattice_alpha'] = structure.lattice.alpha
            feature_dict['Lattice_beta'] = structure.lattice.beta
            feature_dict['Lattice_gamma'] = structure.lattice.gamma
            
            feature_dict['Cell_Volume'] = structure.volume
            feature_dict['Packing_Efficiency'] = density_features[2]
            
            distances = structure.distance_matrix.flatten()
            feature_dict['Min_Interatomic_Distance'] = min(d for d in distances if d > 0)
            
            # C. 配位环境
            vec = sum([valence_electrons.get(el, 0) * amt for el, amt in elements.items()]) / total_atoms
            feature_dict['VEC'] = vec
            feature_dict['WS_Electron_Density'] = feature_dict['VEC'] / structure.volume
            
            cns = [cnn.get_cn(structure, i, use_weights=False) for i in range(len(structure))]
            feature_dict['Mean_CN'] = sum(cns) / len(cns) if cns else 0
            feature_dict['Max_CN'] = max(cns) if cns else 0
            feature_dict['Min_CN'] = min(cns) if cns else 0
            
            bond_lengths = [structure.get_distance(i, j) for i in range(len(structure)) 
                            for j in range(i+1, len(structure)) if structure.get_distance(i, j) > 0]
            feature_dict['Mean_Bond_Length'] = np.mean(bond_lengths) if bond_lengths else np.nan
            feature_dict['Max_Bond_Length'] = np.max(bond_lengths) if bond_lengths else np.nan
            
            cn_counts = {}
            for cn in cns:
                cn_counts[cn] = cn_counts.get(cn, 0) + 1
            most_common_cn = max(cn_counts.items(), key=lambda x: x[1])[0] if cn_counts else 0
            feature_dict['Dominant_CN'] = most_common_cn
            
            features_data.append(feature_dict)
            print(f"Processed {cif_file} successfully")
            
        except Exception as e:
            print(f"Error processing {cif_file}: {str(e)}")
            continue
    
    features_df = pd.DataFrame(features_data)
    print("Features DataFrame columns:", features_df.columns.tolist())
    if features_df.empty:
        raise ValueError("No CIF files were processed successfully. Check the CIF directory or file contents.")
    
    excel_df = pd.read_excel(excel_file)
    print("Excel columns:", excel_df.columns.tolist())
    
    # 检查 Excel 文件中是否有 'ICSD_ID' 或 'COD_ID'
    id_column = 'MP_ID' if 'MP_ID' in excel_df.columns else 'COD_ID'
    if 'Formula' not in excel_df.columns or id_column not in excel_df.columns:
        raise ValueError(f"Excel file must contain 'Formula' and either 'MP_ID' or 'COD_ID' columns. Found: {excel_df.columns.tolist()}")
    
    excel_df['Formula'] = excel_df['Formula'].astype(str)
    excel_df[id_column] = excel_df[id_column].astype(str)
    
    # 使用动态的 ID 列名进行合并
    merged_df = features_df.merge(excel_df[['Formula', id_column]], 
                                 left_on=['Formula', 'MP_ID'], 
                                 right_on=['Formula', id_column], 
                                 how='left')
    
    # 如果 Excel 中的 ID 列名与 'COD_ID' 不同，丢弃多余的列
    if id_column != 'MP_ID':
        merged_df = merged_df.drop(columns=[id_column])
    
    merged_df.to_csv(output_csv, index=False)
    print(f"Features saved to {output_csv}")
    return merged_df

# 使用示例
if __name__ == "__main__":
    cif_dir = "D:/博士生一年级/MP中没有理论计算结果的结构来自ICSD/structures/"#"D:/毕设课题学习/毕设课题学习/毕设论文代码和数据集/Zn10Pd3/"  # 替换为实际路径
    excel_path = "D:/博士生一年级/MP中没有理论计算结果的结构来自ICSD/formula.xlsx"#"D:/毕设课题学习/毕设课题学习/毕设论文代码和数据集/Zn10Pd3_simple_descriptors.xlsx"  # 替换为实际Excel文件路径
    process_cif_files(cif_dir, excel_path)