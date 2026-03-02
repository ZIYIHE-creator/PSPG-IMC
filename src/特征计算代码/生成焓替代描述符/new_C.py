import pandas as pd
import numpy as np
from pymatgen.core import Structure, Composition
from pymatgen.core.periodic_table import Element
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.structure import (DensityFeatures, 
                                          GlobalSymmetryFeatures,
                                          ChemicalOrdering,
                                          SiteStatsFingerprint
                                         )
from pymatgen.analysis.local_env import VoronoiNN, CrystalNN
import os
import warnings
import re
from mendeleev import element
from matminer.featurizers.site import (
    AverageBondAngle,
    AverageBondLength,
    CoordinationNumber,
    CrystalNNFingerprint,
    LocalPropertyDifference,
    OPSiteFingerprint,
)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen.core.structure")
warnings.filterwarnings("ignore", category=UserWarning, module="matminer.featurizers.composition")
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen.analysis.local_env")

def process_cif_files(cif_directory, output_csv):
    features_data = []
    
    # Initialize feature extractors
    elem_prop = ElementProperty.from_preset("magpie", impute_nan=True)
    density_feat = DensityFeatures()
    symmetry_feat = GlobalSymmetryFeatures()
    chem_order = ChemicalOrdering()
    cnn = CrystalNN()
    
    # Customize SiteStatsFingerprint for multiple statistics
    coord_feat = SiteStatsFingerprint(
        CoordinationNumber(nn=VoronoiNN(weight="area"), use_weights="effective"),
        stats=["mean", "avg_dev", "minimum", "maximum", "range"]
    )
    bond_length_feat = SiteStatsFingerprint(
        AverageBondLength(VoronoiNN()),
        stats=["mean", "avg_dev", "minimum", "maximum", "range"]
    )
    local_diff_feat = SiteStatsFingerprint(
        LocalPropertyDifference.from_preset("ward-prb-2017"),
        stats=["mean", "avg_dev", "minimum", "maximum", "range"]
    )
    print(coord_feat.feature_labels())
    print(bond_length_feat.feature_labels())
    print(local_diff_feat.feature_labels())

    properties = [
        "Number", "MendeleevNumber", "AtomicWeight", "CovalentRadius", 
        "Electronegativity", "NsUnfilled", "NpUnfilled", "NdUnfilled", 
        "NfUnfilled", "NUnfilled", "GSvolume_pa", "GSbandgap", 
        "GSmagmom", "SpaceGroupNumber", "Column", "Row"
    ]
    # Check CIF directory
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
            
            # Extract COD_ID or ICSD_ID
            cod_id_match = re.search(r'\d+$', filename_base)
            cod_id = cod_id_match.group(0) if cod_id_match else filename_base
            
            feature_dict = {'MP_ID': cod_id, 'Formula': formula}
            
            # A. Element Properties (Site Statistics)
            elem_stats = elem_prop.featurize(composition)
            elements = composition.get_el_amt_dict()
            total_atoms = composition.num_atoms
            
            # Calculate properties using mendeleev
            melting_temp_sum = sum(element(el).melting_point * amt for el, amt in elements.items() if element(el).melting_point is not None)
            avg_melting_temp = melting_temp_sum / total_atoms if melting_temp_sum else np.nan
            valence_sum = sum(element(el).nvalence() * amt for el, amt in elements.items())
            avg_valence = valence_sum / total_atoms
            avg_principal_quantum = sum(element(el).ec.max_n() * amt for el, amt in elements.items()) / total_atoms
            
            # Compute coordination features
            coord_features = coord_feat.featurize(structure)
            # Compute bond length features
            bond_length_features = bond_length_feat.featurize(structure)
            # Compute local property difference features
            local_diff_features = local_diff_feat.featurize(structure)
            
            # Compute Voronoi cell volumes for Cell_Volume_Variation
            voronoi_nn = VoronoiNN(weight="area")
            # 遍历每个原子，收集 Voronoi 体积
            voronoi_volumes = []
            for n in range(len(structure)):
                polyhedra = voronoi_nn.get_voronoi_polyhedra(structure, n)
                for poly in polyhedra.values():
                    if poly is not None:
                        voronoi_volumes.append(poly["volume"])
            
            # 计算体积变化（标准差）
            cell_volume_variation = np.std(voronoi_volumes) if voronoi_volumes else np.nan            
            # Compute Warren-Cowley parameters for three shells
            def compute_wc_for_shells(structure, max_shells=3):
                wc_params = []
                for site_idx in range(len(structure)):
                    neighbors = voronoi_nn.get_nn_info(structure, site_idx)
                    shell_sizes = [len(neighbors)]  # First shell
                    for shell in range(2, max_shells + 1):
                        distances = [info["weight"] for info in neighbors]
                        if distances:
                            max_dist = max(distances)
                            shell_neighbors = [n for n in neighbors if max_dist < n["weight"] <= max_dist * shell]
                            shell_sizes.append(len(shell_neighbors))
                    wc = 1 - (sum(shell_sizes) / len(structure)) / total_atoms  # Rough approximation
                    wc_params.append(wc)
                return np.mean(wc_params) if wc_params else np.nan
            
            mean_wc_magnitude = compute_wc_for_shells(structure)
            
            # Map local_diff_features to feature dictionary with all statistics
            local_diff_labels = local_diff_feat.feature_labels()
            local_diff_stat_features = {}
            # 统计量列表
            stats_list = ["mean", "avg_dev", "minimum", "maximum", "range"]
            
            # 需要输出的属性名列表，对应 LocalPropertyDifference 里的属性（保持和你的 properties 对应）
            props_list = [
                "Number", "MendeleevNumber", "AtomicWeight", "MeltingT",
                "Column", "Row", "CovalentRadius", "Electronegativity",
                "NsUnfilled", "NpUnfilled", "NdUnfilled", "NfUnfilled",
                "NUnfilled", "GSvolume_pa", "GSbandgap",
                "GSmagmom", "SpaceGroupNumber"
            ]
            # 2) 遍历所有属性和统计量，批量取值
            for prop in props_list:
                for stat in stats_list:
                    label = f"{stat} local difference in {prop}"
                    try:
                        val = local_diff_features[local_diff_labels.index(label)]
                    except ValueError:
                        val = np.nan
                    key_name = f"{prop}_{stat}"
                    local_diff_stat_features[key_name] = val
            feature_dict.update(local_diff_stat_features)
            # feature_dict.update({
            #     'Effective_Coordination_Number': coord_features[coord_feat.feature_labels().index('mean')],
            #     'Mean_Bond_Length': bond_length_features[bond_length_feat.feature_labels().index('mean')],
            #     'Bond_Length_Variation': bond_length_features[bond_length_feat.feature_labels().index('std_dev')],
            #     'Cell_Volume_Variation': cell_volume_variation,
            #     'Mean_WC_Magnitude': mean_wc_magnitude,
            #     'Packing_Efficiency': density_feat.featurize(structure)[2],
            #     'Atomic_Number': local_diff_features[local_diff_labels.index('mean Number')],
            #     'Mendeleev_Number': local_diff_features[local_diff_labels.index('mean MendeleevNumber')],
            #     'Atomic_Weight': local_diff_features[local_diff_labels.index('mean AtomicWeight')],
            #     'Melting_Temperature': avg_melting_temp,
            #     'Column': local_diff_features[local_diff_labels.index('mean Column')],
            #     'Row': local_diff_features[local_diff_labels.index('mean Row')],
            #     'Covalent_Radius': local_diff_features[local_diff_labels.index('mean CovalentRadius')],
            #     'Electronegativity': local_diff_features[local_diff_labels.index('mean Electronegativity')],
            #     'Valence_Electron_Count': avg_valence,
            #     'NsUnfilled': local_diff_features[local_diff_labels.index('mean NsUnfilled')],
            #     'NpUnfilled': local_diff_features[local_diff_labels.index('mean NpUnfilled')],
            #     'NdUnfilled': local_diff_features[local_diff_labels.index('mean NdUnfilled')],
            #     'NfUnfilled': local_diff_features[local_diff_labels.index('mean NfUnfilled')],
            #     'NUnfilled': local_diff_features[local_diff_labels.index('mean NUnfilled')],
            #     'Ground_State_Volume': local_diff_features[local_diff_labels.index('mean GSvolume_pa')],
            #     'Ground_State_Band_Gap': local_diff_features[local_diff_labels.index('mean GSbandgap')],
            #     'Ground_State_Magnetic_Moment': local_diff_features[local_diff_labels.index('mean GSmagmom')],
            #     'Space_Group_Number': local_diff_features[local_diff_labels.index('mean SpaceGroupNumber')],
            #     'Number_of_Components': len(composition.elements),
            #     'p-norms_of_Component_Fractions': np.nan,
            #     'Fraction_of_Valence_Electrons': avg_valence / total_atoms,
            #     'Can_Form_Ionic_Compound': np.nan,
            #     'Ionic_Character': np.nan,
            #     'Avg_Principal_Quantum': avg_principal_quantum,
            #     'Min_Coordination_Number': coord_features[coord_feat.feature_labels().index('minimum')],
            #     'Max_Coordination_Number': coord_features[coord_feat.feature_labels().index('maximum')],
            #     'Range_Coordination_Number': coord_features[coord_feat.feature_labels().index('range')],
            #     'Avg_Dev_Coordination_Number': coord_features[coord_feat.feature_labels().index('avg_dev')],
            #     'Electronegativity_Diff': local_diff_features[local_diff_labels.index('mean Electronegativity')],
            #     'Atomic_Radius_Diff': local_diff_features[local_diff_labels.index('mean CovalentRadius')]
            # })
            
            # B. Structural Features
            density_features = density_feat.featurize(structure)
            feature_dict.update({
                'Density': density_features[0],
                'Space_Group': symmetry_feat.featurize(structure)[0],
                'Crystal_System': symmetry_feat.featurize(structure)[1],
                'N_Symmetry_Ops': symmetry_feat.featurize(structure)[4],
                'Warren_Cowley_Order': chem_order.featurize(structure)[0],
                'Lattice_a': structure.lattice.a,
                'Lattice_b': structure.lattice.b,
                'Lattice_c': structure.lattice.c,
                'Lattice_alpha': structure.lattice.alpha,
                'Lattice_beta': structure.lattice.beta,
                'Lattice_gamma': structure.lattice.gamma,
                'Cell_Volume': structure.volume
            })
            
            # C. Coordination Environment
            vec = avg_valence
            cns = [cnn.get_cn(structure, i, use_weights=False) for i in range(len(structure))]
            bond_lengths = [structure.get_distance(i, j) for i in range(len(structure)) 
                           for j in range(i+1, len(structure)) if structure.get_distance(i, j) > 0]
            
            feature_dict.update({
                'VEC': vec,
                'WS_Electron_Density': vec / structure.volume,
                'Mean_CN': np.mean(cns) if cns else 0,
                'Max_CN': np.max(cns) if cns else 0,
                'Min_CN': np.min(cns) if cns else 0,
                'Mean_Bond_Length': np.mean(bond_lengths) if bond_lengths else np.nan,
                'Max_Bond_Length': np.max(bond_lengths) if bond_lengths else np.nan,
                'Dominant_CN': max(set(cns), key=cns.count) if cns else 0
            })
            
            features_data.append(feature_dict)
            print(f"Processed {cif_file} successfully")
            
        except Exception as e:
            print(f"Error processing {cif_file}: {str(e)}")
            continue
    
    features_df = pd.DataFrame(features_data)


    if features_df.empty:
        raise ValueError("No CIF files were processed successfully.")
    
    # 直接保存，不做任何合并
    features_df.to_csv(output_csv, index=False)
    print(f"Features saved to {output_csv}")

# Example usage
if __name__ == "__main__":
    cif_dir = "D:/毕设课题学习/毕设课题学习/毕设论文代码和数据集/数据集/模型准确性验证/测试结构/CIF/"#金属间化合物结构文件/Material Project(546个)/CIF/"
    output_csv = "D:/毕设课题学习/毕设课题学习/毕设论文代码和数据集/数据集/模型准确性验证/new_c_for_9_IMCs.csv"
    process_cif_files(cif_dir, output_csv)