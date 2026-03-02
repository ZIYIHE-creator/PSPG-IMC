import json
import os
import pandas as pd
from pymatgen.core import Structure, Element
import mendeleev
import numpy as np

# Unified properties list, removed 'Electron affinity' since 'Electron affinity' is included
# Added new electronegativities: Allen, Allred-Rochow, Gordy (using Ghosh as proxy), Martynov-Batsanov (replacing original MB calculation), Mulliken
# Added new radii: metallic_radius_c12, vdw_radius_alvarez
# Note: crystal_radius is part of Shannon radii, already handled; not adding as separate property
properties = [
    'Ionic radii', 'Atomic radius', 'Shannon radii', 'Melting point', 'Boiling point', 'Electron affinity',
    'group', 'period', 'mass', 'IP1', 'IP2', 'IP3', 'ΔH_fusion', 'ΔH_vapor', 'λ',
    'V_molar', 'r_cov', 'χ_pualing', 'Z_eff', 'ΔH_atom', 'α_p', 'r_vdW',
    'C', 'ρ', 'η', 'Nv', 'r_s', 'r_p', 'r_d',
    'χ_allen', 'χ_allred_rochow', 'χ_gordy', 'χ_martynov_batsanov', 'χ_mulliken',
    'metallic_radius_c12', 'vdw_radius_alvarez'
]

# ATOMIC_PROPERTIES dictionary (unchanged)
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
    'Dy': {'group': 3, 'ΔH_fusion': 11.06, 'ΔH_vapor': 280, 'IP1': 5.94, 'Electron affinity': 0.0},
    'Er': {'group': 3, 'ΔH_fusion': 19.90, 'ΔH_vapor': 280, 'IP1': 6.11, 'Electron affinity': 0.0},
    'Eu': {'group': 3, 'ΔH_fusion': 9.21, 'ΔH_vapor': 176, 'χ_pualing': 1.2, 'IP1': 5.67, 'Electron affinity': 0.86},
    'Gd': {'group': 3, 'ΔH_fusion': 10.05, 'ΔH_vapor': 301.3, 'λ': 10.6, 'Electron affinity': 0.0, 'η': 3.0749, 'IP1': 6.15},
    'Mg': {'Electron affinity': 0.0, 'η': 3.82312, 'IP1': 7.65},
    'Ca': {'λ': 200, 'IP1': 6.11, 'Electron affinity': 0.02},
    'Sr': {'λ': 35.3, 'IP1': 5.70, 'Electron affinity': 0.05},
    'Ba': {'λ': 18.4, 'IP1': 5.21, 'Electron affinity': 0.14},
    'Mn': {'λ': 7.82, 'Electron affinity': 0.0, 'η': 3.71701, 'IP1': 7.43},
    'Zn': {'Electron affinity': 0.0, 'η': 4.6971, 'IP1': 9.39},
    'Y': {'λ': 17.2, 'IP1': 6.22, 'Electron affinity': 0.31},
    'Ru': {'ΔH_vapor': 591.6, 'IP1': 7.36, 'Electron affinity': 1.05},
    'Cd': {'Electron affinity': 0.0, 'η': 4.4969, 'IP1': 8.99},
    'Ce': {'group': 3, 'IP1': 5.54, 'Electron affinity': 0.55},
    'Pr': {'group': 3, 'IP1': 5.47, 'Electron affinity': 0.96},
    'Nd': {'group': 3, 'λ': 16.5, 'IP1': 5.53, 'Electron affinity': 1.92},
    'Sm': {'group': 3, 'λ': 13.3, 'Electron affinity': 0.15, 'η': 2.746855, 'IP1': 5.64},
    'Tb': {'group': 3, 'ΔH_fusion': 10.15, 'χ_pualing': 1.2, 'IP1': 5.86, 'Electron affinity': 0.0},
    'Ho': {'group': 3, 'ΔH_fusion': 16.8, 'λ': 16.2, 'Electron affinity': 0.0, 'η': 3.01075, 'IP1': 6.02},
    'Tm': {'group': 3, 'ΔH_fusion': 16.84, 'λ': 17.0, 'IP1': 6.18, 'Electron affinity': 0.0},
    'Yb': {'group': 3, 'λ': 38.5, 'χ_pualing': 1.1, 'IP1': 6.25, 'Electron affinity': 0.0},
    'Lu': {'group': 3, 'ΔH_fusion': 22, 'λ': 16.4, 'IP1': 5.43, 'Electron affinity': 0.0},
    'Th': {'group': 3, 'λ': 54.0, 'Electron affinity': 0.608, 'η': 2.736, 'IP1': 6.31},
    'U': {'group': 3, 'Electron affinity': 0.309, 'η': 2.94255, 'IP1': 6.19}
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
                        return float(value)  # Ensure the value is numeric
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
            value = elem.en_pauling
        elif prop == 'Z_eff':
            value = elem.zeff()
        elif prop == 'ΔH_atom':
            value = elem.heat_of_formation
        elif prop == 'α_p':
            value = elem.dipole_polarizability
        elif prop == 'r_vdW':
            value = elem.vdw_radius
        elif prop == 'C':
            value = elem.specific_heat
        elif prop == 'ρ':
            value = elem.density
        elif prop == 'η':
            value = elem.hardness()
        elif prop == 'Nv':
            value = elem.nvalence()
        elif prop == 'χ_allen':
            value = elem.en_allen
        elif prop == 'χ_allred_rochow':
            value = elem.electronegativity('allred-rochow')
        elif prop == 'χ_gordy':
            value = elem.electronegativity('gordy')
        elif prop == 'χ_martynov_batsanov':
            value = elem.electronegativity(scale='martynov-batsanov')
        elif prop == 'χ_mulliken':
            value = elem.electronegativity('mulliken')
        elif prop == 'metallic_radius_c12':
            value = elem.metallic_radius_c12
        elif prop == 'vdw_radius_alvarez':
            value = elem.vdw_radius_alvarez

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

def get_elements_from_cif(cif_file):
    """Extract unique elements from a CIF file."""
    try:
        structure = Structure.from_file(cif_file)
        elements = [str(elem) for elem in structure.composition.elements]
        return elements
    except Exception as e:
        print(f"Error reading CIF file {cif_file}: {str(e)}")
        return []

def get_composition_proportions(cif_file):
    """Get composition proportions from a CIF file."""
    try:
        structure = Structure.from_file(cif_file)
        composition = structure.composition
        total_atoms = sum(composition.as_dict().values())
        proportions = {str(elem): composition[str(elem)] / total_atoms for elem in composition.elements}
        return proportions
    except Exception as e:
        print(f"Error processing composition for {cif_file}: {str(e)}")
        return {}

def get_properties_from_cif(cif_file, properties, pseudo_radii, json_data=None):
    """Get properties for elements in a CIF file, including proportions."""
    elements = get_elements_from_cif(cif_file)
    proportions = get_composition_proportions(cif_file)
    results = {}
    for elem in elements:
        results[elem] = {
            'proportion': proportions.get(elem, 0.0),
            **{prop: get_element_property(elem, prop, pseudo_radii, json_data) for prop in properties}
        }
    return results

def calculate_statistics(prop_data, excluded_props=['Ionic radii', 'Shannon radii']):
    """Calculate proportion-weighted mean and max diff for each property (excluding specified ones)."""
    stats = {}
    elements = list(prop_data.keys())
    if len(elements) < 2:
        print("Warning: Less than 2 elements; max_diff will be 0.")
    
    for prop in properties:
        if prop in excluded_props:
            continue  # Skip excluded properties
        
        values = []
        proportions = []
        for elem in elements:
            value = prop_data[elem].get(prop, None)
            if value is not None and isinstance(value, (int, float)):
                values.append(value)
                proportions.append(prop_data[elem]['proportion'])
        
        if len(values) > 0:
            # Proportion-weighted mean
            if sum(proportions) > 0:
                weighted_mean = np.average(values, weights=proportions)
            else:
                weighted_mean = np.mean(values) if len(values) > 0 else None
            
            # Max diff (max - min)
            max_diff = max(values) - min(values) if len(values) >= 2 else 0
            
            stats[prop] = {
                'weighted_mean': weighted_mean,
                'max_diff': max_diff
            }
        else:
            stats[prop] = {
                'weighted_mean': None,
                'max_diff': None
            }
    
    return stats

def save_statistics_to_csv(results, output_csv='D:/毕设课题学习/毕设课题学习/毕设论文代码和数据集/数据集/模型准确性验证/new_c_3.csv'):
    """Save calculated statistics for all compounds to a CSV file."""
    data = []
    for cif_file, data_dict in results.items():
        formula = data_dict['formula']
        stats = data_dict.get('statistics', {})
        
        row = {'formula': formula}
        for prop, stat_values in stats.items():
            row[f'{prop}_weighted_mean'] = stat_values['weighted_mean']
            row[f'{prop}_max_diff'] = stat_values['max_diff']
        
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Statistics saved to {output_csv}")

def main():
    # File paths (replace with actual paths)
    json_file_path = 'D:/毕设课题学习/毕设课题学习/毕设论文代码和数据集/数据集/property_of_periodic_table_from_pymatgen.json'
    cif_dir = "D:/毕设课题学习/毕设课题学习/毕设论文代码和数据集/数据集/模型准确性验证/测试结构/CIF/"#金属间化合物结构文件/Material Project(546个)/CIF/"
    xlsx_file = "D:/毕设课题学习/毕设课题学习/毕设论文代码和数据集/数据集/zunger pseudopotentials radii.xlsx"
    element_symbol = 'Zn'  # Element for testing JSON-based property extraction

    # Load JSON data
    json_data = {}
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
        print(f"Successfully loaded JSON file: {json_file_path}")
        # Test Electron affinity retrieval for a specific element
        if element_symbol in json_data:
            ea_value = json_data[element_symbol].get('Electron affinity', None)
            print(f"Electron affinity for {element_symbol} from JSON: {ea_value}")
        else:
            print(f"Element {element_symbol} not found in JSON data.")
    except FileNotFoundError:
        print(f"Warning: JSON file {json_file_path} not found. Skipping JSON-based property extraction.")
    except json.JSONDecodeError:
        print("Warning: Invalid JSON format. Skipping JSON-based property extraction.")
    except Exception as e:
        print(f"Warning: Error loading JSON file: {str(e)}. Skipping JSON-based property extraction.")

    # Load pseudopotential radii
    pseudo_radii = load_pseudo_radii(xlsx_file)

    # Process CIF files
    cif_files = [f for f in os.listdir(cif_dir) if f.endswith('.cif')]
    print(f"\nCIF files found: {cif_files}")
    results = {}

    if not cif_files:
        print("No .cif files found in the specified directory.")
        return

    for cif_file in cif_files:
        try:
            filename_base = cif_file.replace('.cif', '')
            full_path = os.path.join(cif_dir, cif_file)
            print(f"\nProcessing: {full_path}")
            structure = Structure.from_file(full_path)
            composition = structure.composition
            formula = str(composition.reduced_formula)
            prop_data = get_properties_from_cif(full_path, properties, pseudo_radii, json_data)
            if prop_data:
                stats = calculate_statistics(prop_data)
                results[cif_file] = {
                    'formula': formula,
                    'properties': prop_data,
                    'statistics': stats
                }
        except Exception as e:
            print(f"Error processing {cif_file}: {str(e)}")
            continue

    # Save statistics to CSV
    save_statistics_to_csv(results)

    # Print CIF processing results
    print("\nCIF Processing Results:")
    for cif_file, data in results.items():
        print(f"\nCIF File: {cif_file}")
        print(f"Formula: {data['formula']}")
        print("Properties:")
        for elem, props in data['properties'].items():
            print(f"  {elem}:")
            for prop, value in props.items():
                if prop == 'Shannon radii' and value:
                    print(f"    {prop}:")
                    for oxidation_state in value:
                        for coordination in value[oxidation_state]:
                            crystal_radius = value[oxidation_state][coordination].get('', {}).get('crystal_radius', None)
                            ionic_radius = value[oxidation_state][coordination].get('', {}).get('ionic_radius', None)
                            print(f"      Oxidation state {oxidation_state}, Coordination {coordination}:")
                            print(f"        Crystal radius: {crystal_radius if crystal_radius is not None else 'Not available'} ang")
                            print(f"        Ionic radius: {ionic_radius if ionic_radius is not None else 'Not available'} ang")
                else:
                    unit = {
                        'Ionic radii': 'ang', 'Atomic radius': 'pm', 'Melting point': 'K',
                        'Boiling point': 'K', 'Electron affinity': 'eV', 'mass': 'amu',
                        'IP1': 'eV', 'IP2': 'eV', 'IP3': 'eV', 'ΔH_fusion': 'kJ/mol',
                        'ΔH_vapor': 'kJ/mol', 'λ': 'W/m·K', 'V_molar': 'cm³/mol',
                        'r_cov': 'pm', 'χ_pualing': '', 'Z_eff': '', 'ΔH_atom': 'kJ/mol',
                        'α_p': 'a.u.', 'r_vdW': 'pm', 'C': 'J/g·K', 'ρ': 'g/cm³',
                        'η': 'eV', 'Nv': '', 'r_s': 'a.u.', 'r_p': 'a.u.', 'r_d': 'a.u.',
                        'χ_allen': '', 'χ_allred_rochow': '', 'χ_gordy': '', 'χ_martynov_batsanov': '',
                        'χ_mulliken': '', 'metallic_radius_c12': 'pm', 'vdw_radius_alvarez': 'pm'
                    }.get(prop, '')
                    print(f"    {prop}: {value if value is not None else 'Not available'} {unit}")
        print("Statistics:")
        for prop, stat in data['statistics'].items():
            print(f"  {prop}:")
            print(f"    Weighted Mean: {stat['weighted_mean']}")
            print(f"    Max Diff: {stat['max_diff']}")

if __name__ == "__main__":
    main()