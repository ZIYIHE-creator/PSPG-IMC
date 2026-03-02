import json
import os
import pandas as pd
from pymatgen.core import Structure, Element
import mendeleev
import numpy as np
from itertools import combinations

# Unified properties list for difference calculations
diff_properties = [
    #'period', 'group', 'Atomic radius', 'r_cov', 'r_s+r_p', 'r_vdW', 'λ',
    #'Melting point', 'Boiling point', 'Electron affinity', 'χ_pualing', 'χ_MB',
    'ΔH_atom','α_p'
]

# ATOMIC_PROPERTIES dictionary (unchanged from original)
ATOMIC_PROPERTIES = {
    'Al': {
        'group': 13, 'period': 3, 'mass': 26.98, 'IP1': 5.99, 'IP2': 18.83, 'IP3': 28.45,
        'ΔH_fusion': 10.71, 'ΔH_vapor': 294.0, 'λ': 237.0, 'V_molar': 10.0, 'r_cov': 1.21,
        'χ_pualing': 1.61, 'Z_eff': 3.0, 'ΔH_atom': 330.0, 'EA': 0.43, 'α_p': 6.8, 'r_vdW': 1.84,
        'C': 0.897, 'ρ': 2.70
    },
    'Ni': {
        'group': 10, 'period': 4, 'mass': 58.69, 'IP1': 7.64, 'IP2': 18.17, 'IP3': 35.19,
        'ΔH_fusion': 17.48, 'ΔH_vapor': 378.0, 'λ': 90.9, 'V_molar': 6.59, 'r_cov': 1.24,
        'χ_pualing': 1.91, 'Z_eff': 4.0, 'ΔH_atom': 430.0, 'EA': 1.16, 'α_p': 6.4, 'r_vdW': 1.63,
        'C': 0.444, 'ρ': 8.90
    },
    'Ti': {
        'group': 4, 'period': 4, 'mass': 47.87, 'IP1': 6.83, 'IP2': 13.58, 'IP3': 27.49,
        'ΔH_fusion': 14.15, 'ΔH_vapor': 425.0, 'λ': 21.9, 'V_molar': 10.64, 'r_cov': 1.60,
        'χ_pualing': 1.54, 'Z_eff': 3.0, 'ΔH_atom': 473.0, 'EA': 0.08, 'α_p': 8.0, 'r_vdW': 2.15,
        'C': 0.523, 'ρ': 4.51
    },
    'Dy': {'group': 3, 'ΔH_fusion': 11.06, 'ΔH_vapor': 280, 'IP1': 5.94, 'EA': 0.0},
    'Er': {'group': 3, 'ΔH_fusion': 19.90, 'ΔH_vapor': 280, 'IP1': 6.11, 'EA': 0.0},
    'Eu': {'group': 3, 'ΔH_fusion': 9.21, 'ΔH_vapor': 176, 'χ_pualing': 1.2, 'IP1': 5.67, 'EA': 0.86},
    'Gd': {'group': 3, 'ΔH_fusion': 10.05, 'ΔH_vapor': 301.3, 'λ': 10.6, 'EA': 0.0, 'η': 3.0749, 'IP1': 6.15},
    'Mg': {'EA': 0.0, 'η': 3.82312, 'IP1': 7.65},
    'Ca': {'λ': 200, 'IP1': 6.11, 'EA': 0.02},
    'Sr': {'λ': 35.3, 'IP1': 5.70, 'EA': 0.05},
    'Ba': {'λ': 18.4, 'IP1': 5.21, 'EA': 0.14},
    'Mn': {'λ': 7.82, 'EA': 0.0, 'η': 3.71701, 'IP1': 7.43},
    'Zn': {'EA': 0.0, 'η': 4.6971, 'IP1': 9.39},
    'Y': {'λ': 17.2, 'IP1': 6.22, 'EA': 0.31},
    'Ru': {'ΔH_vapor': 591.6, 'IP1': 7.36, 'EA': 1.05},
    'Cd': {'EA': 0.0, 'η': 4.4969, 'IP1': 8.99},
    'Ce': {'group': 3, 'IP1': 5.54, 'EA': 0.55},
    'Pr': {'group': 3, 'IP1': 5.47, 'EA': 0.96},
    'Nd': {'group': 3, 'λ': 16.5, 'IP1': 5.53, 'EA': 1.92},
    'Sm': {'group': 3, 'λ': 13.3, 'EA': 0.15, 'η': 2.746855, 'IP1': 5.64},
    'Tb': {'group': 3, 'ΔH_fusion': 10.15, 'χ_pualing': 1.2, 'IP1': 5.86, 'EA': 0.0},
    'Ho': {'group': 3, 'ΔH_fusion': 16.8, 'λ': 16.2, 'EA': 0.0, 'η': 3.01075, 'IP1': 6.02},
    'Tm': {'group': 3, 'ΔH_fusion': 16.84, 'λ': 17.0, 'IP1': 6.18, 'EA': 0.0},
    'Yb': {'group': 3, 'λ': 38.5, 'χ_pualing': 1.1, 'IP1': 6.25, 'EA': 0.0},
    'Lu': {'group': 3, 'ΔH_fusion': 22, 'λ': 16.4, 'IP1': 5.43, 'EA': 0.0},
    'Th': {'group': 3, 'λ': 54.0, 'EA': 0.608, 'η': 2.736, 'IP1': 6.31},
    'U': {'group': 3, 'EA': 0.309, 'η': 2.94255, 'IP1': 6.19}
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
    if json_data and prop in ['Melting point', 'Boiling point', 'Electron affinity']:
        if element_symbol in json_data:
            element_data = json_data[element_symbol]
            return element_data.get(prop, None)

    # Handle r_s + r_p
    if prop == 'r_s+r_p':
        if pseudo_radii and element_symbol in pseudo_radii:
            r_s = pseudo_radii[element_symbol].get('r_s', None)
            r_p = pseudo_radii[element_symbol].get('r_p', None)
            if r_s is not None and r_p is not None:
                return r_s + r_p
        return None

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
        elif prop == 'ΔH_atom':
            value = elem.heat_of_formation
        elif prop == 'α_p':
            value = elem.dipole_polarizability
        elif prop == 'r_cov':
            value = elem.covalent_radius_cordero
        elif prop == 'r_vdW':
            value = elem.vdw_radius
        elif prop == 'λ':
            value = elem.thermal_conductivity
        elif prop == 'χ_pualing':
            value = elem.electronegativity_pauling()
        elif prop == 'EA':
            value = elem.electron_affinity
        elif prop == 'χ_MB':
            ip1 = elem.ionenergies.get(1, 0)
            ea = elem.electron_affinity
            if ip1 is not None and ea is not None and ip1 != 0:
                value = (ip1 + ea) / 2
            else:
                ip1 = ATOMIC_PROPERTIES.get(element_symbol, {}).get('IP1', 0)
                ea = ATOMIC_PROPERTIES.get(element_symbol, {}).get('EA', 0)
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

def calculate_property_differences(cif_file, properties, pseudo_radii, json_data=None):
    """Calculate property differences for elements in a CIF file based on stoichiometric formulas."""
    elements = get_elements_from_cif(cif_file)
    proportions = get_composition_proportions(cif_file)
    if len(elements) < 2:
        print(f"Error: CIF file {cif_file} contains fewer than 2 elements. Skipping.")
        return None

    N = len(elements)
    # Calculate sum of beta_i (composition proportions)
    beta_sum = sum(proportions.get(elem, 0.0) for elem in elements)
    if beta_sum == 0:
        print(f"Warning: Sum of beta_i is zero for {cif_file}. Using 1 as fallback.")
        beta_sum = 1
    results = {}

    # Collect properties for all elements
    element_properties = {}
    for elem in elements:
        element_properties[elem] = {}
        for prop in properties:
            value = get_element_property(elem, prop, pseudo_radii, json_data)
            element_properties[elem][prop] = value

    # Generate all unique pairs for red and diff calculations
    element_pairs = list(combinations(elements, 2))

    for prop in properties:
        # Calculate stoichiometric weighted mean (avg)
        avg_numerator=0
        for elem in elements :
            if element_properties[elem].get(prop) is not None :
                avg_numerator += proportions.get(elem, 0.0) * element_properties[elem].get(prop, 0.0)                                 
            else : avg_numerator = None ;break
        
        if beta_sum != 0 and avg_numerator != None: 
            avg = (1 / beta_sum) * avg_numerator 
        else : avg = None
        # Calculate stoichiometric weighted harmonic mean (red)
        red_numerator = 0
        valid_pairs = 0
        for elem1, elem2 in element_pairs:
            β_i = proportions.get(elem1, 0.0)
            β_j = proportions.get(elem2, 0.0)
            y_i = element_properties[elem1].get(prop)
            y_j = element_properties[elem2].get(prop)
            if y_i is not None and y_j is not None and (β_i + β_j) != 0:
                red_numerator += ((y_i + y_j) / (β_i + β_j)) * β_i * β_j
                valid_pairs += 1
            else : red_numerator = None ;break
        red = (1 / ((N-1) * beta_sum)) * red_numerator if beta_sum != 0 and N > 1 and valid_pairs > 0 else None

        # Calculate stoichiometric weighted mean difference (diff)
        max_diff = None
        for elem1, elem2 in element_pairs:
            β_i = proportions.get(elem1, 0.0)
            β_j = proportions.get(elem2, 0.0)
            y_i = element_properties[elem1].get(prop)
            y_j = element_properties[elem2].get(prop)
            if y_i is not None and y_j is not None:
                diff_numerator = (β_i + β_j) * abs(y_i - y_j)
                diff_value = (1 / ((N-1) * beta_sum)) * diff_numerator if beta_sum != 0 and N > 1 else None
                if diff_value is not None:
                    max_diff = diff_value if max_diff is None else max(max_diff, diff_value)
        diff = max_diff

        results[prop] = {
            'avg': avg,
            'red': red,
            'diff': diff
        }

    return results

def main():
    # File paths (replace with actual paths)
    json_file_path = 'D:/毕设课题学习/毕设课题学习/毕设论文代码和数据集/数据集/property_of_periodic_table_from_pymatgen.json'
    cif_dir = "D:/毕设课题学习/毕设课题学习/毕设论文代码和数据集/数据集/金属间化合物结构文件/Material Project(546个)/CIF/"
    xlsx_file = "D:/毕设课题学习/毕设课题学习/毕设论文代码和数据集/数据集/zunger pseudopotentials radii.xlsx"
    output_csv = "D:/毕设课题学习/毕设课题学习/毕设论文代码和数据集/property_differences_of_MP2.csv"

    # Load JSON data
    json_data = {}
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
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
            prop_diff = calculate_property_differences(full_path, diff_properties, pseudo_radii, json_data)
            if prop_diff:
                results[cif_file] = {
                    'formula': formula,
                    'property_differences': prop_diff
                }
        except Exception as e:
            print(f"Error processing {cif_file}: {str(e)}")
            continue

    # Prepare data for CSV
    csv_data = []
    units = {
        'period': '', 'group': '', 'Atomic radius': 'ang', 'r_cov': 'pm',
        'r_s+r_p': 'a.u.', 'r_vdW': 'pm', 'λ': 'W/m·K',
        'Melting point': 'K', 'Boiling point': 'K', 'Electron affinity': 'eV',
        'χ_pualing': '', 'χ_MB': 'eV','ΔH_atom':'kJ/mol','α_p':'bohr^3'
    }

    for cif_file, data in results.items():
        row = {'CIF File': cif_file, 'Formula': data['formula']}
        for prop in diff_properties:
            prop_data = data['property_differences'].get(prop, {})
            unit = units.get(prop, '')
            row[f'{prop}_avg ({unit})'] = prop_data.get('avg', 'Not available')
            row[f'{prop}_red ({unit})'] = prop_data.get('red', 'Not available')
            row[f'{prop}_diff ({unit})'] = prop_data.get('diff', 'Not available')
        csv_data.append(row)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(csv_data)
    try:
        df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"\nResults saved to {output_csv}")
    except Exception as e:
        print(f"Error saving CSV file {output_csv}: {str(e)}")

    # Print results
    print("\nProperty Differences Results:")
    for cif_file, data in results.items():
        print(f"\nCIF File: {cif_file}")
        print(f"Formula: {data['formula']}")
        print("Property Differences:")
        for prop, diff in data['property_differences'].items():
            print(f"  Property: {prop}")
            unit = units.get(prop, '')
            print(f"    Weighted Mean (avg): "
                  f"{diff['avg'] if diff['avg'] is not None else 'Not available'} {unit}")
            print(f"    Harmonic Mean (red): "
                  f"{diff['red'] if diff['red'] is not None else 'Not available'} {unit}")
            print(f"    Mean Difference (diff): "
                  f"{diff['diff'] if diff['diff'] is not None else 'Not available'} {unit}")

if __name__ == "__main__":
    main()