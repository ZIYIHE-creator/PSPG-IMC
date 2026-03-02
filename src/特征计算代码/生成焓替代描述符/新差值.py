# -*- coding: utf-8 -*-
import json
import os
import pandas as pd
from pymatgen.core import Structure
import mendeleev
import numpy as np
from itertools import combinations

# Properties list for difference calculations and individual properties
diff_properties = [
    'Atomic radius', 'V_molar', 'a', 'χ_pualing', 'IP1', 'Z', 'α_p', 'ΔH_atom', 'φ(V)', 'nws^1/3'
]

# ATOMIC_PROPERTIES dictionary (unchanged)
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

def load_additional_properties(xlsx_file):
    """Load chemical potential (mu) and WS electron density (rho_ws) from an XLSX file."""
    try:
        df = pd.read_excel(xlsx_file)
        required_columns = ['Atom', 'φ(V)', 'nws^1/3']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"XLSX file {xlsx_file} must contain columns: {required_columns}")
        additional_props = {row['Atom']: {'φ(V)': row['φ(V)'], 'nws^1/3': row['nws^1/3']} for _, row in df.iterrows()}
        return additional_props
    except Exception as e:
        print(f"Error reading XLSX file {xlsx_file}: {str(e)}")
        return {}

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

def get_element_property(element_symbol, prop, pseudo_radii=None, additional_props=None, json_data=None):
    """Get a specified property for an element, including pseudopotential radii, chemical potential, WS electron density, and JSON data."""
    if json_data and prop in ['Melting point', 'Boiling point', 'Electron affinity']:
        if element_symbol in json_data:
            element_data = json_data[element_symbol]
            return element_data.get(prop, None)

    if prop in ['r_s', 'r_p', 'r_d']:
        if pseudo_radii and element_symbol in pseudo_radii:
            return pseudo_radii[element_symbol].get(prop, None)
        return None

    if prop in ['φ(V)', 'nws^1/3']:
        if additional_props and element_symbol in additional_props:
            return additional_props[element_symbol].get(prop, None)
        return None

    try:
        elem = mendeleev.element(element_symbol)
        value = None
        if prop == 'Atomic radius':
            value = elem.atomic_radius_rahm
        elif prop == 'V_molar':
            value = elem.atomic_volume
        elif prop == 'a':
            value = elem.lattice_constant
        elif prop == 'χ_pualing':
            value = elem.electronegativity_pauling()
        elif prop == 'IP1':
            value = elem.ionenergies.get(1, None)
        elif prop == 'Z':
            value = elem.atomic_number
        elif prop == 'α_p':
            value = elem.dipole_polarizability
        elif prop == 'ΔH_atom':
            value = elem.heat_of_formation

        if value is not None and value != 0:
            return value
    except Exception as e:
        print(f"Warning: Error retrieving property '{prop}' for element '{element_symbol}' from mendeleev: {str(e)}")

    if element_symbol in ATOMIC_PROPERTIES and prop in ATOMIC_PROPERTIES[element_symbol]:
        value = ATOMIC_PROPERTIES[element_symbol][prop]
        if value is not None:
            return value

    print(f"Warning: Property '{prop}' not available for element '{element_symbol}'. Returning None.")
    return None

def get_elements_from_cif(cif_file):
    """Extract the first two elements (alphabetically sorted) from a CIF file."""
    try:
        structure = Structure.from_file(cif_file)
        elements = sorted([str(elem) for elem in structure.composition.elements])[:2]
        if len(elements) < 2:
            print(f"Error: CIF file {cif_file} contains fewer than 2 elements. Skipping.")
            return []
        return elements
    except Exception as e:
        print(f"Error reading CIF file {cif_file}: {str(e)}")
        return []

def extract_mp_id(cif_file):
    """Extract MP_ID from CIF filename (assuming format like mp-12345.cif)."""
    try:
        base_name = os.path.splitext(cif_file)[0]
        if base_name.startswith('mp-'):
            return base_name
        return base_name
    except Exception as e:
        print(f"Error extracting MP_ID from {cif_file}: {str(e)}")
        return 'Unknown'

def calculate_property_differences(cif_file, properties, pseudo_radii, additional_props, json_data=None):
    """Calculate property differences and collect individual properties for the two selected elements."""
    elements = get_elements_from_cif(cif_file)
    if len(elements) < 2:
        return None

    # Collect properties for the two elements
    element_properties = {}
    for elem in elements:
        element_properties[elem] = {}
        for prop in properties:
            value = get_element_property(elem, prop, pseudo_radii, additional_props, json_data)
            element_properties[elem][prop] = value

    # Calculate property differences
    results = {
        'element_properties': element_properties,
        'property_differences': {}
    }
    elem1, elem2 = elements
    for prop in properties:
        y_i = element_properties[elem1].get(prop)
        y_j = element_properties[elem2].get(prop)
        diff = abs(y_i - y_j) if y_i is not None and y_j is not None else None
        results['property_differences'][prop] = {'max_diff': diff}

    return results

def main():
    # File paths
    json_file_path = 'D:/毕设课题学习/毕设课题学习/毕设论文代码和数据集/数据集/property_of_periodic_table_from_pymatgen.json'
    cif_dir = "D:/毕设课题学习/毕设课题学习/毕设论文代码和数据集/数据集/金属间化合物结构文件/Material Project(546个)/CIF/"
    pseudo_radii_xlsx = "D:/毕设课题学习/毕设课题学习/毕设论文代码和数据集/数据集/zunger pseudopotentials radii.xlsx"
    additional_props_xlsx = "D:/毕设课题学习/毕设课题学习/毕设论文代码和数据集/数据集/化学势和ws电子密度.xlsx"
    output_csv = "D:/毕设课题学习/毕设课题学习/毕设论文代码和数据集/property_differences_with_mu_rho_ws.csv"

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

    # Load pseudopotential radii and additional properties
    pseudo_radii = load_pseudo_radii(pseudo_radii_xlsx)
    additional_props = load_additional_properties(additional_props_xlsx)

    # Process CIF files
    cif_files = [f for f in os.listdir(cif_dir) if f.endswith('.cif')]
    print(f"\nCIF files found: {cif_files}")
    results = {}

    if not cif_files:
        print("No .cif files found in the specified directory.")
        return

    for cif_file in cif_files:
        try:
            full_path = os.path.join(cif_dir, cif_file)
            print(f"\nProcessing: {full_path}")
            structure = Structure.from_file(full_path)
            formula = str(structure.composition.reduced_formula)
            mp_id = extract_mp_id(cif_file)
            prop_data = calculate_property_differences(full_path, diff_properties, pseudo_radii, additional_props, json_data)
            if prop_data:
                results[cif_file] = {
                    'mp_id': mp_id,
                    'formula': formula,
                    'element_properties': prop_data['element_properties'],
                    'property_differences': prop_data['property_differences']
                }
        except Exception as e:
            print(f"Error processing {cif_file}: {str(e)}")
            continue

    # Define fixed column headers
    units = {
        'Atomic radius': 'ang', 'V_molar': 'cm^3/mol', 'a': 'ang', 'χ_pualing': '',
        'IP1': 'eV', 'Z': '', 'α_p': 'bohr^3', 'ΔH_atom': 'kJ/mol', 'φ(V)': 'eV', 'nws^1/3': 'electrons/ang^3'
    }
    columns = ['MP_ID', 'Formula']
    columns.extend([f'{prop}_max_diff ({units.get(prop, "")})' for prop in diff_properties])
    columns.extend([f'{prop}_Elem1 ({units.get(prop, "")})' for prop in diff_properties])
    columns.extend([f'{prop}_Elem2 ({units.get(prop, "")})' for prop in diff_properties])

    # Prepare data for CSV
    csv_data = []
    for cif_file, data in results.items():
        row = {'MP_ID': data['mp_id'], 'Formula': data['formula']}
        elements = sorted(data['element_properties'].keys())
        if len(elements) != 2:
            print(f"Warning: {cif_file} does not contain exactly 2 elements. Skipping.")
            continue

        elem1, elem2 = elements
        # Add property differences
        for prop in diff_properties:
            prop_data = data['property_differences'].get(prop, {})
            unit = units.get(prop, '')
            row[f'{prop}_max_diff ({unit})'] = prop_data.get('max_diff', 'Not available')

        # Add individual properties for Elem1
        for prop in diff_properties:
            unit = units.get(prop, '')
            value = data['element_properties'][elem1].get(prop)
            row[f'{prop}_Elem1 ({unit})'] = value if value is not None else 'Not available'

        # Add individual properties for Elem2
        for prop in diff_properties:
            unit = units.get(prop, '')
            value = data['element_properties'][elem2].get(prop)
            row[f'{prop}_Elem2 ({unit})'] = value if value is not None else 'Not available'

        csv_data.append(row)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(csv_data, columns=columns)
    try:
        df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"\nResults saved to {output_csv}")
    except Exception as e:
        print(f"Error saving CSV file {output_csv}: {str(e)}")

    # Print results
    print("\nResults Summary:")
    for cif_file, data in results.items():
        elements = sorted(data['element_properties'].keys())
        if len(elements) != 2:
            continue
        elem1, elem2 = elements
        print(f"\nCIF File: {cif_file}")
        print(f"MP_ID: {data['mp_id']}")
        print(f"Formula: {data['formula']}")
        print("Property Differences:")
        for prop, diff in data['property_differences'].items():
            unit = units.get(prop, '')
            print(f"  {prop}_max_diff: {diff['max_diff'] if diff['max_diff'] is not None else 'Not available'} {unit}")
        print(f"Element 1 ({elem1}) Properties:")
        for prop in diff_properties:
            unit = units.get(prop, '')
            value = data['element_properties'][elem1].get(prop)
            print(f"  {prop}: {value if value is not None else 'Not available'} {unit}")
        print(f"Element 2 ({elem2}) Properties:")
        for prop in diff_properties:
            unit = units.get(prop, '')
            value = data['element_properties'][elem2].get(prop)
            print(f"  {prop}: {value if value is not None else 'Not available'} {unit}")

if __name__ == "__main__":
    main()