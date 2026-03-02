# -*- coding: utf-8 -*-
import pandas as pd
import json
from mp_api.client import MPRester
from pymatgen.core import Element

# Read material IDs from CSV file
input_csv = "D:/博士生一年级/SISSO计算对比/test_sets.csv"  # Replace with your input CSV file name
output_csv = "D:/博士生一年级/SISSO计算对比/test_sets_features.csv"  # Output CSV file name


# Assuming the CSV has a column 'material_id' containing IDs like 'mp-149'
df = pd.read_csv(input_csv)
material_ids = df['MP_ID'].tolist()

# Initialize lists to store results
results = []

# Initialize MPRester with your API key
with MPRester("zs6Ib4YBrCPGcOwQQVIDfa3xsEkRGVS7") as mpr:
    for material_id in material_ids:
        # Query summary for structure-related properties
        summary_docs = mpr.materials.summary.search(
            material_ids=[material_id],
            fields=[
                "volume",
                "nsites",
                "structure"
            ]
        )
        
        # Query thermodynamic properties
        thermo_docs = mpr.materials.thermo.search(
            material_ids=[material_id],
            fields=[
                "formation_energy_per_atom",
                "energy_per_atom"
            ]
        )
        
        # Query magnetic properties
        mag_docs = mpr.materials.magnetism.search(
            material_ids=[material_id],
            fields=["total_magnetization"]
        )
        
        if not summary_docs:
            continue
            
        summary_doc = summary_docs[0]
        volume = summary_doc.volume
        num_atoms = summary_doc.nsites
        structure = summary_doc.structure
        
        # Compute reduced atomic mass and get element atomic masses
        comp = structure.composition
        reduced_comp = comp.reduced_composition
        elements = [str(el) for el in reduced_comp.elements]
        alphas = [reduced_comp[el] for el in elements]
        N = len(elements)
        
        element_masses = {el: float(Element(el).atomic_mass) for el in elements}
        
        x_red = None
        if N >= 2:
            sum_alpha = sum(alphas)
            sum_harm = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    a_i = alphas[i]
                    a_j = alphas[j]
                    x_i = element_masses[elements[i]]
                    x_j = element_masses[elements[j]]
                    if x_i + x_j > 0:
                        harm = (x_i * x_j) / (x_i + x_j)
                        term = (a_i + a_j) * harm
                        sum_harm += term
            if sum_alpha > 0 and (N - 1) > 0:
                x_red = sum_harm / ((N - 1) * sum_alpha)
        
        # Extract thermodynamic properties
        thermo_doc = thermo_docs[0] if thermo_docs else None
        formation_energy = thermo_doc.formation_energy_per_atom if thermo_doc else None
        energy_per_atom = thermo_doc.energy_per_atom if thermo_doc else None
        
        # Extract magnetic properties
        mag_doc = mag_docs[0] if mag_docs else None
        total_magnetization = mag_doc.total_magnetization if mag_doc else None
        
        # Store results
        results.append({
            'material_id': material_id,
            'formation_energy_per_atom': formation_energy,
            'energy_per_atom': energy_per_atom,
            'total_magnetization': total_magnetization,
            'volume': volume,
            'num_atoms': num_atoms,
            'reduced_atomic_mass': x_red,
            'formula': structure.composition.reduced_formula,
            'element_masses': json.dumps(element_masses)
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
results_df.to_csv(output_csv, index=False)

print(f"Results saved to {output_csv}")