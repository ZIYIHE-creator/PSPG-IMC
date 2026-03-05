import pandas as pd
from mp_api.client import MPRester

# Read material IDs from CSV file
input_csv = "D:/博士生一年级/Zn10Pd3 test/C_Zn10Pd3.csv"#"D:/毕设课题学习/毕设课题学习/毕设论文代码和数据集/material_ids.csv"  # Replace with your input CSV file name
output_csv = "D:/博士生一年级/Zn10Pd3 test/MP_Zn10Pd3.csv"#"D:/毕设课题学习/毕设课题学习/毕设论文代码和数据集/material_properties.csv"  # Output CSV file name

# Assuming the CSV has a column 'material_id' containing IDs like 'mp-149'
df = pd.read_csv(input_csv)
material_ids = df['MP_ID'].tolist()

# Initialize lists to store results
results = []

# Initialize MPRester with your API key
with MPRester("zs6Ib4YBrCPGcOwQQVIDfa3xsEkRGVS7") as mpr:
    for material_id in material_ids:
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
            'total_magnetization': total_magnetization
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
results_df.to_csv(output_csv, index=False)

print(f"Results saved to {output_csv}")