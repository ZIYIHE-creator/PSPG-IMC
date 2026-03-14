# Enabling data‑efficient prediction of Gibbs free energies of formation for intermetallic compounds with PSPG‑IMC

This repository contains **PSPG‑IMC**, a lightweight and physically guided machine‑learning workflow designed to predict the **Gibbs free energy of formation ($\Delta$$G_{f}$)** for intermetallic compounds (IMCs).  
The pipeline leverages **XGBoost ensemble models** and **physics‑informed feature engineering**, enabling accurate predictions on **small datasets** without requiring additional DFT calculations.

---

## 📂 Dataset

The dataset used in PSPG‑IMC includes:

- **Target values (ΔG_f)** generated from **CALPHAD** calculations and thermodynamic parameters extracted from the literature  
- **Descriptors** derived from crystal structures and compositions  
- Additional engineered features based on atomic statistics, alloy rules, and physical heuristics  

All dataset files are provided in the `Datasets/` directory.

---

## 🧹 Feature Extraction

Feature extraction is performed using the modules in the `Featurizer/` directory.

Each `.py` module can:

- Automatically read structure files (e.g., CIF)  
- Parse composition and structural information  
- Generate descriptors such as C, PLMF, new_PLMF, or MP features  

To extract features, run the corresponding featurizer script.  
For example:

python Featurizer/PLMF/new_PLMF.py

Users may freely combine or replace descriptors by selecting modules from: `Model training/cp_app.descriptors`

---

## 🧠 Model Training & Prediction

The main workflow is demonstrated in: `Model training/main.ipynb` 

This notebook includes:

- Training an XGBoost model using the **C + new_PLMF_200D** descriptor set  
- Predicting ΔG_f for **7 out‑of‑distribution (OOD)** intermetallic compounds  
- Visualizing model performance and feature importance  

If you wish to use **different descriptors**:

1. Select alternative descriptor modules from `cp_app.descriptors`  
2. Extract features for your dataset  
3. Replace the feature file in `main.ipynb`  
4. Retrain the model following the same workflow  

For **new structures**, simply:

1. Run the appropriate featurizer to generate descriptors  
2. Load the trained model or retrain a new one  
3. Perform prediction in the notebook  

---

## 📬 Contact

If you have any questions, please contact:

**Ziyi He**  
Email: *zy-he25@mails.tsinghua.edu.cn*






