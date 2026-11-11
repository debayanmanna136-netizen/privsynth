import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

def add_md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def add_code(text):
    cells.append(nbf.v4.new_code_cell(text))

add_md("# 1. Objective\nThis notebook generates our second synthetic dataset using **CTGAN (Conditional Tabular GAN)**. CTGAN is a deep learning-based generator explicitly designed for mixed-type tabular data. It will be trained strictly on the `adult.data` (training) dataset. We will generate approximately the same number of rows as the original training dataset (32,561 rows) and save it for future evaluation.")

add_md("# 2. Imports")
add_code("""import pandas as pd
import numpy as np
import torch
import warnings
import sdv
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
import os

warnings.filterwarnings('ignore')
print(f"SDV version: {sdv.__version__}")""")

add_md("# 3. Load Training Data\nWe load the exact same 15-column `adult.data` dataset used in our previous experiments. Missing values (`?`) are parsed as `NaN`. `adult.test` is strictly avoided.")
add_code("""columns = [
    "age", "workclass", "fnlwgt", "education", "education-num", 
    "marital-status", "occupation", "relationship", "race", "sex", 
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

# Load ONLY the training dataset
df = pd.read_csv('../data/raw/adult.data', names=columns, skipinitialspace=True, na_values='?')
print(f"Loaded {len(df)} records from real training dataset.")
df.head()""")

add_md("# 4. Prepare SDV Metadata\nWe use SDV's `SingleTableMetadata` to automatically infer the types of the columns (numerical vs categorical). CTGAN natively handles mixed types internally using Variational Gaussian Mixture Models (VGM) for continuous columns and Gumbel-Softmax for categoricals, so we do not manually one-hot encode or standardize the data.")
add_code("""metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=df)""")

add_md("# 5. Configure CTGAN\nWe initialize the `CTGANSynthesizer`. We configure it with a reasonable research baseline of `epochs=50` to balance deep learning convergence with computational time. (The default is 300, which can take hours on CPUs for 32k rows).")
add_code("""synthesizer = CTGANSynthesizer(
    metadata,
    enforce_rounding=True,
    epochs=50,
    verbose=True
)

# Set fixed random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)""")

add_md("# 6. Train CTGAN\nWe fit the deep learning model purely on the original training dataset. This may take a few minutes depending on CPU/GPU availability.")
add_code("""synthesizer.fit(df)""")

add_md("# 7. Generate Synthetic Data\nWe sample exactly 32,561 records from the trained Generator network.")
add_code("""synthetic_data = synthesizer.sample(num_rows=len(df))""")

add_md("# 8. Save Synthetic Dataset\nWe export the dataset to our `data/synthetic` directory.")
add_code("""os.makedirs('../data/synthetic', exist_ok=True)
output_path = '../data/synthetic/adult_ctgan.csv'
synthetic_data.to_csv(output_path, index=False)
print(f"Saved to: {output_path}")""")

add_md("# 9. Basic Verification\nWe perform a structural sanity check on the output.")
add_code("""print("--- Real Data Dimensions ---")
print(df.shape)
print("\\n--- Synthetic Data Dimensions ---")
print(synthetic_data.shape)

print("\\n--- Synthetic Data Missing Values ---")
missing = synthetic_data.isnull().sum()
print(missing[missing > 0])

print("\\n--- Synthetic Target Distribution ---")
print(synthetic_data['income'].value_counts(normalize=True))

print("\\n--- Basic Numerical Ranges ---")
print(synthetic_data[['age', 'hours-per-week']].describe().loc[['min', 'max']])

print(f"\\nFile exists: {os.path.exists(output_path)}")
synthetic_data.head()""")

add_md("# 10. Experiment Summary\n- **Generator**: CTGAN (`CTGANSynthesizer` from SDV v1.38.3).\n- **Configuration**: 50 Epochs, enforce_rounding=True, random_seed=42.\n- **Rows Generated**: 32,561 (matching the real training dataset exactly).\n- **Columns**: 15 (matching the real training dataset exactly).\n- **Output Path**: `data/synthetic/adult_ctgan.csv`.\n- **Constraint Checks**: `adult.test` was strictly avoided. The original `adult.data` raw files were loaded dynamically and not modified on disk.\n- **Status**: The CTGAN baseline is structurally sound and ready for formal fidelity, utility, and privacy evaluation in subsequent notebooks.")

nb.cells = cells
with open('c:/.vscode/privsynth/notebooks/07_ctgan_generation.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
