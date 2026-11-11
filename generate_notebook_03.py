import nbformat as nbf
import os

os.makedirs('c:/.vscode/privsynth/data/synthetic', exist_ok=True)

nb = nbf.v4.new_notebook()
cells = []

def add_md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def add_code(text):
    cells.append(nbf.v4.new_code_cell(text))

add_md("# 1. Objective\nThis notebook generates our first synthetic dataset using a Gaussian Copula synthesizer from the SDV library. The generator is fitted strictly on the `adult.data` (training) dataset. We will generate approximately the same number of rows as the original training dataset (32,561 rows) and save it for future utility testing.")

add_md("# 2. Load Real Training Data")
add_code("""import pandas as pd
import warnings
warnings.filterwarnings('ignore')

columns = [
    "age", "workclass", "fnlwgt", "education", "education-num", 
    "marital-status", "occupation", "relationship", "race", "sex", 
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

# We load ONLY the training dataset. We parse '?' as NaN.
df = pd.read_csv('../data/raw/adult.data', names=columns, skipinitialspace=True, na_values='?')
print(f"Loaded {len(df)} records from real training dataset.")
df.head()""")

add_md("# 3. Prepare Data for the Generator\nWe use SDV's `SingleTableMetadata` to automatically infer the types of the columns. SDV is highly flexible and natively handles mixed data types and missing values (NaNs).")
add_code("""from sdv.metadata import SingleTableMetadata

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=df)""")

add_md("# 4. Configure Gaussian Copula\nWe initialize the `GaussianCopulaSynthesizer` using our detected metadata.")
add_code("""from sdv.single_table import GaussianCopulaSynthesizer

synthesizer = GaussianCopulaSynthesizer(
    metadata,
    enforce_min_max_values=True,
    enforce_rounding=True
)""")

add_md("# 5. Fit Generator\nWe fit the synthesizer purely on the original training dataset.")
add_code("""# This computes the marginal distributions and correlation matrix
synthesizer.fit(df)""")

add_md("# 6. Generate Synthetic Data\nWe generate the exact same number of rows (32,561) using a fixed seed for reproducibility.")
add_code("""import sdv
print(f"Using SDV version: {sdv.__version__}")

# We use a fixed seed for reproducible generation
import torch
import numpy as np
torch.manual_seed(42)
np.random.seed(42)

synthetic_data = synthesizer.sample(num_rows=len(df))""")

add_md("# 7. Basic Sanity Checks\nWe check dimensions, column names, and a quick summary to ensure the dataset looks structurally identical.")
add_code("""print("--- Real Data Dimensions ---")
print(df.shape)
print("\\n--- Synthetic Data Dimensions ---")
print(synthetic_data.shape)

print("\\n--- Synthetic Data Missing Values ---")
print(synthetic_data.isnull().sum()[synthetic_data.isnull().sum() > 0])

print("\\n--- Synthetic Target Distribution ---")
print(synthetic_data['income'].value_counts(normalize=True))

synthetic_data.head()""")

add_md("# 8. Save Synthetic Dataset\nWe export the dataset to our `data/synthetic` directory.")
add_code("""import os
os.makedirs('../data/synthetic', exist_ok=True)
synthetic_data.to_csv('../data/synthetic/adult_gaussian_copula.csv', index=False)
print("Saved to: data/synthetic/adult_gaussian_copula.csv")""")

add_md("# 9. Initial Observations\n- **Generator**: SDV `GaussianCopulaSynthesizer`.\n- **Data Handling**: Missing values (`?`) were natively passed to the synthesizer as NaNs. The synthesizer learned the marginal distributions and the covariance.\n- **Rows Generated**: 32,561 (matching training data size).\n- **Columns**: 15 (matching training data structure).\n- **Seed/Version**: Fixed seed used. (Version documented in cell output).")

nb.cells = cells
with open('c:/.vscode/privsynth/notebooks/03_gaussian_copula_generation.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
