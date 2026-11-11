import pandas as pd
import sys
import os

# Ensure privsynth is importable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from privsynth.config import CTGAN_DEFAULT_CONFIG
from privsynth.pipeline.pipeline import run_pipeline
from privsynth.evaluation.fidelity import evaluate_fidelity
from privsynth.evaluation.privacy import evaluate_privacy_proximity

print("Loading Adult dataset...")
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num", 
    "marital-status", "occupation", "relationship", "race", "sex", 
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]
df = pd.read_csv('data/raw/adult.data', names=columns, skipinitialspace=True, na_values='?')
df_small = df.head(100) # Fast test

print("Running pipeline (CTGAN small)...")
result = run_pipeline(df_small, generator="ctgan")

print("\\nProfile results:", list(result['profile'].keys()))
print("Validation results:", result['validation'])

print("\\nTesting Fidelity...")
num_cols = df_small.select_dtypes(include=['number']).columns.tolist()
cat_cols = df_small.select_dtypes(exclude=['number']).columns.tolist()
cat_cols = [c for c in cat_cols if c != 'income'] # Exclude target for simple test

fid = evaluate_fidelity(df_small, result['synthetic_data'], num_cols, cat_cols)
print("Fidelity output keys:", list(fid.keys()))

print("\\nTesting Privacy...")
priv = evaluate_privacy_proximity(df_small, result['synthetic_data'], num_cols, cat_cols)
print("Privacy metrics:", priv)

print("\\nTEST SUCCESSFUL!")
