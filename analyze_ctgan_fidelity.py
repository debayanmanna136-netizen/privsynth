import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon

columns = [
    "age", "workclass", "fnlwgt", "education", "education-num", 
    "marital-status", "occupation", "relationship", "race", "sex", 
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

df_real = pd.read_csv('c:/.vscode/privsynth/data/raw/adult.data', names=columns, skipinitialspace=True, na_values='?')
df_syn = pd.read_csv('c:/.vscode/privsynth/data/synthetic/adult_ctgan.csv', skipinitialspace=True, na_values='?')

num_cols = df_real.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df_real.select_dtypes(include=['object', 'category']).columns.tolist()

print("--- KS Statistic (Numerical) ---")
for col in num_cols:
    real_vals = df_real[col].dropna()
    syn_vals = df_syn[col].dropna()
    stat, p = ks_2samp(real_vals, syn_vals)
    print(f"{col}: KS Stat = {stat:.4f}")

print("\n--- JS Divergence (Categorical) ---")
for col in cat_cols:
    real_counts = df_real[col].value_counts(normalize=True).sort_index()
    syn_counts = df_syn[col].value_counts(normalize=True).sort_index()
    all_cats = list(set(real_counts.index).union(set(syn_counts.index)))
    real_counts = real_counts.reindex(all_cats, fill_value=0)
    syn_counts = syn_counts.reindex(all_cats, fill_value=0)
    js_div = jensenshannon(real_counts, syn_counts)
    print(f"{col}: JS Div = {js_div:.4f}")

print("\n--- Correlation MAE (Numerical) ---")
real_corr = df_real[num_cols].corr()
syn_corr = df_syn[num_cols].corr()
mae_corr = np.abs(real_corr - syn_corr).mean().mean()
print(f"Numerical Correlation MAE: {mae_corr:.4f}")
