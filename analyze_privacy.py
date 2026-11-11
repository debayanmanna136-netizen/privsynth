import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors

columns = [
    "age", "workclass", "fnlwgt", "education", "education-num", 
    "marital-status", "occupation", "relationship", "race", "sex", 
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

df_real = pd.read_csv('c:/.vscode/privsynth/data/raw/adult.data', names=columns, skipinitialspace=True, na_values='?')
df_syn = pd.read_csv('c:/.vscode/privsynth/data/synthetic/adult_gaussian_copula.csv', skipinitialspace=True, na_values='?')

num_cols = df_real.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df_real.select_dtypes(include=['object', 'category']).columns.tolist()

# 1. Prep data (Standardize and OHE)
num_transformer = Pipeline(steps=[('scaler', StandardScaler())])
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

# Fit only on real data
X_real_prep = preprocessor.fit_transform(df_real)
X_syn_prep = preprocessor.transform(df_syn)

# 2. Nearest Neighbors: Synthetic -> Real
nn_syn = NearestNeighbors(n_neighbors=1, metric='euclidean', n_jobs=-1)
nn_syn.fit(X_real_prep)
distances_syn, indices_syn = nn_syn.kneighbors(X_syn_prep)
d_syn = distances_syn.flatten()

# 3. Nearest Neighbors: Real -> Real
nn_real = NearestNeighbors(n_neighbors=2, metric='euclidean', n_jobs=-1)
nn_real.fit(X_real_prep)
distances_real, indices_real = nn_real.kneighbors(X_real_prep)
# The closest point is itself (dist=0). The second closest is n_neighbors=2
d_real = distances_real[:, 1]

def print_stats(name, d):
    print(f"--- {name} ---")
    print(f"Mean: {np.mean(d):.4f}")
    print(f"Median: {np.median(d):.4f}")
    print(f"Min: {np.min(d):.4f}")
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    for p in percentiles:
        print(f"P{p}: {np.percentile(d, p):.4f}")

print_stats("Synthetic -> Real", d_syn)
print_stats("Real -> Real", d_real)

# Find 5 most similar synthetic records
print("\n--- Top 5 closest Synthetic records to Real ---")
closest_idx = np.argsort(d_syn)[:5]
for i in closest_idx:
    print(f"Syn idx: {i}, Dist: {d_syn[i]:.4f}")
