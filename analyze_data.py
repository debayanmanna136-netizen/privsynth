import pandas as pd
import numpy as np

columns = [
    "age", "workclass", "fnlwgt", "education", "education-num", 
    "marital-status", "occupation", "relationship", "race", "sex", 
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

df_train = pd.read_csv('c:/.vscode/privsynth/data/raw/adult.data', names=columns, skipinitialspace=True, na_values='?')
df_test = pd.read_csv('c:/.vscode/privsynth/data/raw/adult.test', names=columns, skiprows=1, skipinitialspace=True, na_values='?')
df = pd.concat([df_train, df_test], ignore_index=True)
df['income'] = df['income'].str.rstrip('.')

print(f"ROWS: {df.shape[0]}")
print(f"COLS: {df.shape[1]}")

numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"NUM_COLS: {len(numerical_cols)}")
print(f"CAT_COLS: {len(categorical_cols)}")

missing = df.isnull().sum()
missing = missing[missing > 0]
for col, val in missing.items():
    print(f"MISSING: {col} {val} {val/len(df)*100:.2f}%")

print(f"INCOME_<=50K: {(df['income'] == '<=50K').sum()} {(df['income'] == '<=50K').sum()/len(df)*100:.2f}%")
print(f"INCOME_>50K: {(df['income'] == '>50K').sum()} {(df['income'] == '>50K').sum()/len(df)*100:.2f}%")

print(f"DUPLICATES: {df.duplicated().sum()}")
