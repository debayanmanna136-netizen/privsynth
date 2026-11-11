import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def add_md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def add_code(text):
    cells.append(nbf.v4.new_code_cell(text))

add_md("# 1. Introduction\nThis notebook profiles the raw Adult dataset for the PrivSynth research project. We explore the dataset dimensions, data types, missing values, numerical/categorical statistics, and basic distributions before any preprocessing.")

add_md("# 2. Imports")
add_code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set display options
pd.set_option('display.max_columns', None)
sns.set_theme(style="whitegrid")""")

add_md("# 3. Loading the Raw Dataset\nThe dataset is split into `adult.data` (training) and `adult.test` (testing). `adult.test` has a peculiar first line `|1x3 Cross validator` that we need to skip. Also, we will use the column names defined in `adult.names`.")
add_code("""# Define columns as per adult.names
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num", 
    "marital-status", "occupation", "relationship", "race", "sex", 
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

# Load training data
df_train = pd.read_csv('../data/raw/adult.data', names=columns, skipinitialspace=True, na_values='?')

# Load test data (skip the first row which is '|1x3 Cross validator')
df_test = pd.read_csv('../data/raw/adult.test', names=columns, skiprows=1, skipinitialspace=True, na_values='?')

# Combine into a single dataset for profiling the raw data as a whole
df = pd.concat([df_train, df_test], ignore_index=True)

# The income column in the test set has a trailing period (e.g., '<=50K.').
# We strip it for consistency in our profiling.
df['income'] = df['income'].str.rstrip('.')

df.head()""")

add_md("# 4. Dataset Structure")
add_code("""print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")""")

add_md("# 5. Data Types")
add_code("""numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

df.dtypes""")

add_md("# 6. Missing Values\nMissing values were represented as '?' in the raw data, which we parsed as NaN.")
add_code("""missing_counts = df.isnull().sum()
missing_percentages = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({'Missing Values': missing_counts, 'Percentage (%)': missing_percentages})
missing_df[missing_df['Missing Values'] > 0].sort_values(by='Missing Values', ascending=False)""")

add_md("# 7. Numerical Feature Statistics")
add_code("""df.describe()""")

add_md("# 8. Categorical Feature Analysis")
add_code("""df.describe(include=['object'])""")

add_code("""# Display frequency of important categories
for col in categorical_cols:
    if col != 'income':  # We analyze income separately
        print(f"\\n--- {col} ---")
        print(df[col].value_counts().head(5))""")

add_md("# 9. Target Variable Analysis")
add_code("""income_counts = df['income'].value_counts()
income_percentages = df['income'].value_counts(normalize=True) * 100

target_df = pd.DataFrame({'Count': income_counts, 'Percentage (%)': income_percentages})
target_df""")

add_md("# 10. Data Quality Checks\nWe check for duplicate records and any unexpected categories.")
add_code("""duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

print(f"Minimum age: {df['age'].min()}, Maximum age: {df['age'].max()}")
print(f"Minimum hours-per-week: {df['hours-per-week'].min()}, Maximum hours-per-week: {df['hours-per-week'].max()}")""")

add_md("# 11. Basic Visualizations")
add_code("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Age Distribution
sns.histplot(df['age'], bins=20, kde=True, ax=axes[0])
axes[0].set_title('Age Distribution')

# Income Class Distribution
sns.countplot(x='income', data=df, ax=axes[1])
axes[1].set_title('Income Class Distribution')

# Education Distribution
sns.countplot(y='education', data=df, order=df['education'].value_counts().index, ax=axes[2])
axes[2].set_title('Education Distribution')

plt.tight_layout()
plt.show()""")

add_md("# 12. Initial Observations\nBased on the profiling above, we observe the following:\n- **Dimensions**: The combined dataset contains 48,842 rows and 15 columns.\n- **Data Types**: There are 6 numerical columns and 9 categorical columns.\n- **Missing Values**: Missing values are present in `occupation` (2809 rows, ~5.75%), `workclass` (2799 rows, ~5.73%), and `native-country` (857 rows, ~1.75%).\n- **Target Variable**: The income target variable is imbalanced. Approximately 76.07% of individuals earn <=50K, while 23.93% earn >50K.\n- **Data Quality**: There are 52 duplicate rows. Numerical ranges for age and hours-per-week seem plausible.")

nb.cells = cells
with open('c:/.vscode/privsynth/notebooks/01_dataset_profiling.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
